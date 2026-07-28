"""Two-stage uncertainty-weighted GAT training.

Stage 1: Load pre-trained base quantile model, extract widths for all nodes.
         Compute CQR-adjusted widths using val set qhat.
Stage 2: Train a new UncertaintyGAT that uses 1/width as confidence weights
         in the attention mechanism.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.temporal_dataset import TemporalDatasetGlobalSplit
from credipred.gnn.model import Model
from credipred.gnn.uncertainty_gat import UncertaintyGATModel
from credipred.utils.args import DataArguments, ModelArguments
from credipred.utils.checkpoint import snapshot_state_dict
from credipred.utils.logger import Logger


def _quantile_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Pinball loss for quantile regression (mid, lower, upper)."""
    mid, lower, upper = preds[:, 0], preds[:, 1], preds[:, 2]
    loss_mid = F.l1_loss(mid, targets)

    residual_lower = targets - lower
    loss_lower = torch.mean(
        torch.where(
            residual_lower >= 0, alpha * residual_lower, (alpha - 1) * residual_lower
        )
    )
    residual_upper = targets - upper
    loss_upper = torch.mean(
        torch.where(
            residual_upper >= 0,
            (1 - alpha) * residual_upper,
            -alpha * residual_upper,
        )
    )
    return loss_mid + loss_lower + loss_upper


def _train_epoch(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    alpha: float,
) -> Tuple[float, List[float], List[float]]:
    """Train one epoch. Returns (loss, pred_scores, target_scores)."""
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    n_batches = 0
    pred_scores: List[float] = []
    target_scores: List[float] = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_conf = batch.confidence.to(device)
        preds = model(batch.x, batch.edge_index, confidence=batch_conf)
        n_seed = batch.batch_size
        seed_preds = preds[:n_seed]
        seed_targets = batch.y[:n_seed]
        loss = _quantile_loss(seed_preds, seed_targets, alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        for p in seed_preds[:, 0]:
            pred_scores.append(p.item())
        for t in seed_targets:
            target_scores.append(t.item())
    return total_loss / n_batches, pred_scores, target_scores


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    mask_name: str,
    alpha: float,
) -> Tuple[float, float, float]:
    """Evaluate. Returns (loss, mean_baseline_loss, random_baseline_loss)."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_mean_loss = 0
    total_random_loss = 0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        batch_conf = batch.confidence.to(device)
        preds = model(batch.x, batch.edge_index, confidence=batch_conf)
        targets = batch.y
        n_seed = batch.batch_size
        mask = getattr(batch, mask_name)[:n_seed]
        if mask.sum() == 0:
            continue
        seed_preds = preds[:n_seed]
        seed_targets = targets[:n_seed]

        loss = _quantile_loss(seed_preds[mask], seed_targets[mask], alpha)
        mean_preds = torch.full(seed_targets[mask].size(), 0.5).to(device)
        random_preds = torch.rand(seed_targets[mask].size(0)).to(device)
        mean_loss = F.l1_loss(mean_preds, seed_targets[mask])
        random_loss = F.l1_loss(random_preds, seed_targets[mask])

        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_random_loss += random_loss.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_mean_loss / n_batches,
        total_random_loss / n_batches,
    )


def _extract_base_widths_and_confidence(
    data,
    model_arguments,
    weight_directory,
    split_idx,
    device,
    alpha,
):
    """Stage 1: Load base model, extract CQR-adjusted widths, compute confidence."""
    logging.info('=== Stage 1: Extracting widths from base quantile model ===')
    base_model = Model(
        model_name=model_arguments.model,
        normalization=model_arguments.normalization,
        in_channels=data.num_features,
        hidden_channels=model_arguments.hidden_channels,
        out_channels=model_arguments.embedding_dimension,
        num_layers=model_arguments.num_layers,
        dropout=model_arguments.dropout,
        binary=False,
        prediction_dim=3,
    ).to(device)

    base_path = weight_directory / model_arguments.model / 'best_model.pt'
    if not base_path.exists():
        raise FileNotFoundError(f'Base model not found at {base_path}')
    state_dict = torch.load(base_path, map_location=device, weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    logging.info('Loaded base model from %s', base_path)

    # Extract predictions for ALL nodes
    all_nodes = torch.arange(data.num_nodes)
    all_preds = torch.zeros(data.num_nodes, 3, device='cpu')

    extract_batch_size = max(model_arguments.batch_size * 8, 8192)
    extract_loader = NeighborLoader(
        data,
        input_nodes=all_nodes,
        num_neighbors=model_arguments.num_neighbors,
        batch_size=extract_batch_size,
        shuffle=False,
        num_workers=4,
    )

    with torch.no_grad():
        for batch in tqdm(extract_loader, desc='Extracting base widths'):
            batch = batch.to(device)
            preds = base_model(batch.x, batch.edge_index)
            n_seed = batch.batch_size
            original_ids = batch.n_id[:n_seed]
            all_preds[original_ids] = preds[:n_seed].cpu()

    # Compute CQR-adjusted widths
    from credipred.conformal_regression.cqr import compute_cqr_scores, compute_qhat

    val_idx = split_idx['valid']
    labels = data.y.cpu()
    cal_scores = compute_cqr_scores(
        all_preds[val_idx, 1],
        all_preds[val_idx, 2],
        labels[val_idx],
    )
    qhat = compute_qhat(cal_scores, alpha)
    logging.info('CQR qhat from val set: %.4f', qhat)

    raw_widths = all_preds[:, 2] - all_preds[:, 1]
    widths = raw_widths + 2 * qhat

    # Normalize widths → confidence
    widths_norm = (widths - widths.min()) / (widths.max() - widths.min() + 1e-8)
    eps = 0.01
    confidence = 1.0 / (widths_norm + eps)
    confidence = (confidence - confidence.min()) / (
        confidence.max() - confidence.min() + 1e-8
    )

    data.confidence = confidence

    logging.info(
        'Raw width stats: mean=%.4f std=%.4f min=%.4f max=%.4f',
        raw_widths.mean(),
        raw_widths.std(),
        raw_widths.min(),
        raw_widths.max(),
    )
    logging.info(
        'CQR-adjusted width stats: mean=%.4f std=%.4f min=%.4f max=%.4f',
        widths.mean(),
        widths.std(),
        widths.min(),
        widths.max(),
    )
    logging.info(
        'Confidence stats: mean=%.4f std=%.4f min=%.4f max=%.4f',
        confidence.mean(),
        confidence.std(),
        confidence.min(),
        confidence.max(),
    )

    del base_model
    torch.cuda.empty_cache()

    return all_preds, labels, qhat


def run_uncertainty_gat(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: TemporalDatasetGlobalSplit,
) -> None:
    """Train uncertainty-weighted GAT using base model's interval widths."""
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    device = torch.device(
        f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu',
    )

    alpha = model_arguments.quantile_alpha

    # ---- Stage 1: Extract widths and confidence ----
    all_preds, labels, qhat = _extract_base_widths_and_confidence(
        data,
        model_arguments,
        weight_directory,
        split_idx,
        device,
        alpha,
    )

    # ---- Stage 2: Train UncertaintyGAT (same structure as gnn_experiment) ----
    logging.info('=== Stage 2: Training Uncertainty-weighted GAT ===')

    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=split_idx['valid'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx['test'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    logger = Logger(model_arguments.runs)
    loss_tuple_run: List[List[Tuple[float, float, float, float, float]]] = []
    final_avg_preds: List[List[float]] = []
    final_avg_targets: List[List[float]] = []
    global_best_val_loss = float('inf')
    best_state_dict = None

    logging.info('*** Training ***')
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model = UncertaintyGATModel(
            in_channels=data.num_features,
            hidden_channels=model_arguments.hidden_channels,
            out_channels=model_arguments.embedding_dimension,
            num_layers=model_arguments.num_layers,
            dropout=model_arguments.dropout,
            prediction_dim=3,
            normalization=model_arguments.normalization,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_arguments.lr,
            weight_decay=model_arguments.weight_decay,
        )

        loss_tuple_epoch: List[Tuple[float, float, float, float, float]] = []
        epoch_avg_preds: List[List[float]] = []
        epoch_avg_targets: List[List[float]] = []

        for epoch in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            _, batch_preds, batch_targets = _train_epoch(
                model, train_loader, optimizer, alpha
            )
            epoch_avg_preds.append(batch_preds)
            epoch_avg_targets.append(batch_targets)

            train_loss, _, _ = _evaluate(model, train_loader, 'train_mask', alpha)
            valid_loss, valid_mean_loss, valid_random_loss = _evaluate(
                model,
                val_loader,
                'valid_mask',
                alpha,
            )
            test_loss, test_mean_loss, test_random_loss = _evaluate(
                model,
                test_loader,
                'test_mask',
                alpha,
            )

            result = (
                train_loss,
                valid_loss,
                test_loss,
                test_mean_loss,
                test_random_loss,
            )
            loss_tuple_epoch.append(result)
            logger.add_result(
                run,
                (train_loss, valid_loss, test_loss, valid_mean_loss, valid_random_loss),
            )

            if valid_loss < global_best_val_loss:
                global_best_val_loss = valid_loss
                best_state_dict = snapshot_state_dict(model)

        from credipred.utils.plot import mean_across_lists

        final_avg_preds.append(mean_across_lists(epoch_avg_preds))
        final_avg_targets.append(mean_across_lists(epoch_avg_targets))
        loss_tuple_run.append(loss_tuple_epoch)

    # Save best model
    save_dir = weight_directory / model_arguments.model
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'uncertainty_gat_model.pt'
    torch.save(best_state_dict, save_path)
    logging.info('Saved uncertainty GAT model to %s', save_path)

    # Statistics (same format as gnn_experiment.py)
    logging.info('*** Statistics ***')
    logging.info(logger.get_statistics())
    logging.info(logger.get_avg_statistics())
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds,
            targets=final_avg_targets,
            percent=10,
        )
    )
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds,
            targets=final_avg_targets,
            percent=5,
        )
    )
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds,
            targets=final_avg_targets,
            percent=1,
        )
    )

    logging.info('Saving pkl of results')
    from credipred.utils.save import save_loss_results

    save_loss_results(loss_tuple_run, model_arguments.model, 'uncertainty_gat')

    # ---- Post-hoc CQR evaluation: Base vs UQ-GAT ----
    logging.info('=== Post-hoc CQR Evaluation ===')
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    train_idx = split_idx['train']
    all_labeled = torch.cat([train_idx, val_idx, test_idx])

    uq_preds = torch.zeros(data.num_nodes, 3, device='cpu')
    eval_loader = NeighborLoader(
        data,
        input_nodes=all_labeled,
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=False,
        num_workers=4,
    )
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Extracting UQ-GAT preds'):
            batch = batch.to(device)
            batch_conf = batch.confidence.to(device)
            preds = model(batch.x, batch.edge_index, confidence=batch_conf)
            n_seed = batch.batch_size
            original_ids = batch.n_id[:n_seed]
            uq_preds[original_ids] = preds[:n_seed].cpu()

    labels_test = labels[test_idx].numpy()
    mid_test = uq_preds[test_idx, 0].numpy()
    lower_test = uq_preds[test_idx, 1].numpy()
    upper_test = uq_preds[test_idx, 2].numpy()
    widths_test = upper_test - lower_test
    mae_test = np.abs(mid_test - labels_test).mean()

    base_lower = all_preds[test_idx, 1].numpy()
    base_upper = all_preds[test_idx, 2].numpy()
    base_mid = all_preds[test_idx, 0].numpy()
    base_widths = base_upper - base_lower
    base_mae = np.abs(base_mid - labels_test).mean()

    logging.info('=== Comparison: Base vs Uncertainty-GAT ===')
    logging.info('  Base GAT:  MAE=%.4f avg_width=%.4f', base_mae, base_widths.mean())
    logging.info('  UQ-GAT:    MAE=%.4f avg_width=%.4f', mae_test, widths_test.mean())

    from credipred.conformal_regression.cqr import evaluate_intervals

    base_metrics = evaluate_intervals(all_preds, labels, val_idx, test_idx, alpha)
    uq_metrics = evaluate_intervals(uq_preds, labels, val_idx, test_idx, alpha)

    logging.info('=== CQR Comparison ===')
    logging.info(
        '  Base CQR:  coverage=%.4f width=%.4f mae=%.4f qhat=%.4f',
        base_metrics.coverage,
        base_metrics.avg_width,
        base_metrics.mae,
        base_metrics.qhat,
    )
    logging.info(
        '  UQ-GAT CQR: coverage=%.4f width=%.4f mae=%.4f qhat=%.4f',
        uq_metrics.coverage,
        uq_metrics.avg_width,
        uq_metrics.mae,
        uq_metrics.qhat,
    )

    # Per-quintile comparison
    logging.info('=== Per-quintile Coverage (CQR-adjusted) ===')
    for name, preds_tensor, qhat_val in [
        ('Base', all_preds, base_metrics.qhat),
        ('UQ-GAT', uq_preds, uq_metrics.qhat),
    ]:
        lower_adj = preds_tensor[test_idx, 1].numpy() - qhat_val
        upper_adj = preds_tensor[test_idx, 2].numpy() + qhat_val
        w = upper_adj - lower_adj
        covered = (labels_test >= lower_adj) & (labels_test <= upper_adj)
        order = np.argsort(w)
        n = len(w)
        q_size = n // 5
        logging.info('  %s:', name)
        for q in range(5):
            s, e = q * q_size, ((q + 1) * q_size if q < 4 else n)
            q_idx = order[s:e]
            logging.info(
                '    Q%d: cov=%.4f width=%.4f n=%d',
                q + 1,
                covered[q_idx].mean(),
                w[q_idx].mean(),
                len(q_idx),
            )
