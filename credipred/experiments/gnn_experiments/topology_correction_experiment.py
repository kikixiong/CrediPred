"""Post-hoc topology correction for quantile regression.

Loads a pre-trained base quantile model, extracts frozen [mid, lower, upper]
predictions for every node, then trains a small CorrectionGNN that outputs
an additive delta:

    corrected = base_preds + delta

The correction network sees only the 3-dim base predictions as node features
and propagates them along the original graph edges.  Training uses pinball
loss so the correction can tighten or shift prediction intervals.

Finally, CQR evaluation compares base vs corrected intervals.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.conformal_regression.correction_gnn import CorrectionGNN
from credipred.dataset.temporal_dataset import TemporalDatasetGlobalSplit
from credipred.gnn.model import Model
from credipred.utils.args import DataArguments, ModelArguments
from credipred.utils.checkpoint import snapshot_state_dict
from credipred.utils.logger import Logger


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _quantile_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Pinball loss for [mid, lower, upper]."""
    mid, lower, upper = preds[:, 0], preds[:, 1], preds[:, 2]
    loss_mid = F.l1_loss(mid, targets)

    residual_lower = targets - lower
    loss_lower = torch.mean(
        torch.where(
            residual_lower >= 0,
            alpha * residual_lower,
            (alpha - 1) * residual_lower,
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


# ---------------------------------------------------------------------------
# Extract base predictions
# ---------------------------------------------------------------------------


def _extract_base_predictions(
    data,
    model_arguments: ModelArguments,
    weight_directory: Path,
    device: torch.device,
) -> torch.Tensor:
    """Load base quantile model and extract [mid, lower, upper] for all nodes."""
    logging.info('=== Loading base quantile model ===')
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
        raise FileNotFoundError(f'Base quantile model not found at {base_path}')
    state_dict = torch.load(base_path, map_location=device, weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    logging.info('Loaded base model from %s', base_path)

    all_preds = torch.zeros(data.num_nodes, 3, device='cpu')
    extract_loader = NeighborLoader(
        data,
        input_nodes=torch.arange(data.num_nodes),
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=False,
        num_workers=4,
    )

    with torch.no_grad():
        for batch in tqdm(extract_loader, desc='Extracting base predictions'):
            batch = batch.to(device)
            preds = base_model(batch.x, batch.edge_index)
            n_seed = batch.batch_size
            original_ids = batch.n_id[:n_seed]
            all_preds[original_ids] = preds[:n_seed].cpu()

    logging.info(
        'Base prediction stats — mid: mean=%.4f, lower: mean=%.4f, upper: mean=%.4f',
        all_preds[:, 0].mean(),
        all_preds[:, 1].mean(),
        all_preds[:, 2].mean(),
    )
    widths = all_preds[:, 2] - all_preds[:, 1]
    logging.info(
        'Raw interval width: mean=%.4f std=%.4f min=%.4f max=%.4f',
        widths.mean(),
        widths.std(),
        widths.min(),
        widths.max(),
    )

    del base_model
    torch.cuda.empty_cache()
    return all_preds


# ---------------------------------------------------------------------------
# Train / evaluate correction
# ---------------------------------------------------------------------------


def _size_loss_regression(
    corrected: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """CQR-based size loss: minimize conformal-adjusted interval width.

    Following original CF-GNN: compute qhat on a random calibration split
    within the batch, then penalize (upper + qhat) - (lower - qhat).
    """
    lower = corrected[:, 1]
    upper = corrected[:, 2]

    # Conformal scores
    cal_scores = torch.maximum(targets - upper, lower - targets)
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = torch.quantile(cal_scores, q_level, interpolation='higher')

    # Size loss: mean adjusted interval width
    return torch.mean((upper + qhat) - (lower - qhat))


def _reg_loss(
    corrected: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    """Regularization loss: penalize deviation of bounds from base predictions."""
    lower_dev = F.mse_loss(corrected[:, 1], base[:, 1])
    upper_dev = F.mse_loss(corrected[:, 2], base[:, 2])
    return lower_dev + upper_dev


def _train_correction_epoch(
    correction_model: CorrectionGNN,
    train_loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    epoch: int = 0,
    pred_only_epochs: int = 0,
    size_loss_weight: float = 1.0,
    reg_loss_weight: float = 1.0,
) -> Tuple[float, List[float], List[float]]:
    """Train one epoch. Returns (loss, pred_scores, target_scores)."""
    correction_model.train()
    device = next(correction_model.parameters()).device
    total_loss = 0
    n_batches = 0
    pred_scores: List[float] = []
    target_scores: List[float] = []
    use_conformal_loss = epoch > pred_only_epochs

    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)

        n_seed = batch.batch_size
        base = batch.base_preds[:n_seed].to(device)  # [n_seed, 3]
        targets = batch.y[:n_seed]

        # Correction GNN takes base_preds as node features
        delta = correction_model(batch.base_preds.to(device), batch.edge_index)
        seed_delta = delta[:n_seed]
        corrected = base + seed_delta

        # Prediction loss (pinball)
        pred_loss = _quantile_loss(corrected, targets, alpha)

        if use_conformal_loss:
            s_loss = _size_loss_regression(corrected, targets, alpha)
            r_loss = _reg_loss(corrected, base)
            loss = pred_loss + size_loss_weight * s_loss + reg_loss_weight * r_loss
        else:
            loss = pred_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        for p in corrected[:, 0]:
            pred_scores.append(p.item())
        for t in targets:
            target_scores.append(t.item())

    return total_loss / n_batches, pred_scores, target_scores


@torch.no_grad()
def _evaluate_correction(
    correction_model: CorrectionGNN,
    loader: NeighborLoader,
    mask_name: str,
    alpha: float,
) -> Tuple[float, float, float]:
    """Evaluate correction model. Returns (loss, mean_baseline, random_baseline)."""
    correction_model.eval()
    device = next(correction_model.parameters()).device
    total_loss = 0
    total_mean_loss = 0
    total_random_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        n_seed = batch.batch_size
        mask = getattr(batch, mask_name)[:n_seed]
        if mask.sum() == 0:
            continue

        base = batch.base_preds[:n_seed].to(device)
        targets = batch.y[:n_seed]

        delta = correction_model(batch.base_preds.to(device), batch.edge_index)
        seed_delta = delta[:n_seed]
        corrected = base + seed_delta

        loss = _quantile_loss(corrected[mask], targets[mask], alpha)
        mean_preds = torch.full(targets[mask].size(), 0.5).to(device)
        random_preds = torch.rand(targets[mask].size(0)).to(device)
        mean_loss = F.l1_loss(mean_preds, targets[mask])
        random_loss = F.l1_loss(random_preds, targets[mask])

        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_random_loss += random_loss.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_mean_loss / n_batches,
        total_random_loss / n_batches,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_topology_correction(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: TemporalDatasetGlobalSplit,
) -> None:
    """Post-hoc topology correction for quantile regression."""
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    device = torch.device(
        f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu',
    )
    alpha = model_arguments.quantile_alpha

    # ---- Extract frozen base predictions (with cache) ----
    cache_path = weight_directory / model_arguments.model / 'base_preds.pt'
    if cache_path.exists():
        logging.info('Loading cached base predictions from %s', cache_path)
        base_preds = torch.load(cache_path, map_location='cpu', weights_only=True)
    else:
        base_preds = _extract_base_predictions(
            data,
            model_arguments,
            weight_directory,
            device,
        )
        torch.save(base_preds, cache_path)
        logging.info('Saved base predictions cache to %s', cache_path)

    # Attach base predictions as a node-level attribute
    data.base_preds = base_preds

    # ---- Set up data loaders ----
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

    # ---- Training (two-phase: pred only → pred + size + reg) ----
    pred_only_epochs = model_arguments.confgnn_pred_only_epochs
    size_loss_weight = model_arguments.confgnn_size_loss_weight
    reg_loss_weight = model_arguments.confgnn_reg_loss_weight
    total_epochs = model_arguments.epochs

    logging.info('=== Training Topology Correction GNN ===')
    logging.info(
        'Two-phase training: pred-only epochs=%d, total epochs=%d',
        pred_only_epochs,
        total_epochs,
    )
    logging.info(
        'size_loss_weight=%.3f, reg_loss_weight=%.3f', size_loss_weight, reg_loss_weight
    )

    logger = Logger(model_arguments.runs)
    loss_tuple_run: List[List[Tuple[float, float, float, float, float]]] = []
    final_avg_preds: List[List[float]] = []
    final_avg_targets: List[List[float]] = []
    global_best_val_loss = float('inf')
    best_state_dict = None

    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        correction_model = CorrectionGNN(
            hidden_channels=64,
            num_layers=2,
            dropout=model_arguments.dropout,
            normalization=model_arguments.normalization,
        ).to(device)
        optimizer = torch.optim.AdamW(
            correction_model.parameters(),
            lr=model_arguments.lr,
            weight_decay=model_arguments.weight_decay,
        )

        loss_tuple_epoch: List[Tuple[float, float, float, float, float]] = []
        epoch_avg_preds: List[List[float]] = []
        epoch_avg_targets: List[List[float]] = []

        for epoch in tqdm(range(1, 1 + total_epochs), desc='Epochs'):
            _, batch_preds, batch_targets = _train_correction_epoch(
                correction_model,
                train_loader,
                optimizer,
                alpha,
                epoch=epoch,
                pred_only_epochs=pred_only_epochs,
                size_loss_weight=size_loss_weight,
                reg_loss_weight=reg_loss_weight,
            )
            epoch_avg_preds.append(batch_preds)
            epoch_avg_targets.append(batch_targets)

            train_loss, _, _ = _evaluate_correction(
                correction_model,
                train_loader,
                'train_mask',
                alpha,
            )
            valid_loss, valid_mean_loss, valid_random_loss = _evaluate_correction(
                correction_model,
                val_loader,
                'valid_mask',
                alpha,
            )
            test_loss, test_mean_loss, test_random_loss = _evaluate_correction(
                correction_model,
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
                best_state_dict = snapshot_state_dict(correction_model)

        from credipred.utils.plot import mean_across_lists

        final_avg_preds.append(mean_across_lists(epoch_avg_preds))
        final_avg_targets.append(mean_across_lists(epoch_avg_targets))
        loss_tuple_run.append(loss_tuple_epoch)

    # ---- Save best model ----
    save_dir = weight_directory / model_arguments.model
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'correction_gnn_model.pt'
    torch.save(best_state_dict, save_path)
    logging.info('Saved correction GNN to %s', save_path)

    # ---- Statistics ----
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

    save_loss_results(loss_tuple_run, model_arguments.model, 'topology_correction')

    # ---- Post-hoc CQR: Base vs Corrected ----
    logging.info('=== Post-hoc CQR Evaluation: Base vs Corrected ===')
    correction_model.load_state_dict(best_state_dict)
    correction_model.to(device)
    correction_model.eval()

    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    train_idx = split_idx['train']
    all_labeled = torch.cat([train_idx, val_idx, test_idx])

    corrected_preds = base_preds.clone()
    eval_loader = NeighborLoader(
        data,
        input_nodes=all_labeled,
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=False,
        num_workers=4,
    )
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Extracting corrected predictions'):
            batch = batch.to(device)
            delta = correction_model(batch.base_preds.to(device), batch.edge_index)
            n_seed = batch.batch_size
            original_ids = batch.n_id[:n_seed]
            base_seed = batch.base_preds[:n_seed].to(device)
            corrected_preds[original_ids] = (base_seed + delta[:n_seed]).cpu()

    corrected_preds_path = save_dir / 'corrected_preds.pt'
    torch.save(corrected_preds, corrected_preds_path)
    logging.info('Saved corrected predictions to %s', corrected_preds_path)

    labels = data.y.cpu()

    # Raw comparison (no CQR)
    labels_test = labels[test_idx].numpy()
    base_mid = base_preds[test_idx, 0].numpy()
    base_widths = (base_preds[test_idx, 2] - base_preds[test_idx, 1]).numpy()
    base_mae = np.abs(base_mid - labels_test).mean()

    corr_mid = corrected_preds[test_idx, 0].numpy()
    corr_widths = (corrected_preds[test_idx, 2] - corrected_preds[test_idx, 1]).numpy()
    corr_mae = np.abs(corr_mid - labels_test).mean()

    logging.info('=== Raw Comparison (no CQR) ===')
    logging.info('  Base:      MAE=%.4f avg_width=%.4f', base_mae, base_widths.mean())
    logging.info('  Corrected: MAE=%.4f avg_width=%.4f', corr_mae, corr_widths.mean())

    # CQR comparison
    from credipred.conformal_regression.cqr import evaluate_intervals

    base_metrics = evaluate_intervals(base_preds, labels, val_idx, test_idx, alpha)
    corr_metrics = evaluate_intervals(corrected_preds, labels, val_idx, test_idx, alpha)

    logging.info('=== CQR Comparison ===')
    logging.info(
        '  Base CQR:      coverage=%.4f width=%.4f mae=%.4f qhat=%.4f',
        base_metrics.coverage,
        base_metrics.avg_width,
        base_metrics.mae,
        base_metrics.qhat,
    )
    logging.info(
        '  Corrected CQR: coverage=%.4f width=%.4f mae=%.4f qhat=%.4f',
        corr_metrics.coverage,
        corr_metrics.avg_width,
        corr_metrics.mae,
        corr_metrics.qhat,
    )

    # Per-quintile coverage
    logging.info('=== Per-quintile Coverage (CQR-adjusted) ===')
    for name, preds_tensor, qhat_val in [
        ('Base', base_preds, base_metrics.qhat),
        ('Corrected', corrected_preds, corr_metrics.qhat),
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
