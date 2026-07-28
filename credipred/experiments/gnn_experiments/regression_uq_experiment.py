"""Quantile regression training pipeline.

Trains a GAT model with pinball loss to predict [mid, lower, upper] quantiles.
Follows the same structure as gnn_experiment.py but with quantile loss.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.temporal_dataset import TemporalDatasetGlobalSplit
from credipred.gnn.model import Model
from credipred.utils.args import DataArguments, ModelArguments
from credipred.utils.checkpoint import snapshot_state_dict
from credipred.utils.logger import Logger
from credipred.utils.plot import (
    Scoring,
    mean_across_lists,
    plot_avg_loss,
    plot_pred_target_distributions_bin_list,
)
from credipred.utils.save import save_loss_results


def _quantile_loss(
    preds: Tensor, targets: Tensor, alpha: float,
) -> Tensor:
    """Pinball loss for quantile regression (mid, lower, upper)."""
    mid, lower, upper = preds[:, 0], preds[:, 1], preds[:, 2]
    loss_mid = F.l1_loss(mid, targets)

    residual_lower = targets - lower
    loss_lower = torch.mean(
        torch.where(residual_lower >= 0, alpha * residual_lower, (alpha - 1) * residual_lower)
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


def train(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    alpha: float,
) -> Tuple[float, List[float], List[float]]:
    """Train one epoch with quantile loss. Returns (loss, pred_scores, target_scores)."""
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    n_batches = 0
    pred_scores: List[float] = []
    target_scores: List[float] = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index)
        n_seed = batch.batch_size
        seed_preds = preds[:n_seed]
        seed_targets = batch.y[:n_seed]
        loss = _quantile_loss(seed_preds, seed_targets, alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        # Collect mid predictions for statistics
        for p in seed_preds[:, 0]:
            pred_scores.append(p.item())
        for t in seed_targets:
            target_scores.append(t.item())
    avg_loss = total_loss / n_batches
    return avg_loss, pred_scores, target_scores


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    mask_name: str,
    alpha: float,
) -> Tuple[float, float, float, float]:
    """Return midpoint MAE, two MAE baselines, and the quantile objective."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_mean_loss = 0
    total_random_loss = 0
    total_quantile_loss = 0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index)
        targets = batch.y
        n_seed = batch.batch_size
        mask = getattr(batch, mask_name)[:n_seed]
        if mask.sum() == 0:
            continue
        seed_preds = preds[:n_seed]
        seed_targets = targets[:n_seed]

        # Use MAE on mid prediction for fair comparison with MAE baseline
        mid_preds = seed_preds[mask, 0]
        loss = F.l1_loss(mid_preds, seed_targets[mask])
        quantile_loss = _quantile_loss(
            seed_preds[mask],
            seed_targets[mask],
            alpha,
        )
        mean_preds = torch.full(seed_targets[mask].size(), 0.5).to(device)
        random_preds = torch.rand(seed_targets[mask].size(0)).to(device)
        mean_loss = F.l1_loss(mean_preds, seed_targets[mask])
        random_loss = F.l1_loss(random_preds, seed_targets[mask])

        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_random_loss += random_loss.item()
        total_quantile_loss += quantile_loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_mean = total_mean_loss / n_batches
    avg_random = total_random_loss / n_batches
    avg_quantile = total_quantile_loss / n_batches
    return avg_loss, avg_mean, avg_random, avg_quantile


def run_regression_uq(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: TemporalDatasetGlobalSplit,
) -> None:
    """Train quantile regression GAT (same structure as run_gnn_baseline)."""
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = torch.device(
        f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu',
    )

    alpha = model_arguments.quantile_alpha

    logging.info(f'Device found: {device}')
    logging.info(f'Quantile alpha: {alpha}')

    logging.info(f'Training set size: {split_idx["train"].size()}')
    logging.info(f'Validation set size: {split_idx["valid"].size()}')
    logging.info(f'Testing set size: {split_idx["test"].size()}')

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
    logging.info('Train loader created')

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
    logging.info('Valid loader created')

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
    logging.info('Test loader created')

    logger = Logger(model_arguments.runs)
    loss_tuple_run: List[List[Tuple[float, float, float, float, float]]] = []
    final_avg_preds: List[List[float]] = []
    final_avg_targets: List[List[float]] = []
    global_best_val_quantile_loss = float('inf')
    best_state_dict = None

    logging.info('*** Training ***')
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model = Model(
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
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_arguments.lr,
            weight_decay=model_arguments.weight_decay,
        )

        loss_tuple_epoch: List[Tuple[float, float, float, float, float]] = []
        epoch_avg_preds: List[List[float]] = []
        epoch_avg_targets: List[List[float]] = []

        for epoch in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            _, batch_preds, batch_targets = train(model, train_loader, optimizer, alpha)
            epoch_avg_preds.append(batch_preds)
            epoch_avg_targets.append(batch_targets)

            train_loss, _, _, train_quantile_loss = evaluate(
                model,
                train_loader,
                'train_mask',
                alpha,
            )
            (
                valid_loss,
                valid_mean_baseline_loss,
                valid_random_baseline_loss,
                valid_quantile_loss,
            ) = evaluate(model, val_loader, 'valid_mask', alpha)
            (
                test_loss,
                test_mean_baseline_loss,
                test_random_baseline_loss,
                test_quantile_loss,
            ) = evaluate(model, test_loader, 'test_mask', alpha)

            result = (
                train_loss,
                valid_loss,
                test_loss,
                test_mean_baseline_loss,
                test_random_baseline_loss,
            )
            loss_tuple_epoch.append(result)
            logger.add_result(
                run,
                (
                    train_loss,
                    valid_loss,
                    test_loss,
                    valid_mean_baseline_loss,
                    valid_random_baseline_loss,
                ),
            )

            wandb.log({
                'run': run,
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
                'train_quantile_loss': train_quantile_loss,
                'valid_quantile_loss': valid_quantile_loss,
                'test_quantile_loss': test_quantile_loss,
            })

            if valid_quantile_loss < global_best_val_quantile_loss:
                global_best_val_quantile_loss = valid_quantile_loss
                best_state_dict = snapshot_state_dict(model)

        final_avg_preds.append(mean_across_lists(epoch_avg_preds))
        final_avg_targets.append(mean_across_lists(epoch_avg_targets))
        loss_tuple_run.append(loss_tuple_epoch)

    best_model_dir = weight_directory / f'{model_arguments.model}'
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / 'best_model.pt'
    torch.save(best_state_dict, best_model_path)
    logging.info(f'Model: {model_arguments} weights saved to: {best_model_path}')

    logging.info('*** Statistics ***')
    logging.info(logger.get_statistics())
    logging.info(logger.get_avg_statistics())
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds, targets=final_avg_targets, percent=10,
        )
    )
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds, targets=final_avg_targets, percent=5,
        )
    )
    logging.info(
        logger.per_run_within_error(
            preds=final_avg_preds, targets=final_avg_targets, percent=1,
        )
    )

    logging.info('Constructing plots')
    plot_pred_target_distributions_bin_list(
        preds=final_avg_preds,
        targets=final_avg_targets,
        model_name=model_arguments.model,
        bins=100,
    )
    plot_avg_loss(
        loss_tuple_run, model_arguments.model, Scoring.mae, 'quantile_loss_plot.png'
    )

    logging.info('Saving pkl of results')
    save_loss_results(loss_tuple_run, model_arguments.model, 'quantile')
