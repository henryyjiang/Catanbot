"""
Usage:
    python catan_network.py --data training_data/ --epochs 60 --batch-size 2048
    python catan_network.py --data training_data/ --resume checkpoints/best.pt
    python catan_network.py --data training_data/ --eval-only --resume checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class CatanDataset(Dataset):
    def __init__(self, path: str, label_key: str = "labels"):
        data = np.load(path, allow_pickle=True)

        self.features = torch.from_numpy(data["features"].astype(np.float32))
        self.labels = torch.from_numpy(data[label_key].astype(np.float32))

        self.outcome_scores = None
        self.position_scores = None
        self.economic_scores = None
        self.won_game = None

        if "outcome_scores" in data:
            self.outcome_scores = torch.from_numpy(
                data["outcome_scores"].astype(np.float32)
            )
        if "position_scores" in data:
            self.position_scores = torch.from_numpy(
                data["position_scores"].astype(np.float32)
            )
        if "economic_scores" in data:
            self.economic_scores = torch.from_numpy(
                data["economic_scores"].astype(np.float32)
            )
        if "won_game" in data:
            self.won_game = torch.from_numpy(data["won_game"].astype(np.float32))

        assert self.features.shape[1] == 1363, (
            f"Expected 1363 features, got {self.features.shape[1]}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ResidualBlock(nn.Module):
    """Pre-activation residual block (BN - ReLU - Linear - BN - ReLU - Linear).
    """

    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(x))
        out = self.fc1(out)
        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out + residual


class CatanNet(nn.Module):
    """
    Input (1363) → Projection (hidden_dim) → N residual blocks → Heads

    The value head predicts a scalar in [0, 1] representing the probability
    of winning (or composite evaluation score) from the current state.

    The auxiliary head predicts sub-component scores (outcome, position,
    economic) for interpretability and multi-task learning.
    """

    def __init__(
        self,
        input_dim: int = 1363,
        hidden_dim: int = 256,
        num_blocks: int = 12,
        value_hidden: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Map the heterogeneous 1363-dim input into a uniform hidden space.
        # Two layers with BN to give the network capacity to remap features
        # before the residual tower.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.res_tower = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.value_head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, value_hidden),
            nn.BatchNorm1d(value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

        # Predicts the three sub-component scores for multi-task learning.
        # This gives additional gradient signal and interpretability.
        self.aux_head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, value_hidden),
            nn.BatchNorm1d(value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 3),
        )

        self.win_head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.res_tower(h)

        value = torch.sigmoid(self.value_head(h))
        aux = torch.sigmoid(self.aux_head(h))
        win_logit = self.win_head(h)

        return value, aux, win_logit


class CatanLoss(nn.Module):
    """Multi-task loss combining value prediction, auxiliary scores, and win
    classification.
    """

    def __init__(
        self,
        w_value: float = 1.0,
        w_aux: float = 0.3,
        w_win: float = 0.2,
    ):
        super().__init__()
        self.w_value = w_value
        self.w_aux = w_aux
        self.w_win = w_win
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        value_pred: torch.Tensor,
        aux_pred: torch.Tensor,
        win_logit: torch.Tensor,
        label: torch.Tensor,
        outcome_score: Optional[torch.Tensor] = None,
        position_score: Optional[torch.Tensor] = None,
        economic_score: Optional[torch.Tensor] = None,
        won_game: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_value = self.mse(value_pred.squeeze(-1), label)
        total = self.w_value * loss_value
        metrics = {"loss_value": loss_value.item()}

        if (
            outcome_score is not None
            and position_score is not None
            and economic_score is not None
        ):
            aux_target = torch.stack(
                [outcome_score, position_score, economic_score], dim=-1
            )
            loss_aux = self.mse(aux_pred, aux_target)
            total = total + self.w_aux * loss_aux
            metrics["loss_aux"] = loss_aux.item()
        if won_game is not None:
            loss_win = self.bce(win_logit.squeeze(-1), won_game)
            total = total + self.w_win * loss_win
            metrics["loss_win"] = loss_win.item()

        metrics["loss_total"] = total.item()
        return total, metrics


class CatanTrainingDataset(Dataset):
    def __init__(self, path: str):
        path = Path(path)

        if path.is_dir():
            self._load_memmap_dir(path)
        elif path.suffix == ".npz":
            self._load_npz(path)
        else:
            dir_path = Path(str(path).replace(".npz", "") + "_data")
            if dir_path.is_dir():
                self._load_memmap_dir(dir_path)
            else:
                raise FileNotFoundError(
                    f"Cannot find dataset at {path}. Expected a .npz file "
                    f"or a directory of .npy memmap files."
                )

        n = len(self.labels)
        d = self.n_features
        print(f"  Loaded {n:,} samples × {d} features ({self._format})")
        label_arr = self.labels[:] if self._is_memmap else self.labels
        print(f"  Labels: mean={label_arr.mean():.4f}, std={label_arr.std():.4f}")
        if self.has_win:
            won_arr = self.won_game[:] if self._is_memmap else self.won_game
            print(f"  Win rate: {won_arr.mean():.2%}")

    def _load_npz(self, path: Path):
        self._format = "npz"
        self._is_memmap = False
        data = np.load(path, allow_pickle=True)

        self.features = torch.from_numpy(data["features"].astype(np.float32))
        self.labels = torch.from_numpy(data["labels"].astype(np.float32))
        self.n_features = self.features.shape[1]

        self.has_aux = "outcome_scores" in data
        if self.has_aux:
            self.outcome_scores = torch.from_numpy(
                data["outcome_scores"].astype(np.float32)
            )
            self.position_scores = torch.from_numpy(
                data["position_scores"].astype(np.float32)
            )
            self.economic_scores = torch.from_numpy(
                data["economic_scores"].astype(np.float32)
            )

        self.has_win = "won_game" in data
        if self.has_win:
            self.won_game = torch.from_numpy(data["won_game"].astype(np.float32))

        assert self.n_features == 1363, (
            f"Expected 1363 features, got {self.n_features}"
        )

    def _load_memmap_dir(self, dir_path: Path):
        """Load from a directory of .npy memmap files.
        """
        self._format = "memmap"
        self._is_memmap = True

        manifest_path = dir_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            n_samples = manifest["n_samples"]
            feat_dim = manifest["feature_dim"]
        else:
            feat_path = dir_path / "features.npy"
            file_bytes = feat_path.stat().st_size
            feat_dim = 1363
            n_samples = file_bytes // (feat_dim * 4)

        self.n_features = feat_dim
        assert feat_dim == 1363, f"Expected 1363 features, got {feat_dim}"

        self.features = np.memmap(
            dir_path / "features.npy",
            dtype=np.float32, mode="r", shape=(n_samples, feat_dim),
        )
        self.labels = np.memmap(
            dir_path / "labels.npy",
            dtype=np.float32, mode="r", shape=(n_samples,),
        )

        outcome_path = dir_path / "outcome_scores.npy"
        self.has_aux = outcome_path.exists()
        if self.has_aux:
            self.outcome_scores = np.memmap(
                outcome_path,
                dtype=np.float32, mode="r", shape=(n_samples,),
            )
            self.position_scores = np.memmap(
                dir_path / "position_scores.npy",
                dtype=np.float32, mode="r", shape=(n_samples,),
            )
            self.economic_scores = np.memmap(
                dir_path / "economic_scores.npy",
                dtype=np.float32, mode="r", shape=(n_samples,),
            )

        won_path = dir_path / "won_game.npy"
        self.has_win = won_path.exists()
        if self.has_win:
            self.won_game = np.memmap(
                won_path,
                dtype=np.bool_, mode="r", shape=(n_samples,),
            )

        # Pull all scalar arrays into RAM — they total < 1 MB for 42k samples.
        # Only self.features stays as a memmap (the one large array worth keeping lazy).
        # This eliminates per-sample page faults on label/score reads in __getitem__.
        self.labels = np.array(self.labels, dtype=np.float32)
        if self.has_aux:
            self.outcome_scores  = np.array(self.outcome_scores,  dtype=np.float32)
            self.position_scores = np.array(self.position_scores, dtype=np.float32)
            self.economic_scores = np.array(self.economic_scores, dtype=np.float32)
        if self.has_win:
            self.won_game = np.array(self.won_game, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self._is_memmap:
            item = {
                "features": torch.from_numpy(
                    self.features[idx].astype(np.float32).copy()
                ),
                # Scalar arrays are pre-loaded into RAM in _load_memmap_dir,
                # so these are plain numpy accesses — no page faults.
                "label": torch.from_numpy(
                    np.array(self.labels[idx], dtype=np.float32)
                ),
            }
            if self.has_aux:
                item["outcome_score"]  = torch.from_numpy(np.array(self.outcome_scores[idx],  dtype=np.float32))
                item["position_score"] = torch.from_numpy(np.array(self.position_scores[idx], dtype=np.float32))
                item["economic_score"] = torch.from_numpy(np.array(self.economic_scores[idx], dtype=np.float32))
            if self.has_win:
                item["won_game"] = torch.from_numpy(np.array(self.won_game[idx], dtype=np.float32))
        else:
            item = {
                "features": self.features[idx],
                "label": self.labels[idx],
            }
            if self.has_aux:
                item["outcome_score"] = self.outcome_scores[idx]
                item["position_score"] = self.position_scores[idx]
                item["economic_score"] = self.economic_scores[idx]
            if self.has_win:
                item["won_game"] = self.won_game[idx]

        return item

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.max_lr * self.step_count / max(self.warmup_steps, 1)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

class Trainer:
    def __init__(
        self,
        model: CatanNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        print(f"  Device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name()}")

        if config.get("optimizer", "adamw") == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config["lr"],
                momentum=0.9,
                weight_decay=config.get("weight_decay", 1e-4),
                nesterov=True,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["lr"],
                weight_decay=config.get("weight_decay", 1e-4),
                betas=(0.9, 0.999),
            )

        self.criterion = CatanLoss(
            w_value=config.get("w_value", 1.0),
            w_aux=config.get("w_aux", 0.3),
            w_win=config.get("w_win", 0.2),
        )

        total_steps = config["epochs"] * len(train_loader)
        warmup_steps = config.get("warmup_epochs", 3) * len(train_loader)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_lr=config["lr"],
            min_lr=config.get("min_lr", 1e-6),
        )

        self.use_amp = config.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.ckpt_dir = Path(config.get("ckpt_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                log_dir=config.get("log_dir", "runs/catan")
            )
        except ImportError:
            print("  TensorBoard not available, skipping logging")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_metrics = {}
        n_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            label = batch["label"].to(self.device)

            outcome = batch.get("outcome_score")
            position = batch.get("position_score")
            economic = batch.get("economic_score")
            won = batch.get("won_game")

            if outcome is not None:
                outcome = outcome.to(self.device)
            if position is not None:
                position = position.to(self.device)
            if economic is not None:
                economic = economic.to(self.device)
            if won is not None:
                won = won.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    value, aux, win_logit = self.model(features)
                    loss, metrics = self.criterion(
                        value, aux, win_logit, label,
                        outcome, position, economic, won,
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("grad_clip", 1.0),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                value, aux, win_logit = self.model(features)
                loss, metrics = self.criterion(
                    value, aux, win_logit, label,
                    outcome, position, economic, won,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("grad_clip", 1.0),
                )
                self.optimizer.step()

            lr = self.scheduler.step()
            metrics["lr"] = lr

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        return {k: v / n_batches for k, v in total_metrics.items()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_metrics = {}
        n_batches = 0
        all_preds = []
        all_labels = []
        all_win_preds = []
        all_win_labels = []

        for batch in self.val_loader:
            features = batch["features"].to(self.device)
            label = batch["label"].to(self.device)

            outcome = batch.get("outcome_score")
            position = batch.get("position_score")
            economic = batch.get("economic_score")
            won = batch.get("won_game")

            if outcome is not None:
                outcome = outcome.to(self.device)
            if position is not None:
                position = position.to(self.device)
            if economic is not None:
                economic = economic.to(self.device)
            if won is not None:
                won = won.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    value, aux, win_logit = self.model(features)
                    _, metrics = self.criterion(
                        value, aux, win_logit, label,
                        outcome, position, economic, won,
                    )
            else:
                value, aux, win_logit = self.model(features)
                _, metrics = self.criterion(
                    value, aux, win_logit, label,
                    outcome, position, economic, won,
                )

            all_preds.append(value.squeeze(-1).cpu())
            all_labels.append(label.cpu())
            if won is not None:
                all_win_preds.append((win_logit.squeeze(-1) > 0).float().cpu())
                all_win_labels.append(won.cpu())

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        avg = {k: v / n_batches for k, v in total_metrics.items()}

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        if len(preds) > 1:
            correlation = torch.corrcoef(torch.stack([preds, labels]))[0, 1].item()
            avg["correlation"] = correlation

        if all_win_preds:
            wp = torch.cat(all_win_preds)
            wl = torch.cat(all_win_labels)
            avg["win_accuracy"] = (wp == wl).float().mean().item()

        return avg

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "arch": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_blocks": len(self.model.res_tower),
                "value_hidden": self.model.value_head[-1].in_features,
            },
        }
        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(state, path)

        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        return ckpt.get("epoch", 0)

    def train(self, start_epoch: int = 0):
        epochs = self.config["epochs"]
        patience = self.config.get("patience", 15)
        no_improve = 0

        param_count = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n  Parameters: {param_count:,} total, {trainable:,} trainable")
        print(f"  Training for {epochs} epochs, patience={patience}")
        print(f"  AMP: {self.use_amp}")
        print()

        header = (
            f"{'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>10} │ "
            f"{'Val Corr':>8} │ {'Win Acc':>7} │ {'LR':>10} │ {'Time':>6}"
        )
        print(header)
        print("─" * len(header))

        for epoch in range(start_epoch, epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            elapsed = time.time() - t0
            val_loss = val_metrics["loss_total"]
            correlation = val_metrics.get("correlation", 0.0)
            win_acc = val_metrics.get("win_accuracy", 0.0)
            lr = train_metrics.get("lr", 0.0)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            best_marker = " ★" if is_best else ""
            print(
                f"{epoch+1:>5} │ {train_metrics['loss_total']:>10.6f} │ "
                f"{val_loss:>10.6f} │ {correlation:>8.4f} │ "
                f"{win_acc:>7.2%} │ {lr:>10.2e} │ {elapsed:>5.1f}s{best_marker}"
            )

            if self.writer:
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

            if (epoch + 1) % self.config.get("ckpt_every", 10) == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_loss, is_best)
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        self.save_checkpoint(epochs, val_loss, is_best=False)

        if self.writer:
            self.writer.close()

        print(f"\n  Training complete. Best val loss: {self.best_val_loss:.6f}")
        print(f"  Best model saved to: {self.ckpt_dir / 'best.pt'}")


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles optional keys."""
    result = {
        "features": torch.stack([b["features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
    if "outcome_score" in batch[0]:
        result["outcome_score"] = torch.stack([b["outcome_score"] for b in batch])
        result["position_score"] = torch.stack([b["position_score"] for b in batch])
        result["economic_score"] = torch.stack([b["economic_score"] for b in batch])
    if "won_game" in batch[0]:
        result["won_game"] = torch.stack([b["won_game"] for b in batch])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train CatanNet — AlphaZero-style value network for Catan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python catan_network.py --data training_data.npz --epochs 60
  python catan_network.py --data training_data.npz --hidden-dim 512 --num-blocks 20
  python catan_network.py --data training_data.npz --resume checkpoints/best.pt
  python catan_network.py --data training_data.npz --eval-only --resume checkpoints/best.pt
        """,
    )
    # Data
    parser.add_argument("--data", required=True, help="Path to .npz file or memmap directory (training_data/)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction")

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=256, help="Residual tower width")
    parser.add_argument("--num-blocks", type=int, default=12, help="Number of residual blocks")
    parser.add_argument("--value-hidden", type=int, default=128, help="Value head hidden dim")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout in res blocks")

    # Training
    parser.add_argument("--epochs", type=int, default=60, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="LR warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")

    # Loss weights
    parser.add_argument("--w-value", type=float, default=1.0, help="Value loss weight")
    parser.add_argument("--w-aux", type=float, default=0.3, help="Auxiliary loss weight")
    parser.add_argument("--w-win", type=float, default=0.2, help="Win loss weight")

    # Infrastructure
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--ckpt-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--ckpt-every", type=int, default=10, help="Checkpoint interval")
    parser.add_argument("--log-dir", default="runs/catan", help="TensorBoard log dir")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    dataset = CatanTrainingDataset(args.data)

    n = len(dataset)
    n_val = int(n * args.val_split)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    print(f"  Train: {n_train:,} | Val: {n_val:,}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )

    print("\nBuilding model...")
    model = CatanNet(
        input_dim=1363,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        value_hidden=args.value_hidden,
        dropout=args.dropout,
    )
    print(f"  Architecture: {args.hidden_dim}-wide × {args.num_blocks} residual blocks")

    config = {
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "epochs": args.epochs,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "w_value": args.w_value,
        "w_aux": args.w_aux,
        "w_win": args.w_win,
        "amp": not args.no_amp,
        "ckpt_dir": args.ckpt_dir,
        "ckpt_every": args.ckpt_every,
        "log_dir": args.log_dir,
    }

    trainer = Trainer(model, train_loader, val_loader, config)

    start_epoch = 0
    if args.resume:
        print(f"\n  Resuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"  Resumed at epoch {start_epoch}")

    if args.eval_only:
        print("\n  Evaluation only:")
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            print(f"    {k}: {v:.6f}")
        return

    print()
    trainer.train(start_epoch)


if __name__ == "__main__":
    main()
