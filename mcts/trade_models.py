"""
Trade models for MCTS:

1. TradeAcceptanceModel — P(accept | state, trade, responder)
   Binary classifier trained on observed trade accept/reject decisions.

2. TradeProposalPolicy — P(trade | state, proposer)
   Scores candidate trades to focus MCTS search on plausible actions.

Both models are trained from the game replay dataset. The training
pipeline extracts (state, trade, outcome) triples from the event logs.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data.state import CatanState
from data.encoder import StateEncoder
from data.replay import GameReplay
from data.enums import Resource, LogType

from mcts.trade_encoder import Trade, TradeEncoder


# ═══════════════════════════════════════════════
# Training data extraction from game replays
# ═══════════════════════════════════════════════

def extract_trade_events(game_data: dict) -> list[dict]:
    """
    Walk the event log and extract trade-related events.

    The Colonist event format has:
      - TRADE_OFFER (118): a player proposes a trade
      - TRADE_ACCEPTED (116): a player accepts the current offer
      - TRADE_COMPLETED (117): the trade is finalized

    We reconstruct the full context for each trade decision.
    """
    events = game_data['data']['eventHistory']['events']
    trade_events = []
    pending_offer = None

    for i, event in enumerate(events):
        log_type = event.get('type')

        if log_type == LogType.TRADE_OFFER:
            # New trade offer — extract details
            payload = event.get('payload', {})
            pending_offer = {
                'event_idx': i,
                'proposer_color': payload.get('playerColor'),
                'offering': _parse_resource_dict(payload.get('offering', {})),
                'requesting': _parse_resource_dict(payload.get('requesting', {})),
                'responses': {},  # will be filled by accept/reject events
            }

        elif log_type == LogType.TRADE_ACCEPTED and pending_offer:
            payload = event.get('payload', {})
            responder = payload.get('playerColor')
            if responder is not None:
                pending_offer['responses'][responder] = True

        elif log_type == LogType.TRADE_COMPLETED and pending_offer:
            # Trade went through — record it
            payload = event.get('payload', {})
            trade_events.append({
                **pending_offer,
                'completed': True,
                'completion_event_idx': i,
            })
            pending_offer = None

        elif log_type == LogType.TURN_END and pending_offer:
            # Turn ended without completion — offer expired/rejected
            trade_events.append({
                **pending_offer,
                'completed': False,
                'completion_event_idx': i,
            })
            pending_offer = None

    return trade_events


def _parse_resource_dict(raw: dict) -> dict[int, int]:
    """Convert JSON resource dict to {resource_int: count}."""
    result = {}
    for key, val in raw.items():
        try:
            res = int(key)
            if val > 0:
                result[res] = val
        except (ValueError, TypeError):
            continue
    return result


def generate_acceptance_samples(
    game_path: str | Path,
    encoder: Optional[TradeEncoder] = None,
) -> Iterator[dict]:
    """
    Generate (features, accepted) samples for the acceptance model.

    For each trade offer in the game:
    - Replay to the state just before the offer
    - For each opponent of the proposer, generate a sample:
        - label = 1 if they accepted, 0 otherwise
    - Encode with TradeEncoder from the responder's perspective
    """
    encoder = encoder or TradeEncoder()

    with open(game_path, 'r') as f:
        game_data = json.load(f)

    replay = GameReplay(game_data)
    trade_events = extract_trade_events(game_data)

    if not trade_events:
        return

    for te in trade_events:
        if te['proposer_color'] is None:
            continue
        if not te['offering'] or not te['requesting']:
            continue

        # Replay to the state just before this trade offer
        state = replay.replay_to_event(te['event_idx'])

        proposer_color = te['proposer_color']

        # For each non-proposer player, generate an acceptance sample
        for color in state.player_colors:
            if color == proposer_color:
                continue

            trade = Trade(
                proposer_color=proposer_color,
                responder_color=color,
                offering=te['offering'],
                requesting=te['requesting'],
            )

            features = encoder.encode_for_acceptance(state, trade, color)
            accepted = te['responses'].get(color, False)

            yield {
                'features': features,
                'accepted': float(accepted),
                'proposer_color': proposer_color,
                'responder_color': color,
                'trade_completed': te['completed'],
                'turn': state.current_turn,
            }


def generate_proposal_samples(
    game_path: str | Path,
    encoder: Optional[TradeEncoder] = None,
) -> Iterator[dict]:
    """
    Generate samples for the proposal policy.

    For each turn in the game:
    - If a trade was proposed: positive sample for that trade
    - Generate negative samples from the candidate set that weren't proposed

    This trains the policy to predict which trades humans actually propose.
    """
    encoder = encoder or TradeEncoder()

    with open(game_path, 'r') as f:
        game_data = json.load(f)

    replay = GameReplay(game_data)
    trade_events = extract_trade_events(game_data)

    # Index trades by their event_idx for quick lookup
    trade_by_event = {te['event_idx']: te for te in trade_events}

    # Walk through the game turn by turn
    for turn_idx in range(1, len(replay._turn_boundaries)):
        state = replay.replay_to_turn(turn_idx)
        if state.is_setup_phase():
            continue

        current_player = state.current_player_color

        # Check if a trade was proposed during this turn
        turn_start = replay._turn_boundaries[turn_idx]
        turn_end = (
            replay._turn_boundaries[turn_idx + 1]
            if turn_idx + 1 < len(replay._turn_boundaries)
            else len(replay.events)
        )

        proposed_trade = None
        for ei in range(turn_start, turn_end):
            if ei in trade_by_event:
                te = trade_by_event[ei]
                if te['proposer_color'] == current_player:
                    proposed_trade = te
                    break

        if proposed_trade is not None:
            # Positive sample: the trade that was actually proposed
            for color in state.player_colors:
                if color == current_player:
                    continue
                trade = Trade(
                    proposer_color=current_player,
                    responder_color=color,
                    offering=proposed_trade['offering'],
                    requesting=proposed_trade['requesting'],
                )
                features = encoder.encode_for_proposal(state, trade, current_player)
                yield {
                    'features': features,
                    'label': 1.0,
                    'proposer_color': current_player,
                    'turn': state.current_turn,
                }
                break  # One positive per turn

        # Negative sample: "no trade" for turns without trades
        if proposed_trade is None:
            features = encoder.encode_for_proposal(state, None, current_player)
            yield {
                'features': features,
                'label': 0.0,
                'proposer_color': current_player,
                'turn': state.current_turn,
            }


# ═══════════════════════════════════════════════
# PyTorch models
# ═══════════════════════════════════════════════

class TradeAcceptanceModel(nn.Module):
    """
    Predicts P(accept | game_state, trade, responder).

    Architecture: MLP with residual connections.
    Input: concatenation of game state features + trade features.
    Output: sigmoid probability of acceptance.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

        # Residual projection for skip connection
        self.skip = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)

        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)

        # Residual block
        residual = self.skip(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        h = h + residual

        h = F.relu(self.bn3(self.fc3(h)))
        h = self.dropout(h)

        return torch.sigmoid(self.fc4(h)).squeeze(-1)

    def predict(self, features: np.ndarray) -> float:
        """Convenience method for single prediction during MCTS."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            return self.forward(x).item()


class TradeProposalPolicy(nn.Module):
    """
    Scores candidate trades: higher score = more likely a human would propose this.

    Used to prune the MCTS action space — only expand the top-K trades.
    Architecture: Same MLP, but output is a score (not bounded to [0,1]).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        h = F.relu(self.bn3(self.fc3(h)))
        h = self.dropout(h)
        return self.fc4(h).squeeze(-1)

    def score_trades(self, features_list: list[np.ndarray]) -> list[float]:
        """Score a batch of candidate trades. Returns list of scores."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(np.stack(features_list), dtype=torch.float32)
            scores = self.forward(x)
            return scores.tolist()


# ═══════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════

class TradeDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_acceptance_model(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.15,
    device: str = 'cpu',
) -> tuple[TradeAcceptanceModel, dict]:
    """
    Train the acceptance model.
    Returns (model, training_history).
    """
    # Train/val split
    n = len(labels)
    indices = np.random.permutation(n)
    val_n = int(n * val_split)
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    train_ds = TradeDataset(features[train_idx], labels[train_idx])
    val_ds = TradeDataset(features[val_idx], labels[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    input_dim = features.shape[1]
    model = TradeAcceptanceModel(input_dim).to(device)

    # Class weighting — trades are rejected more often than accepted
    pos_count = labels.sum()
    neg_count = len(labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # We use BCEWithLogitsLoss, so we need to bypass the sigmoid in forward
    # Actually, let's use BCELoss since forward already applies sigmoid
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item() * len(y)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        val_loss /= len(val_ds)

        # Accuracy
        preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
        acc = sum(p == l for p, l in zip(preds_binary, all_labels)) / len(all_labels)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} — '
                  f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, '
                  f'val_acc: {acc:.4f}')

    return model, history


def train_proposal_policy(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.15,
    device: str = 'cpu',
) -> tuple[TradeProposalPolicy, dict]:
    """
    Train the proposal policy as a binary classifier:
    1 = trade was proposed, 0 = no trade / different trade.
    """
    n = len(labels)
    indices = np.random.permutation(n)
    val_n = int(n * val_split)
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    train_ds = TradeDataset(features[train_idx], labels[train_idx])
    val_ds = TradeDataset(features[val_idx], labels[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    input_dim = features.shape[1]
    model = TradeProposalPolicy(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item() * len(y)
        val_loss /= len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} — '
                  f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

    return model, history
