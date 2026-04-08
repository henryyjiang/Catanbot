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

from trade_mcts.trade_encoder import Trade, TradeEncoder


# ═══════════════════════════════════════════════
# Training data extraction from game replays
# ═══════════════════════════════════════════════

def extract_trade_events(game_data: dict) -> list[dict]:
    """
    Walk the event log and extract trade-related events.

    The Colonist event format has:
      - TRADE_OFFER (118): a player proposes a trade
      - TRADE_COMPLETED (117): the trade is finalized between specific players
      - Direct trade (115): immediate trade between two players

    We track which players accepted by looking at who the trade completed with.
    """
    events = game_data['data']['eventHistory']['events']
    trade_events = []

    for i, event in enumerate(events):
        sc = event.get('stateChange', {})
        game_log = sc.get('gameLogState', {})
        
        # game_log is a dict of log entries, each with a 'text' field
        for log_entry in game_log.values():
            text = log_entry.get('text', {})
            log_type = text.get('type')

            if log_type == LogType.TRADE_COMPLETED:
                # Trade completed between specific players
                proposer_color = text.get('playerColorCreator')
                responder_color = text.get('playerColorOffered')
                offered = text.get('offeredCardEnums', [])
                wanted = text.get('wantedCardEnums', [])
                
                if proposer_color is not None and responder_color is not None and offered and wanted:
                    trade_events.append({
                        'event_idx': i,
                        'proposer_color': proposer_color,
                        'responder_color': responder_color,
                        'offering': {r: offered.count(r) for r in set(offered)},
                        'requesting': {r: wanted.count(r) for r in set(wanted)},
                        'responses': {responder_color: True},  # This responder accepted
                        'completed': True,
                        'completion_event_idx': i,
                    })

            elif log_type == 115:  # Direct trade acceptance (type 115)
                proposer_color = text.get('playerColor')
                responder_color = text.get('acceptingPlayerColor')
                offered = text.get('givenCardEnums', [])
                wanted = text.get('receivedCardEnums', [])
                
                if proposer_color is not None and responder_color is not None and offered and wanted:
                    trade_events.append({
                        'event_idx': i,
                        'proposer_color': proposer_color,
                        'responder_color': responder_color,
                        'offering': {r: offered.count(r) for r in set(offered)},
                        'requesting': {r: wanted.count(r) for r in set(wanted)},
                        'responses': {responder_color: True},  # This responder accepted
                        'completed': True,
                        'completion_event_idx': i,
                    })

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

    For each trade that was completed or rejected:
    - Replay to the state just before the trade
    - For each opponent of the proposer:
        - label = 1 if they accepted, 0 if they rejected or didn't respond
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
        try:
            state = replay.replay_to_event(te['event_idx'])
        except Exception:
            continue

        proposer_color = te['proposer_color']
        responder_color = te.get('responder_color')

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

            try:
                features = encoder.encode_for_acceptance(state, trade, color)
            except Exception:
                continue

            # Label: 1 if this player accepted, 0 otherwise
            accepted = 1.0 if color == responder_color else 0.0

            yield {
                'features': features,
                'accepted': accepted,
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


class NpzStreamDataset(Dataset):
    """
    Memory-efficient dataset that reads from a .npz file using mmap.
    Never loads the full array into RAM — safe for 5M+ samples.
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path, mmap_mode='r')
        self.features = data['features']  # memory-mapped, not loaded
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(np.array(self.features[idx]), dtype=torch.float32)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


def train_acceptance_model(
    features,
    labels,
    epochs: int = 50,
    batch_size: int = 4096,
    lr: float = 1e-3,
    val_split: float = 0.15,
    device: str = 'cpu',
    npz_path: str = None,
) -> tuple['TradeAcceptanceModel', dict]:
    """
    Train the acceptance model.
    Pass npz_path to stream from disk (memory-efficient for large datasets).
    Returns (model, training_history).
    """
    if npz_path is not None:
        # Stream from disk — safe for 5M+ samples
        full_ds = NpzStreamDataset(npz_path)
        n = len(full_ds)
        val_n = int(n * val_split)
        train_n = n - val_n
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_n, val_n])
        input_dim = full_ds.features.shape[1]
    else:
        # Legacy: load from arrays (may OOM on large datasets)
        n = len(labels)
        indices = np.random.permutation(n)
        val_n = int(n * val_split)
        val_idx, train_idx = indices[:val_n], indices[val_n:]
        train_ds = TradeDataset(features[train_idx], labels[train_idx])
        val_ds = TradeDataset(features[val_idx], labels[val_idx])
        input_dim = features.shape[1]

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    input_dim = input_dim if npz_path is not None else features.shape[1]
    model = TradeAcceptanceModel(input_dim).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    # Use all CPU cores for PyTorch operations
    import multiprocessing
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_num_interop_threads(max(1, multiprocessing.cpu_count() // 2))
    print(f'  Using {torch.get_num_threads()} CPU threads for training...')

    n_batches = len(train_dl)
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for bi, (X, y) in enumerate(train_dl):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            if (bi + 1) % max(1, n_batches // 5) == 0:
                print(f'  Epoch {epoch+1}/{epochs} batch {bi+1}/{n_batches} loss={loss.item():.4f}', flush=True)
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

        print(f'Epoch {epoch+1}/{epochs} — '
              f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, '
              f'val_acc: {acc:.4f}')

    return model, history


def train_proposal_policy(
    features=None,
    labels=None,
    epochs: int = 50,
    batch_size: int = 4096,
    lr: float = 1e-3,
    val_split: float = 0.15,
    device: str = 'cpu',
    npz_path: str = None,
) -> tuple['TradeProposalPolicy', dict]:
    """
    Train the proposal policy as a binary classifier.
    Pass npz_path to stream from disk (memory-efficient for large datasets).
    """
    if npz_path is not None:
        full_ds = NpzStreamDataset(npz_path)
        n = len(full_ds)
        val_n = int(n * val_split)
        train_n = n - val_n
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_n, val_n])
        input_dim = full_ds.features.shape[1]
    else:
        n = len(labels)
        indices = np.random.permutation(n)
        val_n = int(n * val_split)
        val_idx, train_idx = indices[:val_n], indices[val_n:]
        train_ds = TradeDataset(features[train_idx], labels[train_idx])
        val_ds = TradeDataset(features[val_idx], labels[val_idx])
        input_dim = features.shape[1]

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    input_dim = input_dim if npz_path is not None else features.shape[1]
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

        print(f'Epoch {epoch+1}/{epochs} — '
              f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

    return model, history