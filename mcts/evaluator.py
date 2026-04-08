"""
StateEvaluator — wraps CatanNet to score a CatanState from one player's perspective.

Two backends:
  1. Neural network (CatanNet) loaded from a checkpoint — preferred.
  2. Heuristic fallback using data.scoring functions — used when no checkpoint
     is available or for quick testing without loading a model.

Usage:
    # With trained model
    ev = StateEvaluator("checkpoints/best.pt")
    score = ev.evaluate(state, color)   # float in [0, 1]

    # Heuristic only (no checkpoint needed)
    ev = StateEvaluator(checkpoint_path=None)
    score = ev.evaluate(state, color)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from data.encoder import StateEncoder
from data.state import CatanState


class StateEvaluator:
    """
    Evaluates a CatanState from a given player's perspective.

    Returns a float in [0, 1] representing estimated win probability.
    Higher is better for `color`.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.encoder = StateEncoder()
        self.model = None
        self.device = torch.device("cpu")

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_model(checkpoint_path)
        else:
            if checkpoint_path:
                print(f"[StateEvaluator] Checkpoint not found: {checkpoint_path}")
            print("[StateEvaluator] Using heuristic evaluator (no neural network).")

    def _load_model(self, path: str) -> None:
        from model.catan_network import CatanNet

        # Prefer MPS on Apple Silicon, fall back to CPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        arch = ckpt.get("arch", {})
        self.model = CatanNet(
            input_dim=arch.get("input_dim", 1363),
            hidden_dim=arch.get("hidden_dim", 256),
            num_blocks=arch.get("num_blocks", 12),
            value_hidden=arch.get("value_hidden", 128),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"[StateEvaluator] Loaded CatanNet from {path} (device={self.device})")

    # ─── Core interface ──────────────────────────────────────────────────────

    def evaluate(self, state: CatanState, color: int) -> float:
        """
        Return a win-probability estimate in [0, 1] for `color` in `state`.
        Uses the neural network if loaded, otherwise the heuristic.
        """
        if self.model is not None:
            return self._neural_evaluate(state, color)
        return self._heuristic_evaluate(state, color)

    def evaluate_batch(
        self, states: list[CatanState], color: int
    ) -> list[float]:
        """
        Evaluate multiple states in a single forward pass.
        Falls back to individual heuristic calls if no model is loaded.
        """
        if self.model is None:
            return [self._heuristic_evaluate(s, color) for s in states]

        features = np.stack([
            self.encoder.encode_flat(s, perspective_color=color) for s in states
        ]).astype(np.float32)
        tensor = torch.from_numpy(features).to(self.device)

        with torch.no_grad():
            value, _aux, _win = self.model(tensor)

        return value.squeeze(-1).cpu().numpy().tolist()

    def evaluate_fast(self, state: CatanState, color: int) -> float:
        """
        Always uses the heuristic evaluator, regardless of whether a neural
        network is loaded.  Used inside _simulate_opponents() so that opponent
        action selection stays cheap (no GPU calls per opponent move).
        """
        return self._heuristic_evaluate(state, color)

    # ─── Backends ────────────────────────────────────────────────────────────

    def _neural_evaluate(self, state: CatanState, color: int) -> float:
        features = self.encoder.encode_flat(state, perspective_color=color)
        tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, _aux, _win = self.model(tensor)
        return float(value.squeeze())

    def _heuristic_evaluate(self, state: CatanState, color: int) -> float:
        """
        Lightweight composite score using position and economic quality.
        Does not require knowing the final outcome.
        """
        from data.scoring import relative_position_score, economic_quality_score

        s_pos = relative_position_score(state, color)
        s_eco = economic_quality_score(state, color)

        # Shift weight toward position score in the late game
        progress = min(state.current_turn / 80.0, 1.0)
        w_pos = 0.40 + 0.20 * progress
        w_eco = 0.60 - 0.20 * progress

        return w_pos * s_pos + w_eco * s_eco


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import sys

    DATASET_DIR = os.environ.get("CATAN_DATASET_DIR", "./dataset")
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    if not files:
        print(f"No JSON files found in {DATASET_DIR}")
        sys.exit(1)

    from data.replay import GameReplay

    replay = GameReplay.from_file(files[0])
    print(f"Loaded game: {os.path.basename(files[0])}")

    # ── Test heuristic evaluator ─────────────────────────────────────────────
    print("\n── Heuristic evaluator ──")
    ev_heuristic = StateEvaluator(checkpoint_path=None)

    for turn in [10, 25, 40]:
        state = replay.replay_to_turn(turn)
        scores = {}
        for color in state.player_colors:
            scores[color] = ev_heuristic.evaluate(state, color)
        total = sum(scores.values())
        print(f"  Turn {turn}: {scores}  (sum={total:.3f})")
        assert all(0.0 <= v <= 1.0 for v in scores.values()), "Scores out of [0,1]"

    # ── Test neural evaluator (if checkpoint exists) ─────────────────────────
    ckpt = "checkpoints/best.pt"
    if os.path.exists(ckpt):
        print(f"\n── Neural evaluator ({ckpt}) ──")
        ev_neural = StateEvaluator(checkpoint_path=ckpt)

        for turn in [10, 25, 40]:
            state = replay.replay_to_turn(turn)
            scores = {}
            for color in state.player_colors:
                scores[color] = ev_neural.evaluate(state, color)
            print(f"  Turn {turn}: {scores}")
            assert all(0.0 <= v <= 1.0 for v in scores.values()), "Neural scores out of [0,1]"

        # Batch test
        print("\n── Batch evaluation ──")
        states = [replay.replay_to_turn(t) for t in range(10, 50, 5)]
        color = replay.play_order[0]
        batch_scores = ev_neural.evaluate_batch(states, color)
        single_scores = [ev_neural.evaluate(s, color) for s in states]
        for i, (b, s) in enumerate(zip(batch_scores, single_scores)):
            diff = abs(b - s)
            assert diff < 1e-4, f"Batch/single mismatch at index {i}: {b:.6f} vs {s:.6f}"
        print(f"  Batch vs single: max diff = {max(abs(b-s) for b,s in zip(batch_scores, single_scores)):.2e} — OK")
    else:
        print(f"\nNo checkpoint at {ckpt} — skipping neural tests.")

    print("\nAll evaluator tests passed.")
