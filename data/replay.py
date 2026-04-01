"""
 It loads a game JSON, creates a CatanState, and provides methods to:
Replay to any turn, Generate (state, outcome) training pairs, Batch-encode games for dataset creation
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional, Iterator

from data.state import CatanState
from data.encoder import StateEncoder
from data.enums import VPCategory


class GameReplay:
    def __init__(self, game_data: dict, encoder: Optional[StateEncoder] = None):
        self.game_data = game_data
        self.events = game_data['data']['eventHistory']['events']
        self.end_state = game_data['data']['eventHistory'].get('endGameState', {})
        self.total_turns = game_data['data']['eventHistory'].get('totalTurnCount', 0)
        self.play_order = game_data['data']['playOrder']
        self.encoder = encoder or StateEncoder()

        self.winner_color: Optional[int] = None
        self.player_rankings: dict[int, int] = {}
        self.player_final_vp: dict[int, int] = {}

        if self.end_state and 'players' in self.end_state:
            for color_str, pdata in self.end_state['players'].items():
                color = int(color_str)
                rank = pdata.get('rank', 99)
                self.player_rankings[color] = rank
                if pdata.get('winningPlayer', False):
                    self.winner_color = color
                vp_dict = pdata.get('victoryPoints', {})
                self.player_final_vp[color] = sum(vp_dict.values())

        self._base_state: Optional[CatanState] = None
        self._turn_boundaries: list[int] = []
        self._build_turn_index()

    @classmethod
    def from_file(cls, path: str | Path, encoder: Optional[StateEncoder] = None) -> 'GameReplay':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data, encoder)

    def _build_turn_index(self):
        """Map turn numbers to event indices."""
        self._turn_boundaries = [0]
        current_turn = 0

        for i, event in enumerate(self.events):
            sc = event.get('stateChange', {})
            cs = sc.get('currentState', {})
            new_turn = cs.get('completedTurns')
            if new_turn is not None and new_turn > current_turn:
                # Fill in any skipped turns
                while current_turn < new_turn:
                    self._turn_boundaries.append(i)
                    current_turn += 1

    @property
    def base_state(self) -> CatanState:
        if self._base_state is None:
            self._base_state = CatanState.from_initial_state(self.game_data)
        return self._base_state

    def replay_to_event(self, event_idx: int) -> CatanState:
        """Reconstruct game state after applying events [0, event_idx)."""
        state = self.base_state.copy()
        end = min(event_idx, len(self.events))
        for i in range(end):
            state.apply_event(self.events[i])
        return state

    def replay_to_turn(self, turn: int) -> CatanState:
        """Reconstruct game state at the start of a given turn"""
        if turn <= 0:
            return self.base_state.copy()
        if turn >= len(self._turn_boundaries):
            return self.replay_to_event(len(self.events))
        event_idx = self._turn_boundaries[turn]
        return self.replay_to_event(event_idx)

    def replay_full(self) -> CatanState:
        return self.replay_to_event(len(self.events))

    def get_outcome(self, color: int) -> float:
        """Get the game outcome for a player."""
        if self.winner_color == color:
            return 1.0
        vp = self.player_final_vp.get(color, 0)
        return vp / 10.0

    def get_outcome_binary(self, color: int) -> float:
        return 1.0 if self.winner_color == color else 0.0

    def generate_samples(
        self,
        sample_every_n_events: int = 5,
        min_turn: int = 8,
        include_setup: bool = False,
        perspective: str = 'all',
    ) -> Iterator[dict]:
        state = self.base_state.copy()

        for i, event in enumerate(self.events):
            state.apply_event(event)

            if i % sample_every_n_events != 0:
                continue

            if not include_setup and state.is_setup_phase():
                continue
            if state.current_turn < min_turn:
                continue

            # Determine which players to generate samples for
            if perspective == 'all':
                colors = list(state.players.keys())
            elif perspective == 'winner' and self.winner_color:
                colors = [self.winner_color]
            elif perspective == 'current':
                colors = [state.current_player_color]
            else:
                colors = list(state.players.keys())

            for color in colors:
                encoded = self.encoder.encode(state, perspective_color=color)

                yield {
                    'features': encoded['flat'],
                    'structured': {
                        'hex_features': encoded['hex_features'],
                        'corner_features': encoded['corner_features'],
                        'edge_features': encoded['edge_features'],
                        'player_features': encoded['player_features'],
                        'global_features': encoded['global_features'],
                    },
                    'outcome': self.get_outcome(color),
                    'outcome_binary': self.get_outcome_binary(color),
                    'player_color': color,
                    'turn': state.current_turn,
                    'event_idx': i,
                }

    def generate_turn_samples(
        self,
        perspective: str = 'current',
    ) -> Iterator[dict]:
        """Generate one sample per turn boundary"""
        for turn_idx in range(1, len(self._turn_boundaries)):
            state = self.replay_to_turn(turn_idx)

            if state.is_setup_phase():
                continue

            if perspective == 'current':
                colors = [state.current_player_color]
            elif perspective == 'all':
                colors = list(state.players.keys())
            else:
                colors = list(state.players.keys())

            for color in colors:
                encoded = self.encoder.encode(state, perspective_color=color)
                yield {
                    'features': encoded['flat'],
                    'structured': {
                        'hex_features': encoded['hex_features'],
                        'corner_features': encoded['corner_features'],
                        'edge_features': encoded['edge_features'],
                        'player_features': encoded['player_features'],
                        'global_features': encoded['global_features'],
                    },
                    'outcome': self.get_outcome(color),
                    'outcome_binary': self.get_outcome_binary(color),
                    'player_color': color,
                    'turn': turn_idx,
                }


class DatasetBuilder:
    def __init__(self, encoder: Optional[StateEncoder] = None):
        self.encoder = encoder or StateEncoder()
        self.features: list[np.ndarray] = []
        self.outcomes: list[float] = []
        self.outcomes_binary: list[float] = []
        self.metadata: list[dict] = []
        self.games_processed: int = 0
        self.games_failed: int = 0

    def add_game(
        self,
        path: str | Path,
        sample_every_n_events: int = 5,
        min_turn: int = 8,
        perspective: str = 'all',
    ) -> int:
        try:
            replay = GameReplay.from_file(path, self.encoder)

            if replay.winner_color is None:
                # Skip games with no winner (abandoned, etc.)
                self.games_failed += 1
                return 0

            count = 0
            for sample in replay.generate_samples(
                sample_every_n_events=sample_every_n_events,
                min_turn=min_turn,
                perspective=perspective,
            ):
                self.features.append(sample['features'])
                self.outcomes.append(sample['outcome'])
                self.outcomes_binary.append(sample['outcome_binary'])
                self.metadata.append({
                    'game_file': str(path),
                    'player_color': sample['player_color'],
                    'turn': sample['turn'],
                    'event_idx': sample['event_idx'],
                })
                count += 1

            self.games_processed += 1
            return count

        except Exception as e:
            self.games_failed += 1
            print(f"Failed to process {path}: {e}")
            return 0

    def build(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the final dataset arrays.
        """
        if not self.features:
            feat_dim = self.encoder.total_flat_size
            return (
                np.zeros((0, feat_dim), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        X = np.stack(self.features)
        y_cont = np.array(self.outcomes, dtype=np.float32)
        y_bin = np.array(self.outcomes_binary, dtype=np.float32)

        return X, y_cont, y_bin

    def save(self, path: str | Path) -> None:
        X, y_cont, y_bin = self.build()
        np.savez_compressed(
            path,
            features=X,
            outcomes=y_cont,
            outcomes_binary=y_bin,
        )
        print(
            f"Saved dataset: {X.shape[0]} samples, {X.shape[1]} features "
            f"({self.games_processed} games, {self.games_failed} failed)"
        )

    def summary(self) -> str:
        n = len(self.features)
        if n == 0:
            return "Empty dataset"
        dim = self.features[0].shape[0] if n > 0 else 0
        win_rate = sum(self.outcomes_binary) / n if n > 0 else 0
        avg_outcome = sum(self.outcomes) / n if n > 0 else 0
        return (
            f"Dataset: {n} samples × {dim} features\n"
            f"Games: {self.games_processed} processed, {self.games_failed} failed\n"
            f"Win rate in samples: {win_rate:.2%}\n"
            f"Avg outcome: {avg_outcome:.3f}"
        )
