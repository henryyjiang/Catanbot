"""
Usage:
    pytest test_pipeline.py -v --dataset /dataset
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pytest

from data.enums import Resource, BuildingType, DevCard, VPCategory
from data.topology import BoardTopology
from data.state import CatanState
from data.encoder import StateEncoder
from data.replay import GameReplay, DatasetBuilder

DATASET_DIR = os.environ.get("CATAN_DATASET_DIR", "./dataset")


def _pick_game_file() -> str:
    pattern = os.path.join(DATASET_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        pytest.skip(f"No JSON files found in {DATASET_DIR}")
    return files[0]


def _pick_random_game_files(n: int = 5) -> list[str]:
    pattern = os.path.join(DATASET_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        pytest.skip(f"No JSON files found in {DATASET_DIR}")
    return random.sample(files, min(n, len(files)))


@pytest.fixture(scope="session")
def game_file() -> str:
    return _pick_game_file()


@pytest.fixture(scope="session")
def replay(game_file) -> GameReplay:
    return GameReplay.from_file(game_file)


@pytest.fixture(scope="session")
def encoder() -> StateEncoder:
    return StateEncoder()

class TestTopology:

    def test_hex_count(self, replay):
        assert replay.base_state.topology.num_hexes == 19

    def test_corner_count(self, replay):
        assert replay.base_state.topology.num_corners == 54

    def test_edge_count(self, replay):
        assert replay.base_state.topology.num_edges == 72

    def test_hex_corner_adjacency(self, replay):
        topo = replay.base_state.topology
        for hex_idx, corners in topo.hex_to_corners.items():
            assert 3 <= len(corners) <= 6, (
                f"Hex {hex_idx} has {len(corners)} corners, expected 3–6"
            )

    def test_corner_hex_adjacency(self, replay):
        topo = replay.base_state.topology
        for cidx, hexes in topo.corner_to_hexes.items():
            assert 1 <= len(hexes) <= 3, (
                f"Corner {cidx} touches {len(hexes)} hexes, expected 1–3"
            )

    def test_edge_corner_adjacency(self, replay):
        topo = replay.base_state.topology
        for eidx, corners in topo.edge_to_corners.items():
            assert 1 <= len(corners) <= 2, (
                f"Edge {eidx} has {len(corners)} corners, expected 1–2"
            )

    def test_dice_mapping_covers_18_hexes(self, replay):
        topo = replay.base_state.topology
        total = sum(len(v) for v in topo.dice_to_hexes.values())
        assert total == 18

    def test_ports_exist(self, replay):
        topo = replay.base_state.topology
        assert len(topo.corner_ports) > 0, "Board should have port corners"

class TestStateReplay:

    def test_initial_state_is_empty(self, replay):
        s0 = replay.base_state
        assert s0.current_turn == 0
        assert len(s0.corner_buildings) == 0
        assert len(s0.edge_roads) == 0

    def test_setup_produces_correct_pieces(self, replay):
        for turn in range(8, 15):
            state = replay.replay_to_turn(turn)
            if len(state.corner_buildings) == 8 and len(state.edge_roads) == 8:
                break
        else:
            state = replay.replay_to_turn(12)

        for color in replay.play_order:
            buildings = state.get_buildings_for_player(color)
            roads = state.get_roads_for_player(color)
            assert len(buildings) >= 2, (
                f"Player {color} has {len(buildings)} buildings after setup"
            )
            assert len(roads) >= 2, (
                f"Player {color} has {len(roads)} roads after setup"
            )
            for btype in buildings.values():
                assert btype in (BuildingType.SETTLEMENT, BuildingType.CITY)

    def test_full_replay_applies_all_events(self, replay):
        s_final = replay.replay_full()
        assert s_final.events_applied == len(replay.events)

    def test_replay_to_turn_monotonic_buildings(self, replay):
        prev_buildings = 0
        prev_roads = 0
        for turn in range(0, min(len(replay._turn_boundaries), 50), 5):
            state = replay.replay_to_turn(turn)
            b = len(state.corner_buildings)
            r = len(state.edge_roads)
            assert b >= prev_buildings, (
                f"Buildings decreased at turn {turn}: {prev_buildings} → {b}"
            )
            assert r >= prev_roads, (
                f"Roads decreased at turn {turn}: {prev_roads} → {r}"
            )
            prev_buildings = b
            prev_roads = r

    def test_winner_detected(self, replay):
        if replay.end_state and replay.end_state.get("players"):
            assert replay.winner_color is not None, (
                "endGameState has players but no winner was detected"
            )

    def test_state_summary_runs(self, replay):
        """summary() should not crash at any point."""
        for turn in [0, 10, 30]:
            state = replay.replay_to_turn(turn)
            text = state.summary()
            assert isinstance(text, str)
            assert len(text) > 0
class TestEncoding:

    def test_flat_vector_size(self, replay, encoder):
        state = replay.replay_to_turn(20)
        result = encoder.encode(state)
        assert result["flat"].shape[0] == encoder.total_flat_size

    def test_structured_shapes(self, replay, encoder):
        state = replay.replay_to_turn(20)
        result = encoder.encode(state)

        assert result["hex_features"].shape == (19, encoder.HEX_FEATURES)
        assert result["corner_features"].shape == (54, encoder.CORNER_FEATURES)
        assert result["edge_features"].shape == (72, encoder.EDGE_FEATURES)
        assert result["player_features"].shape == (4, encoder.PLAYER_FEATURES)
        assert result["global_features"].shape == (encoder.GLOBAL_FEATURES,)

    def test_no_nan_or_inf(self, replay, encoder):
        state = replay.replay_to_turn(30)
        flat = encoder.encode_flat(state)
        assert not np.any(np.isnan(flat)), "NaN found in feature vector"
        assert not np.any(np.isinf(flat)), "Inf found in feature vector"

    def test_features_in_reasonable_range(self, replay, encoder):
        state = replay.replay_to_turn(30)
        flat = encoder.encode_flat(state)
        assert flat.min() >= -0.01, f"Min feature {flat.min():.4f} below -0.01"
        assert flat.max() <= 2.0, f"Max feature {flat.max():.4f} above 2.0"

    def test_perspective_rotation_changes_output(self, replay, encoder):
        state = replay.replay_to_turn(30)
        colors = list(state.players.keys())
        if len(colors) < 2:
            pytest.skip("Need at least 2 players")

        enc_a = encoder.encode_flat(state, perspective_color=colors[0])
        enc_b = encoder.encode_flat(state, perspective_color=colors[1])
        assert not np.allclose(enc_a, enc_b), (
            "Different perspectives should produce different encodings"
        )

    def test_feature_names_match_size(self, encoder):
        names = encoder.feature_names()
        assert len(names) == encoder.total_flat_size

    def test_initial_state_encodes_cleanly(self, replay, encoder):
        """Encoding the initial (empty) state should not crash."""
        state = replay.base_state.copy()
        flat = encoder.encode_flat(state)
        assert flat.shape[0] == encoder.total_flat_size
        assert not np.any(np.isnan(flat))


# ─── Sample Generation Tests ────────────────────────────────────────────────

class TestSampleGeneration:

    def test_generates_samples(self, replay):
        samples = list(replay.generate_samples(
            sample_every_n_events=10, min_turn=8, perspective="all"
        ))
        assert len(samples) > 0

    def test_sample_structure(self, replay):
        samples = list(replay.generate_samples(
            sample_every_n_events=20, min_turn=8, perspective="all"
        ))
        s = samples[0]
        for key in ("features", "outcome", "outcome_binary", "player_color", "turn"):
            assert key in s, f"Missing key: {key}"

    def test_outcomes_in_valid_range(self, replay):
        samples = list(replay.generate_samples(
            sample_every_n_events=10, min_turn=8, perspective="all"
        ))
        for s in samples:
            assert 0.0 <= s["outcome"] <= 1.0, (
                f"Outcome {s['outcome']} out of range"
            )
            assert s["outcome_binary"] in (0.0, 1.0), (
                f"Binary outcome {s['outcome_binary']} not 0 or 1"
            )

    def test_different_turns_produce_different_features(self, replay):
        samples = list(replay.generate_turn_samples(perspective="current"))
        if len(samples) < 2:
            pytest.skip("Not enough turn samples to compare")
        assert not np.allclose(samples[0]["features"], samples[-1]["features"])

    def test_turn_samples_generate(self, replay):
        samples = list(replay.generate_turn_samples(perspective="current"))
        assert len(samples) > 0

class TestStateConsistency:

    def test_no_negative_vp(self, replay):
        state = replay.base_state.copy()
        for i, event in enumerate(replay.events):
            state.apply_event(event)
            for color, player in state.players.items():
                assert player.total_vp >= 0, (
                    f"Event {i}: Player {color} has negative VP ({player.total_vp})"
                )

    def test_no_negative_resources(self, replay):
        state = replay.base_state.copy()
        for i, event in enumerate(replay.events):
            state.apply_event(event)
            for color, player in state.players.items():
                assert player.total_resources >= 0, (
                    f"Event {i}: Player {color} has {player.total_resources} resources"
                )

    def test_no_negative_piece_counts(self, replay):
        state = replay.base_state.copy()
        for i, event in enumerate(replay.events):
            state.apply_event(event)
            for color, player in state.players.items():
                assert player.settlements_remaining >= 0, (
                    f"Event {i}: Player {color} settlements_remaining < 0"
                )
                assert player.cities_remaining >= 0, (
                    f"Event {i}: Player {color} cities_remaining < 0"
                )
                assert player.roads_remaining >= 0, (
                    f"Event {i}: Player {color} roads_remaining < 0"
                )

    def test_winner_vp_close_to_endgame(self, replay):
        if not replay.winner_color:
            pytest.skip("No winner detected")
        final = replay.replay_full()
        winner = final.players[replay.winner_color]
        expected = replay.player_final_vp.get(replay.winner_color, 0)
        diff = abs(winner.total_vp - expected)
        assert diff <= 2, (
            f"Winner VP mismatch: tracked={winner.total_vp}, "
            f"expected={expected} (diff={diff})"
        )


class TestMultiGame:
    """Tests that run across several random games to catch edge cases."""

    def test_multiple_games_parse_without_error(self):
        files = _pick_random_game_files(5)
        for path in files:
            replay = GameReplay.from_file(path)
            state = replay.replay_full()
            assert state.events_applied == len(replay.events), (
                f"{path}: applied {state.events_applied}/{len(replay.events)} events"
            )

    def test_multiple_games_encode_without_nan(self):
        enc = StateEncoder()
        files = _pick_random_game_files(5)
        for path in files:
            replay = GameReplay.from_file(path)
            state = replay.replay_to_turn(20)
            flat = enc.encode_flat(state)
            assert not np.any(np.isnan(flat)), f"NaN in {path}"
            assert not np.any(np.isinf(flat)), f"Inf in {path}"

    def test_dataset_builder(self):
        files = _pick_random_game_files(3)
        builder = DatasetBuilder()
        for path in files:
            builder.add_game(path, sample_every_n_events=20, min_turn=8)
        X, y_cont, y_bin = builder.build()
        assert X.shape[0] > 0, "DatasetBuilder produced 0 samples"
        assert X.shape[1] == builder.encoder.total_flat_size
        assert y_cont.shape[0] == X.shape[0]
        assert y_bin.shape[0] == X.shape[0]

class TestStateCopy:
    def test_copy_is_independent(self, replay):
        state_a = replay.replay_to_turn(20)
        state_b = state_a.copy()
        state_b.robber_hex = 999
        state_b.current_turn = 9999
        dummy_color = list(state_b.players.keys())[0]
        state_b.players[dummy_color].resource_cards.append(99)

        assert state_a.robber_hex != 999
        assert state_a.current_turn != 9999
        assert 99 not in state_a.players[dummy_color].resource_cards

    def test_copy_preserves_topology_reference(self, replay):
        state_a = replay.replay_to_turn(20)
        state_b = state_a.copy()
        assert state_a.topology is state_b.topology


def _run_cli():
    parser = argparse.ArgumentParser(description="Catan AI framework tests")
    parser.add_argument(
        "--dataset", default="./dataset",
        help="Path to directory containing game JSON files (default: /dataset)"
    )
    args = parser.parse_args()

    global DATASET_DIR
    DATASET_DIR = args.dataset
    os.environ["CATAN_DATASET_DIR"] = args.dataset
    pattern = os.path.join(DATASET_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    print(f"  Game files found: {len(files)}")
    if not files:
        print(f"\n  ✗ No JSON files found in {DATASET_DIR}")
        print(f"  Usage: python {sys.argv[0]} --dataset /path/to/json/files")
        sys.exit(1)
    print()

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x", 
    ])
    sys.exit(exit_code)


if __name__ == "__main__":
    _run_cli()
