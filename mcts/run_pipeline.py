"""
═══════════════════════════════════════════════════════════════════════
CATAN TRADE MCTS — Full Pipeline
═══════════════════════════════════════════════════════════════════════

This script walks through the complete pipeline:

  1. VALIDATE   — Run the hand tracker against omniscient data to verify
                  partial observation accuracy
  2. EXTRACT    — Pull trade acceptance/proposal training samples from
                  game replays
  3. TRAIN      — Train the acceptance model and proposal policy
  4. SEARCH     — Run MCTS to find optimal trades at any game state

Requires:
  - Your `data/` package (state.py, encoder.py, replay.py, etc.)
  - Game JSON files from Colonist.io in a directory
  - PyTorch

Usage:
  python run_pipeline.py --games-dir /path/to/game/jsons --phase all
  python run_pipeline.py --games-dir /path/to/game/jsons --phase validate
  python run_pipeline.py --games-dir /path/to/game/jsons --phase train
  python run_pipeline.py --games-dir /path/to/game/jsons --phase search --game-file game.json --turn 30
"""

import argparse
import glob
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter

import torch

# ── Your existing data package ──
from data.state import CatanState
from data.encoder import StateEncoder
from data.replay import GameReplay
from data.scoring import compute_label
from data.enums import Resource, LogType

# ── MCTS package ──
from mcts.hand_tracker import HandTracker, HandBelief
from mcts.trade_encoder import Trade, TradeEncoder, generate_candidate_trades
from mcts.trade_models import (
    TradeAcceptanceModel, TradeProposalPolicy,
    generate_acceptance_samples, generate_proposal_samples,
    extract_trade_events,
    train_acceptance_model, train_proposal_policy,
    TradeDataset,
)
from mcts.search import TradeMCTS, find_best_trade


# ═══════════════════════════════════════════════
# Phase 1: VALIDATE — Hand tracker accuracy
# ═══════════════════════════════════════════════

def validate_hand_tracker(games_dir: str, max_games: int = 100):
    """
    Run the partial-observation hand tracker alongside the omniscient
    game state and verify that the true hand is always within the
    particle set.

    This is your ground-truth check before trusting the tracker in MCTS.
    """
    print("\n" + "="*60)
    print("PHASE 1: Hand Tracker Validation")
    print("="*60)

    files = sorted(glob.glob(os.path.join(games_dir, '*.json')))[:max_games]
    print(f"Validating against {len(files)} games...")

    total_checks = 0
    total_hits = 0
    total_particles = []
    total_steal_events = 0
    games_with_errors = 0

    for fi, fpath in enumerate(files):
        try:
            with open(fpath) as f:
                game_data = json.load(f)

            replay = GameReplay(game_data)
            play_order = replay.play_order

            if len(play_order) != 4:
                continue

            # Pick a random player as our observer
            observer = play_order[0]

            # Initialize tracker from observer's perspective
            tracker = HandTracker(observer, play_order)

            # Walk through events
            state = replay.base_state.copy()
            game_errors = 0

            for event in replay.events:
                log_type = event.get('type')
                payload = event.get('payload', {})
                sc = event.get('stateChange', {})

                # ── Update tracker based on event type ──

                if log_type == LogType.RESOURCE_DISTRIBUTED:
                    # Dice roll → resources distributed
                    if 'playerStates' in sc:
                        for color_str, psc in sc['playerStates'].items():
                            color = int(color_str)
                            cards_after = psc.get('resourceCards', {}).get('cards')
                            if cards_after is not None:
                                # Compare with current to find what was gained
                                current = state.players[color].resource_cards
                                gained = _diff_cards(current, cards_after)
                                for res, amt in gained.items():
                                    if amt > 0:
                                        tracker.observe_resource_gain(color, res, amt)

                elif log_type == LogType.ROBBER_STEAL:
                    total_steal_events += 1
                    thief = payload.get('playerColor')
                    victim = payload.get('victimColor')
                    stolen = payload.get('stolenCard')

                    if thief is not None and victim is not None:
                        if thief == observer or victim == observer:
                            # Observer was involved — knows the card
                            tracker.observe_steal(thief, victim, stolen)
                        else:
                            # Observer was NOT involved — uncertainty!
                            tracker.observe_steal(thief, victim, None)

                elif log_type == LogType.BANK_TRADE:
                    color = payload.get('playerColor')
                    gives = _parse_res(payload.get('offering', {}))
                    gets = _parse_res(payload.get('receiving', {}))
                    if color is not None:
                        tracker.observe_bank_trade(color, gives, gets)

                elif log_type == LogType.TRADE_COMPLETED:
                    # Player trade
                    prop = payload.get('proposerColor')
                    resp = payload.get('accepterColor')
                    offering = _parse_res(payload.get('offering', {}))
                    requesting = _parse_res(payload.get('requesting', {}))
                    if prop is not None and resp is not None:
                        tracker.observe_trade(prop, offering, resp, requesting)

                elif log_type == LogType.DISCARD:
                    color = payload.get('playerColor')
                    discarded = _parse_res(payload.get('discardedCards', {}))
                    if color is not None:
                        tracker.observe_discard(color, discarded)

                elif log_type == LogType.MONOPOLY_PLAYED:
                    color = payload.get('playerColor')
                    resource = payload.get('resource')
                    gained = payload.get('totalGained', 0)
                    if color is not None and resource is not None:
                        tracker.observe_monopoly(color, resource, gained)

                elif log_type == LogType.YEAR_OF_PLENTY:
                    color = payload.get('playerColor')
                    cards = payload.get('resources', [])
                    if color is not None:
                        for res in cards:
                            tracker.observe_resource_gain(color, res, 1)

                # Apply event to omniscient state
                state.apply_event(event)

                # ── Verify: true hand is in particle set ──
                # Check every few events to keep it manageable
                if state.events_applied % 10 == 0:
                    for color in play_order:
                        if color == observer:
                            continue

                        true_hand = state.players[color].resource_counts
                        total_checks += 1

                        # Check if any particle matches
                        found = False
                        for particle in tracker.particles:
                            if particle.hands[color].resource_counts == true_hand:
                                found = True
                                break

                        if found:
                            total_hits += 1
                        else:
                            game_errors += 1

                    total_particles.append(tracker.num_particles)

            if game_errors > 0:
                games_with_errors += 1

        except Exception as e:
            print(f"  Error processing {fpath}: {e}")
            continue

        if (fi + 1) % 20 == 0:
            acc = total_hits / max(total_checks, 1) * 100
            avg_p = np.mean(total_particles) if total_particles else 0
            print(f"  [{fi+1}/{len(files)}] Accuracy: {acc:.2f}% | "
                  f"Avg particles: {avg_p:.1f} | "
                  f"Steal events: {total_steal_events}")

    # Final report
    accuracy = total_hits / max(total_checks, 1) * 100
    avg_particles = np.mean(total_particles) if total_particles else 0
    max_particles = max(total_particles) if total_particles else 0

    print(f"\n{'─'*40}")
    print(f"Hand Tracker Validation Results:")
    print(f"  Games tested:    {len(files)}")
    print(f"  Total checks:    {total_checks}")
    print(f"  Accuracy:        {accuracy:.2f}%")
    print(f"  Games w/ errors: {games_with_errors}")
    print(f"  Steal events:    {total_steal_events}")
    print(f"  Avg particles:   {avg_particles:.1f}")
    print(f"  Max particles:   {max_particles}")
    print(f"{'─'*40}")

    if accuracy < 95:
        print("  ⚠ Accuracy below 95% — check event parsing logic")
        print("  Common issues: missed resource events, trade format mismatch")
    else:
        print("  ✓ Hand tracker is reliable for MCTS use")

    return accuracy


def _diff_cards(before: list[int], after: list[int]) -> dict[int, int]:
    """Compute the difference between two card lists."""
    before_counts = Counter(before)
    after_counts = Counter(after)
    diff = {}
    for res in set(list(before_counts.keys()) + list(after_counts.keys())):
        delta = after_counts.get(res, 0) - before_counts.get(res, 0)
        if delta != 0:
            diff[res] = delta
    return diff


def _parse_res(raw: dict) -> dict[int, int]:
    result = {}
    for k, v in raw.items():
        try:
            res = int(k)
            if isinstance(v, (int, float)) and v > 0:
                result[res] = int(v)
        except (ValueError, TypeError):
            continue
    return result


# ═══════════════════════════════════════════════
# Phase 2: EXTRACT — Training data
# ═══════════════════════════════════════════════

def extract_training_data(
    games_dir: str,
    output_dir: str = 'trade_data',
    max_games: int = None,
):
    """
    Extract trade acceptance and proposal training samples from game replays.
    Saves as .npz files for fast loading.
    """
    print("\n" + "="*60)
    print("PHASE 2: Trade Data Extraction")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(games_dir, '*.json')))
    if max_games:
        files = files[:max_games]

    print(f"Processing {len(files)} games...")

    encoder = TradeEncoder()

    # ── Acceptance data ──
    accept_features = []
    accept_labels = []
    accept_meta = []

    # ── Proposal data ──
    proposal_features = []
    proposal_labels = []

    trades_found = 0
    games_ok = 0

    for fi, fpath in enumerate(files):
        try:
            # Acceptance samples
            for sample in generate_acceptance_samples(fpath, encoder):
                accept_features.append(sample['features'])
                accept_labels.append(sample['accepted'])
                accept_meta.append({
                    'turn': sample['turn'],
                    'proposer': sample['proposer_color'],
                    'responder': sample['responder_color'],
                })
                trades_found += 1

            # Proposal samples
            for sample in generate_proposal_samples(fpath, encoder):
                proposal_features.append(sample['features'])
                proposal_labels.append(sample['label'])

            games_ok += 1

        except Exception as e:
            if fi < 5:
                print(f"  Error on {fpath}: {e}")
            continue

        if (fi + 1) % 500 == 0:
            print(f"  [{fi+1}/{len(files)}] games={games_ok}, "
                  f"accept_samples={len(accept_labels)}, "
                  f"proposal_samples={len(proposal_labels)}")

    # Save
    if accept_features:
        X_accept = np.stack(accept_features)
        y_accept = np.array(accept_labels, dtype=np.float32)

        np.savez_compressed(
            os.path.join(output_dir, 'acceptance_data.npz'),
            features=X_accept,
            labels=y_accept,
        )
        accept_rate = y_accept.mean() * 100
        print(f"\n  Acceptance data: {len(y_accept)} samples, "
              f"accept rate: {accept_rate:.1f}%")
        print(f"  Feature dim: {X_accept.shape[1]}")
    else:
        print("  ⚠ No acceptance samples extracted")

    if proposal_features:
        X_proposal = np.stack(proposal_features)
        y_proposal = np.array(proposal_labels, dtype=np.float32)

        np.savez_compressed(
            os.path.join(output_dir, 'proposal_data.npz'),
            features=X_proposal,
            labels=y_proposal,
        )
        trade_rate = y_proposal.mean() * 100
        print(f"  Proposal data: {len(y_proposal)} samples, "
              f"trade rate: {trade_rate:.1f}%")
    else:
        print("  ⚠ No proposal samples extracted")

    print(f"\n  Saved to {output_dir}/")
    return output_dir


# ═══════════════════════════════════════════════
# Phase 3: TRAIN — Trade models
# ═══════════════════════════════════════════════

def train_models(
    data_dir: str = 'trade_data',
    model_dir: str = 'trade_models',
    epochs: int = 50,
    device: str = 'cpu',
):
    """Train both the acceptance model and proposal policy."""
    print("\n" + "="*60)
    print("PHASE 3: Model Training")
    print("="*60)

    os.makedirs(model_dir, exist_ok=True)

    # ── Train acceptance model ──
    accept_path = os.path.join(data_dir, 'acceptance_data.npz')
    if os.path.exists(accept_path):
        print("\nTraining acceptance model...")
        data = np.load(accept_path)
        X, y = data['features'], data['labels']
        print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Accept rate: {y.mean()*100:.1f}%")

        model, history = train_acceptance_model(
            X, y, epochs=epochs, device=device,
        )
        torch.save(model.state_dict(), os.path.join(model_dir, 'acceptance_model.pt'))
        np.savez(os.path.join(model_dir, 'acceptance_history.npz'), **{
            k: np.array(v) for k, v in history.items()
        })
        print(f"  Final val_acc: {history['val_acc'][-1]:.4f}")
        print(f"  Saved to {model_dir}/acceptance_model.pt")
    else:
        print(f"  ⚠ No acceptance data found at {accept_path}")

    # ── Train proposal policy ──
    proposal_path = os.path.join(data_dir, 'proposal_data.npz')
    if os.path.exists(proposal_path):
        print("\nTraining proposal policy...")
        data = np.load(proposal_path)
        X, y = data['features'], data['labels']
        print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

        model, history = train_proposal_policy(
            X, y, epochs=epochs, device=device,
        )
        torch.save(model.state_dict(), os.path.join(model_dir, 'proposal_policy.pt'))
        print(f"  Saved to {model_dir}/proposal_policy.pt")
    else:
        print(f"  ⚠ No proposal data found at {proposal_path}")


# ═══════════════════════════════════════════════
# Phase 4: SEARCH — Run MCTS on a game state
# ═══════════════════════════════════════════════

def run_search(
    game_file: str,
    turn: int = 30,
    perspective: int = 0,  # 0 = first player in play order
    model_dir: str = 'trade_models',
    iterations: int = 2000,
):
    """
    Load a game, replay to a specific turn, and run MCTS to find
    the best trade from one player's perspective.
    """
    print("\n" + "="*60)
    print("PHASE 4: MCTS Trade Search")
    print("="*60)

    # Load game
    replay = GameReplay.from_file(game_file)
    state = replay.replay_to_turn(turn)
    play_order = replay.play_order
    perspective_color = play_order[perspective]

    print(f"\nGame state at turn {turn}:")
    print(state.summary())
    print(f"\nSearching from Player {perspective_color}'s perspective...")

    # Initialize hand tracker (start from game beginning)
    tracker = HandTracker(perspective_color, play_order)
    # Walk through events up to this turn to build belief state
    base_state = replay.base_state.copy()
    turn_event_idx = replay._turn_boundaries[min(turn, len(replay._turn_boundaries) - 1)]

    for i in range(turn_event_idx):
        event = replay.events[i]
        log_type = event.get('type')
        payload = event.get('payload', {})
        sc = event.get('stateChange', {})

        # Simplified tracker updates — extend as needed
        if log_type == LogType.ROBBER_STEAL:
            thief = payload.get('playerColor')
            victim = payload.get('victimColor')
            stolen = payload.get('stolenCard')
            if thief is not None and victim is not None:
                if thief == perspective_color or victim == perspective_color:
                    tracker.observe_steal(thief, victim, stolen)
                else:
                    tracker.observe_steal(thief, victim, None)

        elif log_type == LogType.RESOURCE_DISTRIBUTED and 'playerStates' in sc:
            for color_str, psc in sc['playerStates'].items():
                color = int(color_str)
                cards_after = psc.get('resourceCards', {}).get('cards')
                if cards_after is not None:
                    current = base_state.players[color].resource_cards
                    gained = _diff_cards(current, cards_after)
                    for res, amt in gained.items():
                        if amt > 0:
                            tracker.observe_resource_gain(color, res, amt)

        base_state.apply_event(event)

    print(f"  Hand tracker: {tracker.num_particles} particles, "
          f"uncertainty: {tracker.uncertainty_level:.2f}")

    # Load models if available
    acceptance_model = None
    proposal_policy = None
    te = TradeEncoder()

    accept_path = os.path.join(model_dir, 'acceptance_model.pt')
    if os.path.exists(accept_path):
        acceptance_model = TradeAcceptanceModel(te.total_feature_size)
        acceptance_model.load_state_dict(torch.load(accept_path, weights_only=True))
        acceptance_model.eval()
        print("  Loaded acceptance model")

    proposal_path = os.path.join(model_dir, 'proposal_policy.pt')
    if os.path.exists(proposal_path):
        proposal_policy = TradeProposalPolicy(te.total_feature_size)
        proposal_policy.load_state_dict(torch.load(proposal_path, weights_only=True))
        proposal_policy.eval()
        print("  Loaded proposal policy")

    # Run MCTS
    print(f"\nRunning MCTS with {iterations} iterations...")
    start = time.time()

    best_trade = find_best_trade(
        state=state,
        perspective_color=perspective_color,
        hand_tracker=tracker,
        acceptance_model=acceptance_model,
        proposal_policy=proposal_policy,
        iterations=iterations,
        verbose=True,
    )

    elapsed = time.time() - start
    print(f"Search completed in {elapsed:.2f}s")

    if best_trade is not None:
        print(f"\n→ Best trade: {best_trade}")
    else:
        print(f"\n→ Recommendation: Don't trade this turn")

    # Show what actually happened in the game at this turn
    print(f"\n── What actually happened in the game ──")
    with open(game_file) as f:
        game_data = json.load(f)
    trade_events = extract_trade_events(game_data)
    for te_data in trade_events:
        if te_data['proposer_color'] == perspective_color:
            ev_state = replay.replay_to_event(te_data['event_idx'])
            if abs(ev_state.current_turn - turn) <= 2:
                offering = ', '.join(
                    f'{v} {Resource(k).name}' for k, v in te_data['offering'].items()
                )
                requesting = ', '.join(
                    f'{v} {Resource(k).name}' for k, v in te_data['requesting'].items()
                )
                completed = "✓ completed" if te_data['completed'] else "✗ rejected"
                print(f"  Turn ~{ev_state.current_turn}: "
                      f"offered [{offering}] for [{requesting}] — {completed}")

    return best_trade


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Catan Trade MCTS Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate hand tracker
  python run_pipeline.py --games-dir /dataset --phase validate

  # Extract training data
  python run_pipeline.py --games-dir /dataset --phase extract

  # Train models
  python run_pipeline.py --phase train

  # Run search on a specific game
  python run_pipeline.py --phase search --game-file game.json --turn 30

  # Full pipeline
  python run_pipeline.py --games-dir /dataset --phase all
        """,
    )
    parser.add_argument('--games-dir', type=str, help='Directory with game JSONs')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['validate', 'extract', 'train', 'search', 'all'])
    parser.add_argument('--max-games', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--game-file', type=str, help='Game JSON for search phase')
    parser.add_argument('--turn', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--data-dir', type=str, default='trade_data')
    parser.add_argument('--model-dir', type=str, default='trade_models')

    args = parser.parse_args()

    if args.phase in ('validate', 'all') and args.games_dir:
        validate_hand_tracker(args.games_dir, args.max_games or 100)

    if args.phase in ('extract', 'all') and args.games_dir:
        extract_training_data(args.games_dir, args.data_dir, args.max_games)

    if args.phase in ('train', 'all'):
        train_models(args.data_dir, args.model_dir, args.epochs, args.device)

    if args.phase in ('search', 'all'):
        game_file = args.game_file
        if not game_file and args.games_dir:
            # Just pick the first game for demo
            files = sorted(glob.glob(os.path.join(args.games_dir, '*.json')))
            if files:
                game_file = files[0]
        if game_file:
            run_search(game_file, args.turn, model_dir=args.model_dir,
                       iterations=args.iterations)
        else:
            print("  ⚠ No game file specified for search phase")


if __name__ == '__main__':
    main()
