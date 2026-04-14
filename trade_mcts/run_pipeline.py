"""
═══════════════════════════════════════════════════════════════════════
CATAN TRADE MCTS — Full Pipeline
═══════════════════════════════════════════════════════════════════════
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

from data.state import CatanState
from data.encoder import StateEncoder
from data.replay import GameReplay
from data.scoring import compute_label
from data.enums import Resource, LogType

from trade_mcts.hand_tracker import HandTracker, HandBelief
from trade_mcts.trade_encoder import Trade, TradeEncoder, generate_candidate_trades
from trade_mcts.trade_models import (
    TradeAcceptanceModel, TradeProposalPolicy,
    generate_acceptance_samples, generate_proposal_samples,
    extract_trade_events,
    train_acceptance_model, train_proposal_policy,
    TradeDataset,
)
from trade_mcts.search import TradeMCTS, find_best_trade


# ═══════════════════════════════════════════════
# Phase 1: VALIDATE
# ═══════════════════════════════════════════════

def validate_hand_tracker(games_dir: str, max_games: int = 100):
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

            observer = play_order[0]
            tracker = HandTracker(observer, play_order)
            state = replay.base_state.copy()
            game_errors = 0
            pending_steal = {}

            for event in replay.events:
                sc = event.get('stateChange', {})
                gl = sc.get('gameLogState', {})

                for log_entry in gl.values():
                    text = log_entry.get('text', {})
                    log_type = text.get('type')

                    if log_type == LogType.RESOURCE_RECEIVED:
                        color = text.get('playerColor')
                        resource = text.get('tileInfo', {}).get('resourceType')
                        if color is not None and resource is not None:
                            tracker.observe_resource_gain(color, resource, 1)

                    elif log_type == LogType.ROBBER_STEAL_PRIVATE:
                        thief = text.get('playerColor')
                        cards = text.get('cardEnums', [])
                        if thief == observer and cards:
                            pending_steal[thief] = cards[0]

                    elif log_type == LogType.ROBBER_LOSE_PRIVATE:
                        victim = text.get('playerColor')
                        cards = text.get('cardEnums', [])
                        if victim == observer and cards:
                            pending_steal[('victim', victim)] = cards[0]

                    elif log_type == LogType.ROBBER_STEAL_PUBLIC:
                        total_steal_events += 1
                        thief = text.get('playerColorThief')
                        victim = text.get('playerColorVictim')
                        if thief is not None and victim is not None:
                            if thief == observer:
                                tracker.observe_steal(thief, victim, pending_steal.get(thief))
                            elif victim == observer:
                                tracker.observe_steal(thief, victim, pending_steal.get(('victim', victim)))
                            else:
                                tracker.observe_steal(thief, victim, None)
                        pending_steal.clear()

                    elif log_type == LogType.BANK_TRADE:
                        color = text.get('playerColor')
                        given = {r: text.get('givenCardEnums', []).count(r) for r in set(text.get('givenCardEnums', []))}
                        received = {r: text.get('receivedCardEnums', []).count(r) for r in set(text.get('receivedCardEnums', []))}
                        if color is not None:
                            tracker.observe_bank_trade(color, given, received)

                    elif log_type == LogType.TRADE_COMPLETED:
                        creator = text.get('playerColorCreator')
                        offered_to = text.get('playerColorOffered')
                        offered = text.get('offeredCardEnums', [])
                        wanted = text.get('wantedCardEnums', [])
                        gives_creator = {r: offered.count(r) for r in set(offered)}
                        gives_other = {r: wanted.count(r) for r in set(wanted)}
                        if creator is not None and offered_to is not None:
                            tracker.observe_trade(creator, gives_creator, offered_to, gives_other)

                    elif log_type == 115:  # direct trade acceptance
                        proposer = text.get('playerColor')
                        accepter = text.get('acceptingPlayerColor')
                        offered = text.get('givenCardEnums', [])
                        wanted = text.get('receivedCardEnums', [])
                        gives_proposer = {r: offered.count(r) for r in set(offered)}
                        gives_accepter = {r: wanted.count(r) for r in set(wanted)}
                        if proposer is not None and accepter is not None:
                            tracker.observe_trade(proposer, gives_proposer, accepter, gives_accepter)

                    elif log_type == LogType.DISCARD:
                        color = text.get('playerColor')
                        cards = text.get('cardEnums', [])
                        discarded = {r: cards.count(r) for r in set(cards)}
                        if color is not None:
                            tracker.observe_discard(color, discarded)

                # Sync tracker from playerStates for resource losses
                for color_str, psc in sc.get('playerStates', {}).items():
                    color = int(color_str)
                    cards = psc.get('resourceCards', {}).get('cards')
                    if cards is not None and color in state.players:
                        tracker.observe_set_hand(color, cards)

                state.apply_event(event)

                if state.events_applied % 10 == 0:
                    for color in play_order:
                        if color == observer:
                            continue
                        true_hand = state.players[color].resource_counts
                        total_checks += 1
                        found = any(
                            p.hands[color].resource_counts == true_hand
                            for p in tracker.particles
                        )
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
    else:
        print("  ✓ Hand tracker is reliable for MCTS use")

    return accuracy


def _diff_cards(before: list, after: list) -> dict:
    before_counts = Counter(before)
    after_counts = Counter(after)
    diff = {}
    for res in set(list(before_counts.keys()) + list(after_counts.keys())):
        delta = after_counts.get(res, 0) - before_counts.get(res, 0)
        if delta != 0:
            diff[res] = delta
    return diff


def _parse_res(raw: dict) -> dict:
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
# Phase 2: EXTRACT
# ═══════════════════════════════════════════════

def _process_single_game(args):
    fpath, encoder = args
    accept_feats, accept_lbls = [], []
    proposal_feats, proposal_lbls = [], []
    try:
        for sample in generate_acceptance_samples(fpath, encoder):
            accept_feats.append(sample['features'])
            accept_lbls.append(sample['accepted'])
        for sample in generate_proposal_samples(fpath, encoder):
            proposal_feats.append(sample['features'])
            proposal_lbls.append(sample['label'])
    except Exception:
        pass
    return accept_feats, accept_lbls, proposal_feats, proposal_lbls


def extract_training_data(
    games_dir: str,
    output_dir: str = 'trade_data',
    max_games: int = None,
    num_workers: int = None,
    batch_size: int = 300,
):
    import multiprocessing

    print("\n" + "="*60)
    print("PHASE 2: Trade Data Extraction")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(games_dir, '*.json')))
    if max_games:
        files = files[:max_games]

    if num_workers is None:
        num_workers = 11  # optimal for Ryzen AI 9 HX 370 (12 cores, leave 1 for OS)

    print(f"Processing {len(files)} games using {num_workers} CPU cores...")

    encoder = TradeEncoder()
    args_list = [(f, encoder) for f in files]

    accept_path = os.path.join(output_dir, 'acceptance_data.npz')
    proposal_path = os.path.join(output_dir, 'proposal_data.npz')

    tmp_dir = os.path.join(output_dir, '_tmp_chunks')
    os.makedirs(tmp_dir, exist_ok=True)

    games_ok = 0
    total_accept = 0
    total_proposal = 0
    chunk_idx = 0

    a_feats, a_lbls = [], []
    p_feats, p_lbls = [], []

    def save_batch():
        nonlocal chunk_idx
        if a_feats:
            np.savez(
                os.path.join(tmp_dir, f'accept_{chunk_idx}.npz'),
                features=np.array(a_feats, dtype=np.float32),
                labels=np.array(a_lbls, dtype=np.float32),
            )
        if p_feats:
            np.savez(
                os.path.join(tmp_dir, f'proposal_{chunk_idx}.npz'),
                features=np.array(p_feats, dtype=np.float32),
                labels=np.array(p_lbls, dtype=np.float32),
            )
        chunk_idx += 1
        a_feats.clear(); a_lbls.clear()
        p_feats.clear(); p_lbls.clear()

    def process_all(file_iterator):
        nonlocal games_ok, total_accept, total_proposal
        for fi, result in enumerate(file_iterator):
            af, al, pf, pl = result
            a_feats.extend(af); a_lbls.extend(al)
            p_feats.extend(pf); p_lbls.extend(pl)
            total_accept += len(af)
            total_proposal += len(pf)
            if af or pf:
                games_ok += 1
            if (fi + 1) % batch_size == 0:
                save_batch()
            if (fi + 1) % 500 == 0:
                print(f"  [{fi+1}/{len(files)}] games={games_ok}, "
                      f"accept_samples={total_accept}, "
                      f"proposal_samples={total_proposal}")

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            process_all(pool.imap_unordered(_process_single_game, args_list, chunksize=4))
    except Exception as e:
        print(f"\n  ⚠ Multiprocessing failed ({e}), falling back to single-threaded mode...")
        a_feats.clear(); a_lbls.clear()
        p_feats.clear(); p_lbls.clear()
        games_ok = 0; total_accept = 0; total_proposal = 0
        process_all(map(_process_single_game, args_list))

    save_batch()

    print(f"\n  Merging {chunk_idx} chunks...")

    def merge_chunks(chunk_files, out_path, label):
        """Merge chunk files into one npz by streaming — never loads all at once."""
        if not chunk_files:
            print(f"  ⚠ No {label} samples extracted")
            return
        # Count total rows first
        total_rows = sum(np.load(f)['features'].shape[0] for f in chunk_files)
        first = np.load(chunk_files[0])
        n_feat = first['features'].shape[1]
        # Allocate output arrays
        X_out = np.empty((total_rows, n_feat), dtype=np.float32)
        y_out = np.empty(total_rows, dtype=np.float32)
        idx = 0
        for f in chunk_files:
            d = np.load(f)
            n = d['features'].shape[0]
            X_out[idx:idx+n] = d['features']
            y_out[idx:idx+n] = d['labels']
            idx += n
        np.savez_compressed(out_path, features=X_out, labels=y_out)
        if label == 'acceptance':
            print(f"  Acceptance data: {total_rows} samples, accept rate: {y_out.mean()*100:.1f}%")
            print(f"  Feature dim: {n_feat}")
        else:
            print(f"  Proposal data: {total_rows} samples, trade rate: {y_out.mean()*100:.1f}%")
        del X_out, y_out

    accept_files = sorted(glob.glob(os.path.join(tmp_dir, 'accept_*.npz')))
    merge_chunks(accept_files, accept_path, 'acceptance')

    proposal_files = sorted(glob.glob(os.path.join(tmp_dir, 'proposal_*.npz')))
    merge_chunks(proposal_files, proposal_path, 'proposal')

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  Saved to {output_dir}/")
    return output_dir


# ═══════════════════════════════════════════════
# Phase 3: TRAIN
# ═══════════════════════════════════════════════

def train_models(
    data_dir: str = 'trade_data',
    model_dir: str = 'trade_models',
    epochs: int = 50,
    device: str = 'cpu',
):
    print("\n" + "="*60)
    print("PHASE 3: Model Training")
    print("="*60)

    os.makedirs(model_dir, exist_ok=True)

    accept_path = os.path.join(data_dir, 'acceptance_data.npz')
    if os.path.exists(accept_path):
        print("\nTraining acceptance model...")
        # Use mmap streaming — never loads full dataset into RAM
        meta = np.load(accept_path, mmap_mode='r')
        n, d = meta['features'].shape
        rate = float(meta['labels'].mean()) * 100
        print(f"  Data: {n} samples, {d} features")
        print(f"  Accept rate: {rate:.1f}%")
        model, history = train_acceptance_model(
            features=None, labels=None, epochs=epochs, device=device, npz_path=accept_path
        )
        torch.save(model.state_dict(), os.path.join(model_dir, 'acceptance_model.pt'))
        np.savez(os.path.join(model_dir, 'acceptance_history.npz'), **{
            k: np.array(v) for k, v in history.items()
        })
        print(f"  Final val_acc: {history['val_acc'][-1]:.4f}")
        print(f"  Saved to {model_dir}/acceptance_model.pt")
    else:
        print(f"  ⚠ No acceptance data found at {accept_path}")

    proposal_path = os.path.join(data_dir, 'proposal_data.npz')
    if os.path.exists(proposal_path):
        print("\nTraining proposal policy...")
        meta = np.load(proposal_path, mmap_mode='r')
        n, d = meta['features'].shape
        print(f"  Data: {n} samples, {d} features")
        model, history = train_proposal_policy(
            features=None, labels=None, epochs=epochs, device=device, npz_path=proposal_path
        )
        torch.save(model.state_dict(), os.path.join(model_dir, 'proposal_policy.pt'))
        print(f"  Saved to {model_dir}/proposal_policy.pt")
    else:
        print(f"  ⚠ No proposal data found at {proposal_path}")


# ═══════════════════════════════════════════════
# Phase 4: SEARCH
# ═══════════════════════════════════════════════

def run_search(
    game_file: str,
    turn: int = 30,
    perspective: int = 0,
    model_dir: str = 'trade_models',
    iterations: int = 2000,
):
    print("\n" + "="*60)
    print("PHASE 4: MCTS Trade Search")
    print("="*60)

    replay = GameReplay.from_file(game_file)
    state = replay.replay_to_turn(turn)
    play_order = replay.play_order
    perspective_color = play_order[perspective]

    print(f"\nGame state at turn {turn}:")
    print(state.summary())
    print(f"\nSearching from Player {perspective_color}'s perspective...")

    tracker = HandTracker(perspective_color, play_order)
    base_state = replay.base_state.copy()
    turn_event_idx = replay._turn_boundaries[min(turn, len(replay._turn_boundaries) - 1)]

    pending_steal = {}
    for i in range(turn_event_idx):
        event = replay.events[i]
        sc = event.get('stateChange', {})
        gl = sc.get('gameLogState', {})

        for log_entry in gl.values():
            text = log_entry.get('text', {})
            log_type = text.get('type')

            if log_type == LogType.RESOURCE_RECEIVED:
                color = text.get('playerColor')
                resource = text.get('tileInfo', {}).get('resourceType')
                if color is not None and resource is not None:
                    tracker.observe_resource_gain(color, resource, 1)

            elif log_type == LogType.ROBBER_STEAL_PRIVATE:
                thief = text.get('playerColor')
                cards = text.get('cardEnums', [])
                if thief == perspective_color and cards:
                    pending_steal[thief] = cards[0]

            elif log_type == LogType.ROBBER_LOSE_PRIVATE:
                victim = text.get('playerColor')
                cards = text.get('cardEnums', [])
                if victim == perspective_color and cards:
                    pending_steal[('victim', victim)] = cards[0]

            elif log_type == LogType.ROBBER_STEAL_PUBLIC:
                thief = text.get('playerColorThief')
                victim = text.get('playerColorVictim')
                if thief is not None and victim is not None:
                    if thief == perspective_color:
                        tracker.observe_steal(thief, victim, pending_steal.get(thief))
                    elif victim == perspective_color:
                        tracker.observe_steal(thief, victim, pending_steal.get(('victim', victim)))
                    else:
                        tracker.observe_steal(thief, victim, None)
                pending_steal.clear()

            elif log_type == LogType.BANK_TRADE:
                color = text.get('playerColor')
                given = {r: text.get('givenCardEnums', []).count(r) for r in set(text.get('givenCardEnums', []))}
                received = {r: text.get('receivedCardEnums', []).count(r) for r in set(text.get('receivedCardEnums', []))}
                if color is not None:
                    tracker.observe_bank_trade(color, given, received)

            elif log_type == LogType.TRADE_COMPLETED:
                creator = text.get('playerColorCreator')
                offered_to = text.get('playerColorOffered')
                offered = text.get('offeredCardEnums', [])
                wanted = text.get('wantedCardEnums', [])
                if creator is not None and offered_to is not None:
                    tracker.observe_trade(
                        creator, {r: offered.count(r) for r in set(offered)},
                        offered_to, {r: wanted.count(r) for r in set(wanted)}
                    )

            elif log_type == 115:
                proposer = text.get('playerColor')
                accepter = text.get('acceptingPlayerColor')
                offered = text.get('givenCardEnums', [])
                wanted = text.get('receivedCardEnums', [])
                if proposer is not None and accepter is not None:
                    tracker.observe_trade(
                        proposer, {r: offered.count(r) for r in set(offered)},
                        accepter, {r: wanted.count(r) for r in set(wanted)}
                    )

            elif log_type == LogType.DISCARD:
                color = text.get('playerColor')
                cards = text.get('cardEnums', [])
                if color is not None:
                    tracker.observe_discard(color, {r: cards.count(r) for r in set(cards)})

        for color_str, psc in sc.get('playerStates', {}).items():
            color = int(color_str)
            cards = psc.get('resourceCards', {}).get('cards')
            if cards is not None:
                tracker.observe_set_hand(color, cards)

        base_state.apply_event(event)

    print(f"  Hand tracker: {tracker.num_particles} particles, "
          f"uncertainty: {tracker.uncertainty_level:.2f}")

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

    return best_trade


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Catan Trade MCTS Pipeline")
    parser.add_argument('--games-dir', type=str)
    parser.add_argument('--phase', type=str, default='all',
                        choices=['validate', 'extract', 'train', 'search', 'all'])
    parser.add_argument('--max-games', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--game-file', type=str)
    parser.add_argument('--turn', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--data-dir', type=str, default='trade_data')
    parser.add_argument('--model-dir', type=str, default='trade_models')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of CPU cores (default: 11 for Ryzen AI 9 HX 370)')

    args = parser.parse_args()

    if args.phase in ('validate', 'all') and args.games_dir:
        validate_hand_tracker(args.games_dir, args.max_games or 100)

    if args.phase in ('extract', 'all') and args.games_dir:
        extract_training_data(args.games_dir, args.data_dir, args.max_games, args.workers)

    if args.phase in ('train', 'all'):
        train_models(args.data_dir, args.model_dir, args.epochs, args.device)

    if args.phase in ('search', 'all'):
        game_file = args.game_file
        if not game_file and args.games_dir:
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