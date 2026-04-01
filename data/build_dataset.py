"""
onvert Colonist game JSONs into training-ready numpy arrays.

Usage:
    python build_dataset.py --input /dataset --output training_data.npz --workers 8 --max-size-gb 8
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np

from data.encoder import StateEncoder
from data.replay import GameReplay
from data.scoring import compute_label
from data.state import CatanState


FEATURE_DIM = StateEncoder().total_flat_size  # 1363

BYTES_PER_SAMPLE = FEATURE_DIM * 4 + 22

GROW_CHUNK = 500_000

SKIP_NO_WINNER = 'no_winner'
SKIP_PLAYER_COUNT = 'not_4_players'
SKIP_BOARD_GEOMETRY = 'non_standard_board'
SKIP_SPECIAL_TILES = 'special_tile_types'
SKIP_TOO_SHORT = 'too_few_turns'
SKIP_NO_SAMPLES = 'no_samples_generated'
SKIP_EXCEPTION = 'exception'

def process_single_game(args: tuple) -> dict:
    """Process one game file. Returns samples or a skip-reason dict."""
    file_path, config = args

    def _skip(reason, detail=''):
        return {'skipped': True, 'reason': reason, 'detail': detail}

    try:
        with open(file_path, 'r') as f:
            game_data = json.load(f)

        encoder = StateEncoder()
        replay = GameReplay(game_data, encoder)

        if replay.winner_color is None:
            return _skip(SKIP_NO_WINNER)
        if len(replay.play_order) != 4:
            return _skip(SKIP_PLAYER_COUNT)

        topo = replay.base_state.topology
        if topo.num_hexes != 19 or topo.num_corners != 54 or topo.num_edges != 72:
            return _skip(SKIP_BOARD_GEOMETRY)
        if any(t >= 6 for t in topo.hex_types.values()):
            return _skip(SKIP_SPECIAL_TILES)

        game_id = game_data['data'].get('gameSettings', {}).get('id', Path(file_path).stem)
        final_state = replay.replay_full()
        total_turns = max(final_state.current_turn, len(replay._turn_boundaries) - 1, 1)

        if total_turns < config.get('min_game_turns', 20):
            return _skip(SKIP_TOO_SHORT)

        sample_mode = config.get('sample_mode', 'turn')
        min_turn = config.get('min_turn', 8)
        perspective = config.get('perspective', 'all')
        event_interval = config.get('event_interval', 5)
        w_out = config.get('w_outcome', 0.50)
        w_pos = config.get('w_position', 0.30)
        w_eco = config.get('w_economic', 0.20)

        if sample_mode == 'turn':
            samples_iter = _gen_turn(replay, min_turn, perspective, total_turns, w_out, w_pos, w_eco)
        else:
            samples_iter = _gen_event(replay, min_turn, perspective, event_interval, total_turns, w_out, w_pos, w_eco)

        feat, lab, out, pos, eco, won = [], [], [], [], [], []
        m_turn, m_col, m_vp, m_lead = [], [], [], []

        for s in samples_iter:
            feat.append(s['features']); lab.append(s['label'])
            out.append(s['outcome_score']); pos.append(s['position_score'])
            eco.append(s['economic_score']); won.append(s['won_game'])
            m_turn.append(s['turn']); m_col.append(s['player_color'])
            m_vp.append(s['current_vp']); m_lead.append(s['vp_lead'])

        if not feat:
            return _skip(SKIP_NO_SAMPLES)

        return {
            'skipped': False, 'n': len(feat),
            'features': np.array(feat, dtype=np.float32),
            'labels': np.array(lab, dtype=np.float32),
            'outcome_scores': np.array(out, dtype=np.float32),
            'position_scores': np.array(pos, dtype=np.float32),
            'economic_scores': np.array(eco, dtype=np.float32),
            'won_game': np.array(won, dtype=np.bool_),
            'meta_turns': np.array(m_turn, dtype=np.int16),
            'meta_colors': np.array(m_col, dtype=np.int8),
            'meta_vp': np.array(m_vp, dtype=np.int8),
            'meta_vp_lead': np.array(m_lead, dtype=np.int8),
        }
    except Exception as e:
        return _skip(SKIP_EXCEPTION, str(e))


def _colors_for(state, perspective, winner_color):
    if perspective == 'current': return [state.current_player_color]
    elif perspective == 'winner': return [winner_color] if winner_color else []
    else: return list(state.players.keys())

def _make_sample(encoder, state, color, replay, total_turns, w_o, w_p, w_e):
    won = (color == replay.winner_color)
    enc = encoder.encode(state, perspective_color=color)
    li = compute_label(state=state, player_color=color, won_game=won,
                       current_turn=state.current_turn, total_turns=total_turns,
                       final_vp=replay.player_final_vp.get(color, 0),
                       w_outcome=w_o, w_position=w_p, w_economic=w_e)
    return {'features': enc['flat'], 'label': li['label'],
            'outcome_score': li['outcome_score'], 'position_score': li['position_score'],
            'economic_score': li['economic_score'], 'won_game': won,
            'turn': state.current_turn, 'player_color': color,
            'current_vp': li['current_vp'], 'vp_lead': li['vp_lead']}

def _gen_turn(replay, min_turn, perspective, total_turns, w_o, w_p, w_e):
    enc = StateEncoder()
    for ti in range(1, len(replay._turn_boundaries)):
        state = replay.replay_to_turn(ti)
        if state.current_turn < min_turn: continue
        for c in _colors_for(state, perspective, replay.winner_color):
            yield _make_sample(enc, state, c, replay, total_turns, w_o, w_p, w_e)

def _gen_event(replay, min_turn, perspective, interval, total_turns, w_o, w_p, w_e):
    enc = StateEncoder()
    state = replay.base_state.copy()
    for i, event in enumerate(replay.events):
        state.apply_event(event)
        if i % interval != 0: continue
        if state.current_turn < min_turn: continue
        for c in _colors_for(state, perspective, replay.winner_color):
            yield _make_sample(enc, state, c, replay, total_turns, w_o, w_p, w_e)

class DiskWriter:
    ARRAYS = [
        ('features',        np.float32, True),   # 2D
        ('labels',          np.float32, False),
        ('outcome_scores',  np.float32, False),
        ('position_scores', np.float32, False),
        ('economic_scores', np.float32, False),
        ('won_game',        np.bool_,   False),
        ('meta_turns',      np.int16,   False),
        ('meta_colors',     np.int8,    False),
        ('meta_vp',         np.int8,    False),
        ('meta_vp_lead',    np.int8,    False),
    ]

    def __init__(self, tmp_dir: str, feat_dim: int):
        self.tmp_dir = tmp_dir
        self.feat_dim = feat_dim
        self.n = 0
        self.cap = 0
        self._maps: dict[str, np.memmap] = {}

    def _grow(self, new_cap: int):
        for name, dtype, is_2d in self.ARRAYS:
            path = os.path.join(self.tmp_dir, f'{name}.npy')
            shape = (new_cap, self.feat_dim) if is_2d else (new_cap,)

            if name in self._maps:
                old = self._maps[name]
                old.flush()
                new = np.memmap(path, dtype=dtype, mode='r+', shape=shape)
                self._maps[name] = new
            else:
                m = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
                self._maps[name] = m

        self.cap = new_cap

    def write(self, result: dict):
        n = result['n']
        needed = self.n + n
        if needed > self.cap:
            self._grow(max(needed, self.cap + GROW_CHUNK))

        s, e = self.n, self.n + n
        for name, _, _ in self.ARRAYS:
            self._maps[name][s:e] = result[name]
        self.n = e

    def flush(self):
        for m in self._maps.values():
            m.flush()

    def save(self, output_path: str):
        self.flush()
        n = self.n
        est_gb = (n * BYTES_PER_SAMPLE) / 1024**3

        print(f"  Saving {n:,} samples to {output_path}...")

        if est_gb < 4.0:
            # Compress into a single .npz
            arrays = {name: self._maps[name][:n] for name, _, _ in self.ARRAYS}
            np.savez_compressed(output_path, **arrays)
        else:
            out_dir = output_path.replace('.npz', '') + '_data'
            os.makedirs(out_dir, exist_ok=True)
            print(f"  Dataset is {est_gb:.1f} GB — saving as .npy files in {out_dir}/")

            chunk = 500_000
            for name, dtype, is_2d in self.ARRAYS:
                shape = (n, self.feat_dim) if is_2d else (n,)
                dst_path = os.path.join(out_dir, f'{name}.npy')
                dst = np.memmap(dst_path, dtype=dtype, mode='w+', shape=shape)
                src = self._maps[name]
                for i in range(0, n, chunk):
                    dst[i:min(i+chunk, n)] = src[i:min(i+chunk, n)]
                dst.flush()
                del dst

            with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
                json.dump({
                    'n_samples': n, 'feature_dim': self.feat_dim,
                    'load_example': (
                        f"features = np.memmap('features.npy', dtype='float32', "
                        f"mode='r', shape=({n}, {self.feat_dim}))\n"
                        f"labels = np.memmap('labels.npy', dtype='float32', "
                        f"mode='r', shape=({n},))"
                    ),
                }, f, indent=2)

    def stats(self) -> dict:
        n = self.n
        if n == 0: return {}
        L = self._maps['labels'][:n]
        O = self._maps['outcome_scores'][:n]
        P = self._maps['position_scores'][:n]
        E = self._maps['economic_scores'][:n]
        W = self._maps['won_game'][:n]
        return {
            'label_mean': float(L.mean()), 'label_std': float(L.std()),
            'label_min': float(L.min()), 'label_max': float(L.max()),
            'outcome_mean': float(O.mean()), 'outcome_std': float(O.std()),
            'position_mean': float(P.mean()), 'position_std': float(P.std()),
            'economic_mean': float(E.mean()), 'economic_std': float(E.std()),
            'win_count': int(W.sum()), 'win_pct': float(W.sum() / n * 100),
        }

def build_dataset(
    input_dir, output_path, workers=1, sample_mode='turn', perspective='all',
    min_turn=8, event_interval=5, min_game_turns=20,
    max_games=None, max_samples=None, max_size_gb=None,
    w_outcome=0.50, w_position=0.30, w_economic=0.20,
):
    pattern = os.path.join(input_dir, '*.json')
    files = sorted(glob.glob(pattern))
    if max_games: files = files[:max_games]
    if not files:
        print(f"No JSON files found in {input_dir}"); sys.exit(1)

    # Compute sample cap
    sample_cap = max_samples
    if max_size_gb is not None:
        size_cap = int((max_size_gb * 1024**3) / BYTES_PER_SAMPLE)
        if sample_cap is None or size_cap < sample_cap:
            sample_cap = size_cap

    if sample_cap:
        print(f"  Sample cap:  {sample_cap:,} (~{sample_cap * BYTES_PER_SAMPLE / 1024**3:.1f} GB)")

    config = dict(sample_mode=sample_mode, perspective=perspective, min_turn=min_turn,
                  event_interval=event_interval, min_game_turns=min_game_turns,
                  w_outcome=w_outcome, w_position=w_position, w_economic=w_economic)

    tmp_dir = tempfile.mkdtemp(prefix='catan_ds_')
    writer = DiskWriter(tmp_dir, FEATURE_DIM)
    start = time.time()
    games_ok = 0; total_samples = 0; cap_hit = False
    skip_reasons: dict[str, int] = {}

    work_items = [(f, config) for f in files]

    def _handle(result):
        nonlocal games_ok, total_samples, cap_hit
        if result.get('skipped', False):
            r = result.get('reason', 'unknown')
            skip_reasons[r] = skip_reasons.get(r, 0) + 1
            return

        n = result['n']
        if sample_cap and total_samples + n > sample_cap:
            remain = sample_cap - total_samples
            if remain <= 0: cap_hit = True; return
            for k in [a[0] for a in DiskWriter.ARRAYS]:
                result[k] = result[k][:remain]
            result['n'] = remain; n = remain

        writer.write(result)
        total_samples += n; games_ok += 1
        if sample_cap and total_samples >= sample_cap:
            cap_hit = True

    try:
        if workers > 1:
            with Pool(workers) as pool:
                for i, result in enumerate(pool.imap_unordered(
                        process_single_game, work_items, chunksize=16)):
                    _handle(result)
                    if (i+1) % 500 == 0 or cap_hit:
                        el = time.time() - start
                        gb = total_samples * BYTES_PER_SAMPLE / 1024**3
                        sk = sum(skip_reasons.values())
                        print(f"  {i+1}/{len(files)} ({games_ok} ok, {sk} skip) "
                              f"— {total_samples:,} samples ({gb:.1f} GB) "
                              f"— {(i+1)/el:.0f} g/s")
                        writer.flush()
                    if cap_hit:
                        print(f"\n  ⚠ Cap reached ({sample_cap:,}). Stopping.")
                        pool.terminate(); break
        else:
            for i, item in enumerate(work_items):
                _handle(process_single_game(item))
                if (i+1) % 500 == 0 or cap_hit:
                    el = time.time() - start
                    gb = total_samples * BYTES_PER_SAMPLE / 1024**3
                    sk = sum(skip_reasons.values())
                    print(f"  {i+1}/{len(files)} ({games_ok} ok, {sk} skip) "
                          f"— {total_samples:,} samples ({gb:.1f} GB) "
                          f"— {(i+1)/el:.0f} g/s")
                    writer.flush()
                if cap_hit:
                    print(f"\n  ⚠ Cap reached ({sample_cap:,}). Stopping."); break

        if total_samples == 0:
            print("No valid samples generated!"); sys.exit(1)

        writer.save(output_path)
        st = writer.stats()
        elapsed = time.time() - start
        sk_total = sum(skip_reasons.values())

        # File size
        if os.path.exists(output_path):
            sz = os.path.getsize(output_path) / 1024**2
        else:
            d = output_path.replace('.npz', '') + '_data'
            sz = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1024**2 if os.path.isdir(d) else 0

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def main():
    p = argparse.ArgumentParser(description="Build training dataset from Colonist game JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  python build_dataset.py --input /dataset --output data.npz --workers 8 --max-size-gb 8
  python build_dataset.py --input /dataset --output data.npz --perspective current --max-size-gb 10
  python build_dataset.py --input /dataset --output data.npz --max-samples 2000000
  python build_dataset.py --input /dataset --output test.npz --max-games 100""")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sample-mode", choices=['turn', 'event'], default='turn')
    p.add_argument("--perspective", choices=['all', 'current', 'winner'], default='all')
    p.add_argument("--min-turn", type=int, default=8)
    p.add_argument("--event-interval", type=int, default=5)
    p.add_argument("--min-game-turns", type=int, default=20)
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-size-gb", type=float, default=None)
    p.add_argument("--w-outcome", type=float, default=0.50)
    p.add_argument("--w-position", type=float, default=0.30)
    p.add_argument("--w-economic", type=float, default=0.20)
    a = p.parse_args()
    build_dataset(a.input, a.output, a.workers, a.sample_mode, a.perspective,
                  a.min_turn, a.event_interval, a.min_game_turns, a.max_games,
                  a.max_samples, a.max_size_gb, a.w_outcome, a.w_position, a.w_economic)

if __name__ == "__main__":
    main()
