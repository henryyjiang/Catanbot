"""
Baseline evaluation: MCTS agent vs 3 opponents over N games.

Two modes
---------
  vs-random    (default) Neural MCTS bot vs 3 purely random agents.
  vs-heuristic           Neural MCTS bot vs 3 heuristic-MCTS agents (no neural net).

Usage
-----
    cd /path/to/Catanbot
    python simulation/run_baseline.py [options]

Options
-------
    --mode MODE         Opponent type: vs-random | vs-heuristic (default: vs-random)
    --games N           Number of games to play (default: 100)
    --checkpoint PATH   Path to CatanNet checkpoint .pt file (default: checkpoints/best.pt)
    --iterations N      MCTS iterations per action (default: 100)
    --opponent-rounds N MCTS opponent simulation rounds (0=fast, 1=stronger; default: 0)
    --dataset DIR       Dataset directory with game JSONs (default: dataset)
    --output DIR        Output directory for results (default: simulation)
    --seed N            Random seed for reproducibility (default: none)
    --no-neural         Force heuristic evaluator even if checkpoint exists

Results are saved to:
    simulation/results_{mode}.json      (raw statistics)
    simulation/graphs_{mode}/           (PNG plots)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from glob import glob
from typing import Optional

import numpy as np

# ── Make sure project root is on sys.path ─────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.game_engine import CatanGameEngine, GameStats, create_fresh_state
from simulation.agents import RandomAgent, MCTSAgent


# ── Constants ─────────────────────────────────────────────────────────────────

PLAYER_COLORS = [1, 2, 3, 4]
MCTS_COLOR = 1           # the MCTS bot always plays as color 1
OPPONENT_COLORS = [2, 3, 4]

# Color palette for plots (MCTS = blue, opponents = shades of orange/red)
PLOT_COLORS = {
    MCTS_COLOR: "#2196F3",
    2:           "#FF7043",
    3:           "#FFA726",
    4:           "#EF5350",
}


def _make_agent_labels(mode: str) -> dict[int, str]:
    if mode == "vs-heuristic":
        return {MCTS_COLOR: "Neural MCTS", 2: "Heuristic 2", 3: "Heuristic 3", 4: "Heuristic 4"}
    return {MCTS_COLOR: "MCTS Bot", 2: "Random 2", 3: "Random 3", 4: "Random 4"}


# ── Board loading ─────────────────────────────────────────────────────────────

def _load_random_board(game_files: list[str]) -> dict:
    path = random.choice(game_files)
    with open(path) as f:
        return json.load(f)


# ── Graph generation ──────────────────────────────────────────────────────────

def generate_graphs(
    all_stats: list[GameStats],
    graphs_dir: str,
    num_games: int,
    agent_labels: dict[int, str],
    mode: str = "vs-random",
) -> None:
    os.makedirs(graphs_dir, exist_ok=True)

    wins = {c: sum(1 for s in all_stats if s.winner == c) for c in PLAYER_COLORS}
    total = len(all_stats)

    mode_title = "vs Heuristic MCTS Agents" if mode == "vs-heuristic" else "vs Random Agents"
    baseline_label = "Equal-skill baseline (25%)"

    # ── 1. Win-rate bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_list = [PLOT_COLORS[c] for c in PLAYER_COLORS]
    labels = [agent_labels[c] for c in PLAYER_COLORS]
    win_rates = [wins[c] / total * 100 for c in PLAYER_COLORS]
    bars = ax.bar(labels, win_rates, color=colors_list, edgecolor="black", linewidth=0.8)
    ax.axhline(25, color="gray", linestyle="--", linewidth=1, label=baseline_label)
    for bar, rate, w in zip(bars, win_rates, [wins[c] for c in PLAYER_COLORS]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%\n({w}/{total})",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title(f"Win Rate: {agent_labels[MCTS_COLOR]} {mode_title}\n({num_games} games)", fontsize=13)
    ax.set_ylim(0, max(win_rates) * 1.25 + 5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "win_rate_bar.png"), dpi=150)
    plt.close()

    # ── 2. Cumulative win rate over games (MCTS only) ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    cumulative_wins = np.cumsum([1 if s.winner == MCTS_COLOR else 0 for s in all_stats])
    game_nums = np.arange(1, total + 1)
    cum_rate = cumulative_wins / game_nums * 100
    ax.plot(game_nums, cum_rate, color=PLOT_COLORS[MCTS_COLOR], linewidth=2, label=agent_labels[MCTS_COLOR])
    ax.axhline(25, color="gray", linestyle="--", linewidth=1, label=baseline_label)
    ax.fill_between(game_nums, 25, cum_rate, where=(cum_rate >= 25),
                    alpha=0.15, color=PLOT_COLORS[MCTS_COLOR])
    ax.set_xlabel("Game Number", fontsize=12)
    ax.set_ylabel("Cumulative Win Rate (%)", fontsize=12)
    ax.set_title(f"{agent_labels[MCTS_COLOR]} Cumulative Win Rate Over Games", fontsize=13)
    ax.set_xlim(1, total)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "cumulative_win_rate.png"), dpi=150)
    plt.close()

    # ── 3. Final VP distribution (box plots) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    vp_data = [[s.final_vp.get(c, 0) for s in all_stats] for c in PLAYER_COLORS]
    bp = ax.boxplot(
        vp_data,
        tick_labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(10, color="crimson", linestyle="--", linewidth=1, label="Win threshold (10 VP)")
    ax.set_ylabel("Final Victory Points", fontsize=12)
    ax.set_title(f"Final VP Distribution\n({num_games} games)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "final_vp_distribution.png"), dpi=150)
    plt.close()

    # ── 4. Average VP progression over turns ─────────────────────────────────
    # Aggregate vp_at_turns across all games
    turn_keys_all: set[int] = set()
    for s in all_stats:
        turn_keys_all.update(s.vp_at_turns.keys())
    turn_keys = sorted(turn_keys_all)

    if turn_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        for color in PLAYER_COLORS:
            avg_vp_at_turn: list[float] = []
            for t in turn_keys:
                vps = [s.vp_at_turns[t][color] for s in all_stats if t in s.vp_at_turns and color in s.vp_at_turns[t]]
                avg_vp_at_turn.append(np.mean(vps) if vps else 0.0)
            ax.plot(
                turn_keys, avg_vp_at_turn,
                color=PLOT_COLORS[color],
                linewidth=2,
                label=agent_labels[color],
                marker="o", markersize=3,
            )
        ax.axhline(10, color="crimson", linestyle="--", linewidth=1, label="Win threshold")
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Average Victory Points", fontsize=12)
        ax.set_title("Average VP Progression Over Turns", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "vp_progression.png"), dpi=150)
        plt.close()

    # ── 5. Game length histogram ──────────────────────────────────────────────
    game_lengths = [s.total_turns for s in all_stats]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(game_lengths, bins=20, color="#4CAF50", edgecolor="black", linewidth=0.7, alpha=0.85)
    ax.axvline(np.mean(game_lengths), color="crimson", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(game_lengths):.1f} turns")
    ax.axvline(np.median(game_lengths), color="orange", linestyle="--", linewidth=2,
               label=f"Median: {np.median(game_lengths):.1f} turns")
    ax.set_xlabel("Game Length (player turns)", fontsize=12)
    ax.set_ylabel("Number of Games", fontsize=12)
    ax.set_title("Game Length Distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "game_length_hist.png"), dpi=150)
    plt.close()

    # ── 6. Average resources collected ───────────────────────────────────────
    avg_resources = {c: np.mean([s.resources_collected.get(c, 0) for s in all_stats]) for c in PLAYER_COLORS}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels,
        [avg_resources[c] for c in PLAYER_COLORS],
        color=colors_list,
        edgecolor="black", linewidth=0.8,
    )
    for bar, c in zip(bars, PLAYER_COLORS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{avg_resources[c]:.1f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Avg Resources Collected", fontsize=12)
    ax.set_title("Average Resources Collected Per Player", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "resources_collected.png"), dpi=150)
    plt.close()

    # ── 7. Action breakdown for MCTS bot ─────────────────────────────────────
    roads    = np.mean([s.roads_built.get(MCTS_COLOR, 0) for s in all_stats])
    settles  = np.mean([s.settlements_built.get(MCTS_COLOR, 0) for s in all_stats])
    cities   = np.mean([s.cities_built.get(MCTS_COLOR, 0) for s in all_stats])
    dev_cards = np.mean([s.dev_cards_bought.get(MCTS_COLOR, 0) for s in all_stats])

    r_roads    = np.mean([s.roads_built.get(2, 0) for s in all_stats])
    r_settles  = np.mean([s.settlements_built.get(2, 0) for s in all_stats])
    r_cities   = np.mean([s.cities_built.get(2, 0) for s in all_stats])
    r_devs     = np.mean([s.dev_cards_bought.get(2, 0) for s in all_stats])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(4)
    width = 0.35
    opp_label = "Avg Heuristic" if mode == "vs-heuristic" else "Avg Random"
    ax.bar(x - width/2, [roads, settles, cities, dev_cards], width,
           label=agent_labels[MCTS_COLOR], color=PLOT_COLORS[MCTS_COLOR], edgecolor="black", linewidth=0.8)
    ax.bar(x + width/2, [r_roads, r_settles, r_cities, r_devs], width,
           label=opp_label, color=PLOT_COLORS[2], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Roads", "Settlements", "Cities", "Dev Cards"], fontsize=11)
    ax.set_ylabel("Avg per Game", fontsize=12)
    ax.set_title(f"Average Build Actions per Game: {agent_labels[MCTS_COLOR]} vs {opp_label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "build_actions.png"), dpi=150)
    plt.close()

    # ── 8. Win rate by game number bucket (early / mid / late convergence) ────
    bucket_size = max(1, total // 5)
    bucket_labels = []
    bucket_rates = []
    for start in range(0, total, bucket_size):
        end = min(start + bucket_size, total)
        batch = all_stats[start:end]
        rate = sum(1 for s in batch if s.winner == MCTS_COLOR) / len(batch) * 100
        bucket_labels.append(f"{start+1}–{end}")
        bucket_rates.append(rate)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(bucket_rates) + 1), bucket_rates,
            color=PLOT_COLORS[MCTS_COLOR], linewidth=2, marker="s", markersize=8)
    ax.axhline(25, color="gray", linestyle="--", linewidth=1, label=baseline_label)
    ax.set_xticks(range(1, len(bucket_labels) + 1))
    ax.set_xticklabels(bucket_labels, rotation=15, fontsize=9)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title(f"{agent_labels[MCTS_COLOR]} Win Rate by Game Bucket", fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "win_rate_by_bucket.png"), dpi=150)
    plt.close()

    print(f"\nGraphs saved to: {graphs_dir}/")
    for fname in [
        "win_rate_bar.png",
        "cumulative_win_rate.png",
        "final_vp_distribution.png",
        "vp_progression.png",
        "game_length_hist.png",
        "resources_collected.png",
        "build_actions.png",
        "win_rate_by_bucket.png",
    ]:
        print(f"  {fname}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_baseline(
    num_games: int = 100,
    checkpoint_path: Optional[str] = None,
    iterations: int = 100,
    opponent_rounds: int = 0,
    dataset_dir: str = "dataset",
    output_dir: str = "simulation",
    seed: Optional[int] = None,
    force_heuristic: bool = False,
    mode: str = "vs-random",
) -> list[GameStats]:
    if mode not in ("vs-random", "vs-heuristic"):
        raise ValueError(f"Invalid mode '{mode}'. Choose: vs-random | vs-heuristic")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    game_files = glob(os.path.join(dataset_dir, "*.json"))
    if not game_files:
        raise FileNotFoundError(f"No JSON game files found in '{dataset_dir}'")

    # Resolve checkpoint
    ckpt = None
    if not force_heuristic and checkpoint_path:
        ckpt = checkpoint_path if os.path.exists(checkpoint_path) else None
        if checkpoint_path and not ckpt:
            print(f"[WARNING] Checkpoint not found: {checkpoint_path} — using heuristic.")

    agent_labels = _make_agent_labels(mode)

    # Build agents
    mcts_agent = MCTSAgent(
        checkpoint_path=ckpt,
        iterations=iterations,
        opponent_rounds=opponent_rounds,
    )
    if mode == "vs-heuristic":
        # Opponents use MCTS with heuristic evaluator (no neural net)
        opponent_agent = MCTSAgent(
            checkpoint_path=None,
            iterations=iterations,
            opponent_rounds=opponent_rounds,
        )
        agents = {
            MCTS_COLOR: mcts_agent,
            **{c: opponent_agent for c in OPPONENT_COLORS},
        }
    else:
        agents = {
            MCTS_COLOR: mcts_agent,
            **{c: RandomAgent() for c in OPPONENT_COLORS},
        }

    mode_desc = "3 Heuristic MCTS Agents" if mode == "vs-heuristic" else "3 Random Agents"
    print("=" * 65)
    print(f"Baseline Evaluation: {agent_labels[MCTS_COLOR]} vs {mode_desc}")
    print("=" * 65)
    print(f"  Mode           : {mode}")
    print(f"  Games          : {num_games}")
    print(f"  MCTS color     : {MCTS_COLOR}  ({agent_labels[MCTS_COLOR]})")
    print(f"  Evaluator      : {'CatanNet (' + ckpt + ')' if ckpt else 'Heuristic only'}")
    print(f"  MCTS iterations: {iterations}  (all players in vs-heuristic)")
    print(f"  Opponent rounds: {opponent_rounds}")
    print(f"  Dataset        : {dataset_dir}/ ({len(game_files)} board layouts)")
    print(f"  Seed           : {seed}")
    if mode == "vs-heuristic":
        print(f"  NOTE           : All 4 players run MCTS — expect ~4x slower than vs-random")
    print("=" * 65)
    print()

    all_stats: list[GameStats] = []
    start_time = time.time()

    for game_idx in range(num_games):
        game_start = time.time()
        try:
            game_data = _load_random_board(game_files)
            initial_state = create_fresh_state(game_data, PLAYER_COLORS)

            engine = CatanGameEngine(initial_state, agents, max_turns=500)
            winner, stats = engine.run_game()
            all_stats.append(stats)

        except Exception as e:
            print(f"[ERROR] Game {game_idx + 1} failed: {e}")
            continue

        # Progress line
        done = len(all_stats)
        mcts_wins = sum(1 for s in all_stats if s.winner == MCTS_COLOR)
        win_rate = mcts_wins / done * 100
        elapsed = time.time() - start_time
        avg_per_game = elapsed / done
        remaining = avg_per_game * (num_games - done)
        game_secs = time.time() - game_start

        bot_label = agent_labels[MCTS_COLOR]
        print(
            f"Game {done:4d}/{num_games}  |  "
            f"Winner: P{stats.winner}  |  "
            f"VP: {' '.join(f'P{c}={stats.final_vp.get(c,0)}' for c in PLAYER_COLORS)}  |  "
            f"Turns: {stats.total_turns:3d}  |  "
            f"{bot_label} WR: {win_rate:5.1f}%  |  "
            f"{game_secs:4.1f}s/game  ETA: {remaining/60:.1f}min"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(all_stats)
    elapsed = time.time() - start_time

    print()
    print("=" * 65)
    print(f"Final Results  ({total} completed games, {elapsed:.0f}s total)")
    print("=" * 65)
    for c in PLAYER_COLORS:
        w = sum(1 for s in all_stats if s.winner == c)
        avg_vp = np.mean([s.final_vp.get(c, 0) for s in all_stats])
        avg_res = np.mean([s.resources_collected.get(c, 0) for s in all_stats])
        tag = " ← BOT" if c == MCTS_COLOR else ""
        print(
            f"  Player {c} ({agent_labels[c]:12s}): "
            f"{w:3d} wins ({w/total*100:5.1f}%)  "
            f"avg VP={avg_vp:.2f}  avg res={avg_res:.0f}{tag}"
        )
    avg_len = np.mean([s.total_turns for s in all_stats])
    avg_sevens = np.mean([s.sevens_rolled for s in all_stats])
    print(f"\n  Avg game length : {avg_len:.1f} turns")
    print(f"  Avg sevens/game : {avg_sevens:.1f}")
    print("=" * 65)

    # ── Save raw results ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"results_{mode}.json")
    summary = {
        "num_games": total,
        "checkpoint": ckpt,
        "iterations": iterations,
        "opponent_rounds": opponent_rounds,
        "seed": seed,
        "mode": mode,
        "win_counts": {str(c): sum(1 for s in all_stats if s.winner == c) for c in PLAYER_COLORS},
        "win_rates": {str(c): sum(1 for s in all_stats if s.winner == c) / total for c in PLAYER_COLORS},
        "avg_final_vp": {str(c): float(np.mean([s.final_vp.get(c, 0) for s in all_stats])) for c in PLAYER_COLORS},
        "avg_resources_collected": {str(c): float(np.mean([s.resources_collected.get(c, 0) for s in all_stats])) for c in PLAYER_COLORS},
        "avg_game_length": float(avg_len),
        "avg_sevens_per_game": float(avg_sevens),
        "elapsed_seconds": elapsed,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ── Generate graphs ───────────────────────────────────────────────────────
    graphs_dir = os.path.join(output_dir, f"graphs_{mode}")
    generate_graphs(all_stats, graphs_dir, total, agent_labels, mode)

    return all_stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MCTS bot vs opponents and collect statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", type=str, default="vs-random",
                   choices=["vs-random", "vs-heuristic"],
                   help="Opponent type: vs-random (pure random) | vs-heuristic (heuristic MCTS)")
    p.add_argument("--games", type=int, default=100, metavar="N",
                   help="Number of games to play")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                   metavar="PATH", help="CatanNet checkpoint file")
    p.add_argument("--iterations", type=int, default=100, metavar="N",
                   help="MCTS iterations per action")
    p.add_argument("--opponent-rounds", type=int, default=0, metavar="N",
                   help="Greedy opponent simulation rounds per leaf (0=fast, 1=stronger)")
    p.add_argument("--dataset", type=str, default="dataset", metavar="DIR",
                   help="Directory containing Colonist.io game JSON files")
    p.add_argument("--output", type=str, default="simulation", metavar="DIR",
                   help="Output directory for results and graphs")
    p.add_argument("--seed", type=int, default=None, metavar="N",
                   help="Random seed for reproducibility")
    p.add_argument("--no-neural", action="store_true",
                   help="Use heuristic evaluator only (ignore checkpoint)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_baseline(
        num_games=args.games,
        checkpoint_path=args.checkpoint,
        iterations=args.iterations,
        opponent_rounds=args.opponent_rounds,
        dataset_dir=args.dataset,
        output_dir=args.output,
        seed=args.seed,
        force_heuristic=args.no_neural,
        mode=args.mode,
    )
