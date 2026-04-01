"""
evaluate_model.py — Evaluate a trained CatanNet on held-out game replays.

Usage:
    python evaluate_model.py \\
        --checkpoint checkpoints/best.pt \\
        --games /path/to/game/jsons \\
        --max-games 200 \\
        --output eval_results/ \\
        --plot
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _try_import_catan_modules():
    try:
        from data.encoder import StateEncoder
        from data.replay import GameReplay
        from data.state import CatanState
        return True
    except ImportError:
        script_dir = Path(__file__).resolve().parent
        for candidate in [script_dir, script_dir / "catan", script_dir.parent]:
            if (candidate / "encoder.py").exists():
                sys.path.insert(0, str(candidate))
                return True
    return False


_try_import_catan_modules()
from data.encoder import StateEncoder
from data.replay import GameReplay
from data.scoring import compute_label

torch = None
CatanNet = None


def _import_torch():
    global torch, CatanNet
    import torch as _torch
    torch = _torch
    from model.catan_network import CatanNet as _CatanNet
    CatanNet = _CatanNet

def is_valid_game(replay: GameReplay) -> tuple[bool, str]:
    if replay.winner_color is None:
        return False, "no_winner"
    if len(replay.play_order) != 4:
        return False, "not_4_players"

    topo = replay.base_state.topology
    if topo.num_hexes != 19 or topo.num_corners != 54 or topo.num_edges != 72:
        return False, "non_standard_board"
    if any(t == 6 for t in topo.hex_types.values()):  # 6 == TileType.GOLD
        return False, "special_tiles"

    # Bug fix: avoid replay_full() just for the turn count — use the already-built index
    total_turns = max(len(replay._turn_boundaries) - 1, replay.total_turns, 1)
    if total_turns < 20:
        return False, "too_short"

    return True, "ok"

def evaluate_single_game(
    replay: GameReplay,
    model,
    encoder: StateEncoder,
    device,
    min_turn: int = 8,
    _timings: Optional[dict] = None,
) -> dict:
    """Replay one game turn-by-turn and collect per-turn model predictions.

    If _timings is a dict, it will be populated with:
        replay_s   — seconds spent in replay_to_turn()
        infer_s    — seconds spent on model forward passes
        n_turns    — number of turns evaluated
    """
    total_turns = max(
        len(replay._turn_boundaries) - 1,
        replay.total_turns,
        1,
    )
    colors = replay.play_order

    turns = []
    predictions = {c: [] for c in colors}
    vp_tracks = {c: [] for c in colors}
    winner_ranks = []

    t_replay = 0.0
    t_infer  = 0.0

    for ti in range(1, len(replay._turn_boundaries)):
        _t0 = time.time()
        state = replay.replay_to_turn(ti)
        t_replay += time.time() - _t0

        if state.current_turn < min_turn:
            continue

        turn_preds = {}
        for color in colors:
            encoded = encoder.encode(state, perspective_color=color)
            feat = torch.from_numpy(encoded["flat"]).unsqueeze(0).to(device)

            _t0 = time.time()
            with torch.no_grad():
                value, _, _ = model(feat)
            t_infer += time.time() - _t0

            pred = value.item()
            turn_preds[color] = pred
            predictions[color].append(pred)
            vp_tracks[color].append(state.players[color].total_vp)

        turns.append(state.current_turn)

        sorted_colors = sorted(colors, key=lambda c: turn_preds[c], reverse=True)
        winner_rank = sorted_colors.index(replay.winner_color) + 1
        winner_ranks.append(winner_rank)

    if _timings is not None:
        _timings["replay_s"] = _timings.get("replay_s", 0.0) + t_replay
        _timings["infer_s"]  = _timings.get("infer_s",  0.0) + t_infer
        _timings["n_turns"]  = _timings.get("n_turns",  0)   + len(turns)

    return {
        "turns": turns,
        "predictions": {str(c): v for c, v in predictions.items()},
        "vp": {str(c): v for c, v in vp_tracks.items()},
        "winner_color": replay.winner_color,
        "player_colors": colors,
        "total_turns": total_turns,
        "player_final_vp": {str(c): v for c, v in replay.player_final_vp.items()},
        "winner_rank": winner_ranks,
    }


def compute_aggregate_metrics(game_results: list[dict]) -> dict:
    phase_bins = {"early": [], "mid": [], "late": []}

    for gr in game_results:
        n = len(gr["turns"])
        if n == 0:
            continue
        for i, rank in enumerate(gr["winner_rank"]):
            progress = i / max(n - 1, 1)
            if progress < 0.33:
                phase_bins["early"].append(rank)
            elif progress < 0.66:
                phase_bins["mid"].append(rank)
            else:
                phase_bins["late"].append(rank)

    ranking_accuracy = {}
    for phase, ranks in phase_bins.items():
        if not ranks:
            ranking_accuracy[phase] = {"top1": 0.0, "top2": 0.0, "n": 0}
            continue
        arr = np.array(ranks)
        ranking_accuracy[phase] = {
            "top1": float((arr == 1).mean()),
            "top2": float((arr <= 2).mean()),
            "mean_rank": float(arr.mean()),
            "n": len(ranks),
        }

    final_preds = []
    final_outcomes = []
    final_won = []

    for gr in game_results:
        for color_str in gr["predictions"]:
            preds_list = gr["predictions"][color_str]
            if not preds_list:
                continue
            final_pred = preds_list[-1]  # last turn prediction
            color = int(color_str)
            won = 1.0 if color == gr["winner_color"] else 0.0
            final_vp = gr["player_final_vp"].get(color_str, 0)
            outcome = 1.0 if won else final_vp / 10.0

            final_preds.append(final_pred)
            final_outcomes.append(outcome)
            final_won.append(won)

    final_preds = np.array(final_preds)
    final_outcomes = np.array(final_outcomes)
    final_won = np.array(final_won)

    pred_outcome_corr = float(np.corrcoef(final_preds, final_outcomes)[0, 1]) if len(final_preds) > 1 else 0.0
    pred_win_corr = float(np.corrcoef(final_preds, final_won)[0, 1]) if len(final_preds) > 1 else 0.0

    n_bins = 10  # Bug fix: n_bins was used but never defined
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration = {"bin_centers": [], "predicted_mean": [], "actual_win_rate": [], "count": []}

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (final_preds >= lo) & (final_preds < hi + (1e-9 if i == n_bins - 1 else 0))
        count = int(mask.sum())
        if count > 0:
            calibration["bin_centers"].append(float((lo + hi) / 2))
            calibration["predicted_mean"].append(float(final_preds[mask].mean()))
            calibration["actual_win_rate"].append(float(final_won[mask].mean()))
            calibration["count"].append(count)

    n_points = 50
    progress_grid = np.linspace(0, 1, n_points)
    winner_curves = []
    loser_curves = []

    for gr in game_results:
        n = len(gr["turns"])
        if n < 5:
            continue
        progress = np.linspace(0, 1, n)

        for color_str, preds_list in gr["predictions"].items():
            color = int(color_str)
            is_winner = color == gr["winner_color"]
            interp = np.interp(progress_grid, progress, preds_list)
            if is_winner:
                winner_curves.append(interp)
            else:
                loser_curves.append(interp)

    avg_winner_curve = np.mean(winner_curves, axis=0).tolist() if winner_curves else []
    avg_loser_curve = np.mean(loser_curves, axis=0).tolist() if loser_curves else []
    std_winner_curve = np.std(winner_curves, axis=0).tolist() if winner_curves else []
    std_loser_curve = np.std(loser_curves, axis=0).tolist() if loser_curves else []

    rank_at_progress = {k: [] for k in range(n_points)}
    for gr in game_results:
        n = len(gr["winner_rank"])
        if n < 5:
            continue
        progress = np.linspace(0, 1, n)
        for pi in range(n_points):
            closest = int(np.argmin(np.abs(progress - progress_grid[pi])))
            rank_at_progress[pi].append(gr["winner_rank"][closest])

    rank_curve_top1 = []
    rank_curve_top2 = []
    for pi in range(n_points):
        if rank_at_progress[pi]:
            arr = np.array(rank_at_progress[pi])
            rank_curve_top1.append(float((arr == 1).mean()))
            rank_curve_top2.append(float((arr <= 2).mean()))
        else:
            rank_curve_top1.append(0.0)
            rank_curve_top2.append(0.0)

    all_winner_ranks = []
    for gr in game_results:
        all_winner_ranks.extend(gr["winner_rank"])
    all_winner_ranks = np.array(all_winner_ranks) if all_winner_ranks else np.array([0])

    end_correct = 0
    end_total = 0
    for gr in game_results:
        if gr["winner_rank"]:
            end_total += 1
            if gr["winner_rank"][-1] == 1:
                end_correct += 1

    return {
        "n_games": len(game_results),
        "n_turns_total": sum(len(gr["turns"]) for gr in game_results),
        "ranking_accuracy_by_phase": ranking_accuracy,
        "overall_top1_accuracy": float((all_winner_ranks == 1).mean()),
        "overall_top2_accuracy": float((all_winner_ranks <= 2).mean()),
        "overall_mean_winner_rank": float(all_winner_ranks.mean()),
        "end_of_game_accuracy": end_correct / max(end_total, 1),
        "pred_outcome_correlation": pred_outcome_corr,
        "pred_win_correlation": pred_win_corr,
        "calibration": calibration,
        "value_trajectory": {
            "progress_grid": progress_grid.tolist(),
            "avg_winner": avg_winner_curve,
            "avg_loser": avg_loser_curve,
            "std_winner": std_winner_curve,
            "std_loser": std_loser_curve,
        },
        "ranking_curve": {
            "progress_grid": progress_grid.tolist(),
            "top1_accuracy": rank_curve_top1,
            "top2_accuracy": rank_curve_top2,
        },
        "scatter": {
            "final_preds": final_preds.tolist(),
            "final_outcomes": final_outcomes.tolist(),
            "final_won": final_won.tolist(),
        },
    }

def generate_plots(metrics: dict, game_results: list[dict], output_dir: Path):
    """Generate matplotlib PNG charts from the aggregated metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "font.size": 11,
    })

    vt = metrics["value_trajectory"]
    if vt["avg_winner"] and vt["avg_loser"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        progress = np.array(vt["progress_grid"]) * 100

        w = np.array(vt["avg_winner"])
        w_std = np.array(vt["std_winner"])
        l = np.array(vt["avg_loser"])
        l_std = np.array(vt["std_loser"])

        ax.plot(progress, w, color="#2ecc71", linewidth=2.5, label="Winner (avg)")
        ax.fill_between(progress, w - w_std, w + w_std, color="#2ecc71", alpha=0.15)
        ax.plot(progress, l, color="#e74c3c", linewidth=2.5, label="Losers (avg)")
        ax.fill_between(progress, l - l_std, l + l_std, color="#e74c3c", alpha=0.15)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Baseline (0.5)")
        ax.set_xlabel("Game Progress (%)")
        ax.set_ylabel("Model Value Prediction")
        ax.set_title("Value Trajectory: Winner vs Losers")
        ax.legend(loc="upper left")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)

        fig.tight_layout()
        fig.savefig(plot_dir / "value_trajectory.png")
        plt.close(fig)
        print(f"    Saved value_trajectory.png")

    rc = metrics["ranking_curve"]
    if rc["top1_accuracy"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        progress = np.array(rc["progress_grid"]) * 100

        ax.plot(progress, rc["top1_accuracy"], color="#3498db", linewidth=2.5, label="Top-1 (winner ranked #1)")
        ax.plot(progress, rc["top2_accuracy"], color="#9b59b6", linewidth=2.5, label="Top-2 (winner in top 2)")
        ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random chance (25%)")
        ax.axhline(y=0.50, color="gray", linestyle=":", alpha=0.4, label="Random top-2 (50%)")

        ax.set_xlabel("Game Progress (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Winner Ranking Accuracy Over Game Progress")
        ax.legend(loc="upper left")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)

        fig.tight_layout()
        fig.savefig(plot_dir / "ranking_accuracy_curve.png")
        plt.close(fig)
        print(f"    Saved ranking_accuracy_curve.png")

    sc = metrics["scatter"]
    if sc["final_preds"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        preds = np.array(sc["final_preds"])
        outcomes = np.array(sc["final_outcomes"])
        won = np.array(sc["final_won"])

        ax = axes[0]
        ax.scatter(outcomes, preds, alpha=0.3, s=12, c="#3498db", edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
        ax.set_xlabel("Actual Outcome (VP/10 or 1.0 for winner)")
        ax.set_ylabel("Model Prediction")
        ax.set_title(f"Prediction vs Outcome (r = {metrics['pred_outcome_correlation']:.3f})")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax = axes[1]
        ax.hist(preds[won == 1], bins=30, alpha=0.7, color="#2ecc71", label="Winners", density=True)
        ax.hist(preds[won == 0], bins=30, alpha=0.7, color="#e74c3c", label="Losers", density=True)
        ax.set_xlabel("Model Prediction (final turn)")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Distribution: Winners vs Losers")
        ax.legend()

        fig.tight_layout()
        fig.savefig(plot_dir / "prediction_vs_outcome.png")
        plt.close(fig)
        print(f"    Saved prediction_vs_outcome.png")

    cal = metrics["calibration"]
    if cal["bin_centers"]:
        fig, ax = plt.subplots(figsize=(8, 8))

        centers = np.array(cal["bin_centers"])
        actual = np.array(cal["actual_win_rate"])
        predicted = np.array(cal["predicted_mean"])
        counts = np.array(cal["count"])

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
        ax.bar(centers, actual, width=0.08, alpha=0.6, color="#3498db", label="Actual win rate")
        ax.scatter(predicted, actual, s=counts / max(counts.max(), 1) * 200 + 20,
                   color="#e74c3c", zorder=5, label="Bin centers (size=count)")

        ax.set_xlabel("Predicted Value (binned)")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title("Calibration Curve")
        ax.legend(loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(plot_dir / "calibration.png")
        plt.close(fig)
        print(f"    Saved calibration.png")

    sample_games = game_results[:min(6, len(game_results))]
    if sample_games:
        n_cols = min(3, len(sample_games))
        n_rows = (len(sample_games) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        player_colors_map = {0: "#e74c3c", 1: "#3498db", 2: "#f39c12", 3: "#2ecc71"}

        for gi, gr in enumerate(sample_games):
            ax = axes[gi]
            turns = gr["turns"]
            if not turns:
                continue

            for pi, color_str in enumerate(gr["predictions"]):
                color = int(color_str)
                preds = gr["predictions"][color_str]
                is_winner = color == gr["winner_color"]
                plot_color = player_colors_map.get(pi, "gray")
                lw = 2.5 if is_winner else 1.2
                ls = "-" if is_winner else "--"
                final_vp = gr["player_final_vp"].get(color_str, "?")
                label = f"P{color} {'★' if is_winner else ''} ({final_vp}VP)"
                ax.plot(turns, preds, color=plot_color, linewidth=lw,
                        linestyle=ls, label=label, alpha=0.9 if is_winner else 0.6)

            ax.set_xlabel("Turn")
            ax.set_ylabel("Model Value")
            ax.set_title(f"Game {gi+1} ({gr['total_turns']} turns)")
            ax.legend(fontsize=8, loc="upper left")
            ax.set_ylim(0, 1)

        for gi in range(len(sample_games), len(axes)):
            axes[gi].set_visible(False)

        fig.tight_layout()
        fig.savefig(plot_dir / "sample_game_trajectories.png")
        plt.close(fig)
        print(f"    Saved sample_game_trajectories.png")
    ra = metrics["ranking_accuracy_by_phase"]
    if all(ra[p]["n"] > 0 for p in ["early", "mid", "late"]):
        fig, ax = plt.subplots(figsize=(8, 5))

        phases = ["early", "mid", "late"]
        x = np.arange(len(phases))
        width = 0.35

        top1 = [ra[p]["top1"] for p in phases]
        top2 = [ra[p]["top2"] for p in phases]

        bars1 = ax.bar(x - width / 2, top1, width, label="Top-1 Accuracy", color="#3498db")
        bars2 = ax.bar(x + width / 2, top2, width, label="Top-2 Accuracy", color="#9b59b6")

        ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Game Phase")
        ax.set_ylabel("Accuracy")
        ax.set_title("Winner Ranking Accuracy by Game Phase")
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"Early\n(0–33%)\nn={ra[p]['n']}" for p in phases
        ])

        ax.set_xticklabels([
            f"Early\nn={ra['early']['n']}",
            f"Mid\nn={ra['mid']['n']}",
            f"Late\nn={ra['late']['n']}",
        ])
        ax.legend()
        ax.set_ylim(0, 1)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        fig.savefig(plot_dir / "phase_accuracy.png")
        plt.close(fig)
        print(f"    Saved phase_accuracy.png")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CatanNet on held-out game replays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --checkpoint checkpoints/best.pt --games /dataset --max-games 200 --plot
  python evaluate_model.py --checkpoint checkpoints/best.pt --games /dataset --output eval_results/
        """,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pt file")
    parser.add_argument("--games", required=True, help="Directory of game JSON files")
    parser.add_argument("--max-games", type=int, default=200, help="Max games to evaluate")
    parser.add_argument("--min-turn", type=int, default=8, help="Skip turns before this")
    parser.add_argument("--output", default="eval_results", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for game selection")

    args = parser.parse_args()

    _import_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    arch = ckpt.get("arch", {})
    cfg = ckpt.get("config", {})
    model = CatanNet(
        input_dim=arch.get("input_dim", 1363),
        hidden_dim=arch.get("hidden_dim", cfg.get("hidden_dim", 256)),
        num_blocks=arch.get("num_blocks", cfg.get("num_blocks", 12)),
        value_hidden=arch.get("value_hidden", cfg.get("value_hidden", 128)),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.6f}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Device: {device}")

    pattern = os.path.join(args.games, "*.json")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print(f"No JSON files found in {args.games}")
        sys.exit(1)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_files)
    candidate_files = all_files[:args.max_games * 3]

    print(f"\n  Found {len(all_files)} game files, evaluating up to {args.max_games}")

    encoder = StateEncoder()
    game_results = []
    skipped = 0
    errors = 0

    t0 = time.time()
    for i, fpath in enumerate(candidate_files):
        if len(game_results) >= args.max_games:
            break

        try:
            with open(fpath) as f:
                game_data = json.load(f)
            replay = GameReplay(game_data, encoder)

            valid, reason = is_valid_game(replay)
            if not valid:
                skipped += 1
                continue

            result = evaluate_single_game(
                replay, model, encoder, device, min_turn=args.min_turn
            )
            game_results.append(result)

            if len(game_results) % 25 == 0:
                elapsed = time.time() - t0
                rate = len(game_results) / elapsed
                print(f"    {len(game_results)}/{args.max_games} games ({rate:.1f} g/s, {skipped} skipped)")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Error processing {Path(fpath).name}: {e}")

    elapsed = time.time() - t0
    print(f"\n  Evaluated {len(game_results)} games in {elapsed:.1f}s ({skipped} skipped, {errors} errors)")

    if not game_results:
        print("No valid games to evaluate!")
        sys.exit(1)

    print("\nComputing metrics...")
    metrics = compute_aggregate_metrics(game_results)

    print(f"\n{'═' * 56}")
    print(f"  EVALUATION RESULTS ({metrics['n_games']} games, {metrics['n_turns_total']:,} turns)")
    print(f"{'═' * 56}")
    print(f"  Overall winner top-1 accuracy:  {metrics['overall_top1_accuracy']:.1%}")
    print(f"  Overall winner top-2 accuracy:  {metrics['overall_top2_accuracy']:.1%}")
    print(f"  Mean winner rank:               {metrics['overall_mean_winner_rank']:.2f} / 4")
    print(f"  End-of-game accuracy:           {metrics['end_of_game_accuracy']:.1%}")
    print(f"  Prediction-outcome correlation:  {metrics['pred_outcome_correlation']:.3f}")
    print(f"  Prediction-win correlation:      {metrics['pred_win_correlation']:.3f}")
    print()

    ra = metrics["ranking_accuracy_by_phase"]
    for phase in ["early", "mid", "late"]:
        p = ra[phase]
        print(f"  {phase.capitalize():6s}  top-1={p['top1']:.1%}  top-2={p['top2']:.1%}  "
              f"mean_rank={p.get('mean_rank', 0):.2f}  (n={p['n']})")

    print()
    print("  Baselines:  random = 25% top-1, 50% top-2, mean rank 2.5")
    print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    games_path = output_dir / "game_results.json"
    with open(games_path, "w") as f:
        json.dump(game_results, f)
    print(f"  Saved game results to {games_path}")

    if args.plot:
        print("\nGenerating plots...")
        try:
            generate_plots(metrics, game_results, output_dir)
            print(f"  Plots saved to {output_dir / 'plots'}/")
        except ImportError:
            print("  matplotlib not installed — skipping plots.")
            print("  Install with: pip install matplotlib")
        except Exception as e:
            print(f"  Error generating plots: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
