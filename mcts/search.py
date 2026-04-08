"""
MCTS search engine for Catan main-phase decisions.

Architecture
------------
Value-MCTS with opponent simulation.  The tree is always one-ply for OUR
decisions (root → our action → child state).  The evaluation at each leaf is
deepened by running `opponent_rounds` rounds of greedy opponent play using the
fast heuristic evaluator, then scoring the resulting state with CatanNet.

Why this design:
  - Our decisions are what we control → put them in the tree so MCTS can
    allocate budget to promising branches.
  - Opponent decisions are uncertain → simulate them greedily with a fast
    heuristic rather than branching the tree (which would explode exponentially).
  - After opponent simulation the leaf state reflects one full round of
    responses, so CatanNet sees a position where the resources we gained (e.g.
    from Monopoly) have already been partially spent by us or contested by
    opponents.  This fixes the systematic undervaluation of card-gaining actions
    that occurs with one-ply evaluation.

Parameters that matter most
---------------------------
  iterations     Higher → stabler visit counts → more confident best action.
                 800 is a good default; use 1500+ for tournament-quality play.
  opponent_rounds  1 = simulate one full round of all opponents after our
                 action (recommended).  0 = original one-ply (fast but biased).
  exploration_constant  C in PUCT.  1.4 works well; lower → more greedy.

For multi-action turns, call find_best_action() in a loop until PassTurn is
returned. The caller tracks dev_card_played_this_turn across calls.

Usage
-----
    from mcts import find_best_action, StateEvaluator, apply_action
    from mcts.actions import PassTurn, PlayKnight, PlayMonopoly

    ev = StateEvaluator("checkpoints/best.pt")
    dev_played = False
    while True:
        action = find_best_action(state, my_color, ev, verbose=True)
        if isinstance(action, PassTurn):
            break
        state = apply_action(state, action, my_color)
        if isinstance(action, (PlayKnight, PlayMonopoly)):
            dev_played = True
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data.state import CatanState
from mcts.actions import Action, PassTurn
from mcts.evaluator import StateEvaluator
from mcts.move_generator import get_legal_actions
from mcts.state_transition import apply_action


# How many candidate actions we evaluate per opponent when simulating their
# turn.  Capped to keep opponent simulation cheap (heuristic is fast but
# enumerating 20+ actions × 3 opponents × 800 iterations still adds up).
_MAX_OPP_CANDIDATES = 8


# ─── Tree node ────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    """One node in the search tree (always a direct child of root)."""

    state: CatanState          # state AFTER applying this node's action
    action: Optional[Action]   # action that produced this state; None at root
    parent: Optional["MCTSNode"]
    prior: float = 1.0         # PUCT prior — uniform unless a policy net is used

    visit_count: int = 0
    total_value: float = 0.0
    children: list["MCTSNode"] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count else 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ─── MCTS engine ──────────────────────────────────────────────────────────────

class CatanMCTS:
    """
    Monte Carlo Tree Search for Catan main-phase decisions.

    Parameters
    ----------
    state : CatanState
        Current game state (after dice roll, before any main-phase action).
    color : int
        Player colour we are optimising for.
    evaluator : StateEvaluator
        Wraps CatanNet (or heuristic) — used for final leaf evaluation.
    exploration_constant : float
        C in PUCT.  1.4 is a good starting value.
    opponent_rounds : int
        How many rounds of greedy opponent play to simulate after our action
        before calling the evaluator.  0 = one-ply (fast, slightly biased).
        1 = one full round of opponents (recommended default).
    dev_card_played_this_turn : bool
        True if a dev card has already been played this turn.
    dirichlet_alpha : float
        Concentration of Dirichlet noise added to root priors.  0 = off.
    dirichlet_weight : float
        Fraction of root prior replaced by noise (AlphaZero uses 0.25).
    """

    def __init__(
        self,
        state: CatanState,
        color: int,
        evaluator: StateEvaluator,
        exploration_constant: float = 1.4,
        opponent_rounds: int = 1,
        dev_card_played_this_turn: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
    ):
        self.color = color
        self.evaluator = evaluator
        self.exploration_constant = exploration_constant
        self.opponent_rounds = opponent_rounds
        self.dev_card_played_this_turn = dev_card_played_this_turn
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

        self.root = MCTSNode(state=state, action=None, parent=None, prior=1.0)

    # ─── Public API ──────────────────────────────────────────────────────────

    def search(
        self,
        iterations: int = 800,
        time_limit: Optional[float] = None,
    ) -> Action:
        """
        Run MCTS for `iterations` iterations (or `time_limit` seconds).
        Returns the recommended Action.
        """
        self._expand_root()
        if not self.root.children:
            return PassTurn()

        if self.dirichlet_alpha > 0 and len(self.root.children) > 1:
            self._add_dirichlet_noise(self.root)

        start = time.time()
        for _ in range(iterations):
            if time_limit and (time.time() - start) > time_limit:
                break
            node = self._select(self.root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        return self._best_action()

    def get_action_stats(self) -> list[dict]:
        """Per-action statistics for root's children, sorted by visit count."""
        stats = [
            {
                "action": str(c.action),
                "visits": c.visit_count,
                "q_value": round(c.q_value, 4),
                "prior": round(c.prior, 4),
            }
            for c in self.root.children
        ]
        stats.sort(key=lambda x: x["visits"], reverse=True)
        return stats

    def print_analysis(self, top_n: int = 10) -> None:
        stats = self.get_action_stats()[:top_n]
        total = max(sum(s["visits"] for s in stats), 1)
        print(f"\n{'='*62}")
        print(f"MCTS — Player {self.color} | "
              f"{self.root.visit_count} iters | "
              f"{len(self.root.children)} candidates | "
              f"opp_rounds={self.opponent_rounds}")
        print(f"{'='*62}")
        for i, s in enumerate(stats):
            pct = s["visits"] / total * 100
            print(f"  #{i+1:2d} [{s['visits']:4d} visits, {pct:4.1f}%] "
                  f"Q={s['q_value']:.4f}  {s['action']}")
        print(f"{'='*62}")
        if stats:
            print(f"  BEST: {stats[0]['action']}")
        print(f"{'='*62}\n")

    # ─── Opponent simulation ─────────────────────────────────────────────────

    def _simulate_opponents(self, state: CatanState, rounds: int) -> CatanState:
        """
        Simulate `rounds` rounds of opponent play using a fast greedy heuristic.

        Each opponent evaluates up to _MAX_OPP_CANDIDATES non-pass actions
        using evaluate_fast() (heuristic only — no GPU calls) and takes
        whichever improves their position most.  If no action helps, they pass.

        Dice rolls are not simulated — we assume resource production averages
        out across MCTS iterations and the evaluator captures production rate
        through the economic score.

        Stochastic actions (knight steal, monopoly) introduce realistic
        variance across iterations, which is desirable.
        """
        for _ in range(rounds):
            for opp_color in state.player_colors:
                if opp_color == self.color:
                    continue

                opp_actions = get_legal_actions(state, opp_color)
                # Score the current state as the baseline (passing)
                baseline = self.evaluator.evaluate_fast(state, opp_color)
                best_score = baseline
                best_action: Action = PassTurn()

                # Evaluate non-pass candidates (capped for speed)
                candidates = [a for a in opp_actions if not isinstance(a, PassTurn)]
                random.shuffle(candidates)  # avoid systematic bias from ordering
                for action in candidates[:_MAX_OPP_CANDIDATES]:
                    try:
                        next_state = apply_action(state, action, opp_color)
                        score = self.evaluator.evaluate_fast(next_state, opp_color)
                        if score > best_score:
                            best_score = score
                            best_action = action
                    except Exception:
                        continue

                state = apply_action(state, best_action, opp_color)

        return state

    # ─── Internal MCTS methods ───────────────────────────────────────────────

    def _expand_root(self) -> None:
        """Populate root's children — one per legal action for our color."""
        actions = get_legal_actions(
            self.root.state,
            self.color,
            dev_card_played_this_turn=self.dev_card_played_this_turn,
        )
        if not actions:
            return
        prior = 1.0 / len(actions)
        for action in actions:
            child_state = apply_action(self.root.state, action, self.color)
            self.root.children.append(MCTSNode(
                state=child_state,
                action=action,
                parent=self.root,
                prior=prior,
            ))

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        n = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        w = self.dirichlet_weight
        for child, eta in zip(node.children, noise):
            child.prior = (1 - w) * child.prior + w * float(eta)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a child using PUCT, preferring unvisited nodes first."""
        unvisited = [c for c in node.children if c.visit_count == 0]
        if unvisited:
            return random.choice(unvisited)
        return self._best_puct_child(node)

    def _best_puct_child(self, node: MCTSNode) -> MCTSNode:
        c = self.exploration_constant
        sqrt_parent = math.sqrt(node.visit_count)
        best_score = -float("inf")
        best_child = node.children[0]
        for child in node.children:
            puct = child.q_value + c * child.prior * sqrt_parent / (1 + child.visit_count)
            if puct > best_score:
                best_score = puct
                best_child = child
        return best_child

    def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluate a leaf node.

        If opponent_rounds > 0, first simulate that many rounds of opponent
        greedy play, then score the resulting state with the full evaluator
        (neural network if loaded).  This makes the value estimate reflect
        what the position looks like after opponents have responded to our move,
        rather than scoring the state immediately after our action alone.
        """
        if self.opponent_rounds > 0:
            sim_state = self._simulate_opponents(node.state, self.opponent_rounds)
            return self.evaluator.evaluate(sim_state, self.color)
        return self.evaluator.evaluate(node.state, self.color)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    def _best_action(self) -> Action:
        """Robust child: most visits, Q-value as tiebreaker."""
        if not self.root.children:
            return PassTurn()
        return max(self.root.children, key=lambda c: (c.visit_count, c.q_value)).action


# ─── Convenience function ─────────────────────────────────────────────────────

def find_best_action(
    state: CatanState,
    color: int,
    evaluator: StateEvaluator,
    iterations: int = 800,
    time_limit: Optional[float] = None,
    opponent_rounds: int = 1,
    exploration_constant: float = 1.4,
    dev_card_played_this_turn: bool = False,
    verbose: bool = False,
) -> Action:
    """
    Find the best main-phase action for `color` and return it.

    Call apply_action(state, action, color) to get the next state, then call
    this again for the next action decision.  Repeat until PassTurn is returned.

    Parameters
    ----------
    iterations : int
        Number of MCTS iterations.  800 balances speed and accuracy for
        real-time play.  Use 1500+ for stronger offline analysis.
    opponent_rounds : int
        Rounds of greedy opponent simulation in each leaf evaluation.
        1 is the recommended default (full 2-ply equivalent without tree bloat).
        Set to 0 to revert to fast one-ply if latency is critical.
    time_limit : float | None
        Hard wall-clock cap in seconds (overrides iterations if hit first).
    """
    mcts = CatanMCTS(
        state=state,
        color=color,
        evaluator=evaluator,
        exploration_constant=exploration_constant,
        opponent_rounds=opponent_rounds,
        dev_card_played_this_turn=dev_card_played_this_turn,
    )
    best = mcts.search(iterations=iterations, time_limit=time_limit)
    if verbose:
        mcts.print_analysis()
    return best


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import os
    import sys

    DATASET_DIR = os.environ.get("CATAN_DATASET_DIR", "./dataset")
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    if not files:
        print(f"No JSON files found in {DATASET_DIR}")
        sys.exit(1)

    from data.replay import GameReplay
    from mcts.actions import PlayKnight, PlayMonopoly, PlayRoadBuilding, PlayYearOfPlenty

    replay = GameReplay.from_file(files[0])
    print(f"Loaded game: {os.path.basename(files[0])}")
    print(f"Players: {replay.play_order}")

    ckpt = "checkpoints/best.pt"
    ev = StateEvaluator(ckpt if os.path.exists(ckpt) else None)

    errors = 0

    # ── Main search tests ────────────────────────────────────────────────────
    for turn in [15, 25, 40]:
        state = replay.replay_to_turn(turn)
        color = state.current_player_color
        print(f"\n── Turn {turn} | Player {color} ──")
        print(state.summary())

        t0 = time.time()
        mcts = CatanMCTS(state=state, color=color, evaluator=ev, opponent_rounds=1)
        best = mcts.search(iterations=500)
        elapsed = time.time() - t0

        mcts.print_analysis(top_n=6)
        print(f"  Search time: {elapsed:.2f}s for 500 iters (opponent_rounds=1)")

        if best is None:
            print("  ERROR: got None action")
            errors += 1
            continue

        new_state = apply_action(state, best, color)
        for c, p in new_state.players.items():
            if p.total_resources < 0:
                print(f"  ERROR: negative resources for player {c} after {best}")
                errors += 1

    # ── Demonstrate opponent_rounds effect on Turn 25 Monopoly decision ──────
    print("\n── Monopoly sensitivity: opponent_rounds=0 vs 1 (Turn 25) ──")
    state25 = replay.replay_to_turn(25)
    color25 = state25.current_player_color
    print(f"Player {color25} | "
          f"resources: {dict(state25.players[color25].resource_counts)} | "
          f"dev cards: {state25.players[color25].dev_cards}")

    for rounds in [0, 1]:
        mcts = CatanMCTS(
            state=state25, color=color25, evaluator=ev, opponent_rounds=rounds
        )
        mcts.search(iterations=500)
        stats = mcts.get_action_stats()
        best_stat = stats[0]
        print(f"  opp_rounds={rounds}: best={best_stat['action']} "
              f"(Q={best_stat['q_value']:.4f}, visits={best_stat['visits']})")

    # ── Full turn simulation ─────────────────────────────────────────────────
    print("\n── Full turn simulation (find_best_action loop, Turn 25) ──")
    state = replay.replay_to_turn(25)
    color = state.current_player_color
    print(f"Player {color} | resources: {dict(state.players[color].resource_counts)}")

    dev_played = False
    step = 0
    while step < 10:
        action = find_best_action(
            state, color, ev,
            iterations=500,
            opponent_rounds=1,
            dev_card_played_this_turn=dev_played,
        )
        print(f"  Step {step+1}: {action}")
        if isinstance(action, PassTurn):
            break
        if isinstance(action, (PlayKnight, PlayMonopoly, PlayRoadBuilding, PlayYearOfPlenty)):
            dev_played = True
        state = apply_action(state, action, color)
        step += 1
    print(f"  Turn ended after {step} action(s)")

    if errors == 0:
        print("\nAll search tests passed.")
    else:
        print(f"\n{errors} error(s) found.")
        sys.exit(1)
