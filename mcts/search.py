"""
Monte Carlo Tree Search for Catan trading decisions.

Tree structure at a trade decision point:

    [DECISION] Choose trade (or no-trade) from candidate set
         │
         ├── Trade A → [CHANCE] Opponent accepts? (from acceptance model)
         │      ├── Accept  → [EVAL] Score resulting state
         │      └── Reject  → [EVAL] Score state without trade
         │
         ├── Trade B → [CHANCE] Opponent accepts?
         │      ├── Accept  → [EVAL]
         │      └── Reject  → [EVAL]
         │
         ├── ...
         │
         └── No Trade → [EVAL] Score current state

The value at leaf nodes comes from your existing win-prediction / scoring
system (data.scoring.compute_label or a trained neural evaluator).

Key design decisions:
  - Partial observation: opponent hands sampled from HandTracker particles
  - Trade actions pruned by TradeProposalPolicy (top-K)
  - Acceptance probability from TradeAcceptanceModel (chance node)
  - UCB1 for tree policy with tunable exploration constant
  - Leaf evaluation uses your composite scoring function
"""

from __future__ import annotations

import math
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

from data.state import CatanState
from data.scoring import compute_label
from data.encoder import StateEncoder

from mcts.hand_tracker import HandTracker
from mcts.trade_encoder import Trade, TradeEncoder, generate_candidate_trades
from mcts.trade_models import TradeAcceptanceModel, TradeProposalPolicy


# ═══════════════════════════════════════════════
# Tree nodes
# ═══════════════════════════════════════════════

@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    # What action led to this node
    action: Optional[Trade]  # None = no-trade or root
    action_type: str = 'root'  # 'root', 'trade', 'no_trade', 'accept', 'reject'

    # Statistics
    visit_count: int = 0
    total_value: float = 0.0

    # Tree structure
    parent: Optional['MCTSNode'] = None
    children: list['MCTSNode'] = field(default_factory=list)

    # For chance nodes
    probability: float = 1.0  # P(this outcome) for chance nodes

    # Prior from proposal policy (used in PUCT)
    prior: float = 0.0

    @property
    def q_value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_chance(self) -> bool:
        return self.action_type in ('accept', 'reject')


# ═══════════════════════════════════════════════
# MCTS Engine
# ═══════════════════════════════════════════════

class TradeMCTS:
    """
    MCTS engine for Catan trade decisions.

    Usage:
        mcts = TradeMCTS(
            state=current_game_state,
            perspective_color=my_color,
            hand_tracker=my_tracker,
            acceptance_model=acceptance_model,
            proposal_policy=proposal_policy,
        )
        best_trade = mcts.search(iterations=1000)
    """

    def __init__(
        self,
        state: CatanState,
        perspective_color: int,
        hand_tracker: HandTracker,
        acceptance_model: Optional[TradeAcceptanceModel] = None,
        proposal_policy: Optional[TradeProposalPolicy] = None,
        value_fn: Optional[Callable[[CatanState, int], float]] = None,
        exploration_constant: float = 1.4,
        max_candidates: int = 15,
    ):
        self.state = state
        self.perspective_color = perspective_color
        self.hand_tracker = hand_tracker
        self.acceptance_model = acceptance_model
        self.proposal_policy = proposal_policy
        self.exploration_constant = exploration_constant
        self.max_candidates = max_candidates

        # Value function — defaults to scoring.compute_label
        self.value_fn = value_fn or self._default_value_fn

        # Encoders
        self.trade_encoder = TradeEncoder()
        self.state_encoder = StateEncoder()

        # Root node
        self.root = MCTSNode(action=None, action_type='root')

        # Generate and score candidate trades
        self.candidates = self._generate_scored_candidates()

    def _default_value_fn(self, state: CatanState, color: int) -> float:
        """
        Default leaf evaluation using the existing scoring system.
        Uses only the components that don't need future knowledge.
        """
        from data.scoring import relative_position_score, economic_quality_score

        s_pos = relative_position_score(state, color)
        s_eco = economic_quality_score(state, color)

        # Weight toward position in mid/late game, economic early
        progress = min(state.current_turn / 80.0, 1.0)
        w_pos = 0.4 + 0.2 * progress
        w_eco = 0.6 - 0.2 * progress

        return w_pos * s_pos + w_eco * s_eco

    def _generate_scored_candidates(self) -> list[tuple[Optional[Trade], float]]:
        """Generate candidate trades and score them with the proposal policy."""
        raw_candidates = generate_candidate_trades(
            self.state,
            self.perspective_color,
            max_candidates=self.max_candidates * 3,  # generate more, then prune
        )

        if self.proposal_policy is None:
            # No policy — use all candidates with uniform priors
            return [(t, 1.0 / len(raw_candidates)) for t in raw_candidates]

        # Score each candidate
        scored = []
        features_list = []
        for trade in raw_candidates:
            feat = self.trade_encoder.encode_for_proposal(
                self.state, trade, self.perspective_color
            )
            features_list.append(feat)

        scores = self.proposal_policy.score_trades(features_list)

        # Softmax to get priors
        scores_arr = np.array(scores)
        scores_arr = scores_arr - scores_arr.max()  # numerical stability
        exp_scores = np.exp(scores_arr)
        priors = exp_scores / exp_scores.sum()

        scored = list(zip(raw_candidates, priors.tolist()))

        # Keep top-K by prior
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:self.max_candidates]

        # Renormalize priors
        total = sum(p for _, p in scored)
        scored = [(t, p / total) for t, p in scored]

        return scored

    # ─── Core MCTS loop ───

    def search(
        self,
        iterations: int = 1000,
        time_limit: Optional[float] = None,
    ) -> Optional[Trade]:
        """
        Run MCTS and return the best trade action.
        Returns None if "no trade" is the best action.
        """
        start_time = time.time()

        for i in range(iterations):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            # 1. Selection — walk tree using UCB/PUCT
            node = self._select(self.root)

            # 2. Expansion — add children if this is a leaf
            if node.visit_count > 0 and node.is_leaf:
                self._expand(node)
                # Pick a child to evaluate
                if node.children:
                    node = self._pick_child_for_rollout(node)

            # 3. Evaluation — score the leaf state
            value = self._evaluate(node)

            # 4. Backpropagation
            self._backpropagate(node, value)

        # Pick the most-visited child of root (robust child selection)
        return self._best_action()

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk tree by UCB1/PUCT until reaching a leaf or unexpanded node."""
        while not node.is_leaf:
            if any(c.visit_count == 0 for c in node.children):
                # Expand unvisited children first
                unvisited = [c for c in node.children if c.visit_count == 0]
                return random.choice(unvisited)
            node = self._best_ucb_child(node)
        return node

    def _best_ucb_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB1/PUCT score."""
        best_score = -float('inf')
        best_child = node.children[0]
        log_parent = math.log(node.visit_count + 1)

        for child in node.children:
            if child.visit_count == 0:
                return child

            # PUCT formula (used by AlphaGo/AlphaZero)
            exploitation = child.q_value
            exploration = self.exploration_constant * child.prior * (
                math.sqrt(log_parent) / (1 + child.visit_count)
            )

            # For chance nodes, weight by probability
            score = exploitation + exploration
            if child.is_chance:
                score *= child.probability

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode):
        """Expand a leaf node by adding children."""
        if node.action_type == 'root':
            # Root → trade decision nodes
            for trade, prior in self.candidates:
                action_type = 'no_trade' if trade is None else 'trade'
                child = MCTSNode(
                    action=trade,
                    action_type=action_type,
                    parent=node,
                    prior=prior,
                )
                node.children.append(child)

        elif node.action_type == 'trade':
            # Trade decision → chance nodes (accept / reject)
            trade = node.action
            accept_prob = self._get_acceptance_probability(trade)

            accept_node = MCTSNode(
                action=trade,
                action_type='accept',
                parent=node,
                probability=accept_prob,
                prior=accept_prob,
            )
            reject_node = MCTSNode(
                action=trade,
                action_type='reject',
                parent=node,
                probability=1.0 - accept_prob,
                prior=1.0 - accept_prob,
            )
            node.children = [accept_node, reject_node]

        # no_trade and chance nodes are leaf-evaluated, not expanded further

    def _pick_child_for_rollout(self, node: MCTSNode) -> MCTSNode:
        """For chance nodes, sample according to probability.
        For decision nodes, pick unvisited or best UCB."""
        # If this is a trade node with accept/reject children, sample by probability
        if node.action_type == 'trade' and node.children:
            r = random.random()
            cumulative = 0.0
            for child in node.children:
                cumulative += child.probability
                if r < cumulative:
                    return child
            return node.children[-1]

        # Otherwise pick unvisited or random
        unvisited = [c for c in node.children if c.visit_count == 0]
        if unvisited:
            return random.choice(unvisited)
        return random.choice(node.children)

    def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluate a leaf node by simulating the trade result.

        For 'no_trade' or 'reject': evaluate current state as-is.
        For 'accept': apply the trade to a copy of the state, then evaluate.
        """
        # Sample hands from belief state for this evaluation
        sampled_hands = self.hand_tracker.sample_all_hands()

        if node.action_type in ('no_trade', 'reject', 'root'):
            # Evaluate the current state
            eval_state = self.state.copy()
            # Inject sampled hands for opponents
            for color, hand in sampled_hands.items():
                if color != self.perspective_color:
                    cards = []
                    for res, count in hand.items():
                        cards.extend([res] * count)
                    eval_state.players[color].resource_cards = cards
            return self.value_fn(eval_state, self.perspective_color)

        elif node.action_type == 'accept':
            # Apply the trade and evaluate
            trade = node.action
            eval_state = self.state.copy()

            # Inject sampled hands
            for color, hand in sampled_hands.items():
                if color != self.perspective_color:
                    cards = []
                    for res, count in hand.items():
                        cards.extend([res] * count)
                    eval_state.players[color].resource_cards = cards

            # Apply trade
            proposer = eval_state.players.get(trade.proposer_color)
            responder = eval_state.players.get(trade.responder_color)

            if proposer and responder:
                # Remove offered resources from proposer, add to responder
                for res, amt in trade.offering.items():
                    for _ in range(amt):
                        if res in proposer.resource_cards:
                            proposer.resource_cards.remove(res)
                        responder.resource_cards.append(res)

                # Remove requested resources from responder, add to proposer
                for res, amt in trade.requesting.items():
                    for _ in range(amt):
                        if res in responder.resource_cards:
                            responder.resource_cards.remove(res)
                        proposer.resource_cards.append(res)

            return self.value_fn(eval_state, self.perspective_color)

        return 0.5  # fallback

    def _get_acceptance_probability(self, trade: Trade) -> float:
        """Get P(accept) from the acceptance model, or use a heuristic."""
        if self.acceptance_model is None:
            # Heuristic fallback: 50/50 baseline, penalize if proposer is leading
            proposer = self.state.players.get(trade.proposer_color)
            responder = self.state.players.get(trade.responder_color)
            if proposer and responder:
                vp_gap = proposer.total_vp - responder.total_vp
                # Less likely to trade with the leader
                base_prob = 0.3
                if vp_gap > 0:
                    base_prob -= vp_gap * 0.05
                elif vp_gap < 0:
                    base_prob += abs(vp_gap) * 0.03
                return max(0.05, min(0.8, base_prob))
            return 0.3

        # Use the trained model
        responder_color = trade.responder_color
        if responder_color is None:
            return 0.3

        features = self.trade_encoder.encode_for_acceptance(
            self.state, trade, responder_color
        )
        return self.acceptance_model.predict(features)

    def _backpropagate(self, node: MCTSNode, value: float):
        """Propagate value up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    def _best_action(self) -> Optional[Trade]:
        """Return the best trade based on visit count (robust child)."""
        if not self.root.children:
            return None

        # Most-visited child
        best = max(self.root.children, key=lambda c: c.visit_count)
        return best.action

    # ─── Diagnostics ───

    def get_action_stats(self) -> list[dict]:
        """Return statistics for each candidate action at the root."""
        stats = []
        for child in self.root.children:
            trade = child.action
            stats.append({
                'trade': str(trade) if trade else 'No Trade',
                'visits': child.visit_count,
                'q_value': child.q_value,
                'prior': child.prior,
                'action_type': child.action_type,
            })
        stats.sort(key=lambda x: x['visits'], reverse=True)
        return stats

    def print_analysis(self, top_n: int = 10):
        """Print a human-readable analysis of the search results."""
        stats = self.get_action_stats()[:top_n]

        total_visits = sum(s['visits'] for s in stats)
        print(f"\n{'='*60}")
        print(f"MCTS Trade Analysis — Player {self.perspective_color}")
        print(f"Total iterations: {self.root.visit_count}")
        print(f"Candidates explored: {len(self.root.children)}")
        print(f"{'='*60}")

        for i, s in enumerate(stats):
            visit_pct = (s['visits'] / max(total_visits, 1)) * 100
            print(f"\n  #{i+1}: {s['trade']}")
            print(f"      Visits: {s['visits']} ({visit_pct:.1f}%) | "
                  f"Q: {s['q_value']:.4f} | Prior: {s['prior']:.4f}")

            # If it's a trade node, show accept/reject stats
            node = [c for c in self.root.children
                    if str(c.action) == (str(s['trade']) if s['trade'] != 'No Trade' else str(None))
                    and c.action_type == s['action_type']]
            if node and node[0].children:
                for cc in node[0].children:
                    print(f"        {cc.action_type}: "
                          f"visits={cc.visit_count}, "
                          f"P={cc.probability:.3f}, "
                          f"Q={cc.q_value:.4f}")

        print(f"\n{'='*60}")
        best = stats[0] if stats else None
        if best:
            if best['trade'] == 'No Trade':
                print(f"RECOMMENDATION: Don't trade this turn")
            else:
                print(f"RECOMMENDATION: {best['trade']}")
        print(f"{'='*60}\n")


# ═══════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════

def find_best_trade(
    state: CatanState,
    perspective_color: int,
    hand_tracker: HandTracker,
    acceptance_model: Optional[TradeAcceptanceModel] = None,
    proposal_policy: Optional[TradeProposalPolicy] = None,
    iterations: int = 1000,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Optional[Trade]:
    """
    One-call interface to find the best trade for a player.

    Args:
        state: Current game state
        perspective_color: The player considering a trade
        hand_tracker: Belief state tracker for opponent hands
        acceptance_model: Trained acceptance classifier (optional)
        proposal_policy: Trained proposal policy (optional)
        iterations: Number of MCTS iterations
        time_limit: Max seconds to search (optional)
        verbose: Print analysis

    Returns:
        Best Trade object, or None if no trade is recommended.
    """
    mcts = TradeMCTS(
        state=state,
        perspective_color=perspective_color,
        hand_tracker=hand_tracker,
        acceptance_model=acceptance_model,
        proposal_policy=proposal_policy,
    )

    best = mcts.search(iterations=iterations, time_limit=time_limit)

    if verbose:
        mcts.print_analysis()

    return best
