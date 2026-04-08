"""
Agent classes for Catan simulation.

BaseAgent defines the interface every agent must implement:
  - choose_setup_settlement(state, color, valid_corners) -> int
  - choose_setup_road(state, color, settlement_corner, valid_edges) -> int
  - choose_action(state, color, dev_played) -> Action
  - choose_robber_hex(state, color, valid_hexes) -> int

RandomAgent  : uniformly random choices for all decisions.
MCTSAgent    : heuristic corner/road selection during setup; MCTS + evaluator
               (neural or heuristic) during the main phase.
"""

from __future__ import annotations

import random
from typing import Optional

from data.enums import BuildingType, Resource
from data.state import CatanState
from mcts.actions import PassTurn
from mcts.move_generator import get_legal_actions


# Dice probability lookup (used for setup scoring)
_DICE_PROB: dict[int, float] = {
    2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
    8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36,
}


class BaseAgent:
    """Abstract agent interface."""

    def choose_setup_settlement(
        self, state: CatanState, color: int, valid_corners: list[int]
    ) -> int:
        raise NotImplementedError

    def choose_setup_road(
        self,
        state: CatanState,
        color: int,
        settlement_corner: int,
        valid_edges: list[int],
    ) -> int:
        raise NotImplementedError

    def choose_action(self, state: CatanState, color: int, dev_played: bool):
        raise NotImplementedError

    def choose_robber_hex(
        self, state: CatanState, color: int, valid_hexes: list[int]
    ) -> int:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """
    Uniformly random agent.

    All decisions — setup placement, road direction, main-phase actions,
    robber placement — are chosen uniformly at random from the legal options.
    """

    def choose_setup_settlement(self, state, color, valid_corners):
        return random.choice(valid_corners)

    def choose_setup_road(self, state, color, settlement_corner, valid_edges):
        return random.choice(valid_edges)

    def choose_action(self, state, color, dev_played):
        actions = get_legal_actions(state, color, dev_played)
        return random.choice(actions)

    def choose_robber_hex(self, state, color, valid_hexes):
        return random.choice(valid_hexes)


class MCTSAgent(BaseAgent):
    """
    MCTS-based agent using the trained state evaluator.

    Setup phase uses a lightweight production-rate heuristic to choose the
    highest-value corner and the road that maximises future expansion options.

    Main phase uses find_best_action() (CatanMCTS) with the neural network
    evaluator (or heuristic fallback if no checkpoint is provided).

    Parameters
    ----------
    checkpoint_path : str or None
        Path to CatanNet checkpoint (.pt).  None → heuristic evaluator only.
    iterations : int
        MCTS iterations per action call.  100 is fast; 400+ is stronger.
    opponent_rounds : int
        Rounds of greedy opponent simulation per leaf evaluation.
        0 = fast one-ply; 1 = full two-ply equivalent (recommended, but slower).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        iterations: int = 100,
        opponent_rounds: int = 0,
    ):
        from mcts.evaluator import StateEvaluator

        self.evaluator = StateEvaluator(checkpoint_path)
        self.iterations = iterations
        self.opponent_rounds = opponent_rounds

    # ── Setup heuristics ──────────────────────────────────────────────────────

    def _corner_production_score(self, state: CatanState, cidx: int) -> float:
        """
        Score a corner by expected resource production + diversity + port bonus.
        Used for initial settlement placement.
        """
        topo = state.topology
        score = 0.0
        resource_types: set = set()

        for hex_idx in topo.corner_to_hexes.get(cidx, []):
            dice_num = topo.hex_dice_numbers.get(hex_idx, 0)
            resource = topo.hex_resources.get(hex_idx)
            if resource is not None and dice_num in _DICE_PROB:
                score += _DICE_PROB[dice_num]
                resource_types.add(int(resource))

        # Bonus for resource diversity (up to 3 hex types per corner)
        score += len(resource_types) * 0.03

        # Bonus for port access
        if cidx in topo.corner_ports:
            score += 0.04

        return score

    def choose_setup_settlement(self, state, color, valid_corners):
        """Pick the corner with the highest production score."""
        return max(valid_corners, key=lambda c: self._corner_production_score(state, c))

    def choose_setup_road(self, state, color, settlement_corner, valid_edges):
        """
        Pick the road edge that opens up the most valid new settlement spots.
        Falls back to a random choice if all edges score equally.
        """
        topo = state.topology
        occupied = set(state.corner_buildings.keys())

        def expansion_score(eidx: int) -> int:
            count = 0
            for cidx in topo.edge_to_corners.get(eidx, []):
                if cidx == settlement_corner or cidx in occupied:
                    continue
                if any(adj in occupied for adj in topo.get_adjacent_corners(cidx)):
                    continue
                count += 1
            return count

        return max(valid_edges, key=expansion_score)

    # ── Main-phase decision ────────────────────────────────────────────────────

    def choose_action(self, state, color, dev_played):
        from mcts.search import find_best_action

        return find_best_action(
            state,
            color,
            self.evaluator,
            iterations=self.iterations,
            opponent_rounds=self.opponent_rounds,
            dev_card_played_this_turn=dev_played,
        )

    # ── Robber placement ───────────────────────────────────────────────────────

    def choose_robber_hex(self, state, color, valid_hexes):
        """
        Place robber on the hex that maximises disruption to opponents
        (highest expected production from opponents on that hex).
        """
        topo = state.topology

        def opponent_production(hex_idx: int) -> float:
            dice_num = topo.hex_dice_numbers.get(hex_idx, 0)
            prob = _DICE_PROB.get(dice_num, 0.0)
            if prob == 0.0:
                return 0.0
            production = 0.0
            for cidx in topo.hex_to_corners.get(hex_idx, []):
                if cidx in state.corner_buildings:
                    owner, btype = state.corner_buildings[cidx]
                    if owner != color:
                        mult = 1 if btype == BuildingType.SETTLEMENT else 2
                        production += prob * mult
            return production

        return max(valid_hexes, key=opponent_production)
