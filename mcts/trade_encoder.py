"""
Trade representation and feature encoding for the trade models.

A trade is represented as:
  - proposer_color: who is offering the trade
  - responder_color: who the trade is directed at (or None for open offers)
  - offering: dict[int, int] — resources the proposer gives
  - requesting: dict[int, int] — resources the proposer wants

The TradeEncoder produces a fixed-size feature vector for each
(game_state, trade, responder) triple, suitable for both the
acceptance model and the proposal policy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from data.enums import Resource, BuildingType, DevCard, VPCategory
from data.state import CatanState, PlayerState
from data.encoder import StateEncoder
from data.topology import BoardTopology
from data.scoring import DICE_PROB


@dataclass
class Trade:
    """Represents a single trade proposal."""
    proposer_color: int
    responder_color: Optional[int]  # None = open offer to any player
    offering: dict[int, int]  # resource_type → amount proposer gives
    requesting: dict[int, int]  # resource_type → amount proposer wants

    @property
    def offering_total(self) -> int:
        return sum(self.offering.values())

    @property
    def requesting_total(self) -> int:
        return sum(self.requesting.values())

    @property
    def is_valid(self) -> bool:
        """Basic validity check."""
        return (
            self.offering_total > 0
            and self.requesting_total > 0
            and not any(r in self.offering for r in self.requesting)
        )

    def reversed(self) -> 'Trade':
        """Return the same trade from the responder's perspective."""
        return Trade(
            proposer_color=self.responder_color or 0,
            responder_color=self.proposer_color,
            offering=dict(self.requesting),
            requesting=dict(self.offering),
        )

    def __repr__(self):
        off = ', '.join(f'{v} {Resource(k).name}' for k, v in self.offering.items() if v > 0)
        req = ', '.join(f'{v} {Resource(k).name}' for k, v in self.requesting.items() if v > 0)
        return f'Trade({self.proposer_color}→{self.responder_color}: gives [{off}] for [{req}])'


# ──────────────────────────────────────────────────
# Canonical trade actions — the pruned action space
# ──────────────────────────────────────────────────

def generate_candidate_trades(
    state: CatanState,
    proposer_color: int,
    max_candidates: int = 20,
    include_no_trade: bool = True,
) -> list[Optional[Trade]]:
    """
    Generate a set of plausible candidate trades for MCTS expansion.

    Strategy:
    1. Identify what the proposer likely NEEDS (resources for next build)
    2. Identify what the proposer can OFFER (surplus resources)
    3. Generate 1:1 and 2:1 trades from surplus → needed
    4. For each opponent, check if the trade is even feasible

    Returns a list of Trade objects. None represents "no trade" action.
    """
    candidates: list[Optional[Trade]] = []
    if include_no_trade:
        candidates.append(None)  # Always include the option to not trade

    proposer = state.players.get(proposer_color)
    if proposer is None:
        return candidates

    hand = proposer.resource_counts
    other_colors = [c for c in state.player_colors if c != proposer_color]

    # Determine what we need based on what we could build
    needed = _identify_needed_resources(hand)
    surplus = _identify_surplus_resources(hand, needed)

    if not needed or not surplus:
        return candidates

    # Generate 1:1 trades: offer 1 surplus for 1 needed
    for need_res in needed:
        for surp_res in surplus:
            if need_res == surp_res:
                continue
            for responder_color in other_colors:
                trade = Trade(
                    proposer_color=proposer_color,
                    responder_color=responder_color,
                    offering={surp_res: 1},
                    requesting={need_res: 1},
                )
                candidates.append(trade)

    # Generate 2:1 trades if we have enough surplus
    for need_res in needed:
        for surp_res in surplus:
            if need_res == surp_res:
                continue
            if hand.get(surp_res, 0) >= 2:
                for responder_color in other_colors:
                    trade = Trade(
                        proposer_color=proposer_color,
                        responder_color=responder_color,
                        offering={surp_res: 2},
                        requesting={need_res: 1},
                    )
                    candidates.append(trade)

    # Limit to max_candidates (keep None + top scored)
    if len(candidates) > max_candidates:
        # Keep None, then take a diverse sample
        no_trade = [c for c in candidates if c is None]
        trades = [c for c in candidates if c is not None]
        # Prioritize 1:1 trades and diversity of opponents
        trades = trades[:max_candidates - len(no_trade)]
        candidates = no_trade + trades

    return candidates


def _identify_needed_resources(hand: dict[int, int]) -> list[int]:
    """What resources does this player need for common builds?"""
    needed = set()

    lum = hand.get(Resource.LUMBER, 0)
    brk = hand.get(Resource.BRICK, 0)
    wol = hand.get(Resource.WOOL, 0)
    grn = hand.get(Resource.GRAIN, 0)
    ore = hand.get(Resource.ORE, 0)

    # Settlement: lumber, brick, wool, grain
    if lum < 1: needed.add(Resource.LUMBER)
    if brk < 1: needed.add(Resource.BRICK)
    if wol < 1: needed.add(Resource.WOOL)
    if grn < 1: needed.add(Resource.GRAIN)

    # City: 3 ore, 2 grain
    if ore < 3: needed.add(Resource.ORE)
    if grn < 2: needed.add(Resource.GRAIN)

    # Dev card: wool, grain, ore
    if wol < 1: needed.add(Resource.WOOL)
    if grn < 1: needed.add(Resource.GRAIN)
    if ore < 1: needed.add(Resource.ORE)

    # Road: lumber, brick
    if lum < 1: needed.add(Resource.LUMBER)
    if brk < 1: needed.add(Resource.BRICK)

    return [r.value if isinstance(r, Resource) else r for r in needed]


def _identify_surplus_resources(
    hand: dict[int, int], needed: list[int]
) -> list[int]:
    """Resources we have more than 1 of and don't urgently need."""
    surplus = []
    for r in Resource:
        count = hand.get(r.value, 0)
        if count >= 2 and r.value not in needed:
            surplus.append(r.value)
        elif count >= 3:
            # Even if we need it, 3+ means we can spare one
            surplus.append(r.value)
    # If no strict surplus, consider anything with count >= 1 that's not needed
    if not surplus:
        for r in Resource:
            count = hand.get(r.value, 0)
            if count >= 1 and r.value not in needed:
                surplus.append(r.value)
    return surplus


# ──────────────────────────────────────────────
# Trade feature encoder
# ──────────────────────────────────────────────

class TradeEncoder:
    """
    Encodes a (state, trade, responder) triple into features for the
    acceptance model and the proposal policy.

    Feature groups:
    1. Base game state (from StateEncoder, relative to perspective player)
    2. Trade-specific features (what's being offered/requested)
    3. Relationship features (VP gap, production complementarity, etc.)
    4. Contextual features (turn number, game phase, recent trades)

    The key insight: we encode the FULL observable game state because
    acceptance depends on everything — not just the trade itself.
    """

    # Trade-specific feature dimensions
    TRADE_FEATURES = 33

    def __init__(self):
        self.state_encoder = StateEncoder()
        self._total_size = None

    @property
    def total_feature_size(self) -> int:
        if self._total_size is None:
            self._total_size = self.state_encoder.total_flat_size + self.TRADE_FEATURES
        return self._total_size

    def encode_for_acceptance(
        self,
        state: CatanState,
        trade: Trade,
        responder_color: int,
    ) -> np.ndarray:
        """
        Encode from the RESPONDER's perspective — they decide whether to accept.
        The state encoding is relative to the responder, and trade features
        describe what THEY would gain/lose.
        """
        # Base state from responder's perspective
        base = self.state_encoder.encode_flat(state, perspective_color=responder_color)

        # Trade-specific features
        trade_feat = self._encode_trade_features(state, trade, responder_color)

        return np.concatenate([base, trade_feat])

    def encode_for_proposal(
        self,
        state: CatanState,
        trade: Optional[Trade],
        proposer_color: int,
    ) -> np.ndarray:
        """
        Encode from the PROPOSER's perspective — used by the proposal policy
        to score candidate trades.
        """
        base = self.state_encoder.encode_flat(state, perspective_color=proposer_color)

        if trade is None:
            # "No trade" action — zero trade features
            trade_feat = np.zeros(self.TRADE_FEATURES, dtype=np.float32)
        else:
            trade_feat = self._encode_trade_features(state, trade, proposer_color)

        return np.concatenate([base, trade_feat])

    def _encode_trade_features(
        self,
        state: CatanState,
        trade: Trade,
        perspective_color: int,
    ) -> np.ndarray:
        """
        33 trade-specific features:

        [0-4]   Resources being offered (from perspective player's view) / 5
        [5-9]   Resources being requested / 5
        [10]    Net card change for perspective player / 5
        [11]    Total cards offered / 5
        [12]    Total cards requested / 5
        [13]    Is perspective player the proposer? (0/1)
        [14]    VP difference (proposer VP - responder VP) / 10
        [15]    Proposer VP / 10
        [16]    Responder VP / 10
        [17]    Proposer is current leader (0/1)
        [18]    Responder is current leader (0/1)
        [19]    Turn progress (current_turn / 100)
        [20]    Proposer total resources / 20
        [21]    Responder total resources / 20
        [22]    Proposer production rate (normalized)
        [23]    Responder production rate (normalized)
        [24]    Trade gives responder a resource they don't produce (0/1)
        [25]    Trade takes a resource responder produces well (0/1)
        [26]    Responder can build settlement after accepting (0/1)
        [27]    Responder can build city after accepting (0/1)
        [28]    Responder can buy dev card after accepting (0/1)
        [29]    Responder can build road after accepting (0/1)
        [30]    Proposer can build settlement after trade (0/1)
        [31]    Proposer can build city after trade (0/1)
        [32]    Proposer is close to winning (VP >= 8) (0/1)
        """
        feat = np.zeros(self.TRADE_FEATURES, dtype=np.float32)

        proposer = state.players.get(trade.proposer_color)
        responder = state.players.get(trade.responder_color) if trade.responder_color else None

        if proposer is None or responder is None:
            return feat

        is_proposer = (perspective_color == trade.proposer_color)

        # What perspective player gives and gets
        if is_proposer:
            gives = trade.offering
            gets = trade.requesting
        else:
            gives = trade.requesting
            gets = trade.offering

        # [0-4] Resources offered by perspective player
        for r in Resource:
            feat[r.value - 1] = gives.get(r.value, 0) / 5.0

        # [5-9] Resources requested by perspective player
        for r in Resource:
            feat[5 + r.value - 1] = gets.get(r.value, 0) / 5.0

        # [10-12] Net change and totals
        feat[10] = (sum(gets.values()) - sum(gives.values())) / 5.0
        feat[11] = sum(gives.values()) / 5.0
        feat[12] = sum(gets.values()) / 5.0

        # [13] Is proposer
        feat[13] = 1.0 if is_proposer else 0.0

        # [14-16] VP features
        prop_vp = proposer.total_vp
        resp_vp = responder.total_vp
        feat[14] = (prop_vp - resp_vp) / 10.0
        feat[15] = prop_vp / 10.0
        feat[16] = resp_vp / 10.0

        # [17-18] Leader status
        max_vp = max(p.total_vp for p in state.players.values())
        feat[17] = 1.0 if prop_vp >= max_vp else 0.0
        feat[18] = 1.0 if resp_vp >= max_vp else 0.0

        # [19] Turn progress
        feat[19] = min(state.current_turn / 100.0, 1.0)

        # [20-21] Total resources
        feat[20] = proposer.total_resources / 20.0
        feat[21] = responder.total_resources / 20.0

        # [22-23] Production rates
        feat[22] = self._production_rate(state, trade.proposer_color) / 0.9
        feat[23] = self._production_rate(state, trade.responder_color) / 0.9

        # [24-25] Production complementarity
        resp_produces = self._produced_resources(state, trade.responder_color)
        # Trade gives responder something they don't produce
        for res in trade.offering.keys():
            if res not in resp_produces:
                feat[24] = 1.0
                break
        # Trade takes something responder produces well
        for res in trade.requesting.keys():
            if res in resp_produces:
                feat[25] = 1.0
                break

        # [26-29] What responder can build after accepting
        resp_hand_after = dict(responder.resource_counts)
        for res, amt in trade.offering.items():
            resp_hand_after[res] = resp_hand_after.get(res, 0) + amt
        for res, amt in trade.requesting.items():
            resp_hand_after[res] = resp_hand_after.get(res, 0) - amt

        feat[26] = 1.0 if self._can_build_settlement(resp_hand_after) else 0.0
        feat[27] = 1.0 if self._can_build_city(resp_hand_after) else 0.0
        feat[28] = 1.0 if self._can_buy_dev(resp_hand_after) else 0.0
        feat[29] = 1.0 if self._can_build_road(resp_hand_after) else 0.0

        # [30-31] What proposer can build after trade
        prop_hand_after = dict(proposer.resource_counts)
        for res, amt in trade.offering.items():
            prop_hand_after[res] = prop_hand_after.get(res, 0) - amt
        for res, amt in trade.requesting.items():
            prop_hand_after[res] = prop_hand_after.get(res, 0) + amt

        feat[30] = 1.0 if self._can_build_settlement(prop_hand_after) else 0.0
        feat[31] = 1.0 if self._can_build_city(prop_hand_after) else 0.0

        # [32] Proposer close to winning
        feat[32] = 1.0 if prop_vp >= 8 else 0.0

        return feat

    @staticmethod
    def _production_rate(state: CatanState, player_color: int) -> float:
        rate = 0.0
        for cidx, (owner, btype) in state.corner_buildings.items():
            if owner != player_color:
                continue
            mult = 1 if btype == BuildingType.SETTLEMENT else 2
            for hex_idx in state.topology.corner_to_hexes.get(cidx, []):
                if hex_idx == state.robber_hex:
                    continue
                dn = state.topology.hex_dice_numbers.get(hex_idx, 0)
                if dn in DICE_PROB:
                    rate += DICE_PROB[dn] * mult
        return rate

    @staticmethod
    def _produced_resources(state: CatanState, player_color: int) -> set[int]:
        resources = set()
        for cidx, (owner, btype) in state.corner_buildings.items():
            if owner != player_color:
                continue
            for hex_idx in state.topology.corner_to_hexes.get(cidx, []):
                res = state.topology.hex_resources.get(hex_idx)
                if res is not None:
                    resources.add(res)
        return resources

    @staticmethod
    def _can_build_settlement(hand: dict[int, int]) -> bool:
        return (hand.get(Resource.LUMBER, 0) >= 1 and hand.get(Resource.BRICK, 0) >= 1
                and hand.get(Resource.WOOL, 0) >= 1 and hand.get(Resource.GRAIN, 0) >= 1)

    @staticmethod
    def _can_build_city(hand: dict[int, int]) -> bool:
        return hand.get(Resource.ORE, 0) >= 3 and hand.get(Resource.GRAIN, 0) >= 2

    @staticmethod
    def _can_buy_dev(hand: dict[int, int]) -> bool:
        return (hand.get(Resource.WOOL, 0) >= 1 and hand.get(Resource.GRAIN, 0) >= 1
                and hand.get(Resource.ORE, 0) >= 1)

    @staticmethod
    def _can_build_road(hand: dict[int, int]) -> bool:
        return hand.get(Resource.LUMBER, 0) >= 1 and hand.get(Resource.BRICK, 0) >= 1
