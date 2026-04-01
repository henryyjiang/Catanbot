"""
Produces a composite score in [0, 1] for each (state, player) pair that
balances three signals:

  1. OUTCOME — did this player ultimately win? Discounted by temporal
     distance so early-game states aren't hard-labeled.

  2. RELATIVE POSITION — VP lead/deficit relative to opponents, with
     penalties for being the visible leader (Catan's "tall poppy" effect)
     and bonuses for being close behind the leader.

  3. ECONOMIC QUALITY — resource production rate, port access, dev card
     holdings, resource diversity. Measures future potential, not current
     score.
"""
from __future__ import annotations

import math
from typing import Optional

from data.enums import Resource, BuildingType, DevCard, VPCategory
from data.state import CatanState, PlayerState
from data.topology import BoardTopology


DICE_PROB = {
    2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36,
}

def outcome_score(
    won_game: bool,
    current_turn: int,
    total_turns: int,
    final_vp: int,
    vp_to_win: int = 10,
) -> float:
    """
    Early game states are pulled toward 0.5 (maximum uncertainty).
    Late game states approach the true binary outcome.
    Losers still get partial credit based on their final VP.
    """
    if total_turns <= 0:
        return 0.5

    progress = min(current_turn / max(total_turns, 1), 1.0)

    # At progress=0.0 → confidence≈0.05 
    # At progress=0.5 → confidence≈0.50
    # At progress=0.9 → confidence≈0.95
    steepness = 6.0
    midpoint = 0.5
    confidence = 1.0 / (1.0 + math.exp(-steepness * (progress - midpoint)))

    # True outcome: 1.0 for winner, scaled VP for losers
    if won_game:
        true_label = 1.0
    else:
        # Losers get credit for final VP — reaching 8/10 is much better than 2/10, even if both are losses
        true_label = (final_vp / vp_to_win) * 0.45  # cap at 0.45 so winner > loser

    # Blend uncertainty (0.5) with true label based on confidence
    return 0.5 * (1.0 - confidence) + true_label * confidence


def relative_position_score(
    state: CatanState,
    player_color: int,
) -> float:
    """Score based on VP position relative to all opponents.

    Returns a value in [0, 1].
    """
    player = state.players[player_color]
    my_vp = player.total_vp
    opponent_vps = sorted(
        [p.total_vp for c, p in state.players.items() if c != player_color],
        reverse=True
    )

    if not opponent_vps:
        return 0.5

    max_opp_vp = opponent_vps[0]
    avg_opp_vp = sum(opponent_vps) / len(opponent_vps)
    vp_to_win = 10
    lead = (my_vp - max_opp_vp) / max(vp_to_win, 1)

    is_leader = my_vp > max_opp_vp
    pack_lead = (my_vp - avg_opp_vp) / max(vp_to_win, 1)

    # Only penalize significant leads (2+ VP above average)
    tall_poppy_penalty = 0.0
    if is_leader and pack_lead > 0.2:
        # Penalty scales with how much we stick out
        tall_poppy_penalty = min(pack_lead * 0.3, 0.15)

    win_proximity = my_vp / vp_to_win

    raw = (
        lead * 0.4 +           # VP advantage
        win_proximity * 0.4 +
        -tall_poppy_penalty +
        0.1
    )

    return max(0.0, min(1.0, raw + 0.3))


def economic_quality_score(
    state: CatanState,
    player_color: int,
) -> float:
    """Evaluates future potential rather than current score:
      - Total resource production rate (expected cards per roll)
      - Resource diversity (access to all 5 types)
      - Port bonuses (trading efficiency)
      - Dev card holdings (hidden potential)
      - Hand quality (can we actually build something right now?)
    """
    player = state.players[player_color]
    topo = state.topology

    production_rate = 0.0
    resource_access = set()

    for cidx, (owner, btype) in state.corner_buildings.items():
        if owner != player_color:
            continue

        multiplier = 1 if btype == BuildingType.SETTLEMENT else 2

        for hex_idx in topo.corner_to_hexes.get(cidx, []):
            # Skip robber hex
            if hex_idx == state.robber_hex:
                continue

            dice_num = topo.hex_dice_numbers.get(hex_idx, 0)
            resource = topo.hex_resources.get(hex_idx)

            if resource is not None and dice_num in DICE_PROB:
                production_rate += DICE_PROB[dice_num] * multiplier
                resource_access.add(resource)

    # Normalize production rate: excellent production ≈ 0.8-1.0 cards/roll
    prod_score = min(production_rate / 0.9, 1.0)

    # Having access to all 5 resource types is extremely valuable
    diversity_score = len(resource_access) / 5.0

    # ── Port bonuses ──
    # Count effective trade ratio advantage
    base_ratios = {r.value: 4 for r in Resource}
    port_bonus = 0.0
    for r in Resource:
        actual = player.bank_trade_ratios.get(r.value, 4)
        if actual < 4:
            port_bonus += (4 - actual) / 2.0
    port_score = min(port_bonus / 3.0, 1.0)

    # Unplayed dev cards represent hidden potentia;
    dev_score = 0.0
    for card in player.dev_cards:
        if card == DevCard.KNIGHT:
            dev_score += 0.15
        elif card == DevCard.VICTORY_POINT:
            dev_score += 0.25
        elif card == DevCard.ROAD_BUILDING:
            dev_score += 0.12
        elif card == DevCard.YEAR_OF_PLENTY:
            dev_score += 0.10
        elif card == DevCard.MONOPOLY:
            dev_score += 0.10
    dev_score = min(dev_score, 1.0)

    rcounts = player.resource_counts
    hand_score = 0.0

    lum = rcounts.get(Resource.LUMBER, 0)
    brk = rcounts.get(Resource.BRICK, 0)
    wol = rcounts.get(Resource.WOOL, 0)
    grn = rcounts.get(Resource.GRAIN, 0)
    ore = rcounts.get(Resource.ORE, 0)

    if lum >= 1 and brk >= 1:
        hand_score += 0.1
    if lum >= 1 and brk >= 1 and wol >= 1 and grn >= 1:
        hand_score += 0.3
    if ore >= 3 and grn >= 2:
        hand_score += 0.3
    if wol >= 1 and grn >= 1 and ore >= 1:
        hand_score += 0.15
    if player.total_resources > 7:
        hand_score -= 0.1 * ((player.total_resources - 7) / 7)

    hand_score = max(0.0, min(1.0, hand_score))

    score = (
        prod_score * 0.35 +
        diversity_score * 0.20 +
        port_score * 0.10 +
        dev_score * 0.15 +
        hand_score * 0.20
    )

    return max(0.0, min(1.0, score))

def compute_label(
    state: CatanState,
    player_color: int,
    won_game: bool,
    current_turn: int,
    total_turns: int,
    final_vp: int,
    vp_to_win: int = 10,
    w_outcome: float = 0.50,
    w_position: float = 0.30,
    w_economic: float = 0.20,
) -> dict[str, float]:
    s_outcome = outcome_score(
        won_game=won_game,
        current_turn=current_turn,
        total_turns=total_turns,
        final_vp=final_vp,
        vp_to_win=vp_to_win,
    )
    s_position = relative_position_score(state, player_color)
    s_economic = economic_quality_score(state, player_color)

    composite = (
        w_outcome * s_outcome +
        w_position * s_position +
        w_economic * s_economic
    )
    composite = max(0.0, min(1.0, composite))

    player = state.players[player_color]
    opponent_vps = [
        p.total_vp for c, p in state.players.items() if c != player_color
    ]
    max_opp_vp = max(opponent_vps) if opponent_vps else 0
    vp_lead = player.total_vp - max_opp_vp

    topo = state.topology
    prod_rate = 0.0
    for cidx, (owner, btype) in state.corner_buildings.items():
        if owner != player_color:
            continue
        mult = 1 if btype == BuildingType.SETTLEMENT else 2
        for hex_idx in topo.corner_to_hexes.get(cidx, []):
            if hex_idx == state.robber_hex:
                continue
            dn = topo.hex_dice_numbers.get(hex_idx, 0)
            if dn in DICE_PROB:
                prod_rate += DICE_PROB[dn] * mult

    return {
        'label': composite,
        'outcome_score': s_outcome,
        'position_score': s_position,
        'economic_score': s_economic,
        'won_game': won_game,
        'current_vp': player.total_vp,
        'vp_lead': vp_lead,
        'production_rate': prod_rate,
        'turn_progress': current_turn / max(total_turns, 1),
    }
