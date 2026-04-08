"""
Forward state transition: apply an Action to a CatanState copy.

The original state is never modified — apply_action() always works on a
state.copy() and returns the new state.

Handles:
  - BuildSettlement / BuildCity / BuildRoad
  - BuyDevCard (drawn card is the top of the bank deck; hidden to opponents)
  - PlayKnight  (moves robber, steals one random card from target)
  - PlayMonopoly (takes all of one resource from all opponents)
  - PlayRoadBuilding (places up to two free roads)
  - PlayYearOfPlenty (takes two resources from bank)
  - BankTrade
  - PassTurn (no-op; caller handles turn advancement)

After road/settlement/city placements the longest-road and largest-army
awards are recalculated and VP adjusted accordingly.
"""

from __future__ import annotations

import random
from typing import Optional

from data.enums import BuildingType, DevCard, Resource, VPCategory
from data.state import CatanState

from mcts.actions import (
    Action,
    BankTrade,
    BuildCity,
    BuildRoad,
    BuildSettlement,
    BuyDevCard,
    PassTurn,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    CITY_COST,
    DEV_CARD_COST,
    ROAD_COST,
    SETTLEMENT_COST,
)


# ─── Internal helpers ────────────────────────────────────────────────────────

def _remove_resources(player, cost: dict) -> None:
    """Remove resources from the player's hand in-place."""
    for res, amount in cost.items():
        res_val = int(res)
        for _ in range(amount):
            player.resource_cards.remove(res_val)


def _return_to_bank(state: CatanState, cost: dict) -> None:
    """Add spent resources back into the bank."""
    for res, amount in cost.items():
        res_val = int(res)
        state.bank_resources[res_val] = state.bank_resources.get(res_val, 0) + amount


def _take_from_bank(state: CatanState, resource: int, amount: int = 1) -> int:
    """
    Remove up to `amount` of `resource` from the bank.
    Returns how many were actually taken (bank may not have enough).
    """
    available = state.bank_resources.get(resource, 0)
    taken = min(available, amount)
    state.bank_resources[resource] = available - taken
    return taken


def _update_port_access(state: CatanState, color: int, corner_idx: int) -> None:
    """Grant port trade ratios for a newly built settlement or city."""
    if corner_idx not in state.topology.corner_ports:
        return
    _, ratio, resource = state.topology.corner_ports[corner_idx]
    player = state.players[color]
    if resource is not None:
        res_val = int(resource)
        if ratio < player.bank_trade_ratios.get(res_val, 4):
            player.bank_trade_ratios[res_val] = ratio
    else:
        # 3:1 generic port — apply to all resources
        for r in Resource:
            if ratio < player.bank_trade_ratios.get(int(r), 4):
                player.bank_trade_ratios[int(r)] = ratio


# ─── Longest road ────────────────────────────────────────────────────────────

def compute_longest_road(state: CatanState, color: int) -> int:
    """
    Compute the longest continuous road for `color` via DFS.

    Opponent buildings block traversal through their corner — you cannot
    count a road chain that passes through an opposing settlement or city.
    """
    road_set = set(state.get_roads_for_player(color))
    if not road_set:
        return 0

    topo = state.topology

    # Corners where an opponent has built — road chains cannot pass through
    blocked = {
        cidx
        for cidx, (owner, _) in state.corner_buildings.items()
        if owner != color
    }

    # corner → list of player's road edges touching that corner
    corner_to_roads: dict[int, list[int]] = {}
    for eidx in road_set:
        for cidx in topo.edge_to_corners.get(eidx, []):
            corner_to_roads.setdefault(cidx, []).append(eidx)

    def dfs(edge: int, visited: set[int]) -> int:
        best = len(visited)
        for cidx in topo.edge_to_corners.get(edge, []):
            if cidx in blocked:
                continue
            for nxt in corner_to_roads.get(cidx, []):
                if nxt not in visited:
                    visited.add(nxt)
                    result = dfs(nxt, visited)
                    if result > best:
                        best = result
                    visited.remove(nxt)
        return best

    max_len = 0
    for start in road_set:
        length = dfs(start, {start})
        if length > max_len:
            max_len = length

    return max_len


def _update_longest_road(state: CatanState, color: int) -> None:
    """
    Recompute longest road for `color` and adjust the VP award if it changed.

    Rules:
      - First player to reach a chain of 5 earns 2 VP.
      - Another player can steal the award by strictly beating the holder.
      - Once awarded, the holder keeps it even if their road is later cut
        (unless someone else beats them).
    """
    new_length = compute_longest_road(state, color)
    state.players[color].longest_road = new_length

    # Find the current holder (if any)
    current_holder: Optional[int] = None
    current_length = 4  # threshold: must exceed 4 to hold the award
    for c, p in state.players.items():
        if p.victory_points.get(VPCategory.LONGEST_ROAD, 0) > 0:
            current_holder = c
            current_length = p.longest_road
            break

    player = state.players[color]

    if current_holder is None:
        if new_length >= 5:
            player.victory_points[VPCategory.LONGEST_ROAD] = 2
    elif current_holder == color:
        # We already hold it; our length updated but the award stays regardless
        pass
    else:
        # Take the award only by strictly beating the holder
        if new_length > current_length:
            state.players[current_holder].victory_points[VPCategory.LONGEST_ROAD] = 0
            player.victory_points[VPCategory.LONGEST_ROAD] = 2


# ─── Largest army ────────────────────────────────────────────────────────────

def _update_largest_army(state: CatanState, color: int) -> None:
    """
    Update largest army after `color` plays a knight.

    Rules:
      - First player to play 3 knights earns 2 VP.
      - Another player can take it by strictly beating the holder's count.
    """
    player = state.players[color]
    if player.knights_played < 3:
        return

    current_holder: Optional[int] = None
    for c, p in state.players.items():
        if p.has_largest_army:
            current_holder = c
            break

    if current_holder is None:
        player.has_largest_army = True
        player.victory_points[VPCategory.LARGEST_ARMY] = 2
    elif current_holder != color:
        holder = state.players[current_holder]
        if player.knights_played > holder.knights_played:
            holder.has_largest_army = False
            holder.victory_points[VPCategory.LARGEST_ARMY] = 0
            player.has_largest_army = True
            player.victory_points[VPCategory.LARGEST_ARMY] = 2


# ─── Main transition ─────────────────────────────────────────────────────────

def apply_action(state: CatanState, action: Action, color: int) -> CatanState:
    """
    Apply `action` for `color` and return the resulting CatanState.
    The original `state` is not modified.
    """
    new = state.copy()
    player = new.players[color]

    # ── Build Settlement ────────────────────────────────────────────────────
    if isinstance(action, BuildSettlement):
        _remove_resources(player, SETTLEMENT_COST)
        _return_to_bank(new, SETTLEMENT_COST)
        new.corner_buildings[action.corner_idx] = (color, BuildingType.SETTLEMENT)
        player.settlements_remaining -= 1
        player.victory_points[VPCategory.SETTLEMENTS] = (
            player.victory_points.get(VPCategory.SETTLEMENTS, 0) + 1
        )
        _update_port_access(new, color, action.corner_idx)
        # A new settlement at a contested corner can cut opponent roads
        for opp_color in new.players:
            if opp_color != color:
                _update_longest_road(new, opp_color)
        _update_longest_road(new, color)

    # ── Build City ──────────────────────────────────────────────────────────
    elif isinstance(action, BuildCity):
        _remove_resources(player, CITY_COST)
        _return_to_bank(new, CITY_COST)
        new.corner_buildings[action.corner_idx] = (color, BuildingType.CITY)
        player.cities_remaining -= 1
        player.settlements_remaining += 1  # piece returned to supply
        player.victory_points[VPCategory.SETTLEMENTS] = (
            player.victory_points.get(VPCategory.SETTLEMENTS, 0) - 1
        )
        player.victory_points[VPCategory.CITIES] = (
            player.victory_points.get(VPCategory.CITIES, 0) + 2
        )
        # Port access doesn't change when upgrading

    # ── Build Road ──────────────────────────────────────────────────────────
    elif isinstance(action, BuildRoad):
        _remove_resources(player, ROAD_COST)
        _return_to_bank(new, ROAD_COST)
        new.edge_roads[action.edge_idx] = color
        player.roads_remaining -= 1
        _update_longest_road(new, color)

    # ── Buy Dev Card ────────────────────────────────────────────────────────
    elif isinstance(action, BuyDevCard):
        _remove_resources(player, DEV_CARD_COST)
        _return_to_bank(new, DEV_CARD_COST)
        if new.bank_dev_cards:
            drawn = new.bank_dev_cards.pop(0)
            player.dev_cards.append(drawn)

    # ── Play Knight ─────────────────────────────────────────────────────────
    elif isinstance(action, PlayKnight):
        player.dev_cards.remove(DevCard.KNIGHT)
        player.dev_cards_used.append(DevCard.KNIGHT)
        player.knights_played += 1
        new.robber_hex = action.target_hex

        if action.steal_from is not None:
            victim = new.players.get(action.steal_from)
            if victim and victim.resource_cards:
                stolen = random.choice(victim.resource_cards)
                victim.resource_cards.remove(stolen)
                player.resource_cards.append(stolen)

        _update_largest_army(new, color)

    # ── Play Monopoly ───────────────────────────────────────────────────────
    elif isinstance(action, PlayMonopoly):
        player.dev_cards.remove(DevCard.MONOPOLY)
        player.dev_cards_used.append(DevCard.MONOPOLY)
        res_val = action.resource
        for opp_color, opp in new.players.items():
            if opp_color == color:
                continue
            stolen = [c for c in opp.resource_cards if c == res_val]
            for _ in stolen:
                opp.resource_cards.remove(res_val)
            player.resource_cards.extend(stolen)

    # ── Play Road Building ──────────────────────────────────────────────────
    elif isinstance(action, PlayRoadBuilding):
        player.dev_cards.remove(DevCard.ROAD_BUILDING)
        player.dev_cards_used.append(DevCard.ROAD_BUILDING)
        if action.edge1 >= 0:
            new.edge_roads[action.edge1] = color
            player.roads_remaining -= 1
        if action.edge2 >= 0:
            new.edge_roads[action.edge2] = color
            player.roads_remaining -= 1
        _update_longest_road(new, color)

    # ── Play Year of Plenty ─────────────────────────────────────────────────
    elif isinstance(action, PlayYearOfPlenty):
        player.dev_cards.remove(DevCard.YEAR_OF_PLENTY)
        player.dev_cards_used.append(DevCard.YEAR_OF_PLENTY)
        _take_from_bank(new, action.resource1)
        player.resource_cards.append(action.resource1)
        _take_from_bank(new, action.resource2)
        player.resource_cards.append(action.resource2)

    # ── Bank Trade ──────────────────────────────────────────────────────────
    elif isinstance(action, BankTrade):
        for _ in range(action.give_count):
            player.resource_cards.remove(action.give_resource)
        new.bank_resources[action.give_resource] = (
            new.bank_resources.get(action.give_resource, 0) + action.give_count
        )
        _take_from_bank(new, action.receive_resource)
        player.resource_cards.append(action.receive_resource)

    # ── Pass Turn ───────────────────────────────────────────────────────────
    elif isinstance(action, PassTurn):
        pass  # caller handles advancing the turn

    return new


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
    from mcts.move_generator import get_legal_actions

    replay = GameReplay.from_file(files[0])
    print(f"Loaded game: {os.path.basename(files[0])}")

    errors = 0

    for turn in [12, 20, 30, 45]:
        state = replay.replay_to_turn(turn)
        color = state.current_player_color
        actions = get_legal_actions(state, color)
        player_before = state.players[color]

        print(f"\n── Turn {turn} | Player {color} ──")
        print(f"   VP before: {player_before.total_vp}")
        print(f"   Resources: {dict(player_before.resource_counts)}")

        tested = 0
        for action in actions:
            if isinstance(action, PassTurn):
                continue
            try:
                new_state = apply_action(state, action, color)
                player_after = new_state.players[color]

                # State independence check
                if new_state is state:
                    print(f"   ERROR: apply_action returned the same object!")
                    errors += 1
                    continue

                # Resources should not go negative
                for c, p in new_state.players.items():
                    if p.total_resources < 0:
                        print(f"   ERROR: {action} gave player {c} negative resources")
                        errors += 1

                # VP should not decrease (for our own player, actions either maintain or increase)
                if isinstance(action, (BuildSettlement, BuildCity)):
                    if player_after.total_vp <= player_before.total_vp:
                        print(f"   WARNING: {action} did not increase VP "
                              f"({player_before.total_vp} → {player_after.total_vp})")

                # Resources correctly deducted
                if isinstance(action, BuildRoad):
                    spent = player_before.total_resources - player_after.total_resources
                    if spent != 2:
                        print(f"   ERROR: BuildRoad should cost 2 resources, got {spent}")
                        errors += 1

                tested += 1
            except Exception as e:
                print(f"   ERROR applying {action}: {e}")
                errors += 1

        print(f"   Tested {tested} actions successfully")

    # ── Longest road test ────────────────────────────────────────────────────
    print("\n── Longest road smoke test ──")
    for turn in [20, 40]:
        state = replay.replay_to_turn(turn)
        for color in state.player_colors:
            road_count = len(state.get_roads_for_player(color))
            longest = compute_longest_road(state, color)
            assert longest <= road_count, (
                f"Longest road ({longest}) > road count ({road_count}) for player {color}"
            )
            assert longest >= 0
        print(f"   Turn {turn}: longest road OK for all players")

    if errors == 0:
        print("\nAll state_transition tests passed.")
    else:
        print(f"\n{errors} error(s) found.")
        sys.exit(1)
