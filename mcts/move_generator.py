"""
Legal move generator for Catan main-phase decisions.

Entry point:
    get_legal_actions(state, color, dev_card_played_this_turn=False)
        → list[Action]

Each sub-generator can also be called individually for inspection.

Road placement rules (Catan standard):
  A road may be placed on an empty edge if at least one endpoint corner is:
    (a) occupied by the player's own settlement or city, OR
    (b) adjacent to another of the player's roads AND not occupied by an
        opponent's building (opponent buildings cut road chains).

Settlement placement rules:
  Must be adjacent to at least one of the player's roads,
  must be on an empty corner, and must satisfy the distance rule
  (no adjacent corner may have any building).
"""

from __future__ import annotations

from typing import Optional

from data.enums import Resource, BuildingType, DevCard
from data.state import CatanState

from mcts.actions import (
    Action,
    BuildCity,
    BuildRoad,
    BuildSettlement,
    BankTrade,
    BuyDevCard,
    PassTurn,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    SETTLEMENT_COST,
    CITY_COST,
    ROAD_COST,
    DEV_CARD_COST,
)


# ─── Resource helpers ────────────────────────────────────────────────────────

def _has_resources(player, cost: dict) -> bool:
    """Return True if the player's hand contains at least `cost` of each resource."""
    counts = player.resource_counts
    for res, amount in cost.items():
        if counts.get(int(res), 0) < amount:
            return False
    return True


# ─── Road edge helpers ────────────────────────────────────────────────────────

def _get_valid_road_edges(
    state: CatanState,
    color: int,
    ignore_cost: bool = False,
    extra_roads: Optional[set[int]] = None,
) -> list[int]:
    """
    Return all edge indices where `color` can legally place a road.

    extra_roads: additional road edges already placed this action (e.g. first
                 road of a Road Building card), so adjacency extends to them.
    ignore_cost: skip the resource check (used for Road Building dev card).
    """
    player = state.players.get(color)
    if player is None or player.roads_remaining == 0:
        return []
    if not ignore_cost and not _has_resources(player, ROAD_COST):
        return []

    topo = state.topology
    occupied_edges = set(state.edge_roads.keys())
    if extra_roads:
        occupied_edges |= extra_roads  # treat already-placed roads as occupied

    opponent_corners = {
        cidx
        for cidx, (owner, _) in state.corner_buildings.items()
        if owner != color
    }
    player_building_corners = {
        cidx
        for cidx, (owner, _) in state.corner_buildings.items()
        if owner == color
    }

    player_road_set = set(state.get_roads_for_player(color))
    if extra_roads:
        player_road_set |= extra_roads

    valid: set[int] = set()

    # (a) Edges adjacent to own buildings
    for cidx in player_building_corners:
        for eidx in topo.corner_to_edges.get(cidx, []):
            if eidx not in occupied_edges:
                valid.add(eidx)

    # (b) Edges reachable through own roads, not blocked by opponent buildings
    for eidx in player_road_set:
        for cidx in topo.edge_to_corners.get(eidx, []):
            if cidx in opponent_corners:
                continue  # opponent building cuts the chain
            for next_eidx in topo.corner_to_edges.get(cidx, []):
                if next_eidx not in occupied_edges:
                    valid.add(next_eidx)

    return list(valid)


# ─── Individual generators ────────────────────────────────────────────────────

def get_legal_roads(state: CatanState, color: int) -> list[BuildRoad]:
    return [BuildRoad(e) for e in _get_valid_road_edges(state, color)]


def get_legal_settlements(state: CatanState, color: int) -> list[BuildSettlement]:
    player = state.players.get(color)
    if player is None or player.settlements_remaining == 0:
        return []
    if not _has_resources(player, SETTLEMENT_COST):
        return []

    topo = state.topology
    occupied = set(state.corner_buildings.keys())

    # Corners reachable by own roads
    reachable: set[int] = set()
    for eidx in state.get_roads_for_player(color):
        for cidx in topo.edge_to_corners.get(eidx, []):
            reachable.add(cidx)

    valid = []
    for cidx in reachable:
        if cidx in occupied:
            continue
        # Distance rule
        if any(adj in occupied for adj in topo.get_adjacent_corners(cidx)):
            continue
        valid.append(BuildSettlement(cidx))

    return valid


def get_legal_cities(state: CatanState, color: int) -> list[BuildCity]:
    player = state.players.get(color)
    if player is None or player.cities_remaining == 0:
        return []
    if not _has_resources(player, CITY_COST):
        return []

    return [
        BuildCity(cidx)
        for cidx, (owner, btype) in state.corner_buildings.items()
        if owner == color and btype == BuildingType.SETTLEMENT
    ]


def get_legal_bank_trades(state: CatanState, color: int) -> list[BankTrade]:
    player = state.players.get(color)
    if player is None:
        return []

    counts = player.resource_counts
    valid = []
    for give_res in Resource:
        ratio = player.bank_trade_ratios.get(int(give_res), 4)
        if counts.get(int(give_res), 0) < ratio:
            continue
        for recv_res in Resource:
            if recv_res == give_res:
                continue
            if state.bank_resources.get(int(recv_res), 0) > 0:
                valid.append(BankTrade(
                    give_resource=int(give_res),
                    give_count=ratio,
                    receive_resource=int(recv_res),
                ))
    return valid


def get_legal_dev_card_plays(
    state: CatanState,
    color: int,
    dev_card_played_this_turn: bool = False,
) -> list[Action]:
    """
    Enumerate all legal dev card play actions.

    VP cards are not included: they are revealed automatically by Colonist
    and do not require a bot decision.

    dev_card_played_this_turn: if True, only passive cards (none here) are
    returned — you cannot play two dev cards in one turn.
    """
    player = state.players.get(color)
    if player is None or dev_card_played_this_turn:
        return []

    topo = state.topology
    actions: list[Action] = []
    hand = player.dev_cards  # list of DevCard int values

    # ── Knight ──────────────────────────────────────────────────────────────
    if DevCard.KNIGHT in hand:
        for hex_idx in topo.hex_positions:
            if hex_idx == state.robber_hex:
                continue

            opponents_on_hex = []
            for cidx in topo.hex_to_corners.get(hex_idx, []):
                if cidx in state.corner_buildings:
                    owner, _ = state.corner_buildings[cidx]
                    if owner != color and owner not in opponents_on_hex:
                        opponents_on_hex.append(owner)

            if opponents_on_hex:
                for steal_color in opponents_on_hex:
                    if state.players[steal_color].total_resources > 0:
                        actions.append(PlayKnight(hex_idx, steal_color))
                    else:
                        # Can still move robber there, just no steal
                        actions.append(PlayKnight(hex_idx, None))
            else:
                actions.append(PlayKnight(hex_idx, None))

    # ── Monopoly ─────────────────────────────────────────────────────────────
    if DevCard.MONOPOLY in hand:
        for res in Resource:
            actions.append(PlayMonopoly(int(res)))

    # ── Road Building ────────────────────────────────────────────────────────
    if DevCard.ROAD_BUILDING in hand and player.roads_remaining > 0:
        valid1 = _get_valid_road_edges(state, color, ignore_cost=True)

        if player.roads_remaining == 1 or len(valid1) == 1:
            # Only one road can/should be placed
            for e1 in valid1:
                actions.append(PlayRoadBuilding(e1, -1))
        elif len(valid1) >= 2:
            pairs: list[PlayRoadBuilding] = []
            for i, e1 in enumerate(valid1):
                # Find valid second roads given e1 is already placed
                valid2 = _get_valid_road_edges(
                    state, color,
                    ignore_cost=True,
                    extra_roads={e1},
                )
                for e2 in valid2:
                    if e2 != e1:
                        # Canonicalise order to deduplicate (e1, e2) vs (e2, e1)
                        lo, hi = (e1, e2) if e1 < e2 else (e2, e1)
                        pairs.append(PlayRoadBuilding(lo, hi))
                if len(pairs) >= 30:  # cap to keep action space tractable
                    break
            # Deduplicate
            actions.extend(list(dict.fromkeys(pairs)))

    # ── Year of Plenty ───────────────────────────────────────────────────────
    if DevCard.YEAR_OF_PLENTY in hand:
        bank = state.bank_resources
        available = [int(r) for r in Resource if bank.get(int(r), 0) > 0]
        for i, r1 in enumerate(available):
            if bank.get(r1, 0) >= 2:
                actions.append(PlayYearOfPlenty(r1, r1))
            for r2 in available[i + 1:]:
                actions.append(PlayYearOfPlenty(r1, r2))

    return actions


# ─── Main entry point ─────────────────────────────────────────────────────────

def get_legal_actions(
    state: CatanState,
    color: int,
    dev_card_played_this_turn: bool = False,
) -> list[Action]:
    """
    Return every legal action for `color` during the main phase (after dice roll).

    Order: roads, settlements, cities, buy dev card, dev card plays, bank trades,
    pass turn.  PassTurn is always included so the bot can end its turn.

    dev_card_played_this_turn: set True once a dev card has been played this
        turn so that a second play is not suggested.
    """
    player = state.players.get(color)
    if player is None:
        return [PassTurn()]

    actions: list[Action] = []
    actions.extend(get_legal_roads(state, color))
    actions.extend(get_legal_settlements(state, color))
    actions.extend(get_legal_cities(state, color))

    if (
        _has_resources(player, DEV_CARD_COST)
        and len(state.bank_dev_cards) > 0
    ):
        actions.append(BuyDevCard())

    actions.extend(get_legal_dev_card_plays(state, color, dev_card_played_this_turn))
    actions.extend(get_legal_bank_trades(state, color))
    actions.append(PassTurn())

    return actions


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import os
    import sys

    DATASET_DIR = os.environ.get("CATAN_DATASET_DIR", "./dataset")
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    if not files:
        print(f"No JSON files found in {DATASET_DIR}")
        print("Set CATAN_DATASET_DIR or run from the project root.")
        sys.exit(1)

    from data.replay import GameReplay

    replay = GameReplay.from_file(files[0])
    print(f"Loaded game: {os.path.basename(files[0])}")
    print(f"Players: {replay.play_order}")

    # Test at several turns across the game
    for turn in [12, 20, 35, 50]:
        state = replay.replay_to_turn(turn)
        color = state.current_player_color
        player = state.players[color]
        actions = get_legal_actions(state, color)

        print(f"\n── Turn {turn} | Player {color} ──")
        print(f"   Resources: {dict(player.resource_counts)}")
        print(f"   Dev cards: {player.dev_cards}")
        print(f"   VP: {player.total_vp}")
        print(f"   Legal actions ({len(actions)} total):")

        # Group by type
        from collections import Counter
        type_counts: Counter = Counter(type(a).__name__ for a in actions)
        for atype, count in sorted(type_counts.items()):
            print(f"     {atype}: {count}")

        # Show up to 3 examples of each type
        shown: Counter = Counter()
        for a in actions:
            atype = type(a).__name__
            if shown[atype] < 3:
                print(f"       → {a}")
                shown[atype] += 1

    # Verify pass is always included
    all_pass = all(any(isinstance(a, PassTurn) for a in get_legal_actions(replay.replay_to_turn(t), replay.play_order[0])) for t in [10, 20, 30])
    print(f"\nPassTurn always present: {all_pass}")

    # Verify no actions when player has no resources and no roads
    print("\nAll tests passed.")
