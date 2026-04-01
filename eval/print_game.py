"""
Print every turn's full state for manual verification.

Usage:
    python print_game.py dataset/189472638.json
    python print_game.py dataset/189472638.json --turns 10-30
"""
import argparse
import json
import sys
from collections import Counter

from data.enums import (
    Resource, TileType, BuildingType, DevCard, PortType,
    VPCategory, LogType, RESOURCE_NAMES, DEV_CARD_NAMES,
    TILE_RESOURCE, PORT_TRADE_RATIOS,
)
from data.topology import BoardTopology
from data.state import CatanState
from data.replay import GameReplay


RESOURCE_SYMBOLS = {
    Resource.LUMBER: "🪵 Lumber",
    Resource.BRICK:  "🧱 Brick",
    Resource.WOOL:   "🐑 Wool",
    Resource.GRAIN:  "🌾 Grain",
    Resource.ORE:    "�ite Ore",
}

RESOURCE_SHORT = {
    Resource.LUMBER: "Lum",
    Resource.BRICK:  "Brk",
    Resource.WOOL:   "Wol",
    Resource.GRAIN:  "Grn",
    Resource.ORE:    "Ore",
}

TILE_NAMES = {
    TileType.DESERT: "Desert",
    TileType.GRAIN:  "Grain",
    TileType.ORE:    "Ore",
    TileType.LUMBER: "Lumber",
    TileType.BRICK:  "Brick",
    TileType.WOOL:   "Wool",
}

VP_NAMES = {
    VPCategory.SETTLEMENTS:  "Settlements",
    VPCategory.CITIES:       "Cities",
    VPCategory.DEV_CARD_VP:  "Dev Card VP",
    VPCategory.LARGEST_ARMY: "Largest Army",
    VPCategory.LONGEST_ROAD: "Longest Road",
}

DEV_NAMES = {
    DevCard.HIDDEN:         "Hidden",
    DevCard.KNIGHT:         "Knight",
    DevCard.ROAD_BUILDING:  "Road Building",
    DevCard.YEAR_OF_PLENTY: "Year of Plenty",
    DevCard.MONOPOLY:       "Monopoly",
    DevCard.VICTORY_POINT:  "Victory Point",
}

PLAYER_COLORS = {1: "Red", 2: "Blue", 3: "Orange", 4: "White", 5: "Green"}


def color_name(color: int) -> str:
    return PLAYER_COLORS.get(color, f"Color{color}")


def format_resources(cards: list[int]) -> str:
    if not cards:
        return "(empty)"
    counts = Counter(cards)
    parts = []
    for r in Resource:
        c = counts.get(r.value, 0)
        if c > 0:
            parts.append(f"{RESOURCE_SHORT[r]}×{c}")
    return " ".join(parts) if parts else "(empty)"


def format_dev_cards(cards: list[int]) -> str:
    if not cards:
        return "(none)"
    counts = Counter(cards)
    parts = []
    for d in DevCard:
        c = counts.get(d.value, 0)
        if c > 0:
            parts.append(f"{DEV_NAMES[d]}×{c}")
    return " ".join(parts) if parts else "(none)"


def format_trade_ratios(ratios: dict[int, int]) -> str:
    parts = []
    for r in Resource:
        ratio = ratios.get(r.value, 4)
        if ratio < 4:
            parts.append(f"{RESOURCE_SHORT[r]} {ratio}:1")
    return " ".join(parts) if parts else "all 4:1"


def print_board_layout(state: CatanState):
    topo = state.topology
    print("HEX LAYOUT ")

    for hex_idx in sorted(topo.hex_positions.keys()):
        x, y = topo.hex_positions[hex_idx]
        tile_type = topo.hex_types[hex_idx]
        dice = topo.hex_dice_numbers.get(hex_idx, 0)
        name = TILE_NAMES.get(TileType(tile_type), f"Type{tile_type}")
        robber = " [ROBBER]" if state.robber_hex == hex_idx else ""
        dice_str = f"dice={dice}" if dice > 0 else "no dice"
        print(f"  │  Hex {hex_idx:2d} ({x:+d},{y:+d}): {name:6s}  {dice_str}{robber}")



def print_buildings(state: CatanState):
    topo = state.topology

    if not state.corner_buildings and not state.edge_roads:
        print("  (no buildings or roads)")
        return

    # Group buildings by player
    for player_color in state.player_colors:
        buildings = state.get_buildings_for_player(player_color)
        roads = state.get_roads_for_player(player_color)

        if not buildings and not roads:
            continue

        print(f"  {color_name(player_color)}:")

        for cidx, btype in sorted(buildings.items()):
            coord = topo.corner_positions.get(cidx, "?")
            bname = "Settlement" if btype == BuildingType.SETTLEMENT else "City"
            # Show which hexes this corner touches
            adj_hexes = topo.corner_to_hexes.get(cidx, [])
            hex_info = []
            for hx in adj_hexes:
                ht = TileType(topo.hex_types[hx])
                dn = topo.hex_dice_numbers.get(hx, 0)
                hex_info.append(f"{TILE_NAMES[ht]}({dn})")
            hex_str = ", ".join(hex_info)
            port_str = ""
            if cidx in topo.corner_ports:
                pt, ratio, res = topo.corner_ports[cidx]
                if res is not None:
                    port_str = f" [Port: {RESOURCE_SHORT.get(res, '?')} {ratio}:1]"
                else:
                    port_str = f" [Port: {ratio}:1 any]"
            print(f"    {bname} @ corner {cidx} {coord} → {hex_str}{port_str}")

        if roads:
            road_strs = []
            for eidx in sorted(roads):
                coord = topo.edge_positions.get(eidx, "?")
                road_strs.append(f"{eidx}{coord}")
            # Print roads in rows of 4
            for i in range(0, len(road_strs), 4):
                chunk = ", ".join(road_strs[i:i+4])
                prefix = "    Roads: " if i == 0 else "           "
                print(f"{prefix}{chunk}")


def print_player_state(state: CatanState, player_color: int, detailed: bool = True):
    """Print full state for one player."""
    p = state.players[player_color]
    is_current = (player_color == state.current_player_color)
    marker = " CURRENT" if is_current else ""

    print(f"  {color_name(player_color)} (color={player_color}){marker}")
    print(f"  {p.total_vp}", end="")
    if p.victory_points:
        vp_parts = []
        for cat_val, count in sorted(p.victory_points.items()):
            if count > 0:
                cat_name = VP_NAMES.get(VPCategory(cat_val), f"VP{cat_val}")
                vp_parts.append(f"{cat_name}={count}")
        if vp_parts:
            print(f"  ({', '.join(vp_parts)})", end="")
    print()

    print(f"  │  Resources ({p.total_resources}): {format_resources(p.resource_cards)}")

    if detailed:
        print(f"  │  Dev cards ({p.total_dev_cards}): {format_dev_cards(p.dev_cards)}")
        if p.dev_cards_used:
            print(f"  │  Dev used: {format_dev_cards(p.dev_cards_used)}")
        print(f"  │  Trade ratios: {format_trade_ratios(p.bank_trade_ratios)}")
        print(f"  │  Pieces left: {p.settlements_remaining}S {p.cities_remaining}C {p.roads_remaining}R")
        print(f"  │  Longest road: {p.longest_road}", end="")
        if p.has_largest_army:
            print("  ★ LARGEST ARMY", end="")
        # Check longest road award from VP
        if p.victory_points.get(VPCategory.LONGEST_ROAD, 0) > 0:
            print("  ★ LONGEST ROAD AWARD", end="")
        print()
        print(f"  │  Knights played: {p.knights_played}")

def print_turn_state(state: CatanState, turn: int, show_buildings: bool = True,
                     detailed_players: bool = True):
    dice_total = sum(state.last_dice)
    dice_str = f"{state.last_dice[0]}+{state.last_dice[1]}={dice_total}" if dice_total > 0 else "not rolled"

    print()
    print(f"{'═' * 60}")
    print(f"  TURN {turn}  │  Current: {color_name(state.current_player_color)}  │  Dice: {dice_str}")
    print(f"  Robber: hex {state.robber_hex}  │  Events applied: {state.events_applied}")
    print(f"{'═' * 60}")

    # Bank summary
    bank_total = sum(state.bank_resources.values())
    bank_parts = []
    for r in Resource:
        c = state.bank_resources.get(r.value, 0)
        bank_parts.append(f"{RESOURCE_SHORT[r]}={c}")
    print(f"  Bank ({bank_total}): {' '.join(bank_parts)}  │  Dev cards: {len(state.bank_dev_cards)}")
    print()

    # Players
    for player_color in state.player_colors:
        print_player_state(state, player_color, detailed=detailed_players)

    # Buildings
    if show_buildings:
        print()
        print("  BOARD:")
        print_buildings(state)

    print()

def describe_event(event: dict, event_idx: int) -> list[str]:
    sc = event.get('stateChange', {})
    if not sc:
        return []

    lines = []

    # Dice
    ds = sc.get('diceState', {})
    if ds.get('diceThrown') is True:
        d1, d2 = ds.get('dice1', 0), ds.get('dice2', 0)
        lines.append(f"  🎲 Dice: {d1}+{d2}={d1+d2}")

    # Buildings
    for corner_str, data in sc.get('mapState', {}).get('tileCornerStates', {}).items():
        if 'owner' in data and 'buildingType' in data:
            btype = "Settlement" if data['buildingType'] == 1 else "City"
            lines.append(f"  🏠 {color_name(data['owner'])} built {btype} @ corner {corner_str}")

    # Roads
    for edge_str, data in sc.get('mapState', {}).get('tileEdgeStates', {}).items():
        if 'owner' in data:
            lines.append(f"  🛤️  {color_name(data['owner'])} built Road @ edge {edge_str}")

    # Resource distribution
    for log_entry in sc.get('gameLogState', {}).values():
        if not isinstance(log_entry, dict) or 'text' not in log_entry:
            continue
        text = log_entry['text']
        ltype = text.get('type')

        if ltype == LogType.RESOURCE_DISTRIBUTED:
            pc = text.get('playerColor', '?')
            cards = text.get('cardsToBroadcast', [])
            res_str = format_resources(cards)
            lines.append(f"  📦 {color_name(pc)} received: {res_str}")

        elif ltype == LogType.ROBBER_MOVE:
            pc = text.get('playerColor', '?')
            lines.append(f"  👮 {color_name(pc)} moved robber")

        elif ltype == LogType.ROBBER_STEAL:
            pc = text.get('playerColor', '?')
            lines.append(f"  💰 {color_name(pc)} stole a card")

        elif ltype == LogType.KNIGHT_PLAYED:
            pc = text.get('playerColor', log_entry.get('from', '?'))
            lines.append(f"  ⚔️  {color_name(pc)} played Knight")

        elif ltype == LogType.MONOPOLY_PLAYED:
            pc = text.get('playerColor', '?')
            lines.append(f"  🎯 {color_name(pc)} played Monopoly")

        elif ltype == LogType.YEAR_OF_PLENTY:
            pc = text.get('playerColor', '?')
            lines.append(f"  🎁 {color_name(pc)} played Year of Plenty")

        elif ltype == LogType.ROAD_BUILDING:
            pc = text.get('playerColor', '?')
            lines.append(f"  🛤️  {color_name(pc)} played Road Building")

        elif ltype == LogType.TRADE_OFFER:
            pc = text.get('playerColor', '?')
            wanted = text.get('wantedCardEnums', [])
            offered = text.get('offeredCardEnums', [])
            lines.append(
                f"  📢 {color_name(pc)} offers {format_resources(offered)} "
                f"for {format_resources(wanted)}"
            )

        elif ltype == LogType.TRADE_COMPLETED:
            lines.append(f"  🤝 Trade completed")

        elif ltype == LogType.BANK_TRADE:
            pc = text.get('playerColor', '?')
            piece = text.get('pieceEnum', -1)
            piece_names = {0: "Road", 1: "Ship", 2: "Settlement", 3: "City"}
            pname = piece_names.get(piece, f"piece{piece}")
            is_vp = text.get('isVp', False)
            lines.append(f"  🏗️  {color_name(pc)} bought/built {pname}")

        elif ltype == LogType.DEV_CARD_BOUGHT:
            pc = text.get('playerColor', '?')
            lines.append(f"  🃏 {color_name(pc)} bought dev card")

        elif ltype == LogType.DISCARD:
            pc = text.get('playerColor', '?')
            lines.append(f"  🗑️  {color_name(pc)} discarded cards (robber)")

    # Robber movement
    rs = sc.get('mechanicRobberState', {})
    if 'locationTileIndex' in rs:
        lines.append(f"  👮 Robber moved to hex {rs['locationTileIndex']}")

    # Largest army change
    for color_str, data in sc.get('mechanicLargestArmyState', {}).items():
        if data.get('hasLargestArmy'):
            lines.append(f"  ★ {color_name(int(color_str))} takes Largest Army!")
        elif data.get('hasLargestArmy') is None:
            lines.append(f"  ☆ {color_name(int(color_str))} lost Largest Army")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Print full game state at every turn for manual verification."
    )
    parser.add_argument("game_json", help="Path to a Colonist game JSON file")
    parser.add_argument(
        "--turns", default=None,
        help="Turn range to print, e.g. '10-30' or '0-5' (default: all)"
    )
    parser.add_argument(
        "--events", action="store_true",
        help="Also print event-level details between turns"
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Hide detailed player info (dev cards, trade ratios, pieces left)"
    )
    parser.add_argument(
        "--no-buildings", action="store_true",
        help="Hide the building list at each turn (less verbose)"
    )
    args = parser.parse_args()

    # Parse turn range
    turn_min, turn_max = 0, 9999
    if args.turns:
        parts = args.turns.split("-")
        turn_min = int(parts[0])
        turn_max = int(parts[1]) if len(parts) > 1 else turn_min

    # Load game
    replay = GameReplay.from_file(args.game_json)
    settings = replay.game_data['data'].get('gameSettings', {})
    game_id = settings.get('id', 'unknown')

    print()
    print(f"{'█' * 60}")
    print(f"  GAME: {game_id}")
    print(f"  Players: {', '.join(color_name(c) for c in replay.play_order)}")
    print(f"  Turn order: {replay.play_order}")
    print(f"  Events: {len(replay.events)}")
    print(f"  Total turns: {replay.total_turns}")
    if replay.winner_color:
        print(f"  Winner: {color_name(replay.winner_color)} ({replay.winner_color})")
        print(f"  Final VP: {replay.player_final_vp}")
    print(f"{'█' * 60}")

    # Print initial board layout
    base = replay.base_state
    print()
    print_board_layout(base)

    # Replay event by event, printing state at each turn boundary
    state = base.copy()
    last_printed_turn = -1
    show_initial = (turn_min == 0)

    if show_initial:
        print_turn_state(state, 0,
                         show_buildings=not args.no_buildings,
                         detailed_players=not args.compact)
        last_printed_turn = 0

    for i, event in enumerate(replay.events):
        prev_turn = state.current_turn
        state.apply_event(event)
        new_turn = state.current_turn

        # Print event details if requested
        if args.events and turn_min <= new_turn <= turn_max:
            event_lines = describe_event(event, i)
            if event_lines:
                if last_printed_turn != new_turn:
                    print(f"  ── events during turn {new_turn} ──")
                for line in event_lines:
                    print(line)

        # Print full state at turn boundaries
        if new_turn > prev_turn and new_turn != last_printed_turn:
            if turn_min <= new_turn <= turn_max:
                print_turn_state(state, new_turn,
                                 show_buildings=not args.no_buildings,
                                 detailed_players=not args.compact)
                last_printed_turn = new_turn

            if new_turn > turn_max:
                break

    # Always print final state if in range
    final_turn = state.current_turn
    if final_turn >= turn_min and last_printed_turn != final_turn:
        print_turn_state(state, final_turn,
                         show_buildings=not args.no_buildings,
                         detailed_players=not args.compact)

    # End-of-game summary
    if replay.winner_color and final_turn <= turn_max:
        print(f"{'═' * 60}")
        print(f"  GAME OVER — {color_name(replay.winner_color)} wins!")
        print(f"{'═' * 60}")
        print()
        print("  Final rankings:")
        for color, rank in sorted(replay.player_rankings.items(), key=lambda x: x[1]):
            vp = replay.player_final_vp.get(color, 0)
            winner_str = " ★ WINNER" if replay.winner_color == color else ""
            print(f"    #{rank} {color_name(color):8s}  {vp} VP{winner_str}")

        # Dice stats
        dice_stats = replay.end_state.get('diceStats', [])
        if dice_stats:
            print()
            print("  Dice distribution (2-12):")
            for i, count in enumerate(dice_stats):
                num = i + 2
                bar = "█" * count
                print(f"    {num:2d}: {count:3d} {bar}")

        print()


if __name__ == "__main__":
    main()
