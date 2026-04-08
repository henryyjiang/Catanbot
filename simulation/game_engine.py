"""
Full Catan game simulation engine.

Handles the complete game loop:
  - Setup phase: snake-draft settlement + road placement
  - Main phase: dice roll, resource distribution, player actions
  - Robber: discard half on 7, move robber, steal
  - Win condition: first to 10 VP
  - Dev card VP: revealed immediately on purchase
"""

from __future__ import annotations

import random
from typing import Optional

from data.enums import BuildingType, DevCard, Resource, TileType, VPCategory
from data.state import CatanState, PlayerState
from mcts.actions import (
    PassTurn,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    BuyDevCard,
)
from mcts.state_transition import apply_action, _update_port_access, _update_longest_road


# Standard Catan dev card deck: 14 knights, 5 VP, 2 road building, 2 YoP, 2 monopoly
_DEV_CARD_DECK: list[int] = (
    [int(DevCard.KNIGHT)] * 14
    + [int(DevCard.VICTORY_POINT)] * 5
    + [int(DevCard.ROAD_BUILDING)] * 2
    + [int(DevCard.YEAR_OF_PLENTY)] * 2
    + [int(DevCard.MONOPOLY)] * 2
)

# Dice probability map (only numbers that appear on tiles)
_DICE_PROB: dict[int, float] = {
    2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
    8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36,
}

# Safety cap on actions per player per turn (prevents infinite loops)
_MAX_ACTIONS_PER_TURN = 30


def _find_desert_hex(state: CatanState) -> int:
    """Return the index of the desert hex (initial robber location)."""
    for hex_idx, tile_type in state.topology.hex_types.items():
        if tile_type == int(TileType.DESERT):
            return hex_idx
    return next(iter(state.topology.hex_positions))  # fallback to first hex


class GameStats:
    """Collects per-game statistics."""

    def __init__(self, colors: list[int]):
        self.colors = colors
        self.winner: Optional[int] = None
        self.total_turns: int = 0
        self.final_vp: dict[int, int] = {c: 0 for c in colors}
        self.resources_collected: dict[int, int] = {c: 0 for c in colors}
        # VP sampled at fixed turn milestones (turn 10, 20, 30, ...)
        self.vp_at_turns: dict[int, dict[int, int]] = {}  # turn -> {color -> vp}
        self.action_counts: dict[int, int] = {c: 0 for c in colors}
        self.roads_built: dict[int, int] = {c: 0 for c in colors}
        self.settlements_built: dict[int, int] = {c: 0 for c in colors}
        self.cities_built: dict[int, int] = {c: 0 for c in colors}
        self.dev_cards_bought: dict[int, int] = {c: 0 for c in colors}
        self.sevens_rolled: int = 0


class CatanGameEngine:
    """
    Simulates a complete game of Catan.

    Parameters
    ----------
    initial_state : CatanState
        Fresh initial state (no buildings placed, standard bank).
        Should be created via create_fresh_state().
    agents : dict[int, agent]
        Mapping of player color → agent instance.
    max_turns : int
        Hard cap on total player turns before declaring a winner by VP.
    """

    def __init__(
        self,
        initial_state: CatanState,
        agents: dict,
        max_turns: int = 400,
    ):
        self.state = initial_state.copy()
        self.agents = agents
        self.max_turns = max_turns

    # ── Setup helpers ──────────────────────────────────────────────────────────

    def _reset_dev_deck(self) -> None:
        deck = list(_DEV_CARD_DECK)
        random.shuffle(deck)
        self.state.bank_dev_cards = deck

    def _valid_setup_corners(self) -> list[int]:
        """All empty corners that satisfy the distance rule."""
        topo = self.state.topology
        occupied = set(self.state.corner_buildings.keys())
        valid = []
        for cidx in topo.corner_positions:
            if cidx in occupied:
                continue
            if any(adj in occupied for adj in topo.get_adjacent_corners(cidx)):
                continue
            valid.append(cidx)
        return valid

    def _place_setup_settlement(
        self,
        color: int,
        corner_idx: int,
        give_resources: bool,
    ) -> None:
        state = self.state
        player = state.players[color]

        state.corner_buildings[corner_idx] = (color, BuildingType.SETTLEMENT)
        player.settlements_remaining -= 1
        player.victory_points[VPCategory.SETTLEMENTS] = (
            player.victory_points.get(VPCategory.SETTLEMENTS, 0) + 1
        )
        _update_port_access(state, color, corner_idx)

        if give_resources:
            for hex_idx in state.topology.corner_to_hexes.get(corner_idx, []):
                resource = state.topology.hex_resources.get(hex_idx)
                if resource is None:
                    continue
                res_val = int(resource)
                available = state.bank_resources.get(res_val, 0)
                if available > 0:
                    state.bank_resources[res_val] -= 1
                    player.resource_cards.append(res_val)

    def _place_setup_road(self, color: int, edge_idx: int) -> None:
        self.state.edge_roads[edge_idx] = color
        self.state.players[color].roads_remaining -= 1
        _update_longest_road(self.state, color)

    def run_setup_phase(self) -> None:
        """Snake-draft setup: each player places 2 settlements + 2 roads."""
        colors = self.state.player_colors
        # Snake order: 1,2,3,4,4,3,2,1
        order = colors + list(reversed(colors))

        for i, color in enumerate(order):
            is_second = (i >= len(colors))
            agent = self.agents[color]

            valid_corners = self._valid_setup_corners()
            if not valid_corners:
                break

            corner_idx = agent.choose_setup_settlement(self.state, color, valid_corners)
            self._place_setup_settlement(color, corner_idx, give_resources=is_second)

            topo = self.state.topology
            valid_edges = [
                e for e in topo.corner_to_edges.get(corner_idx, [])
                if e not in self.state.edge_roads
            ]
            if valid_edges:
                edge_idx = agent.choose_setup_road(
                    self.state, color, corner_idx, valid_edges
                )
                self._place_setup_road(color, edge_idx)

        self.state.current_turn = len(colors) * 2
        self.state.current_player_color = colors[0]

    # ── Main phase helpers ─────────────────────────────────────────────────────

    def _distribute_resources(self, dice_total: int) -> dict[int, int]:
        """Give resources from the dice roll. Returns {color: cards_received}."""
        state = self.state
        received: dict[int, int] = {}
        for corner_idx, hex_idx, resource in state.topology.get_corners_for_dice(dice_total):
            if hex_idx == state.robber_hex:
                continue
            if corner_idx not in state.corner_buildings:
                continue
            owner, btype = state.corner_buildings[corner_idx]
            amount = 1 if btype == BuildingType.SETTLEMENT else 2
            res_val = int(resource)
            available = state.bank_resources.get(res_val, 0)
            given = min(available, amount)
            state.bank_resources[res_val] -= given
            state.players[owner].resource_cards.extend([res_val] * given)
            received[owner] = received.get(owner, 0) + given
        return received

    def _handle_seven(self, current_color: int, agent) -> None:
        """Discard half from players with >7 cards, move robber, optionally steal."""
        state = self.state

        # Discard: all players with >7 cards lose half (rounded down)
        for color, player in state.players.items():
            if player.total_resources > 7:
                discard_n = player.total_resources // 2
                for _ in range(discard_n):
                    if player.resource_cards:
                        card = random.choice(player.resource_cards)
                        player.resource_cards.remove(card)
                        state.bank_resources[card] = state.bank_resources.get(card, 0) + 1

        # Move robber
        valid_hexes = [h for h in state.topology.hex_positions if h != state.robber_hex]
        if valid_hexes:
            new_hex = agent.choose_robber_hex(state, current_color, valid_hexes)
            state.robber_hex = new_hex

            # Steal one card from a random opponent on that hex
            victims: list[int] = []
            for cidx in state.topology.hex_to_corners.get(new_hex, []):
                if cidx in state.corner_buildings:
                    owner, _ = state.corner_buildings[cidx]
                    if (
                        owner != current_color
                        and state.players[owner].total_resources > 0
                        and owner not in victims
                    ):
                        victims.append(owner)

            if victims:
                victim = random.choice(victims)
                stolen = random.choice(state.players[victim].resource_cards)
                state.players[victim].resource_cards.remove(stolen)
                state.players[current_color].resource_cards.append(stolen)

    def _handle_vp_card(self, color: int) -> None:
        """
        Recalculate VP-card count for `color`.
        VP dev cards are revealed immediately on purchase in real Catan.
        """
        player = self.state.players[color]
        vp_count = sum(
            1 for c in player.dev_cards if c == int(DevCard.VICTORY_POINT)
        )
        player.victory_points[VPCategory.DEV_CARD_VP] = vp_count

    def _check_winner(self) -> Optional[int]:
        for color, player in self.state.players.items():
            if player.total_vp >= 10:
                return color
        return None

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_main_phase(self) -> tuple[int, GameStats]:
        colors = self.state.player_colors
        stats = GameStats(colors)
        color_idx = 0
        milestone_turns = set(range(10, 401, 10))

        for global_turn in range(self.max_turns):
            winner = self._check_winner()
            if winner is not None:
                stats.winner = winner
                stats.total_turns = global_turn
                break

            color = colors[color_idx % len(colors)]
            self.state.current_player_color = color
            agent = self.agents[color]

            # Record VP milestones
            if global_turn in milestone_turns:
                stats.vp_at_turns[global_turn] = {
                    c: self.state.players[c].total_vp for c in colors
                }

            # Roll dice
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            dice_total = d1 + d2
            self.state.last_dice = (d1, d2)

            if dice_total == 7:
                stats.sevens_rolled += 1
                self._handle_seven(color, agent)
            else:
                received = self._distribute_resources(dice_total)
                for c, n in received.items():
                    stats.resources_collected[c] = stats.resources_collected.get(c, 0) + n

            # Player's main-phase actions
            dev_played = False
            actions_this_turn = 0

            while actions_this_turn < _MAX_ACTIONS_PER_TURN:
                try:
                    action = agent.choose_action(self.state, color, dev_played)
                except Exception:
                    break

                if isinstance(action, PassTurn):
                    break

                try:
                    self.state = apply_action(self.state, action, color)
                except Exception:
                    break

                # Reveal VP dev cards immediately
                if isinstance(action, BuyDevCard):
                    self._handle_vp_card(color)
                    stats.dev_cards_bought[color] = stats.dev_cards_bought.get(color, 0) + 1

                if isinstance(action, (PlayKnight, PlayMonopoly, PlayRoadBuilding, PlayYearOfPlenty)):
                    dev_played = True

                # Track action stats
                stats.action_counts[color] = stats.action_counts.get(color, 0) + 1
                aname = type(action).__name__
                if aname == "BuildRoad":
                    stats.roads_built[color] = stats.roads_built.get(color, 0) + 1
                elif aname == "BuildSettlement":
                    stats.settlements_built[color] = stats.settlements_built.get(color, 0) + 1
                elif aname == "BuildCity":
                    stats.cities_built[color] = stats.cities_built.get(color, 0) + 1

                actions_this_turn += 1

            self.state.current_turn += 1
            color_idx += 1

        else:
            # Turn limit reached — declare winner by VP
            winner = max(self.state.players.items(), key=lambda x: x[1].total_vp)[0]
            stats.winner = winner
            stats.total_turns = self.max_turns

        # Final VP snapshot
        for c in colors:
            stats.final_vp[c] = self.state.players[c].total_vp

        return stats.winner, stats

    def run_game(self) -> tuple[int, GameStats]:
        """Run a full game (setup + main phase). Returns (winner_color, stats)."""
        self._reset_dev_deck()
        self.run_setup_phase()
        return self.run_main_phase()


# ── Board factory ─────────────────────────────────────────────────────────────

def create_fresh_state(game_data: dict, player_colors: list[int]) -> CatanState:
    """
    Load a board topology from a Colonist.io game JSON and return a completely
    fresh initial state: no buildings, no roads, standard bank, empty hands.

    The board layout (hex types, dice numbers, ports) is taken from the JSON,
    but all game state is reset so a new simulation can be played on it.
    """
    # Patch the play order so the topology builder uses our colours
    game_data = dict(game_data)  # shallow copy — we only mutate top-level keys
    game_data['data'] = dict(game_data['data'])
    game_data['data']['playOrder'] = player_colors

    init = game_data['data']['eventHistory']['initialState']

    # Build a minimal playerStates so CatanState.from_initial_state is happy
    init['playerStates'] = {
        str(c): {
            'resourceCards': {'cards': []},
            'victoryPointsState': {},
            'bankTradeRatiosState': {str(r): 4 for r in range(1, 6)},
        }
        for c in player_colors
    }

    state = CatanState.from_initial_state(game_data)

    # Hard-reset all player states (from_initial_state may have stale data)
    for color in player_colors:
        state.players[color] = PlayerState(color=color)

    # Standard Catan bank: 19 of each resource
    state.bank_resources = {r: 19 for r in range(1, 6)}

    # Clear any buildings/roads (should already be empty, but be safe)
    state.corner_buildings = {}
    state.edge_roads = {}

    # Robber starts on the desert
    state.robber_hex = _find_desert_hex(state)

    # Reset turn tracking
    state.current_turn = 0
    state.current_player_color = player_colors[0]

    return state
