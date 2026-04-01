
from __future__ import annotations
import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from data.enums import (
    Resource, BuildingType, DevCard, NUM_RESOURCE_TYPES,
    VPCategory, LogType, ActionState
)
from data.topology import BoardTopology


def deep_merge(base: dict, delta: dict) -> dict:
    """Recursively merge delta into base, modifying base in place. """
    for key, value in delta.items():
        if value is None:
            base.pop(key, None)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class PlayerState:
    color: int
    resource_cards: list[int] = field(default_factory=list)
    dev_cards: list[int] = field(default_factory=list)
    dev_cards_used: list[int] = field(default_factory=list)
    victory_points: dict[int, int] = field(default_factory=dict)
    bank_trade_ratios: dict[int, int] = field(default_factory=dict)
    is_connected: bool = True

    settlements_remaining: int = 5
    cities_remaining: int = 4
    roads_remaining: int = 15

    longest_road: int = 0
    has_largest_army: bool = False
    knights_played: int = 0

    @property
    def total_vp(self) -> int:
        return sum(self.victory_points.values())

    @property
    def resource_counts(self) -> dict[int, int]:
        counts = {r.value: 0 for r in Resource}
        for card in self.resource_cards:
            counts[card] = counts.get(card, 0) + 1
        return counts

    @property
    def total_resources(self) -> int:
        return len(self.resource_cards)

    @property
    def total_dev_cards(self) -> int:
        return len(self.dev_cards)


@dataclass
class CatanState:
    topology: BoardTopology

    # Board state: what's built where
    corner_buildings: dict[int, tuple[int, int]] = field(default_factory=dict)
    # corner_idx → (owner_color, BuildingType)
    edge_roads: dict[int, int] = field(default_factory=dict)
    # edge_idx → owner_color

    # Player states
    players: dict[int, PlayerState] = field(default_factory=dict)
    player_colors: list[int] = field(default_factory=list)

    # Bank state
    bank_resources: dict[int, int] = field(default_factory=dict)
    bank_dev_cards: list[int] = field(default_factory=list)

    # Robber
    robber_hex: int = 0

    # Turn tracking
    current_turn: int = 0
    current_player_color: int = 0
    turn_state: int = 0
    action_state: int = 0

    # Dice
    last_dice: tuple[int, int] = (0, 0)

    # Trade state (simplified)
    active_trades: dict = field(default_factory=dict)

    # Event counter
    events_applied: int = 0

    # Raw state dicts for fallback deep-merge
    _raw_player_states: dict = field(default_factory=dict)
    _raw_bank_state: dict = field(default_factory=dict)
    _raw_dev_card_state: dict = field(default_factory=dict)
    _raw_mechanic_states: dict = field(default_factory=dict)

    @classmethod
    def from_initial_state(cls, game_data: dict) -> 'CatanState':
        init = game_data['data']['eventHistory']['initialState']
        map_state = init['mapState']
        play_order = game_data['data']['playOrder']

        topology = BoardTopology.from_initial_state(map_state)

        state = cls(topology=topology)
        state.player_colors = play_order

        for color_str, ps in init['playerStates'].items():
            color = int(color_str)
            player = PlayerState(
                color=color,
                resource_cards=list(ps.get('resourceCards', {}).get('cards', [])),
                bank_trade_ratios={
                    int(k): v
                    for k, v in ps.get('bankTradeRatiosState', {}).items()
                },
            )
            state.players[color] = player

        state.bank_resources = {
            int(k): v
            for k, v in init['bankState'].get('resourceCards', {}).items()
        }
        state.bank_dev_cards = list(
            init.get('mechanicDevelopmentCardsState', {})
            .get('bankDevelopmentCards', {})
            .get('cards', [])
        )

        dev_state = init.get('mechanicDevelopmentCardsState', {})
        for color_str, dcs in dev_state.get('players', {}).items():
            color = int(color_str)
            if color in state.players:
                state.players[color].dev_cards = list(
                    dcs.get('developmentCards', {}).get('cards', [])
                )
                state.players[color].dev_cards_used = list(
                    dcs.get('developmentCardsUsed', [])
                )

        for color_str, ms in init.get('mechanicSettlementState', {}).items():
            color = int(color_str)
            if color in state.players:
                state.players[color].settlements_remaining = ms.get('bankSettlementAmount', 5)

        for color_str, ms in init.get('mechanicCityState', {}).items():
            color = int(color_str)
            if color in state.players:
                state.players[color].cities_remaining = ms.get('bankCityAmount', 4)

        for color_str, ms in init.get('mechanicRoadState', {}).items():
            color = int(color_str)
            if color in state.players:
                state.players[color].roads_remaining = ms.get('bankRoadAmount', 15)

        for color_str, ms in init.get('mechanicLongestRoadState', {}).items():
            color = int(color_str)
            if color in state.players:
                state.players[color].longest_road = ms.get('longestRoad', 0)

        for color_str, ms in init.get('mechanicLargestArmyState', {}).items():
            color = int(color_str)
            if color in state.players:
                state.players[color].has_largest_army = ms.get('hasLargestArmy', False)

        robber_state = init.get('mechanicRobberState', {})
        state.robber_hex = robber_state.get('locationTileIndex', 0)

        cs = init.get('currentState', {})
        state.current_turn = cs.get('completedTurns', 0)
        state.current_player_color = cs.get('currentTurnPlayerColor', play_order[0])
        state.turn_state = cs.get('turnState', 0)
        state.action_state = cs.get('actionState', 0)

        state._raw_player_states = copy.deepcopy(init.get('playerStates', {}))
        state._raw_bank_state = copy.deepcopy(init.get('bankState', {}))
        state._raw_dev_card_state = copy.deepcopy(
            init.get('mechanicDevelopmentCardsState', {})
        )
        state._raw_mechanic_states = {
            'settlement': copy.deepcopy(init.get('mechanicSettlementState', {})),
            'city': copy.deepcopy(init.get('mechanicCityState', {})),
            'road': copy.deepcopy(init.get('mechanicRoadState', {})),
            'longestRoad': copy.deepcopy(init.get('mechanicLongestRoadState', {})),
            'largestArmy': copy.deepcopy(init.get('mechanicLargestArmyState', {})),
            'robber': copy.deepcopy(init.get('mechanicRobberState', {})),
        }

        return state

    def apply_event(self, event: dict) -> None:
        sc = event.get('stateChange', {})
        if not sc:
            self.events_applied += 1
            return

        if 'mapState' in sc:
            self._apply_map_changes(sc['mapState'])

        if 'playerStates' in sc:
            self._apply_player_changes(sc['playerStates'])

        if 'bankState' in sc:
            self._apply_bank_changes(sc['bankState'])

        if 'diceState' in sc:
            ds = sc['diceState']
            if ds.get('diceThrown'):
                self.last_dice = (ds.get('dice1', 0), ds.get('dice2', 0))

        if 'currentState' in sc:
            cs = sc['currentState']
            if 'completedTurns' in cs:
                self.current_turn = cs['completedTurns']
            if 'currentTurnPlayerColor' in cs:
                self.current_player_color = cs['currentTurnPlayerColor']
            if 'turnState' in cs:
                self.turn_state = cs['turnState']
            if 'actionState' in cs:
                self.action_state = cs['actionState']

        if 'mechanicDevelopmentCardsState' in sc:
            self._apply_dev_card_changes(sc['mechanicDevelopmentCardsState'])

        if 'mechanicRobberState' in sc:
            rs = sc['mechanicRobberState']
            if 'locationTileIndex' in rs:
                self.robber_hex = rs['locationTileIndex']

        if 'mechanicSettlementState' in sc:
            for color_str, ms in sc['mechanicSettlementState'].items():
                color = int(color_str)
                if color in self.players and 'bankSettlementAmount' in ms:
                    self.players[color].settlements_remaining = ms['bankSettlementAmount']

        if 'mechanicCityState' in sc:
            for color_str, ms in sc['mechanicCityState'].items():
                color = int(color_str)
                if color in self.players and 'bankCityAmount' in ms:
                    self.players[color].cities_remaining = ms['bankCityAmount']

        if 'mechanicRoadState' in sc:
            for color_str, ms in sc['mechanicRoadState'].items():
                color = int(color_str)
                if color in self.players and 'bankRoadAmount' in ms:
                    self.players[color].roads_remaining = ms['bankRoadAmount']

        if 'mechanicLongestRoadState' in sc:
            for color_str, ms in sc['mechanicLongestRoadState'].items():
                color = int(color_str)
                if color in self.players and 'longestRoad' in ms:
                    self.players[color].longest_road = ms['longestRoad']

        if 'mechanicLargestArmyState' in sc:
            for color_str, ms in sc['mechanicLargestArmyState'].items():
                color = int(color_str)
                if color in self.players:
                    val = ms.get('hasLargestArmy')
                    if val is not None:
                        self.players[color].has_largest_army = bool(val)
                    elif val is None:
                        self.players[color].has_largest_army = False

        if 'tradeState' in sc:
            ts = sc['tradeState']
            if 'activeOffers' in ts:
                deep_merge(self.active_trades, ts['activeOffers'])

        self.events_applied += 1

    def _apply_map_changes(self, map_delta: dict):
        for corner_str, data in map_delta.get('tileCornerStates', {}).items():
            cidx = int(corner_str)
            if 'owner' in data and 'buildingType' in data:
                self.corner_buildings[cidx] = (data['owner'], data['buildingType'])

                # Update port access for the building owner
                if cidx in self.topology.corner_ports:
                    port_type, ratio, resource = self.topology.corner_ports[cidx]
                    owner = data['owner']
                    if owner in self.players:
                        if resource is not None:
                            # Specific resource port
                            current = self.players[owner].bank_trade_ratios.get(
                                resource.value if isinstance(resource, Resource) else resource, 4
                            )
                            res_key = resource.value if isinstance(resource, Resource) else resource
                            if ratio < current:
                                self.players[owner].bank_trade_ratios[res_key] = ratio
                        else:
                            for r in Resource:
                                current = self.players[owner].bank_trade_ratios.get(r.value, 4)
                                if ratio < current:
                                    self.players[owner].bank_trade_ratios[r.value] = ratio

        for edge_str, data in map_delta.get('tileEdgeStates', {}).items():
            eidx = int(edge_str)
            if 'owner' in data:
                self.edge_roads[eidx] = data['owner']

    def _apply_player_changes(self, player_delta: dict):
        """Apply resource card and VP changes."""
        for color_str, changes in player_delta.items():
            color = int(color_str)
            if color not in self.players:
                continue
            player = self.players[color]

            if 'resourceCards' in changes:
                cards = changes['resourceCards'].get('cards')
                if cards is not None:
                    player.resource_cards = list(cards)

            if 'victoryPointsState' in changes:
                for vp_key, vp_val in changes['victoryPointsState'].items():
                    player.victory_points[int(vp_key)] = vp_val

            # Bank trade ratios
            if 'bankTradeRatiosState' in changes:
                for res_str, ratio in changes['bankTradeRatiosState'].items():
                    player.bank_trade_ratios[int(res_str)] = ratio

            if 'isConnected' in changes:
                player.is_connected = changes['isConnected']

    def _apply_bank_changes(self, bank_delta: dict):
        if 'resourceCards' in bank_delta:
            for res_str, count in bank_delta['resourceCards'].items():
                self.bank_resources[int(res_str)] = count

    def _apply_dev_card_changes(self, dev_delta: dict):
        if 'bankDevelopmentCards' in dev_delta:
            cards = dev_delta['bankDevelopmentCards'].get('cards')
            if cards is not None:
                self.bank_dev_cards = list(cards)

        for color_str, changes in dev_delta.get('players', {}).items():
            color = int(color_str)
            if color not in self.players:
                continue
            player = self.players[color]

            if 'developmentCards' in changes:
                cards = changes['developmentCards'].get('cards')
                if cards is not None:
                    player.dev_cards = list(cards)

            if 'developmentCardsUsed' in changes:
                used = changes['developmentCardsUsed']
                if used is not None:
                    player.dev_cards_used = list(used)
                    # Count knights for army tracking
                    player.knights_played = sum(
                        1 for c in player.dev_cards_used
                        if c == DevCard.KNIGHT
                    )

    def copy(self) -> 'CatanState':
        new = CatanState(topology=self.topology)
        new.corner_buildings = dict(self.corner_buildings)
        new.edge_roads = dict(self.edge_roads)
        new.players = {
            c: PlayerState(
                color=p.color,
                resource_cards=list(p.resource_cards),
                dev_cards=list(p.dev_cards),
                dev_cards_used=list(p.dev_cards_used),
                victory_points=dict(p.victory_points),
                bank_trade_ratios=dict(p.bank_trade_ratios),
                is_connected=p.is_connected,
                settlements_remaining=p.settlements_remaining,
                cities_remaining=p.cities_remaining,
                roads_remaining=p.roads_remaining,
                longest_road=p.longest_road,
                has_largest_army=p.has_largest_army,
                knights_played=p.knights_played,
            )
            for c, p in self.players.items()
        }
        new.player_colors = list(self.player_colors)
        new.bank_resources = dict(self.bank_resources)
        new.bank_dev_cards = list(self.bank_dev_cards)
        new.robber_hex = self.robber_hex
        new.current_turn = self.current_turn
        new.current_player_color = self.current_player_color
        new.turn_state = self.turn_state
        new.action_state = self.action_state
        new.last_dice = self.last_dice
        new.active_trades = copy.deepcopy(self.active_trades)
        new.events_applied = self.events_applied
        return new

    def get_buildings_for_player(self, color: int) -> dict[int, int]:
        return {
            cidx: btype
            for cidx, (owner, btype) in self.corner_buildings.items()
            if owner == color
        }

    def get_roads_for_player(self, color: int) -> list[int]:
        return [eidx for eidx, owner in self.edge_roads.items() if owner == color]

    def get_player_by_turn_order(self, position: int) -> PlayerState:
        color = self.player_colors[position]
        return self.players[color]

    def is_setup_phase(self) -> bool:
        return self.current_turn < len(self.player_colors) * 2

    def summary(self) -> str:
        lines = [
            f"Turn {self.current_turn} | Player {self.current_player_color}'s turn",
            f"Dice: {self.last_dice[0]}+{self.last_dice[1]}={sum(self.last_dice)}",
            f"Robber on hex {self.robber_hex}",
            f"Events applied: {self.events_applied}",
            "",
        ]
        for color in self.player_colors:
            p = self.players[color]
            buildings = self.get_buildings_for_player(color)
            settlements = sum(1 for b in buildings.values() if b == BuildingType.SETTLEMENT)
            cities = sum(1 for b in buildings.values() if b == BuildingType.CITY)
            roads = len(self.get_roads_for_player(color))
            lines.append(
                f"Player {color}: {p.total_vp}VP | "
                f"{settlements}S {cities}C {roads}R | "
                f"{p.total_resources} cards | "
                f"{p.total_dev_cards} dev | "
                f"road={p.longest_road} knights={p.knights_played}"
            )
        return "\n".join(lines)
