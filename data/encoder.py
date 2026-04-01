from __future__ import annotations

import numpy as np
from typing import Optional

from data.enums import (
    Resource, TileType, BuildingType, DevCard,
    NUM_RESOURCE_TYPES, VPCategory
)
from data.state import CatanState, PlayerState


class StateEncoder:
    """Encodes CatanState into fixed-size numpy arrays."""
 
    NUM_HEXES = 19
    NUM_CORNERS = 54
    NUM_EDGES = 72
    NUM_PLAYERS = 4

    HEX_FEATURES = 13 
    CORNER_FEATURES = 12  
    EDGE_FEATURES = 5     
    PLAYER_FEATURES = 24  
    GLOBAL_FEATURES = 12

    def __init__(self):
        self._total_flat_size = None

    @property
    def total_flat_size(self) -> int:
        """Total size of the flattened feature vector."""
        if self._total_flat_size is None:
            self._total_flat_size = (
                self.NUM_HEXES * self.HEX_FEATURES +
                self.NUM_CORNERS * self.CORNER_FEATURES +
                self.NUM_EDGES * self.EDGE_FEATURES +
                self.NUM_PLAYERS * self.PLAYER_FEATURES +
                self.GLOBAL_FEATURES
            )
        return self._total_flat_size

    def encode(
        self,
        state: CatanState,
        perspective_color: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if perspective_color is None:
            perspective_color = state.current_player_color

        color_order = self._get_relative_order(state, perspective_color)
        color_to_idx = {c: i for i, c in enumerate(color_order)}

        hex_feat = self._encode_hexes(state, color_to_idx)
        corner_feat = self._encode_corners(state, color_to_idx)
        edge_feat = self._encode_edges(state, color_to_idx)
        player_feat = self._encode_players(state, color_order)
        global_feat = self._encode_globals(state, perspective_color, color_to_idx)

        flat = np.concatenate([
            hex_feat.flatten(),
            corner_feat.flatten(),
            edge_feat.flatten(),
            player_feat.flatten(),
            global_feat,
        ])

        return {
            'hex_features': hex_feat,
            'corner_features': corner_feat,
            'edge_features': edge_feat,
            'player_features': player_feat,
            'global_features': global_feat,
            'flat': flat,
        }

    def _get_relative_order(
        self, state: CatanState, perspective_color: int
    ) -> list[int]:
        colors = state.player_colors
        if perspective_color not in colors:
            return colors
        idx = colors.index(perspective_color)
        return colors[idx:] + colors[:idx]

    def _encode_hexes(
        self, state: CatanState, color_to_idx: dict[int, int]
    ) -> np.ndarray:
        """
          [0-5]  tile type one-hot (desert, grain, ore, lumber, brick, wool)
          [6]    dice number / 12 (normalized)
          [7]    dice probability (dots / 5, normalized)
          [8]    robber present (0/1)
          [9-12] has building adjacent for each player (0/1)
        """
        feat = np.zeros((self.NUM_HEXES, self.HEX_FEATURES), dtype=np.float32)

        for hex_idx in range(self.NUM_HEXES):
            if hex_idx not in state.topology.hex_types:
                continue

            tile_type = state.topology.hex_types[hex_idx]
            dice_num = state.topology.hex_dice_numbers.get(hex_idx, 0)

            # Tile type one-hot
            if tile_type < 6:
                feat[hex_idx, tile_type] = 1.0

            # Dice number normalized
            feat[hex_idx, 6] = dice_num / 12.0

            # Dice probability (number of ways to roll this)
            dots = self._dice_dots(dice_num)
            feat[hex_idx, 7] = dots / 5.0

            # Robber
            feat[hex_idx, 8] = 1.0 if state.robber_hex == hex_idx else 0.0

            # Adjacent buildings per player
            for cidx in state.topology.hex_to_corners.get(hex_idx, []):
                if cidx in state.corner_buildings:
                    owner, btype = state.corner_buildings[cidx]
                    pidx = color_to_idx.get(owner)
                    if pidx is not None and pidx < 4:
                        feat[hex_idx, 9 + pidx] = 1.0

        return feat

    def _encode_corners(
        self, state: CatanState, color_to_idx: dict[int, int]
    ) -> np.ndarray:
        """
          [0-3]  owner one-hot by relative player index (0 if empty)
          [4]    is settlement (0/1)
          [5]    is city (0/1)
          [6]    has port (0/1)
          [7]    port is generic 3:1 (0/1)
          [8]    port is specific 2:1 (0/1)
          [9]    sum of adjacent hex dice probabilities (normalized)
          [10]   number of distinct adjacent resources / 3 (normalized)
          [11]   robber adjacent (0/1)
        """
        feat = np.zeros((self.NUM_CORNERS, self.CORNER_FEATURES), dtype=np.float32)

        for cidx in range(self.NUM_CORNERS):
            if cidx not in state.topology.corner_positions:
                continue

            # Building
            if cidx in state.corner_buildings:
                owner, btype = state.corner_buildings[cidx]
                pidx = color_to_idx.get(owner)
                if pidx is not None and pidx < 4:
                    feat[cidx, pidx] = 1.0
                feat[cidx, 4] = 1.0 if btype == BuildingType.SETTLEMENT else 0.0
                feat[cidx, 5] = 1.0 if btype == BuildingType.CITY else 0.0

            # Port
            if cidx in state.topology.corner_ports:
                port_type, ratio, resource = state.topology.corner_ports[cidx]
                feat[cidx, 6] = 1.0
                feat[cidx, 7] = 1.0 if ratio == 3 else 0.0
                feat[cidx, 8] = 1.0 if ratio == 2 else 0.0

            # Adjacent hex stats
            adj_hexes = state.topology.corner_to_hexes.get(cidx, [])
            total_prob = 0.0
            resources_seen = set()
            robber_adj = False

            for hex_idx in adj_hexes:
                dice_num = state.topology.hex_dice_numbers.get(hex_idx, 0)
                total_prob += self._dice_dots(dice_num)
                res = state.topology.hex_resources.get(hex_idx)
                if res is not None:
                    resources_seen.add(res)
                if state.robber_hex == hex_idx:
                    robber_adj = True

            feat[cidx, 9] = min(total_prob / 15.0, 1.0)  # normalize
            feat[cidx, 10] = len(resources_seen) / 3.0
            feat[cidx, 11] = 1.0 if robber_adj else 0.0

        return feat

    def _encode_edges(
        self, state: CatanState, color_to_idx: dict[int, int]
    ) -> np.ndarray:
        """Encode edge features.

        Per edge (72 edges × 5 features):
          [0-3]  road owner one-hot by relative player index (0 if empty)
          [4]    has road (0/1)
        """
        feat = np.zeros((self.NUM_EDGES, self.EDGE_FEATURES), dtype=np.float32)

        for eidx in range(self.NUM_EDGES):
            if eidx in state.edge_roads:
                owner = state.edge_roads[eidx]
                pidx = color_to_idx.get(owner)
                if pidx is not None and pidx < 4:
                    feat[eidx, pidx] = 1.0
                feat[eidx, 4] = 1.0

        return feat

    def _encode_players(
        self, state: CatanState, color_order: list[int]
    ) -> np.ndarray:
        """Encode per-player features.

        Per player (4 players × 24 features), ordered by relative position:
          [0-4]   resource counts (lumber, brick, wool, grain, ore) / 10
          [5]     total resources / 20
          [6]     total dev cards / 10
          [7]     knights played / 5
          [8]     longest road length / 15
          [9]     has largest army (0/1)
          [10]    has longest road award (0/1)
          [11]    victory points / 10
          [12]    settlements on board / 5
          [13]    cities on board / 4
          [14]    roads on board / 15
          [15]    settlements remaining / 5
          [16]    cities remaining / 4
          [17]    roads remaining / 15
          [18-22] bank trade ratios per resource (1/ratio, so 4:1→0.25, 3:1→0.33, 2:1→0.5)
          [23]    is connected (0/1)
        """
        feat = np.zeros((self.NUM_PLAYERS, self.PLAYER_FEATURES), dtype=np.float32)

        for pidx, color in enumerate(color_order):
            if color not in state.players:
                continue
            p = state.players[color]

            # Resource counts
            rcounts = p.resource_counts
            for r in Resource:
                feat[pidx, r.value - 1] = rcounts.get(r.value, 0) / 10.0

            feat[pidx, 5] = p.total_resources / 20.0
            feat[pidx, 6] = p.total_dev_cards / 10.0
            feat[pidx, 7] = p.knights_played / 5.0
            feat[pidx, 8] = p.longest_road / 15.0
            feat[pidx, 9] = 1.0 if p.has_largest_army else 0.0

            # Longest road award (VP category 4)
            feat[pidx, 10] = 1.0 if p.victory_points.get(
                VPCategory.LONGEST_ROAD, 0
            ) > 0 else 0.0

            feat[pidx, 11] = p.total_vp / 10.0

            # Buildings on board
            buildings = state.get_buildings_for_player(color)
            settlements = sum(1 for b in buildings.values() if b == BuildingType.SETTLEMENT)
            cities = sum(1 for b in buildings.values() if b == BuildingType.CITY)
            roads = len(state.get_roads_for_player(color))

            feat[pidx, 12] = settlements / 5.0
            feat[pidx, 13] = cities / 4.0
            feat[pidx, 14] = roads / 15.0
            feat[pidx, 15] = p.settlements_remaining / 5.0
            feat[pidx, 16] = p.cities_remaining / 4.0
            feat[pidx, 17] = p.roads_remaining / 15.0

            # Bank trade ratios
            for r in Resource:
                ratio = p.bank_trade_ratios.get(r.value, 4)
                feat[pidx, 18 + (r.value - 1)] = 1.0 / max(ratio, 1)

            feat[pidx, 23] = 1.0 if p.is_connected else 0.0

        return feat

    def _encode_globals(
        self,
        state: CatanState,
        perspective_color: int,
        color_to_idx: dict[int, int],
    ) -> np.ndarray:

        """
          [0]    current turn / 100 (normalized)
          [1]    setup phase (0/1)
          [2]    is our turn (0/1)
          [3]    dice total / 12
          [4]    robber hex index / 19
          [5-8]  turn order position of each player / 4
          [9]    bank dev cards remaining / 25
          [10]   total bank resources / 95
          [11]   game progress estimate (total buildings / max buildings)
        """
        feat = np.zeros(self.GLOBAL_FEATURES, dtype=np.float32)

        feat[0] = min(state.current_turn / 100.0, 1.0)
        feat[1] = 1.0 if state.is_setup_phase() else 0.0
        feat[2] = 1.0 if state.current_player_color == perspective_color else 0.0
        feat[3] = sum(state.last_dice) / 12.0
        feat[4] = state.robber_hex / 19.0

        for color, idx in color_to_idx.items():
            if idx < 4:
                feat[5 + idx] = state.player_colors.index(color) / 4.0 if color in state.player_colors else 0

        feat[9] = len(state.bank_dev_cards) / 25.0
        feat[10] = sum(state.bank_resources.values()) / 95.0

        total_buildings = len(state.corner_buildings) + len(state.edge_roads)
        max_buildings = 4 * (5 + 4 + 15)  # 4 players × (settlements + cities + roads)
        feat[11] = total_buildings / max_buildings

        return feat

    def encode_flat(
        self,
        state: CatanState,
        perspective_color: Optional[int] = None,
    ) -> np.ndarray:
        return self.encode(state, perspective_color)['flat']

    @staticmethod
    def _dice_dots(number: int) -> int:
        dots = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        return dots.get(number, 0)

    def feature_names(self) -> list[str]:
        names = []

        # Hex features
        hex_names = [
            'desert', 'grain', 'ore', 'lumber', 'brick', 'wool',
            'dice_num', 'dice_prob', 'robber',
            'p0_adj', 'p1_adj', 'p2_adj', 'p3_adj'
        ]
        for h in range(self.NUM_HEXES):
            for fn in hex_names:
                names.append(f'hex{h}_{fn}')

        # Corner features
        corner_names = [
            'p0_owns', 'p1_owns', 'p2_owns', 'p3_owns',
            'is_settlement', 'is_city', 'has_port', 'port_3_1', 'port_2_1',
            'adj_prob', 'adj_diversity', 'robber_adj'
        ]
        for c in range(self.NUM_CORNERS):
            for fn in corner_names:
                names.append(f'corner{c}_{fn}')

        # Edge features
        edge_names = ['p0_road', 'p1_road', 'p2_road', 'p3_road', 'has_road']
        for e in range(self.NUM_EDGES):
            for fn in edge_names:
                names.append(f'edge{e}_{fn}')

        # Player features
        player_names = [
            'lumber', 'brick', 'wool', 'grain', 'ore',
            'total_res', 'total_dev', 'knights', 'longest_road',
            'has_army', 'has_road_award', 'vp',
            'settlements', 'cities', 'roads',
            'settle_rem', 'city_rem', 'road_rem',
            'trade_lumber', 'trade_brick', 'trade_wool', 'trade_grain', 'trade_ore',
            'connected'
        ]
        for p in range(self.NUM_PLAYERS):
            for fn in player_names:
                names.append(f'p{p}_{fn}')

        # Global features
        global_names = [
            'turn', 'setup_phase', 'is_our_turn', 'dice_total',
            'robber_hex', 'p0_order', 'p1_order', 'p2_order', 'p3_order',
            'bank_dev', 'bank_resources', 'game_progress'
        ]
        names.extend(global_names)

        return names
