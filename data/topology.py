"""
Hex topology for standard Catan board.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from data.enums import TileType, TILE_RESOURCE, Resource


def hex_corners(hx: int, hy: int) -> list[tuple[int, int, int]]:
    """Return the 6 corner (x,y,z) coords for hex at (hx, hy).
    z=0 and z=1 are the two corners 'owned' by this hex.
    """
    return [
        (hx, hy, 0),           # this hex's z=0
        (hx, hy, 1),           # this hex's z=1
        (hx + 1, hy - 1, 1),   # neighbor's z=1 bottom-right
        (hx + 1, hy, 1),       # neighbor's z=1 right
        (hx, hy + 1, 0),       # neighbor's z=0 bottom-left
        (hx - 1, hy + 1, 0),   # neighbor's z=0 left
    ]


def hex_edges(hx: int, hy: int) -> list[tuple[int, int, int]]:
    """Return the 6 edge (x,y,z) coords for hex at (hx, hy).
    Edge z values: 0=top, 1=top-right, 2=bottom-right.
    """
    return [
        (hx, hy, 0),           # top
        (hx, hy, 1),           # top-right
        (hx, hy, 2),           # bottom-right
        (hx, hy + 1, 0),       # bottom
        (hx - 1, hy + 1, 1),   # bottom-left
        (hx - 1, hy, 2),       # top-left
    ]

@dataclass
class BoardTopology:
    hex_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
    hex_types: dict[int, int] = field(default_factory=dict)  
    hex_dice_numbers: dict[int, int] = field(default_factory=dict)   
    hex_resources: dict[int, Optional[int]] = field(default_factory=dict) 
    corner_positions: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    edge_positions: dict[int, tuple[int, int, int]] = field(default_factory=dict) 
    corner_coord_to_idx: dict[tuple[int, int, int], int] = field(default_factory=dict)
    edge_coord_to_idx: dict[tuple[int, int, int], int] = field(default_factory=dict)
    corner_ports: dict[int, tuple[int, int, Optional[int]]] = field(default_factory=dict)

    hex_to_corners: dict[int, list[int]] = field(default_factory=dict)
    hex_to_edges: dict[int, list[int]] = field(default_factory=dict)
    corner_to_hexes: dict[int, list[int]] = field(default_factory=dict)
    corner_to_edges: dict[int, list[int]] = field(default_factory=dict)
    edge_to_corners: dict[int, list[int]] = field(default_factory=dict)

    dice_to_hexes: dict[int, list[int]] = field(default_factory=dict)

    num_hexes: int = 0
    num_corners: int = 0
    num_edges: int = 0

    @classmethod
    def from_initial_state(cls, map_state: dict) -> 'BoardTopology':
        topo = cls()

        for idx_str, hex_data in map_state['tileHexStates'].items():
            idx = int(idx_str)
            hx, hy = hex_data['x'], hex_data['y']
            tile_type = hex_data['type']
            dice_num = hex_data.get('diceNumber', 0)

            topo.hex_positions[idx] = (hx, hy)
            topo.hex_types[idx] = tile_type
            topo.hex_dice_numbers[idx] = dice_num
            topo.hex_resources[idx] = TILE_RESOURCE.get(TileType(tile_type))

            if dice_num > 0:
                topo.dice_to_hexes.setdefault(dice_num, []).append(idx)

        topo.num_hexes = len(topo.hex_positions)

        for idx_str, corner_data in map_state['tileCornerStates'].items():
            idx = int(idx_str)
            coord = (corner_data['x'], corner_data['y'], corner_data['z'])
            topo.corner_positions[idx] = coord
            topo.corner_coord_to_idx[coord] = idx

        topo.num_corners = len(topo.corner_positions)

        for idx_str, edge_data in map_state['tileEdgeStates'].items():
            idx = int(idx_str)
            coord = (edge_data['x'], edge_data['y'], edge_data['z'])
            topo.edge_positions[idx] = coord
            topo.edge_coord_to_idx[coord] = idx

        topo.num_edges = len(topo.edge_positions)

        for hex_idx, (hx, hy) in topo.hex_positions.items():
            corner_coords = hex_corners(hx, hy)
            corner_idxs = []
            for cc in corner_coords:
                if cc in topo.corner_coord_to_idx:
                    cidx = topo.corner_coord_to_idx[cc]
                    corner_idxs.append(cidx)
                    topo.corner_to_hexes.setdefault(cidx, []).append(hex_idx)
            topo.hex_to_corners[hex_idx] = corner_idxs

        for hex_idx, (hx, hy) in topo.hex_positions.items():
            edge_coords = hex_edges(hx, hy)
            edge_idxs = []
            for ec in edge_coords:
                if ec in topo.edge_coord_to_idx:
                    eidx = topo.edge_coord_to_idx[ec]
                    edge_idxs.append(eidx)
            topo.hex_to_edges[hex_idx] = edge_idxs

        topo._build_corner_edge_adjacency()
        topo._parse_ports(map_state)

        return topo

    def _build_corner_edge_adjacency(self):
        """Build corner↔edge adjacency using the Colonist coordinate convention.
        """
        for eidx, (ex, ey, ez) in self.edge_positions.items():
            if ez == 0:
                cc = [(ex, ey, 1), (ex, ey, 0)]
            elif ez == 1:
                cc = [(ex, ey, 0), (ex + 1, ey, 1)]
            elif ez == 2:
                cc = [(ex + 1, ey, 1), (ex, ey + 1, 0)]
            else:
                continue

            corner_idxs = []
            for c in cc:
                cidx = self.corner_coord_to_idx.get(c)
                if cidx is not None:
                    corner_idxs.append(cidx)

            if len(corner_idxs) >= 1:
                self.edge_to_corners[eidx] = corner_idxs

            for cidx in corner_idxs:
                self.corner_to_edges.setdefault(cidx, [])
                if eidx not in self.corner_to_edges[cidx]:
                    self.corner_to_edges[cidx].append(eidx)

    def _parse_ports(self, map_state: dict):
        from data.enums import PortType, PORT_TRADE_RATIOS

        for idx_str, port_data in map_state.get('portEdgeStates', {}).items():
            port_coord = (port_data['x'], port_data['y'], port_data['z'])
            port_type = port_data['type']

            if port_type not in [p.value for p in PortType]:
                continue

            pt = PortType(port_type)
            resource, ratio = PORT_TRADE_RATIOS[pt]

            eidx = self.edge_coord_to_idx.get(port_coord)
            if eidx and eidx in self.edge_to_corners:
                for cidx in self.edge_to_corners[eidx]:
                    self.corner_ports[cidx] = (port_type, ratio, resource)

    def get_adjacent_corners(self, corner_idx: int) -> list[int]:
        adjacent = []
        for eidx in self.corner_to_edges.get(corner_idx, []):
            for cidx in self.edge_to_corners.get(eidx, []):
                if cidx != corner_idx and cidx not in adjacent:
                    adjacent.append(cidx)
        return adjacent

    def get_corners_for_dice(self, dice_total: int) -> list[tuple[int, int, Optional[int]]]:
        """Given a dice roll total, return (corner_idx, hex_idx, resource)
        for each corner that should receive resources.
        """
        results = []
        for hex_idx in self.dice_to_hexes.get(dice_total, []):
            resource = self.hex_resources.get(hex_idx)
            if resource is None:
                continue
            for cidx in self.hex_to_corners.get(hex_idx, []):
                results.append((cidx, hex_idx, resource))
        return results
