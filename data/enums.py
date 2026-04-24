from enum import IntEnum


class Resource(IntEnum):
    LUMBER = 1
    BRICK = 2
    WOOL = 3
    GRAIN = 4
    ORE = 5

RESOURCE_NAMES = {r.value: r.name.lower() for r in Resource}
NUM_RESOURCE_TYPES = 5

class TileType(IntEnum):
    DESERT = 0
    GRAIN = 1
    ORE = 2
    LUMBER = 3 
    BRICK = 4 
    WOOL = 5    
    GOLD = 6        # gold hex - special maps
    OCEAN = 7       # ocean/water
    FOG = 8         # fog of war — special maps

TILE_RESOURCE = {
    TileType.GRAIN: Resource.GRAIN,
    TileType.ORE: Resource.ORE,
    TileType.LUMBER: Resource.LUMBER,
    TileType.BRICK: Resource.BRICK,
    TileType.WOOL: Resource.WOOL,
}

STANDARD_HEX_COUNT = 19
STANDARD_CORNER_COUNT = 54
STANDARD_EDGE_COUNT = 72


class BuildingType(IntEnum):
    SETTLEMENT = 1
    CITY = 2

class EdgeType(IntEnum):
    ROAD = 1


class DevCard(IntEnum):
    HIDDEN = 10        
    KNIGHT = 11
    ROAD_BUILDING = 12
    YEAR_OF_PLENTY = 13
    MONOPOLY = 14
    VICTORY_POINT = 15

DEV_CARD_NAMES = {d.value: d.name for d in DevCard}
NUM_DEV_CARD_TYPES = 5  # knight, road_building, year_of_plenty, monopoly, vp

class PortType(IntEnum):
    GENERIC_3_1 = 1     # 3:1 any resource
    LUMBER_2_1 = 2
    BRICK_2_1 = 3
    WOOL_2_1 = 4
    GRAIN_2_1 = 5
    ORE_2_1 = 6

# Port type
PORT_TRADE_RATIOS = {
    PortType.GENERIC_3_1: (None, 3),
    PortType.LUMBER_2_1: (Resource.LUMBER, 2),
    PortType.BRICK_2_1: (Resource.BRICK, 2),
    PortType.WOOL_2_1: (Resource.WOOL, 2),
    PortType.GRAIN_2_1: (Resource.GRAIN, 2),
    PortType.ORE_2_1: (Resource.ORE, 2),
}

class VPCategory(IntEnum):
    SETTLEMENTS = 0
    CITIES = 1
    DEV_CARD_VP = 2
    LARGEST_ARMY = 3
    LONGEST_ROAD = 4

class LogType(IntEnum):
    PLAYER_JOINED = 0
    TURN_START = 1
    BUILT_PIECE = 4
    BOUGHT_OR_BUILT = 5
    DICE_ROLL = 10
    ROBBER_MOVE = 11
    ROBBER_STEAL = 14
    DISCARD = 15
    KNIGHT_PLAYED = 16
    MONOPOLY_PLAYED = 20
    YEAR_OF_PLENTY = 21
    ROAD_BUILDING = 55
    TURN_END = 44
    RESOURCE_DISTRIBUTED = 47
    BANK_TRADE = 49
    VP_CARD_REVEALED = 68
    DEV_CARD_BOUGHT = 86
    LARGEST_ARMY = 113
    LONGEST_ROAD = 115
    TRADE_OFFER = 118
    TRADE_ACCEPTED = 116
    TRADE_COMPLETED = 117
    PLAYER_DISCONNECTED = 24
    NO_RESOURCES = 45

class ActionState(IntEnum):
    SETUP_SETTLEMENT = 1
    SETUP_ROAD = 3
    ROLL_DICE = 0
    MAIN_PHASE = 24
    MOVE_ROBBER = 27
    STEAL_CARD = 28
    DISCARD_CARDS = 30
    ROAD_BUILDING = 31


class PieceEnum(IntEnum):
    ROAD = 0
    SHIP = 1
    SETTLEMENT = 2
    CITY = 3
