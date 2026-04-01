from enum import IntEnum


class Resource(IntEnum):
    BRICK = 1
    WOOL = 2
    GRAIN = 3
    ORE = 4
    LUMBER = 5

RESOURCE_NAMES = {r.value: r.name.lower() for r in Resource}
NUM_RESOURCE_TYPES = 5

class TileType(IntEnum):
    DESERT = 0
    GRAIN = 1
    ORE = 2
    LUMBER = 3
    BRICK = 4
    WOOL = 5
    GOLD = 6
    OCEAN = 7
    FOG = 8

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
    MONOPOLY = 12
    VICTORY_POINT = 13
    ROAD_BUILDING = 14
    YEAR_OF_PLENTY = 15

DEV_CARD_NAMES = {d.value: d.name for d in DevCard}
NUM_DEV_CARD_TYPES = 5

class PortType(IntEnum):
    GENERIC_3_1 = 1
    LUMBER_2_1 = 2
    BRICK_2_1 = 3
    WOOL_2_1 = 4
    GRAIN_2_1 = 5
    ORE_2_1 = 6

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
    # Verified against actual game JSON files
    PLAYER_JOINED = 0
    TURN_START = 1
    BUILT_PIECE = 4           # Building placed
    BOUGHT_OR_BUILT = 5
    DICE_ROLL = 10            # Dice rolled (firstDice, secondDice, playerColor)
    RESOURCE_RECEIVED = 11    # Per-player: received resource from tile (resourceType in tileInfo, playerColor)
    ROBBER_STEAL_PRIVATE = 14 # Private to thief: cardEnums = stolen card, playerColor = thief
    ROBBER_LOSE_PRIVATE = 15  # Private to victim: cardEnums = lost card, playerColor = victim
    ROBBER_STEAL_PUBLIC = 16  # Public: playerColorThief, playerColorVictim (card hidden)
    DEV_CARD_PLAYED = 20      # cardEnum, playerColor
    YEAR_OF_PLENTY = 21
    DEV_CARD_BOUGHT = 22
    PLAYER_DISCONNECTED = 24
    TURN_END = 44             # End of turn
    GAME_WINNER = 45
    RESOURCE_DISTRIBUTED = 47 # Broadcast: cardsToBroadcast list, playerColor, distributionType
    ROBBER_TILE_INFO = 49     # tileInfo shown when robber moves
    DISCARD = 55              # Discarded on 7: cardEnums, playerColor
    VP_CARD_REVEALED = 66
    TRADE_OFFER_CLOSED = 68
    LARGEST_ARMY = 113
    ROAD_BUILDING_USED = 114
    LONGEST_ROAD = 115
    BANK_TRADE = 116          # givenCardEnums, receivedCardEnums, playerColor
    TRADE_COMPLETED = 117     # playerColorCreator, playerColorOffered, offeredCardEnums, wantedCardEnums
    TRADE_OFFER = 118         # playerColor, offeredCardEnums, wantedCardEnums
    MONOPOLY_PLAYED = 86      # Monopoly result

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