"""
Action dataclasses for Catan main-phase decisions.

Each action is frozen (hashable) so it can be used as a dict key.
The union type `Action` covers every move the bot can make.

Resource costs use Resource enum values as keys so that:
    player.resource_counts[Resource.BRICK]  (IntEnum, equals int 1)
works transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from data.enums import Resource


# ─── Resource costs ────────────────────────────────────────────────────────

SETTLEMENT_COST: dict[Resource, int] = {
    Resource.BRICK: 1,
    Resource.LUMBER: 1,
    Resource.WOOL: 1,
    Resource.GRAIN: 1,
}

CITY_COST: dict[Resource, int] = {
    Resource.GRAIN: 2,
    Resource.ORE: 3,
}

ROAD_COST: dict[Resource, int] = {
    Resource.BRICK: 1,
    Resource.LUMBER: 1,
}

DEV_CARD_COST: dict[Resource, int] = {
    Resource.WOOL: 1,
    Resource.GRAIN: 1,
    Resource.ORE: 1,
}


# ─── Action dataclasses ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuildSettlement:
    corner_idx: int

    def __str__(self) -> str:
        return f"Build Settlement @ corner {self.corner_idx}"


@dataclass(frozen=True)
class BuildCity:
    corner_idx: int

    def __str__(self) -> str:
        return f"Build City @ corner {self.corner_idx}"


@dataclass(frozen=True)
class BuildRoad:
    edge_idx: int

    def __str__(self) -> str:
        return f"Build Road @ edge {self.edge_idx}"


@dataclass(frozen=True)
class BuyDevCard:
    def __str__(self) -> str:
        return "Buy Dev Card"


@dataclass(frozen=True)
class PlayKnight:
    target_hex: int
    steal_from: Optional[int] = None  # player color to steal from; None = no steal

    def __str__(self) -> str:
        steal = f", steal from player {self.steal_from}" if self.steal_from is not None else ""
        return f"Play Knight → hex {self.target_hex}{steal}"


@dataclass(frozen=True)
class PlayMonopoly:
    resource: int  # Resource enum int value

    def __str__(self) -> str:
        return f"Play Monopoly ({Resource(self.resource).name})"


@dataclass(frozen=True)
class PlayRoadBuilding:
    edge1: int
    edge2: int  # -1 if only one road can be placed (roads_remaining == 1)

    def __str__(self) -> str:
        edges = str(self.edge1) if self.edge2 < 0 else f"{self.edge1} + {self.edge2}"
        return f"Road Building: edges {edges}"


@dataclass(frozen=True)
class PlayYearOfPlenty:
    resource1: int  # Resource enum int value
    resource2: int  # Resource enum int value (can equal resource1)

    def __str__(self) -> str:
        r1 = Resource(self.resource1).name
        r2 = Resource(self.resource2).name
        return f"Year of Plenty: {r1} + {r2}"


@dataclass(frozen=True)
class BankTrade:
    give_resource: int   # Resource enum int value
    give_count: int      # 2, 3, or 4 depending on port access
    receive_resource: int  # Resource enum int value

    def __str__(self) -> str:
        give = Resource(self.give_resource).name
        recv = Resource(self.receive_resource).name
        return f"Bank Trade: {self.give_count}× {give} → {recv}"


@dataclass(frozen=True)
class PassTurn:
    def __str__(self) -> str:
        return "Pass Turn"


# ─── Union type ─────────────────────────────────────────────────────────────

Action = Union[
    BuildSettlement,
    BuildCity,
    BuildRoad,
    BuyDevCard,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    BankTrade,
    PassTurn,
]


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Action types and string representations:")
    samples = [
        BuildSettlement(corner_idx=7),
        BuildCity(corner_idx=12),
        BuildRoad(edge_idx=3),
        BuyDevCard(),
        PlayKnight(target_hex=5, steal_from=2),
        PlayKnight(target_hex=9, steal_from=None),
        PlayMonopoly(resource=Resource.GRAIN),
        PlayRoadBuilding(edge1=4, edge2=8),
        PlayRoadBuilding(edge1=4, edge2=-1),
        PlayYearOfPlenty(resource1=Resource.ORE, resource2=Resource.GRAIN),
        BankTrade(give_resource=Resource.BRICK, give_count=4, receive_resource=Resource.ORE),
        BankTrade(give_resource=Resource.LUMBER, give_count=2, receive_resource=Resource.WOOL),
        PassTurn(),
    ]
    for a in samples:
        print(f"  {type(a).__name__:20s} → {a}")

    print("\nResource costs:")
    for name, cost in [
        ("Settlement", SETTLEMENT_COST),
        ("City", CITY_COST),
        ("Road", ROAD_COST),
        ("Dev Card", DEV_CARD_COST),
    ]:
        items = ", ".join(f"{r.name}×{n}" for r, n in cost.items())
        print(f"  {name}: {items}")

    print("\nAll action types are hashable (can be used as dict keys):")
    d = {a: i for i, a in enumerate(samples)}
    print(f"  Created dict with {len(d)} entries — OK")
