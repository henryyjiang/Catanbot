"""
Translate bot Action dataclasses → Colonist.io outgoing WebSocket JSON payloads.

STATUS: Phase 4 placeholder — field names are educated guesses based on the
inbound stateChange format. They MUST be validated against real captured outgoing
WS frames (browser devtools → Network → WS tab) before Phase 5 goes live.

For each action type, play a manual game on Colonist.io, perform that action,
and check the exact JSON sent in the outgoing WS frame. Update the dict below.

Verified fields will be marked with # ✓ verified.
"""

from __future__ import annotations

from typing import Any

from data.enums import Resource
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
)


# ─── Translation ──────────────────────────────────────────────────────────────

def translate_action(action: Action) -> dict[str, Any]:
    """
    Convert a bot Action into the JSON payload to send over the Colonist.io
    WebSocket.

    For PlayRoadBuilding, returns a special dict with type='roadBuilding' and
    both edge indices; the content script handles splitting this into two
    sequential WS sends with an intermediate ack.

    Raises ValueError for unrecognised action types.
    """
    match action:

        case PassTurn():
            return {"type": "endTurn"}                    # TODO: verify field name

        case BuildSettlement(corner_idx=c):
            return {"type": "buildSettlement", "cornerIndex": c}  # TODO: verify

        case BuildCity(corner_idx=c):
            return {"type": "buildCity", "cornerIndex": c}        # TODO: verify

        case BuildRoad(edge_idx=e):
            return {"type": "buildRoad", "edgeIndex": e}          # TODO: verify

        case BuyDevCard():
            return {"type": "buyDevCard"}                          # TODO: verify

        case PlayKnight(target_hex=h, steal_from=victim):
            payload: dict[str, Any] = {
                "type": "playKnight",                              # TODO: verify
                "tileIndex": h,                                    # TODO: verify
            }
            if victim is not None:
                payload["stealFromColor"] = victim                 # TODO: verify
            return payload

        case PlayMonopoly(resource=r):
            return {"type": "playMonopoly", "resourceType": r}    # TODO: verify

        case PlayYearOfPlenty(resource1=r1, resource2=r2):
            return {
                "type": "playYearOfPlenty",                        # TODO: verify
                "resources": [r1, r2],                             # TODO: verify
            }

        case PlayRoadBuilding(edge1=e1, edge2=e2):
            # Handled specially by content.js: sends two buildRoad messages
            # with an intermediate stateChange ack.
            return {"type": "roadBuilding", "edge1": e1, "edge2": e2}

        case BankTrade(give_resource=give, give_count=n, receive_resource=recv):
            return {
                "type": "bankTrade",                               # TODO: verify
                "give": [give] * n,                                # TODO: verify
                "receive": [recv],                                 # TODO: verify
            }

        case _:
            raise ValueError(f"Unknown action type: {type(action).__name__}")


# ─── Trade offer translation ───────────────────────────────────────────────────

def translate_trade_offer(offering: dict[int, int], requesting: dict[int, int]) -> dict[str, Any]:
    """
    Build a trade offer payload.

    offering / requesting: {resource_int: count}
    Returns the WS JSON to propose the trade to other players.

    TODO: verify Colonist field names from devtools.
    """
    def expand(resource_dict: dict[int, int]) -> list[int]:
        cards = []
        for resource, count in resource_dict.items():
            cards.extend([resource] * count)
        return cards

    return {
        "type": "tradeOffer",                                      # TODO: verify
        "offeredCards": expand(offering),                          # TODO: verify
        "wantedCards": expand(requesting),                         # TODO: verify
    }


def translate_trade_response(accepted: bool) -> dict[str, Any]:
    """
    Accept or reject an incoming trade offer.
    TODO: Colonist may use a different mechanism (e.g., separate endpoint or
    a message with the offer ID). Verify from devtools.
    """
    return {"type": "tradeResponse", "accepted": accepted}         # TODO: verify


# ─── Setup action translation ──────────────────────────────────────────────────

def translate_setup_settlement(corner_idx: int) -> dict[str, Any]:
    return {"type": "buildSettlement", "cornerIndex": corner_idx}  # TODO: verify


def translate_setup_road(edge_idx: int) -> dict[str, Any]:
    return {"type": "buildRoad", "edgeIndex": edge_idx}            # TODO: verify


def translate_roll_dice() -> dict[str, Any]:
    return {"type": "rollDice"}                                    # TODO: verify


def translate_move_robber(hex_idx: int) -> dict[str, Any]:
    return {"type": "moveRobber", "tileIndex": hex_idx}            # TODO: verify


def translate_steal(victim_color: int) -> dict[str, Any]:
    return {"type": "stealCard", "stealFromColor": victim_color}   # TODO: verify


def translate_discard(cards: list[int]) -> dict[str, Any]:
    """cards: list of resource int values to discard."""
    return {"type": "discardCards", "cards": cards}                # TODO: verify
