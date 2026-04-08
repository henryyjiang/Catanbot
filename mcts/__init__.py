"""
mcts/ — General move generation and MCTS search for Catan main-phase decisions.

The model backend (CatanNet) is kept completely separate: this package only
imports it through mcts.evaluator.StateEvaluator.  Feed in a CatanState
(parsed from a Colonist.io JSON) and get back the best Action.

Public API
----------
    from mcts.actions import Action, BuildSettlement, BuildRoad, ...
    from mcts.move_generator import get_legal_actions
    from mcts.state_transition import apply_action
    from mcts.evaluator import StateEvaluator
    from mcts.search import CatanMCTS, find_best_action
"""

from mcts.actions import (
    Action,
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
    SETTLEMENT_COST,
    CITY_COST,
    ROAD_COST,
    DEV_CARD_COST,
)
from mcts.move_generator import get_legal_actions
from mcts.state_transition import apply_action
from mcts.evaluator import StateEvaluator
from mcts.search import CatanMCTS, find_best_action

__all__ = [
    "Action",
    "BuildSettlement",
    "BuildCity",
    "BuildRoad",
    "BuyDevCard",
    "PlayKnight",
    "PlayMonopoly",
    "PlayRoadBuilding",
    "PlayYearOfPlenty",
    "BankTrade",
    "PassTurn",
    "SETTLEMENT_COST",
    "CITY_COST",
    "ROAD_COST",
    "DEV_CARD_COST",
    "get_legal_actions",
    "apply_action",
    "StateEvaluator",
    "CatanMCTS",
    "find_best_action",
]
