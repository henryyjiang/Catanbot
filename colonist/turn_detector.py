"""
Turn detection: given the current CatanState and the bot's color, determine
whether it is the bot's turn and which phase it must act in.
"""

from __future__ import annotations

from enum import Enum

from data.enums import ActionState


class TurnPhase(str, Enum):
    NONE          = "none"
    ROLL_DICE     = "roll_dice"
    MAIN_PHASE    = "main_phase"
    MOVE_ROBBER   = "move_robber"
    STEAL_CARD    = "steal_card"
    DISCARD_CARDS = "discard_cards"
    ROAD_BUILDING = "road_building"
    SETUP_SETTLEMENT = "setup_settlement"
    SETUP_ROAD    = "setup_road"


# ActionState values that require the bot to act (active player)
_ACTIVE_PLAYER_STATES: dict[int, TurnPhase] = {
    ActionState.ROLL_DICE:     TurnPhase.ROLL_DICE,
    ActionState.MAIN_PHASE:    TurnPhase.MAIN_PHASE,
    ActionState.MOVE_ROBBER:   TurnPhase.MOVE_ROBBER,
    ActionState.STEAL_CARD:    TurnPhase.STEAL_CARD,
    ActionState.ROAD_BUILDING: TurnPhase.ROAD_BUILDING,
    ActionState.SETUP_SETTLEMENT: TurnPhase.SETUP_SETTLEMENT,
    ActionState.SETUP_ROAD:    TurnPhase.SETUP_ROAD,
}

# DISCARD is addressed to any player with 8+ cards, not just active player.
_DISCARD_STATE = ActionState.DISCARD_CARDS


def get_turn_phase(state, bot_color: int) -> TurnPhase:
    """
    Return the TurnPhase the bot must act in, or TurnPhase.NONE if it is not
    the bot's turn.

    Parameters
    ----------
    state : CatanState
        Current game state (after all events so far have been applied).
    bot_color : int
        The player color assigned to the bot.
    """
    action_state = state.action_state

    # Discard is special: any player with 8+ cards must discard, not only the
    # active player. Check the bot's hand regardless of whose turn it is.
    if action_state == _DISCARD_STATE:
        player = state.players.get(bot_color)
        if player and player.total_resources >= 8:
            return TurnPhase.DISCARD_CARDS
        return TurnPhase.NONE

    # All other phases require the bot to be the active player.
    if state.current_player_color != bot_color:
        return TurnPhase.NONE

    return _ACTIVE_PLAYER_STATES.get(action_state, TurnPhase.NONE)
