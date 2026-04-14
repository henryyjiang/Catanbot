"""
LiveGameSession — stateful manager that bridges the Colonist.io WebSocket
stream to the bot's MCTS decision engine.

Lifecycle:
    session = LiveGameSession(bot_color=2, checkpoint_path="checkpoints/best.pt")
    session.initialize(game_data)           # called once when game starts
    action = session.apply_event(event)     # called for every incoming WS event
    # action is None when it's not the bot's turn, or a WS-ready dict otherwise

All existing bot code (CatanState, find_best_action, etc.) is used unmodified.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from data.enums import ActionState, Resource
from data.state import CatanState
from mcts.evaluator import StateEvaluator
from mcts.move_generator import get_legal_actions
from mcts.search import find_best_action
from mcts.actions import (
    PassTurn, PlayKnight, PlayMonopoly, PlayRoadBuilding, PlayYearOfPlenty,
    BankTrade, BuildSettlement, BuildCity, BuildRoad, BuyDevCard,
)
from simulation.agents import MCTSAgent
from colonist.turn_detector import TurnPhase, get_turn_phase
from colonist.action_translator import (
    translate_action,
    translate_setup_settlement,
    translate_setup_road,
    translate_roll_dice,
    translate_move_robber,
    translate_steal,
    translate_discard,
    translate_trade_response,
    translate_trade_offer,
)

try:
    from trade_mcts.search import find_best_trade
    from trade_mcts.hand_tracker import HandTracker
    from trade_mcts.trade_models import TradeAcceptanceModel, TradeProposalPolicy
    _TRADE_AVAILABLE = True
except ImportError:
    _TRADE_AVAILABLE = False

logger = logging.getLogger(__name__)


class LiveGameSession:
    """
    Maintains the live CatanState for a single game and decides the bot's
    actions for every turn phase.

    Parameters
    ----------
    bot_color : int
        Colonist.io color code assigned to the bot's account in this game.
    checkpoint_path : str or None
        Path to a CatanNet .pt checkpoint.  None → heuristic evaluator.
    mcts_iterations : int
        MCTS iterations per main-phase action.  400 balances speed and quality.
    trade_model_dir : str or None
        Directory containing acceptance_model.pt and proposal_policy.pt.
        None → no player trade proposals (bank trades still work).
    """

    def __init__(
        self,
        bot_color: int,
        checkpoint_path: Optional[str] = "checkpoints/best.pt",
        mcts_iterations: int = 400,
        trade_model_dir: Optional[str] = None,
    ):
        self.bot_color = bot_color
        self.mcts_iterations = mcts_iterations

        self.state: Optional[CatanState] = None
        self._agent = MCTSAgent(
            checkpoint_path=checkpoint_path,
            iterations=mcts_iterations,
            opponent_rounds=1,
        )
        self.evaluator: StateEvaluator = self._agent.evaluator

        # Trade models (optional)
        self._acceptance_model = None
        self._proposal_policy = None
        self._hand_tracker: Optional[Any] = None
        if _TRADE_AVAILABLE and trade_model_dir:
            self._load_trade_models(trade_model_dir)

        # Per-turn tracking
        self._dev_played_this_turn = False
        self._last_setup_corner: Optional[int] = None  # corner placed in SETUP_SETTLEMENT
        self._active_trade_offer: Optional[dict] = None  # pending outgoing trade

        # Phase 6: track that we've already proposed a trade this turn
        self._trade_proposed_this_turn = False

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, game_data: dict) -> None:
        """
        Build the initial CatanState from the Colonist.io game-start payload.

        game_data is whatever the extension POSTs to /game/start.  It may be:
          (a) The raw Colonist initial-state packet (from the first WS message),
              OR
          (b) Already shaped like a replay file ({ data: { eventHistory: ... }}).

        A normalization step handles both shapes.
        """
        normalized = self._normalize_game_data(game_data)
        self.state = CatanState.from_initial_state(normalized)

        other_colors = [c for c in self.state.player_colors if c != self.bot_color]
        if _TRADE_AVAILABLE:
            self._hand_tracker = HandTracker(self.bot_color, other_colors)

        logger.info(
            "Session initialized: bot_color=%d, players=%s",
            self.bot_color, self.state.player_colors,
        )

    def _normalize_game_data(self, raw: dict) -> dict:
        """
        Normalize the game data dict to the shape CatanState.from_initial_state
        expects:
            { data: { playOrder: [...], eventHistory: { initialState: {...} } } }

        Live Colonist.io type=4 message structure (confirmed Phase 1):
            {
              id: '130',
              data: {
                type: 4,
                payload: {
                  playerColor: <bot_color>,
                  playOrder: [...],
                  gameState: { mapState: {...}, playerStates: {...}, ... },
                  ...
                }
              }
            }
        """
        # Shape A: already in replay format — pass through.
        if "data" in raw and "eventHistory" in raw.get("data", {}):
            return raw

        # Shape B: live type=4 initial-state message (confirmed from Phase 1).
        payload = raw.get("data", {}).get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected payload to be a dict, got {type(payload).__name__}. "
                "This message is not a valid initial-state packet."
            )
        game_state = payload.get("gameState")
        play_order = payload.get("playOrder")

        if game_state and play_order is not None:
            return {
                "data": {
                    "playOrder": play_order,
                    "eventHistory": {
                        "initialState": game_state,
                    },
                }
            }

        # Shape C: the whole thing IS the initialState block already.
        if "mapState" in raw:
            colors = [int(k) for k in raw.get("playerStates", {}).keys()]
            return {
                "data": {
                    "playOrder": colors,
                    "eventHistory": {"initialState": raw},
                }
            }

        raise ValueError(
            f"Cannot normalize game data — unrecognised structure. "
            f"Top-level keys: {list(raw.keys())}. "
            f"data keys: {list(raw.get('data', {}).keys())}. "
            f"payload keys: {list(payload.keys()) if payload else '(none)'}."
        )

    # ── Event processing ─────────────────────────────────────────────────────

    def apply_event(self, event: dict) -> Optional[dict]:
        """
        Apply one Colonist.io WebSocket event to the game state, update the
        hand tracker, and decide whether the bot must act.

        Returns a WS-ready action dict if it is the bot's turn, else None.
        For PlayRoadBuilding the caller (content.js) handles the two-send split.
        """
        if self.state is None:
            logger.warning("apply_event called before initialize()")
            return None

        prev_turn_player = self.state.current_player_color
        prev_action_state = self.state.action_state

        # Apply to game state
        self.state.apply_event(event)

        # Update hand tracker on every event
        if self._hand_tracker is not None:
            try:
                self._hand_tracker.observe_event(event)
            except Exception as exc:
                logger.debug("HandTracker update failed: %s", exc)

        # Reset per-turn flags when a new turn for the bot begins
        new_turn = (
            self.state.current_player_color == self.bot_color
            and (
                prev_turn_player != self.bot_color
                or (
                    prev_action_state == ActionState.ROLL_DICE
                    and self.state.action_state != ActionState.ROLL_DICE
                )
            )
        )
        if new_turn:
            self._dev_played_this_turn = False
            self._trade_proposed_this_turn = False
            self._active_trade_offer = None

        # Determine what phase (if any) the bot must act in
        phase = get_turn_phase(self.state, self.bot_color)
        if phase == TurnPhase.NONE:
            return None

        return self._decide(phase, event)

    # ── Decision dispatch ─────────────────────────────────────────────────────

    def _decide(self, phase: TurnPhase, event: dict) -> Optional[dict]:
        """Route the decision to the appropriate handler for the current phase."""
        logger.info("Bot acting: phase=%s, turn=%d", phase, self.state.current_turn)

        try:
            if phase == TurnPhase.ROLL_DICE:
                return translate_roll_dice()

            if phase == TurnPhase.SETUP_SETTLEMENT:
                return self._decide_setup_settlement()

            if phase == TurnPhase.SETUP_ROAD:
                return self._decide_setup_road()

            if phase == TurnPhase.MOVE_ROBBER:
                return self._decide_robber()

            if phase == TurnPhase.STEAL_CARD:
                return self._decide_steal()

            if phase == TurnPhase.DISCARD_CARDS:
                return self._decide_discard()

            if phase == TurnPhase.MAIN_PHASE:
                return self._decide_main_phase()

        except Exception as exc:
            logger.exception("Decision failed (phase=%s): %s", phase, exc)

        return None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _valid_setup_corners(self) -> list[int]:
        topo = self.state.topology
        occupied = set(self.state.corner_buildings.keys())
        return [
            c for c in topo.corner_positions
            if c not in occupied
            and not any(adj in occupied for adj in topo.get_adjacent_corners(c))
        ]

    def _decide_setup_settlement(self) -> dict:
        valid = self._valid_setup_corners()
        corner = self._agent.choose_setup_settlement(self.state, self.bot_color, valid)
        self._last_setup_corner = corner
        logger.info("Setup: place settlement at corner %d", corner)
        return translate_setup_settlement(corner)

    def _decide_setup_road(self) -> dict:
        topo = self.state.topology
        if self._last_setup_corner is None:
            # Fallback: find our most recent settlement
            my_corners = [
                c for c, (owner, _) in self.state.corner_buildings.items()
                if owner == self.bot_color
            ]
            self._last_setup_corner = my_corners[-1] if my_corners else 0

        valid_edges = [
            e for e in topo.corner_to_edges.get(self._last_setup_corner, [])
            if e not in self.state.edge_roads
        ]
        edge = self._agent.choose_setup_road(
            self.state, self.bot_color, self._last_setup_corner, valid_edges
        )
        logger.info("Setup: place road at edge %d", edge)
        return translate_setup_road(edge)

    # ── Robber & steal ────────────────────────────────────────────────────────

    def _decide_robber(self) -> dict:
        topo = self.state.topology
        # Any hex except the current robber position
        valid_hexes = [
            h for h in topo.hex_positions if h != self.state.robber_hex
        ]
        hex_idx = self._agent.choose_robber_hex(self.state, self.bot_color, valid_hexes)
        logger.info("Robber → hex %d", hex_idx)
        return translate_move_robber(hex_idx)

    def _decide_steal(self) -> dict:
        """Steal from the opponent on the robber hex with the most resources."""
        topo = self.state.topology
        robber_hex = self.state.robber_hex
        candidates = set()
        for cidx in topo.hex_to_corners.get(robber_hex, []):
            if cidx in self.state.corner_buildings:
                owner, _ = self.state.corner_buildings[cidx]
                if owner != self.bot_color:
                    candidates.add(owner)

        if not candidates:
            # Nobody to steal from; Colonist may not send STEAL_CARD in this case,
            # but guard anyway.
            logger.warning("STEAL_CARD phase but no valid victims")
            return translate_steal(next(iter(self.state.players)))

        victim = max(candidates, key=lambda c: self.state.players[c].total_resources)
        logger.info("Steal from color %d", victim)
        return translate_steal(victim)

    # ── Discard ───────────────────────────────────────────────────────────────

    def _decide_discard(self) -> dict:
        """
        Discard floor(hand_size / 2) cards when a 7 forces discard.
        Strategy: discard the resources we have the most copies of (least scarce).
        """
        player = self.state.players[self.bot_color]
        hand = list(player.resource_cards)
        n_discard = len(hand) // 2

        # Sort by descending frequency (discard surplus first)
        from collections import Counter
        counts = Counter(hand)
        sorted_cards = sorted(hand, key=lambda r: -counts[r])
        to_discard = sorted_cards[:n_discard]
        logger.info("Discard %d cards: %s", n_discard, to_discard)
        return translate_discard(to_discard)

    # ── Full-turn planner ─────────────────────────────────────────────────────

    def plan_full_turn(self) -> list:
        """
        Compute the complete recommended action sequence for the current turn.

        Runs MCTS on a copy of the state, applies each chosen action, and
        continues until PassTurn is recommended.  Returns a list of Action
        objects (including the final PassTurn).

        Does NOT modify self.state.
        """
        import copy
        from mcts.state_transition import apply_action

        sim_state = copy.deepcopy(self.state)
        plan = []
        dev_played = False

        for _ in range(10):  # safety cap — a turn can't have more than ~10 actions
            action = find_best_action(
                sim_state,
                self.bot_color,
                self.evaluator,
                iterations=self.mcts_iterations,
                time_limit=5.0,
                opponent_rounds=1,
                dev_card_played_this_turn=dev_played,
            )
            plan.append(action)
            if isinstance(action, PassTurn):
                break
            if isinstance(action, (PlayKnight, PlayMonopoly, PlayYearOfPlenty)):
                dev_played = True
            try:
                sim_state = apply_action(sim_state, action, self.bot_color)
            except Exception as exc:
                logger.warning("plan_full_turn: apply_action failed: %s", exc)
                break

        return plan

    # ── Main phase ────────────────────────────────────────────────────────────

    def _decide_main_phase(self) -> Optional[dict]:
        """
        Run MCTS to pick the next action.  Loops until PassTurn is chosen,
        returning the first non-Pass action's WS payload.

        (Each call to apply_event() triggers one action; the next event from
        Colonist.io will trigger the next action in the same turn if needed.)
        """
        # Phase 6: optionally propose a player trade first
        if (
            not self._trade_proposed_this_turn
            and _TRADE_AVAILABLE
            and self._hand_tracker is not None
            and self._acceptance_model is not None
        ):
            trade = self._try_trade()
            if trade is not None:
                self._trade_proposed_this_turn = True
                return trade

        action = find_best_action(
            self.state,
            self.bot_color,
            self.evaluator,
            iterations=self.mcts_iterations,
            time_limit=5.0,
            opponent_rounds=1,
            dev_card_played_this_turn=self._dev_played_this_turn,
        )

        if isinstance(action, PassTurn):
            logger.info("MCTS chose PassTurn")
            return translate_action(action)

        if isinstance(action, (PlayKnight, PlayMonopoly, PlayYearOfPlenty)):
            self._dev_played_this_turn = True

        logger.info("MCTS chose: %s", action)
        return translate_action(action)

    # ── Trade (Phase 6) ───────────────────────────────────────────────────────

    def _try_trade(self) -> Optional[dict]:
        """
        Ask TradeMCTS whether a player trade is worth proposing.
        Returns a WS payload for the offer, or None.
        """
        try:
            trade = find_best_trade(
                self.state,
                self.bot_color,
                self._hand_tracker,
                acceptance_model=self._acceptance_model,
                proposal_policy=self._proposal_policy,
                iterations=500,
                time_limit=3.0,
            )
        except Exception as exc:
            logger.warning("TradeMCTS failed: %s", exc)
            return None

        if trade is None:
            return None

        logger.info("Proposing trade: offer=%s want=%s", trade.offering, trade.requesting)
        return translate_trade_offer(trade.offering, trade.requesting)

    def handle_trade_offer(self, event: dict) -> Optional[dict]:
        """
        Called when Colonist sends a TRADE_OFFER event directed at the bot
        (the bot is a potential responder, not the proposer).

        Returns an accept or reject WS payload.
        """
        # Phase 6 TODO: use acceptance model to decide
        # For now: always reject — safe default until trade models are wired up.
        return translate_trade_response(accepted=False)

    def _load_trade_models(self, model_dir: str) -> None:
        import os
        from trade_mcts.trade_models import TradeAcceptanceModel, TradeProposalPolicy
        import torch

        acc_path = os.path.join(model_dir, "acceptance_model.pt")
        prop_path = os.path.join(model_dir, "proposal_policy.pt")

        if os.path.exists(acc_path):
            self._acceptance_model = TradeAcceptanceModel()
            self._acceptance_model.load_state_dict(torch.load(acc_path, map_location="cpu"))
            self._acceptance_model.eval()
            logger.info("Loaded acceptance model from %s", acc_path)

        if os.path.exists(prop_path):
            self._proposal_policy = TradeProposalPolicy()
            self._proposal_policy.load_state_dict(torch.load(prop_path, map_location="cpu"))
            self._proposal_policy.eval()
            logger.info("Loaded proposal policy from %s", prop_path)

    # ── Private card events (type=43) ────────────────────────────────────────

    def apply_private_event(self, private_event: dict) -> None:
        """
        Apply a type=43 private card event to correct the bot's known hand.

        Colonist sends these only to the player who can see the cards — so for
        the bot's player they carry the exact resource card list.  For opponents,
        the cards are hidden (value 0).

        Structure:
          { givingCards: [res, ...], givingPlayer: color,
            receivingCards: [res, ...], receivingPlayer: color }
        """
        if self.state is None:
            return

        giving_player   = private_event.get("givingPlayer")
        receiving_player = private_event.get("receivingPlayer")
        giving_cards    = private_event.get("givingCards", [])
        receiving_cards = private_event.get("receivingCards", [])

        # Only correct cards for the bot's own player (we don't trust hidden-card
        # values for opponents — those are 0/placeholder in the private event).
        for color, cards in [(giving_player, giving_cards), (receiving_player, receiving_cards)]:
            if color != self.bot_color:
                continue
            if color not in self.state.players:
                continue
            # Replace the known hand with what Colonist told us directly.
            # Filter out 0 (hidden placeholder) — should not appear for our own player.
            known = [c for c in cards if c != 0]
            if known or cards == []:  # accept empty hand
                self.state.players[color].resource_cards = list(known)
                logger.debug(
                    "Private event: corrected player %d hand → %s", color, known
                )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a brief human-readable snapshot of the current game state."""
        if self.state is None:
            return "Session not initialized"
        lines = [f"Turn {self.state.current_turn}, ActionState={self.state.action_state}"]
        for color, player in self.state.players.items():
            marker = " ← bot" if color == self.bot_color else ""
            lines.append(
                f"  Player {color}{marker}: {player.total_vp} VP, "
                f"{player.total_resources} cards, "
                f"{player.total_dev_cards} dev cards"
            )
        return "\n".join(lines)
