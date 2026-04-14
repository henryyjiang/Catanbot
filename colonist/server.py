"""
Catanbot FastAPI server — the Python brain behind the Chrome extension.

Endpoints:
    GET  /health          — liveness check (extension polls this on load)
    POST /game/start      — initialize session from Colonist initial-state packet
    POST /game/event      — process one WS event; returns action if it's our turn
    POST /game/end        — tear down session

Run:
    python -m colonist.server                         # default: localhost:8765
    uvicorn colonist.server:app --port 8765 --reload  # dev mode

The bot_color is configured via the BOT_COLOR environment variable (default 2).
The checkpoint is configured via CHECKPOINT_PATH (default checkpoints/best.pt).
MCTS_ITERATIONS controls search depth (default 400).
TRADE_MODEL_DIR, if set, enables player trade proposals.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from colonist.session import LiveGameSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

BOT_COLOR       = int(os.getenv("BOT_COLOR", "2"))
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best.pt")
MCTS_ITERATIONS = int(os.getenv("MCTS_ITERATIONS", "400"))
TRADE_MODEL_DIR = os.getenv("TRADE_MODEL_DIR", None)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Catanbot", version="1.0")

# Allow the Chrome extension's content script to fetch from localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # extension origin varies; restrict in production if desired
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Global session (one game at a time)
_session: Optional[LiveGameSession] = None


# ─── Request / response models ────────────────────────────────────────────────

class GameStartRequest(BaseModel):
    gameData: dict[str, Any]
    botColor: Optional[int] = None  # override BOT_COLOR env var if provided

class GameEventRequest(BaseModel):
    event: dict[str, Any]

class ActionResponse(BaseModel):
    ourTurn: bool
    action: Optional[dict[str, Any]] = None
    phase: Optional[str] = None
    plan: Optional[list[str]] = None   # human-readable action sequence for advisory mode
    summary: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

def _print_plan(plan_strs: list[str], summary: str) -> None:
    sep = "─" * 52
    print(f"\n┌{sep}┐")
    print(f"│{'  YOUR TURN — RECOMMENDED PLAN':^52}│")
    print(f"├{sep}┤")
    for i, step in enumerate(plan_strs, 1):
        print(f"│  {i}. {step:<49}│")
    print(f"├{sep}┤")
    for line in summary.splitlines():
        print(f"│  {line:<50}│")
    print(f"└{sep}┘\n")


def _print_action(action_payload: dict) -> None:
    """Print a single non-main-phase action recommendation."""
    atype = action_payload.get("type", "?")
    detail = {k: v for k, v in action_payload.items() if k != "type"}
    detail_str = ", ".join(f"{k}={v}" for k, v in detail.items())
    print(f"\n  ▶  {atype}  {detail_str}\n")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "session_active": _session is not None and _session.state is not None,
        "bot_color": BOT_COLOR,
    }


@app.post("/game/start")
def game_start(req: GameStartRequest):
    """
    Initialize a new game session.

    The Chrome extension calls this when it detects the first WS packet that
    contains the initial game state (the packet with 'initialState' or
    'eventHistory' in it).
    """
    global _session

    # Prefer auto-detected color from type=4 payload.playerColor over env var.
    bot_color = req.botColor if req.botColor is not None else BOT_COLOR

    _session = LiveGameSession(
        bot_color=bot_color,
        checkpoint_path=CHECKPOINT_PATH,
        mcts_iterations=MCTS_ITERATIONS,
        trade_model_dir=TRADE_MODEL_DIR,
    )

    try:
        _session.initialize(req.gameData)
    except Exception as exc:
        logger.exception("Failed to initialize session")
        _session = None
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info("Game session started: bot_color=%d", bot_color)
    return {"status": "ready", "botColor": bot_color}


@app.post("/game/event", response_model=ActionResponse)
def game_event(req: GameEventRequest):
    """
    Process one incoming Colonist.io WebSocket event.

    Returns:
        ourTurn=False  when the bot does not need to act.
        ourTurn=True + action=<payload>  when the bot should send an action.
    """
    if _session is None:
        # Session not started yet — ignore (happens during pre-game lobby events).
        return ActionResponse(ourTurn=False)

    try:
        action_payload = _session.apply_event(req.event)
    except Exception as exc:
        logger.exception("apply_event failed")
        return ActionResponse(ourTurn=False)

    if action_payload is None:
        return ActionResponse(ourTurn=False)

    phase = _session.state.action_state if _session.state else None
    plan_strs: list[str] = []

    # For main-phase turns, compute and print the full recommended sequence.
    from data.enums import ActionState
    if _session.state and _session.state.action_state == ActionState.MAIN_PHASE:
        try:
            plan = _session.plan_full_turn()
            plan_strs = [str(a) for a in plan]
            _print_plan(plan_strs, _session.summary())
        except Exception as exc:
            logger.warning("plan_full_turn failed: %s", exc)
            _print_action(action_payload)
    else:
        _print_action(action_payload)

    return ActionResponse(
        ourTurn=True,
        action=action_payload,
        plan=plan_strs or None,
        summary=_session.summary(),
    )


class PrivateEventRequest(BaseModel):
    privateEvent: dict[str, Any]

class TradeOfferRequest(BaseModel):
    tradeOffer: dict[str, Any]

@app.post("/game/trade_offer")
def game_trade_offer(req: TradeOfferRequest):
    """
    Process an incoming trade offer from another player.

    Colonist sends this when a human proposes a trade; the bot must respond
    with accept or reject.  Returns { action: <ws-payload> }.

    The trade offer payload structure from Colonist has not been fully
    captured yet — verify the field names from DevTools → Network → WS.
    Expected shape: { offeringPlayer: color, offeredCards: [...], wantedCards: [...] }
    """
    if _session is None:
        return {"action": None}
    try:
        action_payload = _session.handle_trade_offer(req.tradeOffer)
    except Exception as exc:
        logger.warning("handle_trade_offer failed: %s", exc)
        action_payload = None
    return {"action": action_payload}


@app.post("/game/private_event")
def game_private_event(req: PrivateEventRequest):
    """
    Process a type=43 private card event — updates the bot's known hand.
    { givingCards: [...], givingPlayer: N, receivingCards: [...], receivingPlayer: N }
    """
    if _session is None:
        return {"status": "ignored"}
    try:
        _session.apply_private_event(req.privateEvent)
    except Exception as exc:
        logger.warning("apply_private_event failed: %s", exc)
    return {"status": "ok"}


@app.post("/game/end")
def game_end():
    """Tear down the session after a game ends."""
    global _session
    if _session:
        logger.info("Game ended. Final state:\n%s", _session.summary())
    _session = None
    return {"status": "ended"}


@app.get("/game/state")
def game_state():
    """Diagnostic: return a human-readable snapshot of the current game state."""
    if _session is None:
        return {"error": "No active session"}
    return {"summary": _session.summary()}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("colonist.server:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
