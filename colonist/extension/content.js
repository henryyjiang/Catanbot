/**
 * content.js — isolated content script world.
 *
 * Responsibility:
 *   1. At document_start, inject inject.js into the PAGE context so it can
 *      patch window.WebSocket before Colonist's bundle runs.
 *   2. Listen for WS messages relayed from inject.js via postMessage.
 *   3. Forward game events to the local Python server.
 *   4. In ADVISORY mode: log MCTS recommendations to the browser console.
 *      In AUTO_PLAY mode: send MCTS-chosen actions back to Colonist.
 *
 * Toggle auto-play from the extension popup (stored in chrome.storage.local
 * under the key "autoPlay"). Defaults to false (advisory mode).
 */

const SERVER_URL = 'http://localhost:8765';

// ── Step 1: inject inject.js into the page context ───────────────────────────

const script = document.createElement('script');
script.src = chrome.runtime.getURL('inject.js');
script.onload = () => script.remove();
(document.head || document.documentElement).appendChild(script);

// ── State ─────────────────────────────────────────────────────────────────────

let gameStarted   = false;
let autoPlay      = true;    // updated from chrome.storage / popup messages

// roadBuilding: after sending edge1 we must send edge2 on the next stateChange.
let _pendingRoadEdge2 = null;

// Game room ID captured from the first real outbound game frame.
// Colonist frames: 0x03 0x01 [len] [roomId bytes] msgpack({action,payload,sequence})
let _gameRoomIdBytes = null;

// Outgoing sequence counter — incremented for each action we send.
// Initialised to 0; will be updated to track the game's own counter once we
// see real outbound frames, so our bot actions don't collide.
let _outgoingSequence = 0;

// ── Load persisted auto-play preference ──────────────────────────────────────

chrome.storage.local.get('autoPlay', (data) => {
  // Only override the default (true) if the user has explicitly toggled the
  // popup at some point. 'autoPlay' won't be in storage on first install.
  if ('autoPlay' in data) {
    autoPlay = !!data.autoPlay;
  }
  const mode = autoPlay ? 'AUTO-PLAY' : 'advisory';
  console.log(`[Catanbot] Loaded — ${mode} mode (toggle in extension popup)`);
});

// Listen for popup toggle messages relayed via chrome.runtime.
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'setAutoPlay') {
    autoPlay = !!msg.value;
    chrome.storage.local.set({ autoPlay });
    console.log('[Catanbot] Auto-play', autoPlay ? 'ENABLED' : 'disabled');
  }
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function sendToServer(endpoint, payload) {
  return fetch(`${SERVER_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
    .then(r => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    })
    .catch(err => {
      console.warn(`[Catanbot] Server unreachable (${endpoint}):`, err.message);
      return null;
    });
}

/**
 * Send a bot action to Colonist using the real wire protocol:
 *   [0x03][0x01][roomIdLen][...roomId][msgpack {action, payload, sequence}]
 *
 * _gameRoomIdBytes must be captured first (happens automatically on the first
 * real outbound frame seen after a manual action or game start).
 */
function dispatchAction(action) {
  if (!_gameRoomIdBytes) {
    console.warn('[Catanbot] Room ID not captured yet — waiting for first real game frame.');
    return;
  }
  _outgoingSequence++;
  window.postMessage({
    source:       'catanbot-ext',
    type:         'sendAction',
    action,
    roomIdBytes:  Array.from(_gameRoomIdBytes),
    sequence:     _outgoingSequence,
  }, '*');
}

// ── Message type constants (confirmed from Phase 1 capture) ──────────────────

const MSG_TYPE_INIT    = 4;    // initial game state: payload has playerColor, playOrder, gameState
const MSG_TYPE_DIFF    = 91;   // public state diff: payload.diff = stateChange
const MSG_TYPE_PRIVATE = 43;   // private card info: payload has givingCards/receivingCards

// Colonist sends this type when a player proposes a trade to others.
// The exact type code was not captured in Phase 1; 12 is a common value in
// Colonist traffic for TRADE_OFFER. Verify from DevTools → WS if trade
// responses are never triggered.
const MSG_TYPE_TRADE_OFFER = 12;

function extractStateChange(msg) {
  if (msg?.data?.type !== MSG_TYPE_DIFF) return null;
  return msg?.data?.payload?.diff ?? null;
}

function extractPrivateCardEvent(msg) {
  if (msg?.data?.type !== MSG_TYPE_PRIVATE) return null;
  return msg?.data?.payload ?? null;
}

function extractTradeOffer(msg) {
  if (msg?.data?.type !== MSG_TYPE_TRADE_OFFER) return null;
  return msg?.data?.payload ?? null;
}

function isInitialStateMsg(msg) {
  return msg?.data?.type === MSG_TYPE_INIT;
}

function getBotColorFromMsg(msg) {
  return msg?.data?.payload?.playerColor ?? null;
}

// ── Main message handler ──────────────────────────────────────────────────────

async function handleGameMessage(msg) {
  if (!msg || typeof msg !== 'object') return;

  // Heartbeats have no game data.
  if (msg.id === '136') return;

  console.log('[Catanbot] id=' + msg.id + ' data:', msg.data);

  const stateChange  = extractStateChange(msg);
  const privateEvent = extractPrivateCardEvent(msg);
  const tradeOffer   = extractTradeOffer(msg);

  // ── Detect game start (type=4) ────────────────────────────────────────────
  if (!gameStarted) {
    if (isInitialStateMsg(msg)) {
      const botColor = getBotColorFromMsg(msg);
      console.log('[Catanbot] Game start: botColor=' + botColor,
                  'playOrder=' + msg.data.payload.playOrder,
                  'gameState keys=' + Object.keys(msg.data.payload.gameState ?? {}));
      const result = await sendToServer('/game/start', {
        gameData: msg,
        botColor: botColor,
      });
      if (result?.status === 'ready') {
        gameStarted = true;
        console.log('[Catanbot] Server session ready, playing as color', result.botColor);
      }
    }
    return;
  }

  // ── Private card event (type=43) ──────────────────────────────────────────
  if (privateEvent) {
    await sendToServer('/game/private_event', { privateEvent });
    return;
  }

  // ── Incoming trade offer from another player ──────────────────────────────
  if (tradeOffer) {
    const result = await sendToServer('/game/trade_offer', { tradeOffer });
    if (result?.action && autoPlay) {
      console.log('[Catanbot] Trade response:', result.action);
      dispatchAction(result.action);
    } else if (result?.action) {
      console.log('%c[Catanbot] Trade response (advisory):', 'color:#ffaa00;font-weight:bold', result.action);
    }
    return;
  }

  // ── Public state diff (type=91) ───────────────────────────────────────────
  if (!stateChange) return;

  // ── Road building: send pending second edge on next stateChange ───────────
  if (_pendingRoadEdge2 !== null) {
    const edge2 = _pendingRoadEdge2;
    _pendingRoadEdge2 = null;
    if (autoPlay) {
      console.log('[Catanbot] Road building — sending edge2:', edge2);
      dispatchAction({ type: 'buildRoad', edgeIndex: edge2 });
    }
    // Still fall through to forward the current stateChange to the server.
  }

  // Wrap in the format apply_event() expects: { stateChange: {...} }
  const event  = { stateChange };
  const result = await sendToServer('/game/event', { event });
  if (!result?.ourTurn) return;

  // ── Log recommendations (always) ─────────────────────────────────────────
  if (result.plan && result.plan.length > 0) {
    console.log('%c[Catanbot] YOUR TURN — RECOMMENDED PLAN:', 'color:#00cc00;font-weight:bold;font-size:14px');
    result.plan.forEach((step, i) => console.log(`  ${i + 1}. ${step}`));
    if (result.summary) console.log('%c' + result.summary, 'color:#888');
  } else if (result.action) {
    const a = result.action;
    console.log('%c[Catanbot] ▶  ' + a.type, 'color:#00cc00;font-weight:bold', a);
  }

  if (!autoPlay) {
    console.log('[Catanbot] Auto-play is OFF — enable in popup to send this action automatically.');
    return;
  }

  // ── Auto-play: dispatch action to Colonist ────────────────────────────────
  const action = result.action;
  if (!action) return;

  console.log('%c[Catanbot] ▶▶  DISPATCHING:', 'color:#00ff00;font-weight:bold;font-size:14px', action);

  if (action.type === 'roadBuilding') {
    // PlayRoadBuilding requires two sequential buildRoad messages with an
    // intermediate stateChange acknowledgement from Colonist between them.
    // Send edge1 now; edge2 is queued and dispatched on the next stateChange.
    console.log('[Catanbot] Road building — sending edge1:', action.edge1, '(edge2 queued)');
    dispatchAction({ type: 'buildRoad', edgeIndex: action.edge1 });
    _pendingRoadEdge2 = action.edge2;
  } else {
    dispatchAction(action);
  }
}

// ── Listen for messages from inject.js ───────────────────────────────────────

window.addEventListener('message', (evt) => {
  if (!evt.data || evt.data.source !== 'catanbot-page') return;

  if (evt.data.type === 'wsMessage') {
    handleGameMessage(evt.data.decoded);

  } else if (evt.data.type === 'wsClose') {
    gameStarted       = false;
    _pendingRoadEdge2 = null;

  } else if (evt.data.type === 'wsOutbound' && !evt.data.fromBot) {
    // Parse every real outbound frame to:
    //  (a) capture the game room ID (bytes 3..3+len of type-0x03 frames)
    //  (b) track the sequence counter so bot sends don't collide
    const hex = evt.data.hex;
    if (!hex) return;

    const bytes = hex.split(' ').map(h => parseInt(h, 16));
    if (bytes.length < 3) return;

    // Game action frames start with 0x03.
    if (bytes[0] === 0x03) {
      if (!_gameRoomIdBytes) {
        const roomLen = bytes[2];
        if (bytes.length >= 3 + roomLen) {
          _gameRoomIdBytes = bytes.slice(3, 3 + roomLen);
          // Start our counter well above what the game has already sent.
          _outgoingSequence = 50;
          console.log(
            '%c[Catanbot] Room ID captured: "' + String.fromCharCode(..._gameRoomIdBytes) + '" — auto-play ready.',
            'color:#ffcc00;font-weight:bold',
          );
        }
      }
      // Keep our counter ahead of the game's counter (parsed from values if available).
      const vals = evt.data.values;
      if (Array.isArray(vals)) {
        for (let i = 0; i < vals.length - 1; i++) {
          if (vals[i] === 'sequence' && typeof vals[i + 1] === 'number') {
            _outgoingSequence = Math.max(_outgoingSequence, vals[i + 1] + 10);
          }
        }
      }
    }
  }
});
