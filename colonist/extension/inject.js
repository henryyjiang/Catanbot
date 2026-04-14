/**
 * inject.js — runs in the PAGE's JavaScript context.
 *
 * Colonist.io sends all game data as MessagePack-encoded ArrayBuffers.
 * This script:
 *   1. Patches window.WebSocket to intercept messages.
 *   2. Decodes ArrayBuffer frames with a minimal MessagePack decoder.
 *   3. Relays decoded objects to content.js via postMessage.
 */

(function () {
  'use strict';

  // ── Minimal MessagePack decoder ──────────────────────────────────────────────

  function decodeMsgpack(buffer) {
    const bytes = new Uint8Array(buffer);
    const view  = new DataView(buffer);
    let   pos   = 0;

    const td = new TextDecoder();

    function read() {
      const b = bytes[pos++];

      if (b <= 0x7f) return b;                          // positive fixint
      if (b >= 0xe0) return b - 256;                    // negative fixint

      if ((b & 0xf0) === 0x80) return readMap(b & 0x0f);    // fixmap
      if ((b & 0xf0) === 0x90) return readArr(b & 0x0f);    // fixarray
      if ((b & 0xe0) === 0xa0) return readStr(b & 0x1f);    // fixstr

      switch (b) {
        case 0xc0: return null;
        case 0xc2: return false;
        case 0xc3: return true;

        case 0xca: { const v = view.getFloat32(pos, false); pos += 4; return v; }
        case 0xcb: { const v = view.getFloat64(pos, false); pos += 8; return v; }

        case 0xcc: return bytes[pos++];
        case 0xcd: { const v = view.getUint16(pos, false); pos += 2; return v; }
        case 0xce: { const v = view.getUint32(pos, false); pos += 4; return v; }
        case 0xcf: {
          // uint64 — JS can't represent >2^53 exactly; return as Number anyway
          const hi = view.getUint32(pos, false);
          const lo = view.getUint32(pos + 4, false);
          pos += 8;
          return hi * 0x100000000 + lo;
        }

        case 0xd0: { const v = view.getInt8(pos); pos += 1; return v; }
        case 0xd1: { const v = view.getInt16(pos, false); pos += 2; return v; }
        case 0xd2: { const v = view.getInt32(pos, false); pos += 4; return v; }

        case 0xd9: { const n = bytes[pos++];                        return readStr(n); }
        case 0xda: { const n = view.getUint16(pos, false); pos += 2; return readStr(n); }
        case 0xdb: { const n = view.getUint32(pos, false); pos += 4; return readStr(n); }

        case 0xdc: { const n = view.getUint16(pos, false); pos += 2; return readArr(n); }
        case 0xdd: { const n = view.getUint32(pos, false); pos += 4; return readArr(n); }

        case 0xde: { const n = view.getUint16(pos, false); pos += 2; return readMap(n); }
        case 0xdf: { const n = view.getUint32(pos, false); pos += 4; return readMap(n); }

        // fixext / ext — skip over (not needed for game data)
        case 0xd4: pos += 2;  return null;
        case 0xd5: pos += 3;  return null;
        case 0xd6: pos += 5;  return null;
        case 0xd7: pos += 9;  return null;
        case 0xd8: pos += 17; return null;
        case 0xc7: { const n = bytes[pos++];                        pos += n + 1; return null; }
        case 0xc8: { const n = view.getUint16(pos, false); pos += 2; pos += n + 1; return null; }
        case 0xc9: { const n = view.getUint32(pos, false); pos += 4; pos += n + 1; return null; }

        // bin — return as Uint8Array
        case 0xc4: { const n = bytes[pos++];                        return readBin(n); }
        case 0xc5: { const n = view.getUint16(pos, false); pos += 2; return readBin(n); }
        case 0xc6: { const n = view.getUint32(pos, false); pos += 4; return readBin(n); }

        default:
          throw new Error(`Unknown msgpack byte 0x${b.toString(16)} at pos ${pos - 1}`);
      }
    }

    function readStr(n) {
      const s = td.decode(bytes.subarray(pos, pos + n));
      pos += n;
      return s;
    }
    function readBin(n) {
      const a = bytes.slice(pos, pos + n);
      pos += n;
      return a;
    }
    function readArr(n) {
      const a = [];
      for (let i = 0; i < n; i++) a.push(read());
      return a;
    }
    function readMap(n) {
      const o = {};
      for (let i = 0; i < n; i++) { const k = read(); o[k] = read(); }
      return o;
    }

    return read();
  }

  // ── Minimal MessagePack encoder ──────────────────────────────────────────────
  // Colonist.io uses MessagePack (binary) for both inbound and outbound frames.
  // This encoder handles all value types produced by translate_action():
  //   maps (objects), arrays, strings, integers, booleans, null.
  //
  // NOTE: If actions are silently dropped by Colonist's server, open DevTools →
  // Network → WS, find an outbound frame from a manual action, and compare its
  // hex bytes against what encodeMsgpack() produces for the same action.

  const _te = new TextEncoder();

  function encodeMsgpack(obj) {
    const parts = [];

    function writeUint8(n)  { parts.push(new Uint8Array([n])); }
    function writeUint16(n) {
      const b = new Uint8Array(2);
      new DataView(b.buffer).setUint16(0, n, false);
      parts.push(b);
    }
    function writeUint32(n) {
      const b = new Uint8Array(4);
      new DataView(b.buffer).setUint32(0, n, false);
      parts.push(b);
    }
    function writeInt8(n)  { parts.push(new Uint8Array([n & 0xff])); }
    function writeInt16(n) {
      const b = new Uint8Array(2);
      new DataView(b.buffer).setInt16(0, n, false);
      parts.push(b);
    }
    function writeInt32(n) {
      const b = new Uint8Array(4);
      new DataView(b.buffer).setInt32(0, n, false);
      parts.push(b);
    }

    function write(val) {
      if (val === null || val === undefined) {
        writeUint8(0xc0);                                        // nil

      } else if (val === false) {
        writeUint8(0xc2);                                        // false

      } else if (val === true) {
        writeUint8(0xc3);                                        // true

      } else if (typeof val === 'number' && Number.isInteger(val)) {
        if (val >= 0 && val <= 0x7f)        { writeUint8(val); }                        // positive fixint
        else if (val >= 0 && val <= 0xff)   { writeUint8(0xcc); writeUint8(val); }      // uint8
        else if (val >= 0 && val <= 0xffff) { writeUint8(0xcd); writeUint16(val); }     // uint16
        else if (val >= 0)                  { writeUint8(0xce); writeUint32(val); }     // uint32
        else if (val >= -32)                { writeUint8(val + 256); }                   // negative fixint
        else if (val >= -128)               { writeUint8(0xd0); writeInt8(val); }       // int8
        else if (val >= -32768)             { writeUint8(0xd1); writeInt16(val); }      // int16
        else                                { writeUint8(0xd2); writeInt32(val); }      // int32

      } else if (typeof val === 'string') {
        const enc = _te.encode(val);
        const n   = enc.length;
        if      (n <= 31)    { writeUint8(0xa0 | n); }                                  // fixstr
        else if (n <= 0xff)  { writeUint8(0xd9); writeUint8(n); }                       // str8
        else if (n <= 0xffff){ writeUint8(0xda); writeUint16(n); }                      // str16
        else                 { writeUint8(0xdb); writeUint32(n); }                      // str32
        parts.push(enc);

      } else if (Array.isArray(val)) {
        const n = val.length;
        if      (n <= 15)    { writeUint8(0x90 | n); }                                  // fixarray
        else if (n <= 0xffff){ writeUint8(0xdc); writeUint16(n); }                      // array16
        else                 { writeUint8(0xdd); writeUint32(n); }                      // array32
        for (const item of val) write(item);

      } else if (typeof val === 'object') {
        const keys = Object.keys(val);
        const n    = keys.length;
        if      (n <= 15)    { writeUint8(0x80 | n); }                                  // fixmap
        else if (n <= 0xffff){ writeUint8(0xde); writeUint16(n); }                      // map16
        else                 { writeUint8(0xdf); writeUint32(n); }                      // map32
        for (const k of keys) { write(k); write(val[k]); }
      }
    }

    write(obj);

    const total  = parts.reduce((s, p) => s + p.length, 0);
    const result = new Uint8Array(total);
    let   offset = 0;
    for (const p of parts) { result.set(p, offset); offset += p.length; }
    return result.buffer;
  }

  // ── Colonist action code table ────────────────────────────────────────────────
  // Real wire format (confirmed from captured OUTBOUND[GAME] frames):
  //   [0x03][0x01][roomIdLen][...roomId][msgpack {action:N, payload:X, sequence:N}]
  //
  // action codes discovered so far:
  //   15 = buildSettlement  (payload = cornerIndex)
  //   11 = buildRoad        (payload = edgeIndex)
  //
  // Unknown codes are null — the bot will warn and skip those actions until
  // they are discovered through further gameplay observation.
  const _ACTION_CODES = {
    buildSettlement:  { code: 15,   payloadFn: a => a.cornerIndex },
    buildRoad:        { code: 11,   payloadFn: a => a.edgeIndex   },
    buildCity:        { code: null, payloadFn: a => a.cornerIndex },  // TBD
    buyDevCard:       { code: null, payloadFn: () => null         },  // TBD
    rollDice:         { code: null, payloadFn: () => null         },  // TBD
    endTurn:          { code: null, payloadFn: () => null         },  // TBD
    moveRobber:       { code: null, payloadFn: a => a.tileIndex   },  // TBD
    stealCard:        { code: null, payloadFn: a => a.stealFromColor }, // TBD
    discardCards:     { code: null, payloadFn: a => a.cards       },  // TBD
    playKnight:       { code: null, payloadFn: a => a.tileIndex   },  // TBD
    playMonopoly:     { code: null, payloadFn: a => a.resourceType }, // TBD
    playYearOfPlenty: { code: null, payloadFn: a => a.resources   },  // TBD
    roadBuilding:     { code: null, payloadFn: () => null         },  // TBD
    bankTrade:        { code: null, payloadFn: () => null         },  // TBD
    tradeOffer:       { code: null, payloadFn: () => null         },  // TBD
    tradeResponse:    { code: null, payloadFn: () => null         },  // TBD
  };

  // ── Outbound action dispatch ──────────────────────────────────────────────────
  window.addEventListener('message', (evt) => {
    if (!evt.data || evt.data.source !== 'catanbot-ext') return;
    if (evt.data.type !== 'sendAction') return;

    const ws = window.__catanBotWS;
    if (!ws) {
      console.warn('[Catanbot/page] sendAction: WS not captured yet, dropping:', evt.data.action);
      return;
    }
    if (ws.readyState !== 1 /* OPEN */) {
      console.warn('[Catanbot/page] sendAction: WS readyState=' + ws.readyState + ', dropping:', evt.data.action);
      return;
    }

    const { action, roomIdBytes, sequence } = evt.data;
    const mapping = _ACTION_CODES[action.type];

    if (!mapping) {
      console.warn('[Catanbot/page] Unknown action type:', action.type);
      return;
    }
    if (mapping.code === null) {
      console.warn('[Catanbot/page] Action code not yet discovered for:', action.type,
                   '— play this action manually once to capture it.');
      return;
    }

    const colonistPayload = mapping.payloadFn(action) ?? null;
    const msgpackBuf = encodeMsgpack({ action: mapping.code, payload: colonistPayload, sequence });
    const msgpack    = new Uint8Array(msgpackBuf);
    const roomBytes  = new Uint8Array(roomIdBytes);

    // Wire frame: [0x03][0x01][roomIdLen][...roomId][msgpack...]
    const frame = new Uint8Array(3 + roomBytes.length + msgpack.length);
    frame[0] = 0x03;
    frame[1] = 0x01;
    frame[2] = roomBytes.length;
    frame.set(roomBytes, 3);
    frame.set(msgpack, 3 + roomBytes.length);

    ws.__catanBotSend = true;
    ws.send(frame.buffer);
    ws.__catanBotSend = false;

    console.log('[Catanbot/page] Action sent: ' + action.type +
                ' → Colonist code=' + mapping.code + ' payload=' + colonistPayload +
                ' seq=' + sequence);
  });

  // ── WebSocket patch ──────────────────────────────────────────────────────────

  const _OriginalWebSocket = window.WebSocket;

  // ── Read ALL msgpack values from a buffer ────────────────────────────────────
  // The game protocol packs multiple values sequentially in one frame (not as
  // a wrapping array).  decodeMsgpack() only reads the first.  This reads all.
  function decodeAllMsgpack(buffer) {
    const bytes  = new Uint8Array(buffer);
    const values = [];
    let   pos    = 0;
    // Reuse the inner read() logic by creating a mini-decoder that advances pos.
    while (pos < bytes.length) {
      const start = pos;
      try {
        const val = decodeMsgpack(buffer.slice ? buffer.slice(pos) : buffer.slice(pos));
        values.push(val);
        // Advance pos by however many bytes were consumed.
        // We approximate by encoding val back — but it's simpler to just try one at a time
        // using a fresh slice and track length via trial.
        // Instead: use the known msgpack size rules.
        const b = bytes[pos];
        if      (b <= 0x7f || b >= 0xe0)            pos += 1;   // fixint
        else if ((b & 0xf0) === 0x80) pos += 1 + 2 * (b & 0x0f);  // fixmap (approx)
        else if ((b & 0xf0) === 0x90) pos += 1 + (b & 0x0f);      // fixarray (approx)
        else if ((b & 0xe0) === 0xa0) pos += 1 + (b & 0x1f);      // fixstr
        else if (b === 0xc0 || b === 0xc2 || b === 0xc3) pos += 1; // nil/false/true
        else if (b === 0xcc || b === 0xd0) pos += 2;
        else if (b === 0xcd || b === 0xd1) pos += 3;
        else if (b === 0xca || b === 0xce || b === 0xd2) pos += 5;
        else if (b === 0xcb || b === 0xcf || b === 0xd3) pos += 9;
        else if (b === 0xd9) pos += 2 + bytes[pos + 1];
        else if (b === 0xda) pos += 3 + (bytes[pos+1] << 8 | bytes[pos+2]);
        else { pos = bytes.length; break; }  // unknown — stop
      } catch (e) { break; }
    }
    return values;
  }

  // ── Shared send-intercept logic ───────────────────────────────────────────
  // Called for EVERY ws.send(), whether from the real game or from our bot.
  function _interceptSend(data, fromBot) {
    const tag = fromBot ? '[BOT]' : '[GAME]';

    if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
      const view = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : new Uint8Array(data);

      // Always log raw hex (first 32 bytes max) so nothing is hidden.
      const hex = Array.from(view.subarray(0, Math.min(view.length, 32)))
                       .map(b => b.toString(16).padStart(2, '0')).join(' ');

      // Read ALL msgpack values in the frame (not just the first).
      const buf  = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
      const vals = decodeAllMsgpack(buf);

      console.log('[Catanbot/page] OUTBOUND' + tag,
                  '| hex:', hex,
                  '| values:', vals);

      window.postMessage({
        source:  'catanbot-page',
        type:    'wsOutbound',
        fromBot,
        hex,
        values:  vals,
      }, '*');

    } else if (typeof data === 'string') {
      console.log('[Catanbot/page] OUTBOUND' + tag + ' (text):', data);
      window.postMessage({ source: 'catanbot-page', type: 'wsOutbound', fromBot, hex: null, values: [data] }, '*');
    }
  }

  // ── XHR interceptor (game may use XHR instead of fetch for actions) ─────────
  const _XHROpen = XMLHttpRequest.prototype.open;
  const _XHRSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url) {
    this.__catanMethod = method;
    this.__catanUrl    = url;
    return _XHROpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function (body) {
    if (this.__catanMethod && this.__catanMethod.toUpperCase() !== 'GET') {
      let bodyStr = '(empty)';
      try {
        if (typeof body === 'string') bodyStr = body.slice(0, 300);
        else if (body instanceof ArrayBuffer || ArrayBuffer.isView(body)) {
          const view = ArrayBuffer.isView(body) ? new Uint8Array(body.buffer, body.byteOffset, body.byteLength) : new Uint8Array(body);
          bodyStr = 'hex: ' + Array.from(view.subarray(0, 32)).map(b => b.toString(16).padStart(2,'0')).join(' ');
        }
      } catch(e) {}
      console.log('[Catanbot/page] XHR', this.__catanMethod, this.__catanUrl, 'body:', bodyStr);
      window.postMessage({ source: 'catanbot-page', type: 'xhrAction', method: this.__catanMethod, url: this.__catanUrl, body: bodyStr }, '*');
    }
    return _XHRSend.apply(this, arguments);
  };

  // ── Patch the original WS prototype so ALL sends are intercepted ──────────
  // The game likely stores a reference to the original prototype send and
  // calls it directly, bypassing our subclass override.  Patching the
  // prototype catches every call regardless of how it is invoked.
  const _OriginalSend = _OriginalWebSocket.prototype.send;
  _OriginalWebSocket.prototype.send = function (data) {
    _interceptSend(data, !!this.__catanBotSend);
    _OriginalSend.call(this, data);
  };

  class CatanBotWebSocket extends _OriginalWebSocket {
    constructor(url, protocols) {
      if (protocols !== undefined) {
        super(url, protocols);
      } else {
        super(url);
      }

      console.log('[Catanbot/page] New WebSocket:', url);

      this.addEventListener('message', async (evt) => {
        let decoded;

        if (evt.data instanceof ArrayBuffer) {
          try {
            decoded = decodeMsgpack(evt.data);
          } catch (e) {
            console.warn('[Catanbot/page] msgpack decode failed:', e.message);
            return;
          }
        } else if (typeof evt.data === 'string') {
          try { decoded = JSON.parse(evt.data); }
          catch { return; }
        } else if (evt.data instanceof Blob) {
          try {
            const buf = await evt.data.arrayBuffer();
            decoded = decodeMsgpack(buf);
          } catch (e) {
            console.warn('[Catanbot/page] Blob decode failed:', e.message);
            return;
          }
        } else {
          return;
        }

        console.log('[Catanbot/page] Decoded msg id=' + decoded?.id + ':', decoded);

        window.__catanBotWS = this;

        window.postMessage({
          source: 'catanbot-page',
          type: 'wsMessage',
          decoded: decoded,
        }, '*');
      });

      this.addEventListener('close', () => {
        console.log('[Catanbot/page] WebSocket closed');
        window.postMessage({ source: 'catanbot-page', type: 'wsClose' }, '*');
      });
    }
  }

  window.WebSocket = CatanBotWebSocket;
  console.log('[Catanbot/page] WebSocket patched — prototype.send also patched');
})();
