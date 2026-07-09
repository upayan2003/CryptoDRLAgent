"""
demo_api.py
===========
FastAPI backend for the Crypto RL Agent live demo.
Supports both autonomous agent trading and manual demo orders.

Endpoints:
  GET  /health              - server + model status
  GET  /status              - full portfolio + chart data
  GET  /chart               - OHLCV bars for candlestick chart
  POST /order               - place a manual demo order {side: "BUY"|"SELL"|"CLOSE"}
  POST /reset               - reset entire demo portfolio
  GET  /agent/signal        - what the agent would do right now
  POST /agent/pause         - pause or resume the agent {"paused": true|false}
  POST /agent/stop          - stop the agent entirely
  POST /market/simulate     - inject a simulated price {"delta_pct": 1.0} or {"set_price": 95000}

Deploy to Render.com:
  Start command: uvicorn demo_api:app --host 0.0.0.0 --port $PORT
"""

import os, time, threading, asyncio, json
from typing import Optional
import numpy as np
from collections import deque
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from stable_baselines3 import PPO

# ── APP SETUP ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Crypto DRL Agent Demo API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "best_model.zip"
OBS_WINDOW      = 60
INITIAL_BALANCE = 10_000.0
ORDER_SIZE_USD  = 100.0
FEE_RATE        = 0.0004
INFERENCE_EVERY = 5
EPS             = 1e-9

# ── SHARED STATE ──────────────────────────────────────────────────────────────
portfolio = {
    "balance": INITIAL_BALANCE, "position": 0, "entry_price": 0.0,
    "realized_pnl": 0.0, "unrealized_pnl": 0.0,
    "trades": [], "equity_history": [INITIAL_BALANCE],
    "action_history": [], "last_action": "HOLD",
    "last_price": 0.0, "bars_since_trade": 0,
    "agent_signal": "HOLD", "agent_confidence": 0.0,
    "manual_mode": False,
}

feature_buffer = deque(maxlen=OBS_WINDOW + 10)
lob_buffer     = deque(maxlen=50)
trade_buffer   = deque(maxlen=200)
price_history  = deque(maxlen=500)   # for candlestick
ohlcv_bars     = deque(maxlen=200)   # 5-second OHLCV bars
_bar_prices    = []                  # accumulates intrabar trade prices from WS
_bar_start     = time.time()

model = None

# ── AGENT CONTROL STATE ───────────────────────────────────────────────────────
# FIX: these globals allow the dashboard to pause/stop the inference loop.
_agent_paused  = False
_agent_stopped = False

# ── SIMULATION STATE ───────────────────────────────────────────────────────────
# FIX: allows the dashboard's market simulator to inject synthetic prices.
_sim_mode          = False
_sim_price_override = 0.0

# ── MODEL LOAD ────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global model
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH)
        print(f"[API] Model loaded ✓")
    else:
        print(f"[API] WARNING: {MODEL_PATH} not found — signals will be HOLD")

    threading.Thread(target=start_websocket, daemon=True).start()
    threading.Thread(target=_inference_loop, daemon=True).start()
    print("[API] Background threads started ✓")

# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────
def compute_features(price, prev_price, bid_p, bid_q, ask_p, ask_q, buy_vol, sell_vol):
    log_ret    = float(np.log(price / max(prev_price, EPS)))
    spread     = ask_p - bid_p
    mid_price  = (bid_p + ask_p) / 2.0
    rel_spread = spread / max(mid_price, EPS)
    depth_imb  = (bid_q - ask_q) / max(bid_q + ask_q, EPS)
    tfi        = (buy_vol - sell_vol) / max(buy_vol + sell_vol, EPS)
    buf        = list(feature_buffer)
    vol_roll   = float(np.std([b["log_ret"] for b in buf[-20:]] + [log_ret])) if len(buf) > 2 else 0.0
    prices     = [b["price"] for b in buf[-20:]] + [price]
    ret_5      = float(np.log(prices[-1] / max(prices[-6],  EPS))) if len(prices) >= 6  else 0.0
    ret_20     = float(np.log(prices[-1] / max(prices[-21], EPS))) if len(prices) >= 21 else 0.0
    vwap       = float(np.mean(prices[-100:])) if prices else price
    pvwap      = (price - vwap) / max(vwap, EPS)
    ofi_norm   = 0.0
    if len(lob_buffer) > 1:
        lobs  = list(lob_buffer)[-20:]
        ofi   = sum((lobs[i]["bid_qty"] - lobs[i-1]["bid_qty"]) -
                    (lobs[i]["ask_qty"] - lobs[i-1]["ask_qty"])
                    for i in range(1, len(lobs)))
        depth = sum(l["bid_qty"] + l["ask_qty"] for l in lobs)
        ofi_norm = ofi / max(depth, EPS)
    return {
        "price": price, "log_ret": log_ret, "tfi": tfi,
        "ofi_norm": min(1.0, max(-1.0, ofi_norm)),
        "vol_rolling": vol_roll, "rel_spread": rel_spread,
        "depth_imb": depth_imb, "ret_5": ret_5,
        "ret_20": ret_20, "price_vs_vwap": pvwap,
    }

def build_observation():
    if len(feature_buffer) < OBS_WINDOW:
        return None
    window    = list(feature_buffer)[-OBS_WINDOW:]
    feat_cols = ["tfi","ofi_norm","vol_rolling","rel_spread",
                 "depth_imb","ret_5","ret_20","price_vs_vwap"]
    mkt  = np.array([[b[c] for c in feat_cols] for b in window], dtype=np.float32)
    port = np.full((OBS_WINDOW, 3),
                   [float(portfolio["position"]),
                    float(portfolio["unrealized_pnl"] / INITIAL_BALANCE), 0.0],
                   dtype=np.float32)
    return np.concatenate([mkt, port], axis=1).flatten()

# ── TRADE EXECUTION ───────────────────────────────────────────────────────────
def _close_position(price, label, source="agent"):
    if portfolio["position"] == 0:
        return
    btc = ORDER_SIZE_USD / portfolio["entry_price"]
    pnl = ((price - portfolio["entry_price"]) * btc if portfolio["position"] == 1
           else (portfolio["entry_price"] - price) * btc)
    pnl -= FEE_RATE * ORDER_SIZE_USD
    portfolio["realized_pnl"] += pnl
    portfolio["unrealized_pnl"] = 0.0
    portfolio["trades"].append({
        "time":   datetime.utcnow().strftime("%H:%M:%S"),
        "type":   label, "source": source,
        "price":  round(price, 2),
        "pnl":    round(pnl, 4),
        "position": portfolio["position"],
    })
    portfolio["position"] = 0
    portfolio["entry_price"] = 0.0
    portfolio["bars_since_trade"] = 0

def _open_position(price, direction, source="agent"):
    portfolio["position"]    = direction
    portfolio["entry_price"] = price
    portfolio["realized_pnl"] -= FEE_RATE * ORDER_SIZE_USD
    portfolio["trades"].append({
        "time":   datetime.utcnow().strftime("%H:%M:%S"),
        "type":   "BUY" if direction == 1 else "SELL",
        "source": source,
        "price":  round(price, 2),
        "pnl":    0,
        "position": direction,
    })

def update_unrealized(price):
    if portfolio["position"] == 1:
        btc = ORDER_SIZE_USD / portfolio["entry_price"]
        portfolio["unrealized_pnl"] = (price - portfolio["entry_price"]) * btc
    elif portfolio["position"] == -1:
        btc = ORDER_SIZE_USD / portfolio["entry_price"]
        portfolio["unrealized_pnl"] = (portfolio["entry_price"] - price) * btc
    else:
        portfolio["unrealized_pnl"] = 0.0

# ── INFERENCE LOOP ────────────────────────────────────────────────────────────
def _inference_loop():
    """
    Runs every INFERENCE_EVERY seconds in a background thread.
    Fetches price from Binance REST, computes features, runs agent.
    Uses REST polling instead of WebSocket to avoid asyncio conflicts.
    """
    import requests as req
    # FIX: _bar_start must also be declared global so it can be reassigned.
    global _bar_prices, _bar_start
    prev_price = None

    print("[POLL] Inference loop started")

    while True:
        time.sleep(INFERENCE_EVERY)

        # FIX: respect the pause/stop controls set by the dashboard endpoints.
        if _agent_stopped:
            print("[POLL] Agent stopped — exiting loop")
            break
        if _agent_paused:
            continue

        try:
            # FIX: honour the market simulator's synthetic price when active.
            if _sim_mode and _sim_price_override > 0:
                price = _sim_price_override
            else:
                r     = req.get("https://fapi.binance.com/fapi/v1/ticker/price",
                                params={"symbol": "BTCUSDT"}, timeout=5)
                price = float(r.json()["price"])

            price_history.append({"price": price, "ts": time.time()})

            # ── OHLCV bar ────────────────────────────────────────────────────
            # FIX: _bar_prices is also fed by WebSocket aggTrade ticks (see
            # start_websocket below), so it accumulates real intrabar prices.
            # Emit one bar per REST poll interval (every INFERENCE_EVERY sec)
            # and reset the accumulator for the next bar.
            _bar_prices.append(price)
            if _bar_prices:
                ohlcv_bars.append({
                    "time":  datetime.utcnow().strftime("%H:%M:%S"),
                    "open":  _bar_prices[0],
                    "high":  max(_bar_prices),
                    "low":   min(_bar_prices),
                    "close": _bar_prices[-1],  # last traded price, not always == price
                    "vol":   len(_bar_prices),
                })
                _bar_prices = []
                _bar_start  = time.time()

            if prev_price is None:
                prev_price = price
                print(f"[POLL] First price: {price:.2f} — warming up...")
                continue

            # Aggregate recent trades
            recent   = list(trade_buffer)
            buy_vol  = sum(t["qty"] for t in recent if not t["is_sell"])
            sell_vol = sum(t["qty"] for t in recent if t["is_sell"])
            trade_buffer.clear()

            # Get latest LOB snapshot
            lob = list(lob_buffer)[-1] if lob_buffer else {
                "bid_price": price * 0.9999, "bid_qty": 1.0,
                "ask_price": price * 1.0001, "ask_qty": 1.0,
            }

            bar = compute_features(price, prev_price,
                                   lob["bid_price"], lob["bid_qty"],
                                   lob["ask_price"], lob["ask_qty"],
                                   buy_vol, sell_vol)
            feature_buffer.append(bar)
            prev_price = price

            n = len(feature_buffer)
            print(f"[POLL] price={price:.2f}  bars={n}/{OBS_WINDOW}  "
                  f"tfi={bar['tfi']:+.3f}  ofi={bar['ofi_norm']:+.3f}")

            update_unrealized(price)
            portfolio["last_price"] = price
            portfolio["bars_since_trade"] += 1
            equity = INITIAL_BALANCE + portfolio["realized_pnl"] + portfolio["unrealized_pnl"]
            portfolio["equity_history"].append(round(equity, 4))

            if n < OBS_WINDOW or model is None:
                continue

            obs = build_observation()
            if obs is None:
                continue

            action, _ = model.predict(obs, deterministic=True)
            action     = int(action)
            names      = {0:"HOLD", 1:"BUY", 2:"SELL"}
            portfolio["agent_signal"] = names[action]

            if not portfolio["manual_mode"]:
                if action == 1 and portfolio["position"] <= 0:
                    if portfolio["position"] == -1:
                        _close_position(price, "CLOSE_SHORT", source="agent")
                    _open_position(price, 1, source="agent")
                elif action == 2 and portfolio["position"] >= 0:
                    if portfolio["position"] == 1:
                        _close_position(price, "CLOSE_LONG", source="agent")
                    _open_position(price, -1, source="agent")
                portfolio["last_action"] = names[action]

            portfolio["action_history"].append({
                "time":   datetime.utcnow().strftime("%H:%M:%S"),
                "action": names[action],
                "price":  round(price, 2),
                "source": "manual" if portfolio["manual_mode"] else "agent",
                "tfi":    round(bar["tfi"], 4),
                "ofi":    round(bar["ofi_norm"], 4),
            })

        except Exception as e:
            print(f"[POLL] Error: {e}")

# ── WEBSOCKET ─────────────────────────────────────────────────────────────────
def start_websocket():
    async def _run():
        import websockets
        uri = "wss://fstream.binance.com/stream?streams=btcusdt@bookTicker/btcusdt@aggTrade"
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    print("[WS] Connected")
                    async for msg in ws:
                        d      = json.loads(msg)
                        stream = d.get("stream", "")
                        data   = d.get("data", {})
                        if "bookTicker" in stream:
                            lob_buffer.append({
                                "bid_price": float(data.get("b", 0)),
                                "bid_qty":   float(data.get("B", 0)),
                                "ask_price": float(data.get("a", 0)),
                                "ask_qty":   float(data.get("A", 0)),
                            })
                        elif "aggTrade" in stream:
                            trade_price = float(data.get("p", 0))
                            # FIX: feed real trade prices into the bar accumulator
                            # so each 5-second OHLCV bar reflects intrabar movement
                            # rather than always being a single-price doji.
                            if trade_price > 0:
                                _bar_prices.append(trade_price)
                            trade_buffer.append({
                                "qty":     float(data.get("q", 0)),
                                "is_sell": bool(data.get("m", False)),
                                "price":   trade_price,
                            })
            except Exception as e:
                print(f"[WS] Error: {e} — reconnecting in 3s")
                await asyncio.sleep(3)
    asyncio.run(_run())

# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "running",
        "model_loaded":  model is not None,
        "bars_buffered": len(feature_buffer),
        "ready":         len(feature_buffer) >= OBS_WINDOW,
    }

@app.get("/status")
def status():
    equity = INITIAL_BALANCE + portfolio["realized_pnl"] + portfolio["unrealized_pnl"]
    return {
        "equity":           round(equity, 2),
        "realized_pnl":     round(portfolio["realized_pnl"], 4),
        "unrealized_pnl":   round(portfolio["unrealized_pnl"], 4),
        "total_return_pct": round((equity / INITIAL_BALANCE - 1) * 100, 4),
        "position":         portfolio["position"],
        "entry_price":      round(portfolio["entry_price"], 2),
        "last_action":      portfolio["last_action"],
        "last_price":       portfolio["last_price"],
        "agent_signal":     portfolio["agent_signal"],
        "manual_mode":      portfolio["manual_mode"],
        "n_trades":         len(portfolio["trades"]),
        "equity_history":   portfolio["equity_history"][-120:],
        "trades":           portfolio["trades"][-20:],
        "actions":          portfolio["action_history"][-30:],
        "bars_ready":       len(feature_buffer) >= OBS_WINDOW,
        "bars_buffered":    len(feature_buffer),
        "bars_needed":      OBS_WINDOW,
    }

@app.get("/chart")
def chart():
    """Return OHLCV bars for candlestick chart."""
    return {
        "bars":       list(ohlcv_bars),
        "last_price": portfolio["last_price"],
    }

class OrderRequest(BaseModel):
    side: str   # "BUY", "SELL", "CLOSE"

@app.post("/order")
def manual_order(req: OrderRequest):
    """Place a manual demo order. Switches to manual mode."""
    price = portfolio["last_price"]
    if price == 0:
        return {"error": "No price data yet — wait for WebSocket connection"}

    portfolio["manual_mode"] = True
    side = req.side.upper()

    if side == "BUY":
        if portfolio["position"] == -1:
            _close_position(price, "CLOSE_SHORT", source="manual")
        if portfolio["position"] == 0:
            _open_position(price, 1, source="manual")
            portfolio["last_action"] = "BUY"
            return {"status": "BUY executed", "price": price}
        return {"status": "Already long"}

    elif side == "SELL":
        if portfolio["position"] == 1:
            _close_position(price, "CLOSE_LONG", source="manual")
        if portfolio["position"] == 0:
            _open_position(price, -1, source="manual")
            portfolio["last_action"] = "SELL"
            return {"status": "SELL executed", "price": price}
        return {"status": "Already short"}

    elif side == "CLOSE":
        if portfolio["position"] == 0:
            return {"status": "No open position"}
        label = "CLOSE_LONG" if portfolio["position"] == 1 else "CLOSE_SHORT"
        _close_position(price, label, source="manual")
        portfolio["last_action"] = "CLOSE"
        return {"status": f"{label} executed", "price": price,
                "pnl": round(portfolio["realized_pnl"], 4)}

    return {"error": f"Unknown side: {side}"}

@app.post("/toggle_mode")
def toggle_mode():
    """Switch between agent-auto and manual mode."""
    portfolio["manual_mode"] = not portfolio["manual_mode"]
    mode = "MANUAL" if portfolio["manual_mode"] else "AGENT AUTO"
    return {"mode": mode, "manual_mode": portfolio["manual_mode"]}

@app.post("/reset")
def reset():
    global _agent_stopped, _agent_paused, _sim_mode, _sim_price_override
    portfolio.update({
        "balance": INITIAL_BALANCE, "position": 0, "entry_price": 0.0,
        "realized_pnl": 0.0, "unrealized_pnl": 0.0,
        "trades": [], "equity_history": [INITIAL_BALANCE],
        "action_history": [], "last_action": "HOLD",
        "last_price": 0.0, "bars_since_trade": 0,
        "agent_signal": "HOLD", "agent_confidence": 0.0,
        "manual_mode": False,
    })
    feature_buffer.clear()
    ohlcv_bars.clear()
    _agent_paused       = False
    _agent_stopped      = False
    _sim_mode           = False
    _sim_price_override = 0.0
    return {"status": "reset complete"}

# ── FIX: missing endpoints that the dashboard calls ──────────────────────────

@app.get("/agent/signal")
def agent_signal():
    """Return the current agent signal and readiness."""
    return {
        "signal":     portfolio["agent_signal"],
        "confidence": portfolio["agent_confidence"],
        "ready":      len(feature_buffer) >= OBS_WINDOW,
        "paused":     _agent_paused,
        "stopped":    _agent_stopped,
    }

class PauseRequest(BaseModel):
    paused: bool

@app.post("/agent/pause")
def agent_pause(req: PauseRequest):
    """Pause or resume the inference loop without resetting state."""
    global _agent_paused
    _agent_paused = req.paused
    return {"paused": _agent_paused}

@app.post("/agent/stop")
def agent_stop():
    """Stop the inference loop (requires /reset to restart)."""
    global _agent_stopped
    _agent_stopped = True
    return {"stopped": True}

class SimulateRequest(BaseModel):
    delta_pct: Optional[float] = None   # e.g. 1.0 means +1%
    set_price: Optional[float] = None   # override to an exact value

@app.post("/market/simulate")
def simulate_market(req: SimulateRequest):
    """Inject a synthetic price for demo/testing purposes."""
    global _sim_mode, _sim_price_override
    _sim_mode = True
    current = _sim_price_override if _sim_price_override > 0 else portfolio["last_price"]
    if req.set_price is not None:
        _sim_price_override = float(req.set_price)
    elif req.delta_pct is not None:
        _sim_price_override = current * (1.0 + req.delta_pct / 100.0)
    # Immediately update last_price so the UI reflects the change at once
    if _sim_price_override > 0:
        portfolio["last_price"] = _sim_price_override
    return {"status": "simulated", "price": _sim_price_override}

# ── STARTUP ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
