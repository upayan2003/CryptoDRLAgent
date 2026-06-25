"""
demo_api.py
===========
FastAPI app that serves PPO agent predictions.
Deploy to Render.com for free hosting.

Endpoints:
  GET /health          — check if API is running
  GET /predict         — get agent's current action
  GET /status          — full portfolio status
  POST /reset          — reset demo portfolio
"""

import os
import time
import threading
import numpy as np
from collections import deque
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from stable_baselines3 import PPO

app = FastAPI(title="Crypto RL Agent Demo API")

# Allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "best_model.zip"   # upload alongside this file
OBS_WINDOW      = 60
N_FEATURES      = 8
INITIAL_BALANCE = 10_000.0
ORDER_SIZE_USD  = 100.0
FEE_RATE        = 0.0004
INFERENCE_EVERY = 5        # seconds between predictions (fast for demo)
EPS             = 1e-9

# ── State (in-memory — resets when server restarts) ───────────────────────────
portfolio = {
    "balance":       INITIAL_BALANCE,
    "position":      0,
    "entry_price":   0.0,
    "realized_pnl":  0.0,
    "unrealized_pnl":0.0,
    "trades":        [],
    "equity_history":[INITIAL_BALANCE],
    "action_history":[],
    "last_action":   "HOLD",
    "last_price":    0.0,
    "bars_since_trade": 0,
}

# Rolling feature buffer
feature_buffer = deque(maxlen=OBS_WINDOW + 10)
price_buffer   = deque(maxlen=200)
lob_buffer     = deque(maxlen=50)
trade_buffer   = deque(maxlen=100)

# Load model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"WARNING: {MODEL_PATH} not found — predictions will be random")

# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(price, prev_price, bid_p, bid_q, ask_p, ask_q,
                     buy_vol, sell_vol):
    """Compute the 8 microstructure features for one bar."""
    EPS = 1e-9
    log_ret   = float(np.log(price / max(prev_price, EPS)))
    spread    = ask_p - bid_p
    mid_price = (bid_p + ask_p) / 2.0
    rel_spread= spread / max(mid_price, EPS)
    depth_imb = (bid_q - ask_q) / max(bid_q + ask_q, EPS)
    tfi       = (buy_vol - sell_vol) / max(buy_vol + sell_vol, EPS)
    vol_roll  = float(np.std([b["log_ret"] for b in list(feature_buffer)[-20:]]
                              + [log_ret])) if len(feature_buffer) > 2 else 0.0
    prices    = [b["price"] for b in list(feature_buffer)[-20:]] + [price]
    ret_5     = float(np.log(prices[-1] / max(prices[-6], EPS))) if len(prices) >= 6 else 0.0
    ret_20    = float(np.log(prices[-1] / max(prices[-21], EPS))) if len(prices) >= 21 else 0.0
    vwap      = float(np.mean(prices[-100:])) if prices else price
    pvwap     = (price - vwap) / max(vwap, EPS)
    # OFI from last 20 LOB snapshots
    ofi_norm  = 0.0
    if len(lob_buffer) > 1:
        lobs = list(lob_buffer)[-20:]
        ofi  = sum(
            (lobs[i]["bid_qty"] - lobs[i-1]["bid_qty"]) -
            (lobs[i]["ask_qty"] - lobs[i-1]["ask_qty"])
            for i in range(1, len(lobs))
        )
        depth = sum(l["bid_qty"] + l["ask_qty"] for l in lobs)
        ofi_norm = ofi / max(depth, EPS)

    return {
        "price":    price,
        "log_ret":  log_ret,
        "tfi":      tfi,
        "ofi_norm": min(1.0, max(-1.0, ofi_norm)),
        "vol_rolling": vol_roll,
        "rel_spread":  rel_spread,
        "depth_imb":   depth_imb,
        "ret_5":    ret_5,
        "ret_20":   ret_20,
        "price_vs_vwap": pvwap,
    }

def build_observation():
    """Build (660,) flat observation for the PPO agent."""
    if len(feature_buffer) < OBS_WINDOW:
        return None
    window = list(feature_buffer)[-OBS_WINDOW:]
    feat_cols = ["tfi","ofi_norm","vol_rolling","rel_spread",
                 "depth_imb","ret_5","ret_20","price_vs_vwap"]
    mkt = np.array([[b[c] for c in feat_cols] for b in window],
                   dtype=np.float32)
    pos_norm  = float(portfolio["position"])
    upnl_norm = float(portfolio["unrealized_pnl"] / INITIAL_BALANCE)
    time_norm = 0.0
    port = np.full((OBS_WINDOW, 3), [pos_norm, upnl_norm, time_norm],
                   dtype=np.float32)
    return np.concatenate([mkt, port], axis=1).flatten()

# ── Binance Testnet WebSocket ──────────────────────────────────────────────────

def start_websocket():
    """Connect to Binance Testnet WebSocket for live prices."""
    import websockets, asyncio, json

    async def _run():
        uri = "wss://stream.binancefuture.com/stream?streams=btcusdt@bookTicker/btcusdt@aggTrade"
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    print("WebSocket connected")
                    async for msg in ws:
                        data = json.loads(msg)
                        stream = data.get("stream","")
                        d      = data.get("data", {})

                        if "bookTicker" in stream:
                            lob_buffer.append({
                                "bid_price": float(d.get("b",0)),
                                "bid_qty":   float(d.get("B",0)),
                                "ask_price": float(d.get("a",0)),
                                "ask_qty":   float(d.get("A",0)),
                                "ts": time.time(),
                            })
                        elif "aggTrade" in stream:
                            is_sell = d.get("m", False)
                            qty     = float(d.get("q",0))
                            trade_buffer.append({
                                "qty":    qty,
                                "is_sell": is_sell,
                                "price":  float(d.get("p",0)),
                            })
            except Exception as e:
                print(f"WebSocket error: {e} — reconnecting in 3s")
                await asyncio.sleep(3)

    asyncio.run(_run())

# ── Inference loop ────────────────────────────────────────────────────────────

def inference_loop():
    """Run agent prediction every INFERENCE_EVERY seconds."""
    import requests as req
    prev_price = None

    while True:
        time.sleep(INFERENCE_EVERY)
        try:
            # Get current price from Binance Futures (public, no auth needed)
            r = req.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol":"BTCUSDT"}, timeout=5
            )
            price = float(r.json()["price"])
            price_buffer.append(price)

            if prev_price is None:
                prev_price = price
                continue

            # Aggregate recent trades for TFI
            recent = list(trade_buffer)
            buy_vol  = sum(t["qty"] for t in recent if not t["is_sell"])
            sell_vol = sum(t["qty"] for t in recent if t["is_sell"])
            trade_buffer.clear()

            # Get last LOB snapshot
            lob = list(lob_buffer)[-1] if lob_buffer else {
                "bid_price": price*0.9999, "bid_qty":1.0,
                "ask_price": price*1.0001, "ask_qty":1.0,
            }

            # Compute features
            bar = compute_features(
                price, prev_price,
                lob["bid_price"], lob["bid_qty"],
                lob["ask_price"], lob["ask_qty"],
                buy_vol, sell_vol,
            )
            feature_buffer.append(bar)
            prev_price = price

            # Update unrealized PnL
            if portfolio["position"] == 1:
                btc = ORDER_SIZE_USD / portfolio["entry_price"]
                portfolio["unrealized_pnl"] = (price - portfolio["entry_price"]) * btc
            elif portfolio["position"] == -1:
                btc = ORDER_SIZE_USD / portfolio["entry_price"]
                portfolio["unrealized_pnl"] = (portfolio["entry_price"] - price) * btc
            else:
                portfolio["unrealized_pnl"] = 0.0

            portfolio["last_price"] = price
            equity = (INITIAL_BALANCE + portfolio["realized_pnl"]
                      + portfolio["unrealized_pnl"])
            portfolio["equity_history"].append(round(equity, 4))

            # Run agent prediction
            obs = build_observation()
            if obs is None or model is None:
                continue

            portfolio["bars_since_trade"] += 1

            # Enforce minimum holding period (skip cooldown check for demo speed)
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            action = int(action)
            action_names = {0:"HOLD", 1:"BUY", 2:"SELL"}

            # Execute paper trade
            if action == 1 and portfolio["position"] <= 0:
                if portfolio["position"] == -1:
                    _close(price, "CLOSE_SHORT")
                _open(price, 1)
            elif action == 2 and portfolio["position"] >= 0:
                if portfolio["position"] == 1:
                    _close(price, "CLOSE_LONG")
                _open(price, -1)

            portfolio["last_action"] = action_names[action]
            portfolio["action_history"].append({
                "time":   time.strftime("%H:%M:%S"),
                "action": action_names[action],
                "price":  price,
            })
            print(f"  {time.strftime('%H:%M:%S')}  price={price:.2f}  "
                  f"action={action_names[action]}  pos={portfolio['position']}  "
                  f"equity={equity:.2f}")

        except Exception as e:
            print(f"Inference error: {e}")

def _close(price, label):
    btc = ORDER_SIZE_USD / portfolio["entry_price"]
    if portfolio["position"] == 1:
        pnl = (price - portfolio["entry_price"]) * btc
    else:
        pnl = (portfolio["entry_price"] - price) * btc
    pnl -= FEE_RATE * ORDER_SIZE_USD
    portfolio["realized_pnl"] += pnl
    portfolio["trades"].append({
        "type": label, "price": price,
        "pnl": round(pnl, 4),
        "time": time.strftime("%H:%M:%S"),
    })
    portfolio["position"]    = 0
    portfolio["entry_price"] = 0.0
    portfolio["bars_since_trade"] = 0

def _open(price, direction):
    portfolio["position"]    = direction
    portfolio["entry_price"] = price
    portfolio["realized_pnl"] -= FEE_RATE * ORDER_SIZE_USD
    portfolio["trades"].append({
        "type": "BUY" if direction == 1 else "SELL",
        "price": price, "pnl": 0,
        "time": time.strftime("%H:%M:%S"),
    })

# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}

@app.get("/predict")
def predict():
    return {
        "action":    portfolio["last_action"],
        "price":     portfolio["last_price"],
        "position":  portfolio["position"],
        "equity":    round(INITIAL_BALANCE + portfolio["realized_pnl"]
                           + portfolio["unrealized_pnl"], 2),
        "pnl":       round(portfolio["realized_pnl"], 4),
    }

@app.get("/status")
def status():
    equity = INITIAL_BALANCE + portfolio["realized_pnl"] + portfolio["unrealized_pnl"]
    return {
        "equity":         round(equity, 2),
        "realized_pnl":   round(portfolio["realized_pnl"], 4),
        "unrealized_pnl": round(portfolio["unrealized_pnl"], 4),
        "position":       portfolio["position"],
        "last_action":    portfolio["last_action"],
        "last_price":     portfolio["last_price"],
        "n_trades":       len(portfolio["trades"]),
        "equity_history": portfolio["equity_history"][-60:],
        "trades":         portfolio["trades"][-10:],
        "actions":        portfolio["action_history"][-20:],
    }

@app.post("/reset")
def reset():
    portfolio.update({
        "balance":        INITIAL_BALANCE,
        "position":       0,
        "entry_price":    0.0,
        "realized_pnl":   0.0,
        "unrealized_pnl": 0.0,
        "trades":         [],
        "equity_history": [INITIAL_BALANCE],
        "action_history": [],
        "last_action":    "HOLD",
        "last_price":     0.0,
        "bars_since_trade": 0,
    })
    feature_buffer.clear()
    return {"status": "reset complete"}

# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start WebSocket in background thread
    ws_thread = threading.Thread(target=start_websocket, daemon=True)
    ws_thread.start()

    # Start inference loop in background thread
    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    # Start API server
    uvicorn.run(app, host="0.0.0.0", port=8000)