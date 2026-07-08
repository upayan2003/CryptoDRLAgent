"""
demo_api.py
===========
FastAPI backend for the Crypto RL Agent live demo.
Supports both autonomous agent trading and manual demo orders.

Endpoints:
  GET  /health          - server + model status
  GET  /status          - full portfolio + chart data
  GET  /chart           - OHLCV bars for candlestick chart
  POST /order           - place a manual demo order {side: "BUY"|"SELL"|"CLOSE"}
  POST /reset           - reset entire demo portfolio
  GET  /agent/signal    - what the agent would do right now

Deploy to Render.com:
  Start command: uvicorn demo_api:app --host 0.0.0.0 --port $PORT
"""

import os, time, threading
import numpy as np
from collections import deque
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, requests as req
from stable_baselines3 import PPO

# ── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Crypto DRL Agent API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_PATH      = "best_model.zip"
OBS_WINDOW      = 60
INITIAL_BALANCE = 10_000.0
ORDER_SIZE_USD  = 100.0
FEE_RATE        = 0.0004
POLL_EVERY      = 5          # seconds between REST polls
EPS             = 1e-9

# Binance Futures public REST (no auth needed)
BINANCE_BASE    = "https://fapi.binance.com/fapi/v1"
SYMBOL          = "BTCUSDT"

# ── SHARED STATE ───────────────────────────────────────────────────────────────
portfolio = {
    "balance":         INITIAL_BALANCE,
    "position":        0,
    "entry_price":     0.0,
    "realized_pnl":    0.0,
    "unrealized_pnl":  0.0,
    "trades":          [],
    "equity_history":  [INITIAL_BALANCE],
    "action_history":  [],
    "last_action":     "HOLD",
    "last_price":      0.0,
    "agent_signal":    "HOLD",
    "manual_mode":     False,
    "bars_since_trade":0,
}

feature_buffer = deque(maxlen=OBS_WINDOW + 10)
ohlcv_bars     = deque(maxlen=200)
prev_lob       = None          # previous bookTicker for OFI delta
lob_history    = deque(maxlen=30)
model          = None
_running       = False


# ── MODEL LOAD ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global model
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH)
        print(f"[API] Model loaded ✓")
    else:
        print(f"[API] WARNING: {MODEL_PATH} not found")
    # Start background polling thread here — NOT in __main__
    # This is the critical fix: Render runs uvicorn directly so __main__
    # never executes. startup() always runs regardless.
    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    print("[API] Background polling thread started ✓")


# ── REST POLLING ───────────────────────────────────────────────────────────────

def _fetch_price():
    """Fetch current mark price via REST."""
    try:
        r = req.get(f"{BINANCE_BASE}/ticker/price",
                    params={"symbol": SYMBOL}, timeout=4)
        return float(r.json()["price"])
    except Exception:
        return None

def _fetch_book():
    """Fetch best bid/ask via REST bookTicker endpoint."""
    try:
        r = req.get(f"{BINANCE_BASE}/ticker/bookTicker",
                    params={"symbol": SYMBOL}, timeout=4)
        d = r.json()
        return {
            "bid_price": float(d["bidPrice"]),
            "bid_qty":   float(d["bidQty"]),
            "ask_price": float(d["askPrice"]),
            "ask_qty":   float(d["askQty"]),
        }
    except Exception:
        return None

def _fetch_recent_trades():
    """Fetch last 50 aggregate trades to compute buy/sell volume."""
    try:
        r = req.get(f"{BINANCE_BASE}/aggTrades",
                    params={"symbol": SYMBOL, "limit": 50}, timeout=4)
        trades = r.json()
        buy_vol  = sum(float(t["q"]) for t in trades if not t["m"])
        sell_vol = sum(float(t["q"]) for t in trades if t["m"])
        return buy_vol, sell_vol
    except Exception:
        return 0.0, 0.0


# ── FEATURE COMPUTATION ────────────────────────────────────────────────────────

def _compute_ofi(current_lob, prev_lob_snap):
    """OFI from two consecutive REST bookTicker snapshots."""
    if prev_lob_snap is None:
        return 0.0
    e_bid = 0.0
    if current_lob["bid_price"] > prev_lob_snap["bid_price"]:
        e_bid = current_lob["bid_qty"]
    elif current_lob["bid_price"] == prev_lob_snap["bid_price"]:
        e_bid = (current_lob["bid_qty"]
                 if current_lob["bid_qty"] > prev_lob_snap["bid_qty"]
                 else -prev_lob_snap["bid_qty"])
    else:
        e_bid = -current_lob["bid_qty"]

    e_ask = 0.0
    if current_lob["ask_price"] < prev_lob_snap["ask_price"]:
        e_ask = -current_lob["ask_qty"]
    elif current_lob["ask_price"] == prev_lob_snap["ask_price"]:
        e_ask = (-current_lob["ask_qty"]
                 if current_lob["ask_qty"] < prev_lob_snap["ask_qty"]
                 else prev_lob_snap["ask_qty"])
    else:
        e_ask = current_lob["ask_qty"]

    ofi = e_bid - e_ask
    total_depth = (current_lob["bid_qty"] + current_lob["ask_qty"] +
                   prev_lob_snap["bid_qty"] + prev_lob_snap["ask_qty"])
    return float(np.clip(ofi / max(total_depth, EPS), -1.0, 1.0))

def _compute_bar(price, prev_price, lob, buy_vol, sell_vol):
    """Compute all 8 microstructure features for one 5-second bar."""
    buf        = list(feature_buffer)
    log_ret    = float(np.log(price / max(prev_price, EPS)))
    spread     = lob["ask_price"] - lob["bid_price"]
    mid        = (lob["bid_price"] + lob["ask_price"]) / 2.0
    rel_spread = spread / max(mid, EPS)
    depth_imb  = ((lob["bid_qty"] - lob["ask_qty"]) /
                  max(lob["bid_qty"] + lob["ask_qty"], EPS))
    tfi        = (buy_vol - sell_vol) / max(buy_vol + sell_vol, EPS)
    vol_roll   = (float(np.std([b["log_ret"] for b in buf[-20:]] + [log_ret]))
                  if len(buf) > 2 else 0.0)
    prices     = [b["price"] for b in buf[-20:]] + [price]
    ret_5      = (float(np.log(prices[-1] / max(prices[-6],  EPS)))
                  if len(prices) >= 6  else 0.0)
    ret_20     = (float(np.log(prices[-1] / max(prices[-21], EPS)))
                  if len(prices) >= 21 else 0.0)
    vwap       = float(np.mean([b["price"] for b in buf[-100:]] + [price]))
    pvwap      = (price - vwap) / max(vwap, EPS)
    ofi_norm   = _compute_ofi(lob, prev_lob)
    return {
        "price": price, "log_ret": log_ret,
        "tfi": tfi, "ofi_norm": ofi_norm,
        "vol_rolling": vol_roll, "rel_spread": rel_spread,
        "depth_imb": depth_imb, "ret_5": ret_5,
        "ret_20": ret_20, "price_vs_vwap": pvwap,
    }

def _build_obs():
    if len(feature_buffer) < OBS_WINDOW:
        return None
    window    = list(feature_buffer)[-OBS_WINDOW:]
    feat_cols = ["tfi","ofi_norm","vol_rolling","rel_spread",
                 "depth_imb","ret_5","ret_20","price_vs_vwap"]
    mkt  = np.array([[b[c] for c in feat_cols] for b in window], dtype=np.float32)
    port = np.full(
        (OBS_WINDOW, 3),
        [float(portfolio["position"]),
         float(portfolio["unrealized_pnl"] / INITIAL_BALANCE),
         0.0],
        dtype=np.float32,
    )
    return np.concatenate([mkt, port], axis=1).flatten()


# ── TRADE HELPERS ──────────────────────────────────────────────────────────────

def _close(price, label, source="agent"):
    if portfolio["position"] == 0:
        return
    btc = ORDER_SIZE_USD / portfolio["entry_price"]
    pnl = ((price - portfolio["entry_price"]) * btc
           if portfolio["position"] == 1
           else (portfolio["entry_price"] - price) * btc)
    pnl -= FEE_RATE * ORDER_SIZE_USD
    portfolio["realized_pnl"]  += pnl
    portfolio["unrealized_pnl"] = 0.0
    portfolio["trades"].append({
        "time":     datetime.utcnow().strftime("%H:%M:%S"),
        "type":     label, "source": source,
        "price":    round(price, 2),
        "pnl":      round(pnl, 4),
        "position": portfolio["position"],
    })
    portfolio["position"]    = 0
    portfolio["entry_price"] = 0.0

def _open(price, direction, source="agent"):
    portfolio["position"]     = direction
    portfolio["entry_price"]  = price
    portfolio["realized_pnl"] -= FEE_RATE * ORDER_SIZE_USD
    portfolio["trades"].append({
        "time":     datetime.utcnow().strftime("%H:%M:%S"),
        "type":     "BUY" if direction == 1 else "SELL",
        "source":   source,
        "price":    round(price, 2),
        "pnl":      0,
        "position": direction,
    })

def _update_unrealized(price):
    if portfolio["position"] == 0:
        portfolio["unrealized_pnl"] = 0.0
    else:
        btc = ORDER_SIZE_USD / portfolio["entry_price"]
        portfolio["unrealized_pnl"] = (
            (price - portfolio["entry_price"]) * btc
            if portfolio["position"] == 1
            else (portfolio["entry_price"] - price) * btc
        )


# ── MAIN POLLING LOOP ──────────────────────────────────────────────────────────

def _poll_loop():
    """
    Runs every POLL_EVERY seconds in a background thread.
    Fetches price + bookTicker + recent trades via REST,
    computes features, and runs agent inference.
    Started by startup() — always runs on Render.
    """
    global prev_lob
    prev_price = None
    bar_prices = []
    bar_start  = time.time()

    print("[POLL] Loop started")

    while True:
        time.sleep(POLL_EVERY)
        try:
            price    = _fetch_price()
            lob      = _fetch_book()
            buy_vol, sell_vol = _fetch_recent_trades()

            if price is None or lob is None:
                print("[POLL] Fetch failed — retrying next cycle")
                continue

            bar_prices.append(price)

            # Build OHLCV bar every POLL_EVERY seconds
            ohlcv_bars.append({
                "time":  datetime.utcnow().strftime("%H:%M:%S"),
                "open":  bar_prices[0] if bar_prices else price,
                "high":  max(bar_prices) if bar_prices else price,
                "low":   min(bar_prices) if bar_prices else price,
                "close": price,
                "vol":   len(bar_prices),
            })
            bar_prices = []

            # Skip first iteration — need prev_price for log return
            if prev_price is None:
                prev_price = price
                prev_lob   = lob
                print(f"[POLL] First price: {price:.2f} — warming up...")
                continue

            # Compute features and add to buffer
            bar = _compute_bar(price, prev_price, lob, buy_vol, sell_vol)
            feature_buffer.append(bar)
            lob_history.append(lob)
            prev_price = price
            prev_lob   = lob

            n_buf = len(feature_buffer)
            print(f"[POLL] price={price:.2f}  bars={n_buf}/{OBS_WINDOW}  "
                  f"tfi={bar['tfi']:+.3f}  ofi={bar['ofi_norm']:+.3f}")

            # Update portfolio
            _update_unrealized(price)
            portfolio["last_price"] = price
            equity = (INITIAL_BALANCE
                      + portfolio["realized_pnl"]
                      + portfolio["unrealized_pnl"])
            portfolio["equity_history"].append(round(equity, 4))

            # Agent inference — only when buffer is full
            if n_buf < OBS_WINDOW or model is None:
                continue

            obs    = _build_obs()
            if obs is None:
                continue

            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            action    = int(action)
            names     = {0:"HOLD", 1:"BUY", 2:"SELL"}
            portfolio["agent_signal"] = names[action]

            # Execute if in agent-auto mode
            if not portfolio["manual_mode"]:
                if action == 1 and portfolio["position"] <= 0:
                    if portfolio["position"] == -1:
                        _close(price, "CLOSE_SHORT", source="agent")
                    _open(price, 1, source="agent")
                elif action == 2 and portfolio["position"] >= 0:
                    if portfolio["position"] == 1:
                        _close(price, "CLOSE_LONG", source="agent")
                    _open(price, -1, source="agent")
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


# ── ENDPOINTS ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    n = len(feature_buffer)
    return {
        "status":       "running",
        "model_loaded": model is not None,
        "bars_buffered":n,
        "bars_needed":  OBS_WINDOW,
        "ready":        n >= OBS_WINDOW,
    }

@app.get("/status")
def status():
    equity = (INITIAL_BALANCE
              + portfolio["realized_pnl"]
              + portfolio["unrealized_pnl"])
    n = len(feature_buffer)
    return {
        "equity":           round(equity, 2),
        "realized_pnl":     round(portfolio["realized_pnl"],   4),
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
        "bars_ready":       n >= OBS_WINDOW,
        "bars_buffered":    n,
        "bars_needed":      OBS_WINDOW,
    }

@app.get("/chart")
def chart():
    return {"bars": list(ohlcv_bars), "last_price": portfolio["last_price"]}

class OrderRequest(BaseModel):
    side: str

@app.post("/order")
def manual_order(order: OrderRequest):
    price = portfolio["last_price"]
    if price == 0:
        return {"error": "No price yet — wait ~10 seconds"}
    portfolio["manual_mode"] = True
    side = order.side.upper()
    if side == "BUY":
        if portfolio["position"] == -1:
            _close(price, "CLOSE_SHORT", source="manual")
        if portfolio["position"] == 0:
            _open(price, 1, source="manual")
            portfolio["last_action"] = "BUY"
            return {"status": "BUY executed", "price": round(price,2)}
        return {"status": "Already long"}
    elif side == "SELL":
        if portfolio["position"] == 1:
            _close(price, "CLOSE_LONG", source="manual")
        if portfolio["position"] == 0:
            _open(price, -1, source="manual")
            portfolio["last_action"] = "SELL"
            return {"status": "SELL executed", "price": round(price,2)}
        return {"status": "Already short"}
    elif side == "CLOSE":
        if portfolio["position"] == 0:
            return {"status": "No open position"}
        label = "CLOSE_LONG" if portfolio["position"] == 1 else "CLOSE_SHORT"
        _close(price, label, source="manual")
        portfolio["last_action"] = "CLOSE"
        return {"status": f"{label} executed",
                "price": round(price,2),
                "pnl":   round(portfolio["realized_pnl"],4)}
    return {"error": f"Unknown side: {side}"}

@app.post("/toggle_mode")
def toggle_mode():
    portfolio["manual_mode"] = not portfolio["manual_mode"]
    return {"mode": "MANUAL" if portfolio["manual_mode"] else "AGENT AUTO",
            "manual_mode": portfolio["manual_mode"]}

@app.post("/reset")
def reset():
    portfolio.update({
        "balance": INITIAL_BALANCE, "position": 0, "entry_price": 0.0,
        "realized_pnl": 0.0, "unrealized_pnl": 0.0,
        "trades": [], "equity_history": [INITIAL_BALANCE],
        "action_history": [], "last_action": "HOLD",
        "last_price": 0.0, "agent_signal": "HOLD",
        "manual_mode": False, "bars_since_trade": 0,
    })
    feature_buffer.clear()
    ohlcv_bars.clear()
    return {"status": "reset complete"}

# ── LOCAL RUN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ── ADDITIONAL ENDPOINTS FOR REACT DASHBOARD ──────────────────────────────────

class PauseRequest(BaseModel):
    paused: bool

@app.post("/agent/pause")
def agent_pause(req: PauseRequest):
    """Pause or resume agent inference loop."""
    portfolio["agent_paused"] = req.paused
    return {"paused": req.paused, "status": "paused" if req.paused else "running"}

@app.post("/agent/stop")
def agent_stop():
    """Stop agent and reset portfolio."""
    portfolio["agent_paused"] = True
    return {"status": "stopped"}

class SimRequest(BaseModel):
    delta_pct: float = 0.0
    set_price: float = None

@app.post("/market/simulate")
def market_simulate(req: SimRequest):
    """
    Simulate a price move for demo/testing.
    Injects a synthetic bar into the feature buffer so the agent
    reacts to the simulated price without real market data.
    delta_pct: percentage change e.g. 1.0 = +1%, -0.5 = -0.5%
    set_price: override to exact price value
    """
    current = portfolio["last_price"] or 100_000.0

    if req.set_price:
        new_price = req.set_price
    else:
        new_price = current * (1 + req.delta_pct / 100.0)

    # Build a synthetic bar with the simulated price
    buy_vol  = 1.0 if req.delta_pct > 0 else 0.1
    sell_vol = 0.1 if req.delta_pct > 0 else 1.0

    prev_p = portfolio["last_price"] or new_price
    lob = list(lob_buffer)[-1] if lob_buffer else {
        "bid_price": new_price * 0.9999,
        "bid_qty":   1.0,
        "ask_price": new_price * 1.0001,
        "ask_qty":   1.0,
    }

    bar = _compute_bar(new_price, prev_p, lob, buy_vol, sell_vol)
    feature_buffer.append(bar)

    ohlcv_bars.append({
        "time":  datetime.utcnow().strftime("%H:%M:%S"),
        "open":  prev_p,
        "high":  max(prev_p, new_price),
        "low":   min(prev_p, new_price),
        "close": new_price,
        "vol":   10,
    })

    # Update portfolio price
    portfolio["last_price"] = new_price
    update_unrealized(new_price)
    equity = INITIAL_BALANCE + portfolio["realized_pnl"] + portfolio["unrealized_pnl"]
    portfolio["equity_history"].append(round(equity, 4))

    return {
        "status":    "simulated",
        "new_price": round(new_price, 2),
        "delta_pct": req.delta_pct,
    }
