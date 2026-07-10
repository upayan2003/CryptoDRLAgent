"""
demo_dashboard.py
=================
Professional trading terminal dashboard for the Crypto DRL Agent demo.
Polls the FastAPI backend every 2 seconds for live updates.

Run: streamlit run demo_dashboard.py
"""

import time
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_URL     = "https://cryptodrlagent.onrender.com"
REFRESH_SEC = 2
INITIAL_BAL = 10_000.0

# ── PALETTE ─────────────────────────────────────────────────────────────
PURPLE  = "#7c3aed"
MAGENTA = "#c026d3"
PINK    = "#ec4899"
ORANGE  = "#ea580c"
GOLD    = "#d97706"
GREEN   = "#059669"
RED     = "#dc2626"
NAVY    = "#1e1b4b"
GRAY    = "#6b7280"
LGRAY   = "#e5e7eb"
BG      = "#faf7f2"
CARD    = "#ffffff"
TEXT    = "#1a1a2e"
SUBTLE  = "#f3f0eb"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Crypto DRL Agent — Live Demo",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── LIGHT MODE SUNSET CSS ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Global ─────────────────────────────────────── */
.stApp {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
.main .block-container {{
    padding: 1.2rem 2rem 2rem 2rem;
    max-width: 100%;
}}

/* ── Header ────────────────────────────────────── */
.dash-header {{
    background: linear-gradient(135deg, {PURPLE} 0%, {MAGENTA} 50%, {PINK} 100%);
    border-radius: 12px;
    padding: 16px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(124,58,237,0.25);
}}
.dash-title {{
    color: white;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 1px;
}}
.dash-sub {{
    color: rgba(255,255,255,0.75);
    font-size: 12px;
    margin-top: 3px;
}}

/* ── Metric cards ────────────────────────────── */
.mcard {{
    background: {CARD};
    border: 1.5px solid {LGRAY};
    border-radius: 12px;
    padding: 16px 20px 14px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 4px;
}}
.mcard-label {{
    color: {GRAY};
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}}
.mcard-value {{
    font-size: 22px;
    font-weight: 800;
    line-height: 1.2;
}}
.c-purple  {{ color: {PURPLE};  }}
.c-magenta {{ color: {MAGENTA}; }}
.c-pink    {{ color: {PINK};    }}
.c-orange  {{ color: {ORANGE};  }}
.c-gold    {{ color: {GOLD};    }}
.c-green   {{ color: {GREEN};   }}
.c-red     {{ color: {RED};     }}
.c-navy    {{ color: {NAVY};    }}
.c-gray    {{ color: {GRAY};    }}

/* ── Signal badge ────────────────────────────── */
.sig-buy {{
    background: #dcfce7; color: {GREEN};
    border: 1.5px solid {GREEN};
    padding: 8px 20px; border-radius: 8px;
    font-weight: 800; font-size: 16px;
    text-align: center; letter-spacing: 1px;
}}
.sig-sell {{
    background: #fee2e2; color: {RED};
    border: 1.5px solid {RED};
    padding: 8px 20px; border-radius: 8px;
    font-weight: 800; font-size: 16px;
    text-align: center; letter-spacing: 1px;
}}
.sig-hold {{
    background: {SUBTLE}; color: {GRAY};
    border: 1.5px solid {LGRAY};
    padding: 8px 20px; border-radius: 8px;
    font-weight: 800; font-size: 16px;
    text-align: center; letter-spacing: 1px;
}}

/* ── Control panel card ──────────────────────── */
.ctrl-card {{
    background: {CARD};
    border: 1.5px solid {LGRAY};
    border-radius: 12px;
    padding: 18px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 14px;
}}
.ctrl-title {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: {GRAY};
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid {LGRAY};
}}

/* ── Status dots ─────────────────────────────── */
.dot-green  {{ color: {GREEN};   }}
.dot-gold   {{ color: {GOLD};    }}
.dot-red    {{ color: {RED};     }}
.dot-purple {{ color: {PURPLE};  }}
.status-row {{
    font-size: 12px;
    color: {GRAY};
    margin-top: 10px;
    line-height: 2;
}}

/* ── Agent state badge ───────────────────────── */
.state-running {{ background:#f3e8ff; color:{PURPLE}; border:1.5px solid {PURPLE};
                  padding:4px 14px; border-radius:20px; font-size:11px; font-weight:700; }}
.state-paused  {{ background:#fff7ed; color:{ORANGE}; border:1.5px solid {ORANGE};
                  padding:4px 14px; border-radius:20px; font-size:11px; font-weight:700; }}
.state-stopped {{ background:#fee2e2; color:{RED}; border:1.5px solid {RED};
                  padding:4px 14px; border-radius:20px; font-size:11px; font-weight:700; }}

/* ── Market mode badge ───────────────────────── */
.mkt-auto   {{ background:#fef9c3; color:{GOLD}; border:1.5px solid {GOLD};
               padding:4px 14px; border-radius:20px; font-size:11px; font-weight:700; }}
.mkt-manual {{ background:#fce7f3; color:{PINK}; border:1.5px solid {PINK};
               padding:4px 14px; border-radius:20px; font-size:11px; font-weight:700; }}

/* ── Trade log ───────────────────────────────── */
.trade-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.trade-table th {{
    background: {SUBTLE};
    color: {GRAY};
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 14px;
    text-align: center;
    border-bottom: 1.5px solid {LGRAY};
}}
.trade-table td {{
    padding: 10px 14px;
    text-align: center;
    border-bottom: 1px solid {LGRAY};
    font-weight: 500;
}}
.trade-table tr:hover td {{ background: {SUBTLE}; }}
.pnl-pos {{ color: {GREEN}; font-weight: 700; }}
.pnl-neg {{ color: {RED};   font-weight: 700; }}
.pnl-neu {{ color: {GRAY};  font-weight: 600; }}
.type-buy  {{ color: {GREEN};  font-weight: 700; }}
.type-sell {{ color: {RED};    font-weight: 700; }}
.type-cls  {{ color: {ORANGE}; font-weight: 700; }}
.src-agent  {{ color: {PURPLE}; font-size: 10px; font-weight: 700;
               background:#f3e8ff; padding:2px 8px; border-radius:10px; }}
.src-manual {{ color: {PINK};   font-size: 10px; font-weight: 700;
               background:#fce7f3; padding:2px 8px; border-radius:10px; }}

/* ── Section headers ─────────────────────────── */
.section-header {{
    font-size: 13px;
    font-weight: 700;
    color: {NAVY};
    margin: 18px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ── Hide streamlit defaults ─────────────────── */
#MainMenu {{visibility:hidden;}}
footer     {{visibility:hidden;}}
header     {{visibility:hidden;}}
.stDeployButton {{display:none;}}
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "agent_state"  not in st.session_state: st.session_state.agent_state  = "running"
if "market_mode"  not in st.session_state: st.session_state.market_mode  = "auto"
if "sim_price"    not in st.session_state: st.session_state.sim_price     = None
if "last_refresh" not in st.session_state: st.session_state.last_refresh  = 0


# ── HELPERS ────────────────────────────────────────────────────────────────────
def fetch(ep, default=None):
    try:
        r = requests.get(f"{API_URL}{ep}", timeout=4)
        return r.json()
    except Exception:
        return default

def post(ep, payload=None):
    try:
        r = requests.post(f"{API_URL}{ep}", json=payload or {}, timeout=4)
        return r.json()
    except Exception:
        return {}

def pnl_class(v):
    try:
        v = float(v)
        return "c-green" if v > 0 else ("c-red" if v < 0 else "c-gray")
    except Exception:
        return "c-gray"

def sig_class(s):
    return {"BUY":"sig-buy","SELL":"sig-sell"}.get(s,"sig-hold")

def fmt_pnl(v):
    try:
        v = float(v)
        return f"+${v:.4f}" if v > 0 else f"${v:.4f}"
    except Exception:
        return str(v)

def type_class(t):
    t = str(t).upper()
    if "BUY"  in t: return "type-buy"
    if "SELL" in t: return "type-sell"
    return "type-cls"


# ── FETCH DATA ─────────────────────────────────────────────────────────────────
status = fetch("/status", default={})
health = fetch("/health", default={})
chart  = fetch("/chart",  default={})

price      = status.get("last_price",       0)
equity     = status.get("equity",           INITIAL_BAL)
pos        = status.get("position",         0)
rpnl       = status.get("realized_pnl",     0)
upnl       = status.get("unrealized_pnl",   0)
ret_pct    = status.get("total_return_pct", 0)
signal     = status.get("agent_signal",     "HOLD")
manual_ag  = status.get("manual_mode",      False)
ready      = status.get("bars_ready",       False)
buffered   = status.get("bars_buffered",    0)
needed     = status.get("bars_needed",      60)
entry_p    = status.get("entry_price",      0)
n_trades   = status.get("n_trades",         0)
eq_hist    = status.get("equity_history",   [INITIAL_BAL])
trades     = status.get("trades",           [])
actions    = status.get("actions",          [])
bars       = chart.get("bars",              [])

pos_str    = {0:"FLAT", 1:"▲ LONG", -1:"▼ SHORT"}.get(pos, "FLAT")
server_ok  = health.get("status") == "running"
model_ok   = health.get("model_loaded", False)

# Override price in manual market mode
display_price = (st.session_state.sim_price
                 if st.session_state.market_mode == "manual"
                    and st.session_state.sim_price
                 else price)


# ── HEADER ─────────────────────────────────────────────────────────────────────
state_badge = {
    "running": f'<span class="state-running">▶ RUNNING</span>',
    "paused":  f'<span class="state-paused">⏸ PAUSED</span>',
    "stopped": f'<span class="state-stopped">■ STOPPED</span>',
}.get(st.session_state.agent_state, "")

mkt_badge = (f'<span class="mkt-manual">🎮 MARKET: MANUAL</span>'
             if st.session_state.market_mode == "manual"
             else f'<span class="mkt-auto">📡 MARKET: LIVE</span>')

st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">CRYPTO DRL AGENT — LIVE DEMO</div>
    <div class="dash-sub">
      BTCUSDT Futures &nbsp;|&nbsp; PPO v5 &nbsp;|&nbsp;
      Binance {'Testnet' if True else 'Mainnet'} &nbsp;|&nbsp; $100 Paper Notional
    </div>
  </div>
  <div style="display:flex; gap:10px; align-items:center;">
    {state_badge} &nbsp; {mkt_badge}
  </div>
</div>
""", unsafe_allow_html=True)


# ── TOP METRICS ─────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7 = st.columns(7, gap="medium")

def mcard(col, label, value, css_class, sub=""):
    col.markdown(f"""
    <div class="mcard">
      <div class="mcard-label">{label}</div>
      <div class="mcard-value {css_class}">{value}</div>
      {'<div style="font-size:11px;color:#9ca3af;margin-top:3px;">'+sub+'</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

with c1: mcard(c1, "BTC PRICE",     f"${display_price:,.2f}", "c-navy")
with c2: mcard(c2, "EQUITY",        f"${equity:,.2f}",
               "c-green" if equity >= INITIAL_BAL else "c-red")
with c3:
    sign = "+" if ret_pct >= 0 else ""
    mcard(c3, "TOTAL RETURN", f"{sign}{ret_pct:.3f}%",
          "c-green" if ret_pct >= 0 else "c-red")
with c4:
    sign = "+" if rpnl >= 0 else ""
    mcard(c4, "REALIZED PnL", f"{sign}${rpnl:.4f}",
          "c-green" if rpnl >= 0 else "c-red")
with c5:
    sign = "+" if upnl >= 0 else ""
    mcard(c5, "UNREALIZED PnL", f"{sign}${upnl:.4f}",
          "c-green" if upnl >= 0 else "c-red",
          sub=f"Entry ${entry_p:,.2f}" if entry_p else "")
with c6:
    pos_css = "c-green" if pos==1 else ("c-red" if pos==-1 else "c-gray")
    mcard(c6, "POSITION", pos_str, pos_css)
with c7: mcard(c7, "TOTAL TRADES", str(n_trades), "c-purple")

st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)


# ── MAIN LAYOUT ──────────────────────────────────────────────────────────────────
chart_col, ctrl_col = st.columns([3.2, 1], gap="large")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART COLUMN
# ═══════════════════════════════════════════════════════════════════════════════
with chart_col:

    PLOTLY_LAYOUT = dict(
        paper_bgcolor = "white",
        plot_bgcolor  = "#fafafa",
        font          = dict(color=TEXT, size=11, family="Inter, sans-serif"),
        margin        = dict(l=10, r=10, t=40, b=10),
        height        = 430,
        xaxis         = dict(gridcolor="#ede9e0", showgrid=True,
                             tickfont=dict(size=10)),
        yaxis         = dict(gridcolor="#ede9e0", showgrid=True,
                             tickfont=dict(size=10)),
    )

    tab_price, tab_equity, tab_signals = st.tabs(
        ["📈  Price Chart", "💰  Equity Curve", "📡  Signal Feed"]
    )

    # ── Price chart ────────────────────────────────────────────────────────────
    with tab_price:
        if bars:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.78, 0.22], vertical_spacing=0.02,
            )
            times  = [b["time"]  for b in bars]
            opens  = [b["open"]  for b in bars]
            highs  = [b["high"]  for b in bars]
            lows   = [b["low"]   for b in bars]
            closes = [b["close"] for b in bars]
            vols   = [b["vol"]   for b in bars]

            fig.add_trace(go.Candlestick(
                x=times, open=opens, high=highs, low=lows, close=closes,
                name="BTCUSDT",
                increasing_line_color=GREEN,  increasing_fillcolor="#dcfce7",
                decreasing_line_color=RED,    decreasing_fillcolor="#fee2e2",
            ), row=1, col=1)

            # Trade markers on chart
            for t in trades[-30:]:
                c = GREEN if "BUY" in str(t.get("type","")).upper() else RED
                s = "triangle-up" if "BUY" in str(t.get("type","")).upper() else "triangle-down"
                fig.add_trace(go.Scatter(
                    x=[t.get("time","")], y=[t.get("price",0)],
                    mode="markers", showlegend=False,
                    marker=dict(symbol=s, size=13, color=c,
                                line=dict(color="white", width=1.5)),
                    hovertext=f"{t.get('type','')}  ${t.get('price',0):.2f}  "
                              f"PnL:{fmt_pnl(t.get('pnl',0))}",
                ), row=1, col=1)

            # Volume
            bar_colors = [GREEN if c >= o else RED
                          for c, o in zip(closes, opens)]
            fig.add_trace(go.Bar(
                x=times, y=vols, name="Volume",
                marker_color=bar_colors, opacity=0.6, showlegend=False,
            ), row=2, col=1)

            layout = dict(**PLOTLY_LAYOUT)
            layout["title"] = dict(
                text=f"BTCUSDT  ·  5-second bars  ·  "
                     f"<span style='color:{PURPLE}'>Last ${display_price:,.2f}</span>",
                font=dict(size=13, color=NAVY),
            )
            layout["xaxis_rangeslider_visible"] = False
            layout["xaxis2"] = dict(gridcolor="#ede9e0")
            layout["yaxis2"] = dict(gridcolor="#ede9e0", tickfont=dict(size=9))
            fig.update_layout(**layout)
            st.plotly_chart(fig, width='stretch')
        else:
            pct = int(buffered / max(needed, 1) * 100)
            st.markdown(f"""
            <div style='background:{CARD};border:1.5px solid {LGRAY};border-radius:12px;
                 padding:60px 40px;text-align:center;'>
              <div style='font-size:32px;margin-bottom:12px;'>⏳</div>
              <div style='font-size:16px;font-weight:700;color:{NAVY};'>
                Warming up feature buffer...</div>
              <div style='font-size:13px;color:{GRAY};margin:8px 0 20px 0;'>
                {buffered} / {needed} bars collected &nbsp;·&nbsp; ~{(needed-buffered)*5}s remaining
              </div>
              <div style='background:{LGRAY};border-radius:20px;height:8px;
                   width:80%;margin:0 auto;'>
                <div style='background:linear-gradient(90deg,{PURPLE},{PINK});
                     height:8px;border-radius:20px;width:{pct}%;
                     transition:width 0.5s;'></div>
              </div>
              <div style='font-size:11px;color:{GRAY};margin-top:10px;'>{pct}%</div>
            </div>""", unsafe_allow_html=True)

    # ── Equity curve ───────────────────────────────────────────────────────────
    with tab_equity:
        if len(eq_hist) > 1:
            fig2  = go.Figure()
            x     = list(range(len(eq_hist)))
            above = [e if e >= INITIAL_BAL else None for e in eq_hist]
            below = [e if e <  INITIAL_BAL else None for e in eq_hist]

            fig2.add_trace(go.Scatter(
                x=x, y=above, mode="lines", name="Profit zone",
                line=dict(color=GREEN, width=2.5),
                fill="tonexty", fillcolor="rgba(5,150,105,0.08)",
            ))
            fig2.add_trace(go.Scatter(
                x=x, y=below, mode="lines", name="Loss zone",
                line=dict(color=RED, width=2.5),
                fill="tonexty", fillcolor="rgba(220,38,38,0.08)",
            ))
            fig2.add_trace(go.Scatter(
                x=x, y=eq_hist, mode="lines",
                line=dict(color=PURPLE, width=2),
                name="Equity", showlegend=True,
            ))
            peak = max(eq_hist)
            if peak > INITIAL_BAL:
                fig2.add_hline(
                    y=peak, line_dash="dot", line_color=GOLD, line_width=1.2,
                    annotation_text=f"Peak ${peak:,.2f}",
                    annotation_font=dict(color=GOLD, size=10),
                )
            layout2 = dict(**PLOTLY_LAYOUT)
            layout2["title"] = dict(
                text="Portfolio Equity Curve  ·  $10,000 Starting Capital",
                font=dict(size=13, color=NAVY),
            )
            layout2["xaxis"]["title"] = "Bars (5-sec)"
            layout2["yaxis"]["title"] = "Value (USD)"
            layout2["legend"]  = dict(
                orientation="h", y=1.08, x=0,
                bgcolor="rgba(0,0,0,0)",
            )
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("⏳ Collecting equity data...")

    # ── Signal feed ────────────────────────────────────────────────────────────
    with tab_signals:
        if actions:
            # Build coloured HTML table
            rows_html = ""
            for a in reversed(actions[-50:]):
                act  = str(a.get("action",""))
                src  = str(a.get("source","agent"))
                acls = "type-buy" if act=="BUY" else ("type-sell" if act=="SELL" else "type-cls")
                scls = "src-agent" if src=="agent" else "src-manual"
                rows_html += f"""
                <tr>
                  <td>{a.get('time','')}</td>
                  <td class='{acls}'>{act}</td>
                  <td>${a.get('price',0):,.2f}</td>
                  <td>{a.get('tfi',0):+.4f}</td>
                  <td>{a.get('ofi',0):+.4f}</td>
                  <td><span class='{scls}'>{src.upper()}</span></td>
                </tr>"""
            st.markdown(f"""
            <div style='max-height:380px;overflow-y:auto;
                        border:1.5px solid {LGRAY};border-radius:10px;'>
            <table class='trade-table'>
              <thead><tr>
                <th>TIME</th><th>ACTION</th><th>PRICE</th>
                <th>TFI</th><th>OFI</th><th>SOURCE</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table></div>""", unsafe_allow_html=True)
        else:
            st.info("⏳ No signals yet — buffer filling...")
    
    # ── TRADE LOG ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='section-header'>
    <span style='background:linear-gradient(135deg,{PURPLE},{PINK});
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                font-size:14px;'>📜</span>
    Trade Execution Log
    </div>""", unsafe_allow_html=True)

    if trades:
        rows_html = ""
        for t in reversed(trades):
            pnl_v   = float(t.get("pnl", 0))
            pnl_str = fmt_pnl(pnl_v) if pnl_v != 0 else "—"
            pcls    = "pnl-pos" if pnl_v > 0 else ("pnl-neg" if pnl_v < 0 else "pnl-neu")
            typ     = str(t.get("type",""))
            tcls    = type_class(typ)
            src     = str(t.get("source","agent"))
            scls    = "src-agent" if src == "agent" else "src-manual"
            pos_v   = t.get("position", 0)
            pos_lbl = {1:"LONG", -1:"SHORT", 0:"FLAT"}.get(pos_v, str(pos_v))
            rows_html += f"""
            <tr>
            <td>{t.get('time','')}</td>
            <td class='{tcls}'>{typ}</td>
            <td>${t.get('price',0):,.2f}</td>
            <td>{pos_lbl}</td>
            <td class='{pcls}'>{pnl_str}</td>
            <td><span class='{scls}'>{src.upper()}</span></td>
            </tr>"""

        st.markdown(f"""
        <div style='background:{CARD};border:1.5px solid {LGRAY};border-radius:12px;
                    overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
        <div style='max-height:480px;overflow-y:auto;'>
            <table class='trade-table'>
            <thead><tr>
                <th>TIME</th>
                <th>TYPE</th>
                <th>PRICE</th>
                <th>POSITION</th>
                <th>PnL (USD)</th>
                <th>SOURCE</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
            </table>
        </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:{CARD};border:1.5px solid {LGRAY};border-radius:12px;
                    padding:40px;text-align:center;color:{GRAY};font-size:13px;'>
        No trades yet — agent is warming up or market is flat.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL COLUMN
# ═══════════════════════════════════════════════════════════════════════════════
with ctrl_col:

    # ── Agent signal ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='ctrl-card'>
      <div class='ctrl-title'>🤖 Agent Signal</div>
      <div class='{sig_class(signal)}'>{signal}</div>
      <div style='margin-top:10px;font-size:11px;color:{GRAY};text-align:center;'>
        {"✅ Ready — " + str(buffered) + " bars" if ready
         else f"⏳ {buffered}/{needed} bars"}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Agent controls: Start / Pause / Stop ──────────────────────────────────
    st.markdown(f"""
    <div class='ctrl-card'>
      <div class='ctrl-title'>⚙️ Agent Controls</div>""",
    unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        start_dis = st.session_state.agent_state == "running"
        if st.button("▶", width='stretch',
                     help="Start agent", disabled=start_dis):
            post("/agent/pause", {"paused": False})
            st.session_state.agent_state = "running"
            st.rerun()
    with b2:
        pause_dis = st.session_state.agent_state in ("paused","stopped")
        if st.button("⏸", width='stretch',
                     help="Pause agent", disabled=pause_dis):
            post("/agent/pause", {"paused": True})
            st.session_state.agent_state = "paused"
            st.rerun()
    with b3:
        if st.button("■", width='stretch',
                     help="Stop and reset"):
            post("/agent/stop")
            post("/reset")
            st.session_state.agent_state = "stopped"
            st.rerun()

    state_label = {
        "running":"🟢 Running",
        "paused": "🟡 Paused",
        "stopped":"🔴 Stopped",
    }.get(st.session_state.agent_state,"")
    st.markdown(f"""
      <div style='text-align:center;font-size:11px;
                  color:{GRAY};margin:8px 0 0 0;'>{state_label}</div>
    </div>""", unsafe_allow_html=True)

    # ── Order panel ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='ctrl-card'>
      <div class='ctrl-title'>📋 Manual Orders</div>
      <div style='font-size:11px;color:{GRAY};margin-bottom:10px;'>
        Override agent with your own trades
      </div>
      <div style='font-size:12px;margin-bottom:10px;'>
        <span style='color:{GRAY};'>Position: </span>
        <span style='color:{"#059669" if pos==1 else ("#dc2626" if pos==-1 else GRAY)};
                     font-weight:700;'>
          {pos_str}{f" @ ${entry_p:,.2f}" if entry_p else ""}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    ob1, ob2 = st.columns(2)
    with ob1:
        if st.button("▲  BUY", width='stretch',
                     type="primary" if pos <= 0 else "secondary"):
            r = post("/order", {"side": "BUY"})
            st.success(r.get("status","Done"))
            time.sleep(0.4); st.rerun()
    with ob2:
        if st.button("▼  SELL", width='stretch'):
            r = post("/order", {"side": "SELL"})
            st.error(r.get("status","Done"))
            time.sleep(0.4); st.rerun()

    if st.button("✕  CLOSE POSITION", width='stretch',
                 disabled=(pos == 0)):
        r = post("/order", {"side": "CLOSE"})
        pv = float(r.get("pnl", 0))
        (st.success if pv >= 0 else st.error)(
            f"Closed — PnL: {fmt_pnl(pv)}"
        )
        time.sleep(0.4); st.rerun()

    st.markdown("<div style='margin:6px 0;'></div>", unsafe_allow_html=True)

    # ── Market control ──────────────────────────────────────────────────────────
    mode_toggle = ("🌐 Switch to LIVE Market"
                   if st.session_state.market_mode == "manual"
                   else "🎮 Switch to MANUAL Market")
    if st.button(mode_toggle, width='stretch'):
        st.session_state.market_mode = (
            "auto" if st.session_state.market_mode == "manual" else "manual"
        )
        if st.session_state.market_mode == "manual":
            st.session_state.sim_price = price or 100_000.0
        st.rerun()

    if st.session_state.market_mode == "manual":
        st.markdown(f"""
        <div class='ctrl-card' style='border-color:{PINK};'>
          <div class='ctrl-title' style='color:{PINK};'>🎮 Market Simulator</div>
          <div style='font-size:11px;color:{GRAY};margin-bottom:10px;'>
            Simulate price moves — watch agent react in real-time
          </div>
          <div style='font-size:20px;font-weight:800;color:{NAVY};
                      text-align:center;margin-bottom:12px;'>
            ${st.session_state.sim_price:,.2f}
          </div>""", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            if st.button("🔺 +1%",  width='stretch'):
                st.session_state.sim_price *= 1.01
                post("/market/simulate", {"delta_pct": 1.0})
                st.rerun()
            if st.button("▲ +0.1%", width='stretch'):
                st.session_state.sim_price *= 1.001
                post("/market/simulate", {"delta_pct": 0.1})
                st.rerun()
            if st.button("↑ +0.01%",width='stretch'):
                st.session_state.sim_price *= 1.0001
                post("/market/simulate", {"delta_pct": 0.01})
                st.rerun()
        with m2:
            if st.button("🔻 -1%",  width='stretch'):
                st.session_state.sim_price *= 0.99
                post("/market/simulate", {"delta_pct": -1.0})
                st.rerun()
            if st.button("▼ -0.1%", width='stretch'):
                st.session_state.sim_price *= 0.999
                post("/market/simulate", {"delta_pct": -0.1})
                st.rerun()
            if st.button("↓ -0.01%",width='stretch'):
                st.session_state.sim_price *= 0.9999
                post("/market/simulate", {"delta_pct": -0.01})
                st.rerun()

        # Custom price input
        new_p = st.number_input(
            "Set exact price ($)",
            value=float(st.session_state.sim_price),
            step=100.0, format="%.2f",
            label_visibility="visible",
        )
        if st.button("📌 Set Price", width='stretch'):
            st.session_state.sim_price = new_p
            post("/market/simulate", {"set_price": new_p})
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Live metrics ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='ctrl-card'>
      <div class='ctrl-title'>📊 Live Metrics</div>""",
    unsafe_allow_html=True)

    dd   = (min(eq_hist) / INITIAL_BAL - 1) * 100 if eq_hist else 0
    wins = [t for t in trades if float(t.get("pnl",0)) > 0]
    gw   = sum(float(t.get("pnl",0)) for t in wins)
    gl   = abs(sum(float(t.get("pnl",0)) for t in trades
                   if float(t.get("pnl",0)) < 0))
    pf   = gw / max(gl, 1e-9)
    wr   = len(wins) / max(n_trades, 1) * 100

    rows = [
        ("Max Drawdown",  f"{dd:.3f}%",  "c-red"   if dd < 0  else "c-gray"),
        ("Win Rate",      f"{wr:.1f}%",  "c-green"  if wr>50  else "c-orange"),
        ("Profit Factor", f"{pf:.3f}",   "c-green"  if pf>1   else "c-red"),
        ("Trades",        str(n_trades), "c-purple"),
    ]
    table_html = ""
    for lbl, val, css in rows:
        table_html += f"""
        <div style='display:flex;justify-content:space-between;
                    padding:8px 0;border-bottom:1px solid {LGRAY};
                    font-size:12px;'>
          <span style='color:{GRAY};'>{lbl}</span>
          <span class='{css}' style='font-weight:700;'>{val}</span>
        </div>"""
    st.markdown(table_html + "</div>", unsafe_allow_html=True)

    # ── Server status ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='font-size:11px;color:{GRAY};margin-top:6px;line-height:1.8;'>
      {"🟢" if server_ok else "🔴"} Server &nbsp;
      {"🟢" if model_ok  else "🔴"} Model &nbsp;
      {"🟢" if ready     else "🟡"} Features<br>
      <span style='color:{LGRAY};font-size:10px;'>
        {time.strftime("%H:%M:%S")} UTC — refresh {REFRESH_SEC}s
      </span>
    </div>""", unsafe_allow_html=True)

    if st.button("🔄 Reset Demo", width='stretch'):
        post("/reset")
        st.session_state.sim_price    = None
        st.session_state.agent_state  = "running"
        st.session_state.market_mode  = "auto"
        st.success("Reset!")
        time.sleep(0.4); st.rerun()

# ── AUTO REFRESH ───────────────────────────────────────────────────────────────
if st.session_state.agent_state != "stopped":
    time.sleep(REFRESH_SEC)
    st.rerun()
