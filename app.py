"""
4H Sequence Model (H1) — Backtester, Performance Dashboard & Trade Journal
==========================================================================

Implements the mechanical strategy described in the specification:
  - 3-candle FVG structure (C1, C2, C3)
  - EBP reversal / liquidity sweep on C4
  - Dynamic limit entry on C5 (scale matrix)
  - Fixed 2RR, SL at C4 extreme
  - One trade at a time (new setups ignored while a position is open)
  - C5 limit canceled at close of the bar if unfilled

Run with:  streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components

# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="4H Sequence Model — Backtester", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1500px;}
      .mcard{display:flex;justify-content:space-between;align-items:center;
             background:#ffffff;border:1px solid #eceef3;border-radius:16px;
             padding:18px 20px;box-shadow:0 1px 2px rgba(16,24,40,.04);
             min-height:84px;margin-bottom:4px;}
      .mcard-label{color:#667085;font-size:13px;font-weight:500;margin-bottom:7px;}
      .mcard-value{font-size:26px;font-weight:700;line-height:1;}
      .mcard-icon{width:44px;height:44px;border-radius:11px;display:flex;
                  align-items:center;justify-content:center;font-size:20px;}
      .panel{background:#ffffff;border:1px solid #eceef3;border-radius:16px;
             padding:22px 24px;box-shadow:0 1px 2px rgba(16,24,40,.04);}
      .panel-title{font-size:20px;font-weight:700;color:#101828;margin-bottom:18px;}
      .bk{text-align:center;}
      .bk-label{color:#667085;font-size:14px;margin-bottom:6px;}
      .bk-value{font-size:30px;font-weight:700;}
      .sec-title{font-size:22px;font-weight:700;color:#101828;margin:6px 0 12px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _norm(col: str) -> str:
    return str(col).strip().lower().replace("<", "").replace(">", "").replace(" ", "")


def load_ohlc(file_or_buffer) -> pd.DataFrame:
    """Parse a 60-minute OHLC CSV into a clean dataframe with columns:
    datetime, open, high, low, close. Tolerant of common export formats
    (TradingView, MT4/MT5, generic)."""
    raw = pd.read_csv(file_or_buffer)
    lookup = {_norm(c): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in lookup:
                return lookup[n]
        return None

    o = pick("open", "o")
    h = pick("high", "h")
    l = pick("low", "l")
    c = pick("close", "c", "price")
    if not all([o, h, l, c]):
        raise ValueError(
            "Could not find Open/High/Low/Close columns. "
            f"Found columns: {list(raw.columns)}"
        )

    # --- datetime ---
    dt_col = pick("datetime", "timestamp", "time", "date")
    date_only = pick("date")
    time_only = pick("time")

    # Parse with utc=True so rows carrying different timezone offsets
    # (common in broker/TradingView exports, esp. around DST changes) all
    # normalize to a single timezone instead of raising "Mixed timezones".
    if date_only and time_only and date_only != time_only:
        dt = pd.to_datetime(
            raw[date_only].astype(str) + " " + raw[time_only].astype(str),
            errors="coerce", utc=True,
        )
    elif dt_col is not None:
        series = raw[dt_col]
        if pd.api.types.is_numeric_dtype(series):
            # unix epoch (seconds or ms)
            unit = "ms" if series.max() > 1e12 else "s"
            dt = pd.to_datetime(series, unit=unit, errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        raise ValueError("Could not find a date/time column.")

    # Drop the timezone so every bar shares one clock (UTC wall time).
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)

    df = pd.DataFrame(
        {
            "datetime": dt,
            "open": pd.to_numeric(raw[o], errors="coerce"),
            "high": pd.to_numeric(raw[h], errors="coerce"),
            "low": pd.to_numeric(raw[l], errors="coerce"),
            "close": pd.to_numeric(raw[c], errors="coerce"),
        }
    ).dropna()

    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < 5:
        raise ValueError("Need at least 5 bars of data to evaluate a setup.")
    return df


def sample_data(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Synthetic 60m OHLC so the dashboard is explorable without a file."""
    rng = np.random.default_rng(seed)
    price = 100.0
    rows = []
    start = pd.Timestamp("2024-01-01 00:00")
    for i in range(n):
        drift = rng.normal(0, 0.35)
        op = price
        cl = op + drift
        hi = max(op, cl) + abs(rng.normal(0, 0.25))
        lo = min(op, cl) - abs(rng.normal(0, 0.25))
        rows.append((start + pd.Timedelta(hours=i), op, hi, lo, cl))
        price = cl
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])


# --------------------------------------------------------------------------- #
# Strategy / backtest engine
# --------------------------------------------------------------------------- #
SCALE_TIERS = [(0.00, 0.25, 0.25), (0.25, 0.35, 0.35), (0.35, 0.45, 0.45)]


def scale_mult(close_pct: float, out_of_spec: str):
    """Return the scale-matrix multiplier for a given Close_Pct.
    out_of_spec controls Close_Pct > 0.45 (undefined in spec):
      'extend'  -> use the top 0.45 tier
      'skip'    -> nullify the setup (return None)
    """
    for lo, hi, mult in SCALE_TIERS:
        if lo <= close_pct <= hi if lo == 0.0 else lo < close_pct <= hi:
            return mult
    # close_pct > 0.45
    if out_of_spec == "extend":
        return 0.45
    return None


def detect_setup(df: pd.DataFrame, i: int, out_of_spec: str):
    """Evaluate a setup with C1..C4 at rows i..i+3. Returns a dict or None."""
    c1, c2, c3, c4 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2], df.iloc[i + 3]
    rng = c4["high"] - c4["low"]
    if rng <= 0:
        return None

    # ---------------- Bullish ----------------
    if (
        c2["high"] > c1["high"]
        and c3["low"] > c1["high"]                       # FVG present
        and c4["low"] < c3["low"]                         # sweep
        and c4["close"] > c3["open"]                      # structural close
    ):
        close_pct = (c4["close"] - c4["low"]) / rng
        mult = scale_mult(close_pct, out_of_spec)
        if mult is None:
            return None
        entry = c4["low"] + rng * mult
        sl = c4["low"]
        risk = entry - sl
        tp = entry + risk * 2.0
        return dict(direction="LONG", c4_idx=i + 3, c5_idx=i + 4,
                    entry=entry, sl=sl, tp=tp, risk=risk, close_pct=close_pct)

    # ---------------- Bearish ----------------
    if (
        c2["low"] < c1["low"]
        and c3["high"] < c1["low"]
        and c4["high"] > c3["high"]                       # sweep
        and c4["close"] < c3["open"]                      # structural close
    ):
        close_pct = (c4["high"] - c4["close"]) / rng
        mult = scale_mult(close_pct, out_of_spec)
        if mult is None:
            return None
        entry = c4["high"] - rng * mult
        sl = c4["high"]
        risk = sl - entry
        tp = entry - risk * 2.0
        return dict(direction="SHORT", c4_idx=i + 3, c5_idx=i + 4,
                    entry=entry, sl=sl, tp=tp, risk=risk, close_pct=close_pct)

    return None


def resolve_bar(bar, s, ambiguous):
    """Return 'WIN' / 'LOSS' / 'BE' / None for a single bar against setup s."""
    if s["direction"] == "LONG":
        hit_sl = bar["low"] <= s["sl"]
        hit_tp = bar["high"] >= s["tp"]
    else:
        hit_sl = bar["high"] >= s["sl"]
        hit_tp = bar["low"] <= s["tp"]
    if hit_sl and hit_tp:
        return {"be": "BE", "loss": "LOSS", "win": "WIN"}[ambiguous]
    if hit_sl:
        return "LOSS"
    if hit_tp:
        return "WIN"
    return None


def backtest(df, account, risk_pct, ambiguous="be", out_of_spec="extend"):
    """Walk the series, enforce one-trade-at-a-time, simulate fills & exits."""
    n = len(df)
    balance = float(account)
    trades = []
    i = 0
    while i <= n - 5:
        s = detect_setup(df, i, out_of_spec)
        if s is None:
            i += 1
            continue

        c5 = s["c5_idx"]
        bar5 = df.iloc[c5]

        # ---- Limit valid only during C5 ----
        if s["direction"] == "LONG":
            filled = bar5["low"] <= s["entry"]
        else:
            filled = bar5["high"] >= s["entry"]

        if not filled:
            i += 1  # order canceled at C5 close; keep scanning
            continue

        # ---- Track exit from C5 onward ----
        outcome, exit_idx = None, None
        for j in range(c5, n):
            res = resolve_bar(df.iloc[j], s, ambiguous)
            if res:
                outcome, exit_idx = res, j
                break

        if outcome is None:           # never resolved before data ended
            i += 1
            continue

        r_mult = {"WIN": 2.0, "LOSS": -1.0, "BE": 0.0}[outcome]
        risk_amt = balance * risk_pct / 100.0
        pnl = r_mult * risk_amt
        balance += pnl

        trades.append(
            dict(
                num=len(trades) + 1,
                direction=s["direction"],
                setup_time=df.iloc[s["c4_idx"]]["datetime"],   # C4 close
                entry_time=bar5["datetime"],                   # C5 open
                exit_time=df.iloc[exit_idx]["datetime"],
                entry=round(s["entry"], 6),
                sl=round(s["sl"], 6),
                tp=round(s["tp"], 6),
                outcome=outcome,
                r=r_mult,
                pnl=pnl,
                balance=balance,
                c4_idx=s["c4_idx"],
                entry_idx=c5,
                exit_idx=exit_idx,
            )
        )

        # one trade at a time: ignore setups during the open position
        i = exit_idx + 1

    return pd.DataFrame(trades), balance


# --------------------------------------------------------------------------- #
# Metrics & charts
# --------------------------------------------------------------------------- #
def compute_metrics(trades, account, final):
    wins = int((trades["outcome"] == "WIN").sum())
    losses = int((trades["outcome"] == "LOSS").sum())
    be = int((trades["outcome"] == "BE").sum())
    total = len(trades)
    decided = wins + losses
    win_rate = (wins / decided * 100) if decided else 0.0
    total_r = trades["r"].sum()
    avg_r = trades["r"].mean() if total else 0.0
    pnl = final - account
    ret = (final / account - 1) * 100 if account else 0.0

    # max drawdown on the equity curve
    eq = np.concatenate([[account], trades["balance"].to_numpy()])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()
    return dict(total=total, wins=wins, losses=losses, be=be,
                win_rate=win_rate, total_r=total_r, avg_r=avg_r,
                pnl=pnl, ret=ret, final=final, max_dd=max_dd)


def card(label, value, icon, color="#101828", icon_bg="#eef2ff"):
    st.markdown(
        f'<div class="mcard"><div><div class="mcard-label">{label}</div>'
        f'<div class="mcard-value" style="color:{color}">{value}</div></div>'
        f'<div class="mcard-icon" style="background:{icon_bg}">{icon}</div></div>',
        unsafe_allow_html=True,
    )


GREEN, RED, BLUE = "#16a34a", "#dc2626", "#2563eb"


def equity_chart(trades, account):
    x = [0] + trades["num"].tolist()
    y = [account] + trades["balance"].tolist()
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines",
                               line=dict(color=BLUE, width=2)))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Trade #", yaxis_title="Capital",
        yaxis=dict(tickprefix="$", gridcolor="#f0f1f4"),
        xaxis=dict(gridcolor="#f7f8fa"),
    )
    return fig


def drawdown_chart(trades, account):
    eq = np.concatenate([[account], trades["balance"].to_numpy()])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    x = list(range(len(dd)))
    fig = go.Figure(go.Scatter(x=x, y=dd, mode="lines", fill="tozeroy",
                               line=dict(color=RED, width=1.2),
                               fillcolor="rgba(220,38,38,.18)"))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Trade #", yaxis_title="Drawdown %",
        yaxis=dict(ticksuffix="%", gridcolor="#f0f1f4"),
        xaxis=dict(gridcolor="#f7f8fa"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
st.sidebar.header("Inputs")

uploaded = st.sidebar.file_uploader("Upload 60m OHLC CSV", type=["csv"])
symbol = st.sidebar.text_input(
    "Symbol (for TradingView)", value="XAUUSD",
    help="Just the ticker — e.g. XAUUSD, EURUSD, GBPJPY, NQ1!, ES1!, BTCUSD. "
         "No exchange/broker prefix needed; it's only used for the chart link.",
)

st.sidebar.subheader("Account")
account = st.sidebar.number_input("Account size ($)", min_value=1.0,
                                  value=100_000.0, step=1000.0)
risk_pct = st.sidebar.number_input("Risk per trade (%)", min_value=0.01,
                                    value=1.0, step=0.25)

with st.sidebar.expander("Advanced (spec edge cases)"):
    ambiguous = st.selectbox(
        "If a bar hits both TP and SL",
        options=["be", "loss", "win"],
        format_func=lambda v: {"be": "Break-even (scratch)",
                               "loss": "Conservative (loss)",
                               "win": "Optimistic (win)"}[v],
        help="OHLC bars don't reveal intrabar order. The spec is silent on this.",
    )
    out_of_spec = st.selectbox(
        "If Close_Pct > 0.45 (undefined in spec)",
        options=["extend", "skip"],
        format_func=lambda v: {"extend": "Extend top tier (0.45)",
                               "skip": "Skip / nullify setup"}[v],
    )

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
if uploaded is not None:
    try:
        df = load_ohlc(uploaded)
        st.sidebar.success(f"Loaded {len(df):,} bars "
                           f"({df['datetime'].iloc[0]:%Y-%m-%d} → "
                           f"{df['datetime'].iloc[-1]:%Y-%m-%d})")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    df = sample_data()
    st.info("No file uploaded — showing **sample synthetic data**. "
            "Upload a 60m OHLC CSV in the sidebar to backtest your own data.")

# --------------------------------------------------------------------------- #
# Run backtest
# --------------------------------------------------------------------------- #
trades, final_balance = backtest(df, account, risk_pct, ambiguous, out_of_spec)

if trades.empty:
    st.warning("No trades were generated for this dataset / settings.")
    st.stop()

m = compute_metrics(trades, account, final_balance)

# --------------------------------------------------------------------------- #
# Performance Summary
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec-title">Performance Summary</div>',
            unsafe_allow_html=True)

r1 = st.columns(3)
with r1[0]:
    card("Total Trades", f"{m['total']}", "📊", BLUE, "#eef2ff")
with r1[1]:
    card("Win Rate", f"{m['win_rate']:.1f}%", "🎯", GREEN, "#ecfdf5")
with r1[2]:
    card("Total R", f"{m['total_r']:.2f}", "🏅",
         GREEN if m["total_r"] >= 0 else RED, "#ecfdf5")

r2 = st.columns(3)
with r2[0]:
    card("Average R", f"{m['avg_r']:.2f}", "📈",
         GREEN if m["avg_r"] >= 0 else RED, "#ecfdf5")
with r2[1]:
    card("Total P&L", f"${m['pnl']:,.0f}", "💲",
         GREEN if m["pnl"] >= 0 else RED, "#ecfdf5")
with r2[2]:
    card("Return", f"{m['ret']:.2f}%", "📈",
         GREEN if m["ret"] >= 0 else RED, "#ecfdf5")

# Trade breakdown
st.markdown(
    f'''
    <div class="panel" style="margin-top:14px;">
      <div class="panel-title">Trade Breakdown</div>
      <div style="display:flex;justify-content:space-around;">
        <div class="bk"><div class="bk-label">Winning</div>
             <div class="bk-value" style="color:{GREEN}">{m['wins']}</div></div>
        <div class="bk"><div class="bk-label">Losing</div>
             <div class="bk-value" style="color:{RED}">{m['losses']}</div></div>
        <div class="bk"><div class="bk-label">Break Even</div>
             <div class="bk-value" style="color:#667085">{m['be']}</div></div>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# Final capital / max drawdown
fc = st.columns(2)
with fc[0]:
    card("Final Capital", f"${m['final']:,.3f}", "💰", "#101828", "#f1f5f9")
with fc[1]:
    card("Max Drawdown", f"{m['max_dd']:.2f}%", "📉", RED, "#fef2f2")

# Charts
ch = st.columns(2)
with ch[0]:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Equity Curve</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(equity_chart(trades, account), use_container_width=True)
with ch[1]:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Drawdown Chart</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(drawdown_chart(trades, account), use_container_width=True)

# --------------------------------------------------------------------------- #
# Trade Journal
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec-title" style="margin-top:18px;">Trade Journal</div>',
            unsafe_allow_html=True)

journal = trades[[
    "num", "direction", "setup_time", "entry_time", "exit_time",
    "entry", "sl", "tp", "outcome", "r", "pnl", "balance",
]].copy()
journal.columns = [
    "#", "Side", "Setup (C4 close)", "Entry (C5)", "Exit",
    "Entry", "SL", "TP", "Outcome", "R", "P&L ($)", "Balance ($)",
]

st.dataframe(
    journal.style.format({
        "P&L ($)": "{:,.2f}", "Balance ($)": "{:,.2f}", "R": "{:.1f}",
    }),
    use_container_width=True, hide_index=True, height=380,
)

st.download_button(
    "⬇ Download trade journal (CSV)",
    data=journal.to_csv(index=False).encode(),
    file_name="trade_journal.csv",
    mime="text/csv",
)

# ---- Per-trade visual check on TradingView ----
st.markdown("#### Inspect a trade on TradingView")
sel = st.selectbox(
    "Select trade",
    options=trades["num"].tolist(),
    format_func=lambda x: (
        f"#{x} · {trades.loc[trades['num']==x,'direction'].iloc[0]} · "
        f"{trades.loc[trades['num']==x,'outcome'].iloc[0]} · "
        f"{trades.loc[trades['num']==x,'entry_time'].iloc[0]:%Y-%m-%d %H:%M}"
    ),
)
t = trades[trades["num"] == sel].iloc[0]

info = st.columns(4)
info[0].metric("Side", t["direction"])
info[1].metric("Outcome", t["outcome"])
info[2].metric("Entry", f"{t['entry']:.5f}")
info[3].metric("R", f"{t['r']:.1f}")
info2 = st.columns(4)
info2[0].metric("Stop Loss", f"{t['sl']:.5f}")
info2[1].metric("Take Profit", f"{t['tp']:.5f}")
info2[2].metric("Entry time", f"{t['entry_time']:%Y-%m-%d %H:%M}")
info2[3].metric("Exit time", f"{t['exit_time']:%Y-%m-%d %H:%M}")

tv_url = f"https://www.tradingview.com/chart/?symbol={symbol}&interval=60"
st.markdown(
    f"🔗 [Open **{symbol}** (1H) on TradingView]({tv_url}) — then scroll to "
    f"**{t['entry_time']:%Y-%m-%d %H:%M}** to review this setup."
)

# Embedded live TradingView chart for the selected symbol (1H)
components.html(
    f"""
    <div class="tradingview-widget-container">
      <div id="tv_adv"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
      new TradingView.widget({{
        "width": "100%", "height": 520, "symbol": "{symbol}",
        "interval": "60", "timezone": "Etc/UTC", "theme": "light",
        "style": "1", "locale": "en", "enable_publishing": false,
        "allow_symbol_change": true, "container_id": "tv_adv"
      }});
      </script>
    </div>
    """,
    height=540,
)

st.caption(
    "Educational backtesting tool implementing the supplied specification. "
    "Not financial advice; past simulated results do not predict future returns."
)

