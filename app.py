"""
4H Sequence Model (H1) — Backtester, Performance Dashboard & Trade Journal
==========================================================================

Strategy (per the latest spec revision):
  - 3-candle FVG structure (C1, C2, C3)
  - EBP reversal / liquidity sweep + structural close on C4
  - ENTRY: market entry once C4 (EBP) closes, taken at the C5 open
  - STOP:  C4 extreme (low for longs, high for shorts)
  - TARGET: fixed 1.5R
  - No break-even rule
  - One trade at a time (new setups ignored while a position is open)

Run with:  streamlit run app.py
"""

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
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      html, body, .stApp {font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
                          color:#101828;}
      .stApp {background:#f4f6fb !important;}
      .block-container {padding-top:1.4rem; max-width:1500px;}

      /* keep themed Streamlit widgets readable on the light background */
      section[data-testid="stSidebar"]{background:#ffffff !important;border-right:1px solid #e6e9f0;}
      section[data-testid="stSidebar"] *{color:#101828 !important;}
      label, label *{color:#1d2939 !important;}
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{color:#667085 !important;}

      div[data-baseweb="select"] > div{background:#fff !important;border:1px solid #d9dee8 !important;border-radius:10px !important;}
      div[data-baseweb="select"] *{color:#101828 !important;}
      ul[data-baseweb="menu"]{background:#fff !important;}
      li[role="option"], li[role="option"] *{color:#101828 !important;}
      input, textarea{color:#101828 !important;}

      .stDownloadButton button, .stButton button{
        background:#2563eb !important;color:#fff !important;border:none !important;
        border-radius:10px !important;padding:.55rem 1.1rem !important;font-weight:600 !important;
        box-shadow:0 2px 8px rgba(37,99,235,.28) !important;}
      .stDownloadButton button:hover, .stButton button:hover{background:#1d4ed8 !important;}
      .stDownloadButton button *{color:#fff !important;}

      .mcard{display:flex;justify-content:space-between;align-items:flex-start;
             background:#fff;border:1px solid #eceff4;border-radius:16px;
             padding:18px 20px;box-shadow:0 4px 16px rgba(16,24,40,.06);min-height:88px;}
      .mcard-label{color:#667085;font-size:13px;font-weight:500;margin-bottom:9px;}
      .mcard-value{font-size:27px;font-weight:800;line-height:1;}
      .mcard-icon{width:44px;height:44px;border-radius:12px;display:flex;
                  align-items:center;justify-content:center;flex:none;}
      .mcard-icon svg{width:22px;height:22px;}

      .panel{background:#fff;border:1px solid #eceff4;border-radius:16px;
             padding:22px 26px;box-shadow:0 4px 16px rgba(16,24,40,.06);margin-bottom:14px;}
      .panel-title{font-size:19px;font-weight:700;color:#101828;margin-bottom:18px;}
      .sec-title{font-size:23px;font-weight:800;color:#101828;margin:16px 0 12px 0;}

      .bk-row{display:flex;justify-content:space-around;text-align:center;}
      .bk-label{color:#667085;font-size:14px;margin-bottom:8px;}
      .bk-value{font-size:30px;font-weight:800;}

      .twocol{display:flex;gap:64px;}
      .twocol .lab{color:#667085;font-size:13px;margin-bottom:6px;}
      .twocol .val{font-size:24px;font-weight:800;color:#101828;}

      div[data-testid="stVerticalBlockBorderWrapper"]{
             background:#fff;border:1px solid #eceff4 !important;border-radius:16px;
             box-shadow:0 4px 16px rgba(16,24,40,.06);}

      .badge{display:inline-block;padding:4px 13px;border-radius:999px;font-size:13px;font-weight:700;}
      .pill{display:inline-block;padding:2px 11px;border-radius:999px;font-size:12px;font-weight:700;}
      .b-long{background:#ecfdf5;color:#059669 !important;}
      .b-short{background:#fef2f2;color:#dc2626 !important;}
      .b-win{background:#ecfdf5;color:#16a34a !important;}
      .b-loss{background:#fef2f2;color:#dc2626 !important;}
      .b-be{background:#f1f5f9;color:#475467 !important;}

      .dgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:20px 24px;margin-top:18px;}
      .dgrid .lab{color:#667085;font-size:12px;font-weight:600;margin-bottom:5px;
                  text-transform:uppercase;letter-spacing:.04em;}
      .dgrid .val{font-size:18px;font-weight:700;color:#101828;font-variant-numeric:tabular-nums;}

      .tjwrap{max-height:440px;overflow:auto;border:1px solid #eceff4;border-radius:14px;
              box-shadow:0 4px 16px rgba(16,24,40,.06);background:#fff;margin-bottom:14px;}
      table.tj{border-collapse:collapse;width:100%;font-size:13px;}
      table.tj thead th{position:sticky;top:0;background:#f8fafc;color:#475467;font-weight:600;
              text-align:right;padding:11px 14px;border-bottom:1px solid #e6e9f0;white-space:nowrap;z-index:2;}
      table.tj thead th:nth-child(-n+2){text-align:left;}
      table.tj tbody td{padding:9px 14px;border-bottom:1px solid #f1f3f7;text-align:right;
              color:#101828;font-variant-numeric:tabular-nums;white-space:nowrap;}
      table.tj tbody td:nth-child(-n+2){text-align:left;}
      table.tj tbody tr:hover{background:#f8fafc;}

      a.tvbtn{display:inline-block;background:#101828;color:#fff !important;text-decoration:none;
              padding:.6rem 1.1rem;border-radius:10px;font-weight:600;font-size:14px;}
      a.tvbtn:hover{background:#1f2937;}
    </style>
    """,
    unsafe_allow_html=True,
)

GREEN, RED, BLUE, INK, MUTE = "#16a34a", "#dc2626", "#2563eb", "#101828", "#475467"

# Line-style icons (Lucide-ish) so the cards match the reference design.
ICONS = {
    "bar": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "target": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "award": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M15.5 13 17 22l-5-3-5 3 1.5-9"/></svg>',
    "up": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "dollar": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _norm(col: str) -> str:
    return str(col).strip().lower().replace("<", "").replace(">", "").replace(" ", "")


def load_ohlc(file_or_buffer) -> pd.DataFrame:
    """Parse a 60-minute OHLC CSV into datetime/open/high/low/close.
    Tolerant of TradingView / MT4-MT5 / generic exports, including files with
    mixed timezone offsets."""
    raw = pd.read_csv(file_or_buffer)
    lookup = {_norm(c): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in lookup:
                return lookup[n]
        return None

    o, h, l, c = pick("open", "o"), pick("high", "h"), pick("low", "l"), pick("close", "c", "price")
    if not all([o, h, l, c]):
        raise ValueError(f"Could not find Open/High/Low/Close columns. Found: {list(raw.columns)}")

    dt_col = pick("datetime", "timestamp", "time", "date")
    date_only, time_only = pick("date"), pick("time")

    if date_only and time_only and date_only != time_only:
        dt = pd.to_datetime(raw[date_only].astype(str) + " " + raw[time_only].astype(str),
                            errors="coerce", utc=True)
    elif dt_col is not None:
        series = raw[dt_col]
        if pd.api.types.is_numeric_dtype(series):
            unit = "ms" if series.max() > 1e12 else "s"
            dt = pd.to_datetime(series, unit=unit, errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        raise ValueError("Could not find a date/time column.")

    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)

    df = pd.DataFrame({
        "datetime": dt,
        "open": pd.to_numeric(raw[o], errors="coerce"),
        "high": pd.to_numeric(raw[h], errors="coerce"),
        "low": pd.to_numeric(raw[l], errors="coerce"),
        "close": pd.to_numeric(raw[c], errors="coerce"),
    }).dropna().sort_values("datetime").reset_index(drop=True)

    if len(df) < 5:
        raise ValueError("Need at least 5 bars of data to evaluate a setup.")
    return df


def sample_data(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price, rows, start = 100.0, [], pd.Timestamp("2024-01-01 00:00")
    for i in range(n):
        op = price
        cl = op + rng.normal(0, 0.35)
        hi = max(op, cl) + abs(rng.normal(0, 0.25))
        lo = min(op, cl) - abs(rng.normal(0, 0.25))
        rows.append((start + pd.Timedelta(hours=i), op, hi, lo, cl))
        price = cl
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])


# --------------------------------------------------------------------------- #
# Strategy / backtest engine
# --------------------------------------------------------------------------- #
RR_TARGET = 1.5        # take-profit distance in R


def detect_setup(df, i):
    """Check C1..C4 at rows i..i+3 for a valid structure + EBP close.
    Returns (direction, sl) or None. Entry is taken later at the C5 open."""
    c1, c2, c3, c4 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2], df.iloc[i + 3]

    # Bullish
    if (c2["high"] > c1["high"] and c3["low"] > c1["high"]
            and c4["low"] < c3["low"] and c4["close"] > c3["open"]):
        return "LONG", c4["low"]

    # Bearish
    if (c2["low"] < c1["low"] and c3["high"] < c1["low"]
            and c4["high"] > c3["high"] and c4["close"] < c3["open"]):
        return "SHORT", c4["high"]

    return None


def simulate_exit(df, direction, entry, sl, tp, start_idx, n, ambiguous):
    """First-touch exit from start_idx (the C5 bar) onward."""
    for j in range(start_idx, n):
        bar = df.iloc[j]
        if direction == "LONG":
            hit_sl, hit_tp = bar["low"] <= sl, bar["high"] >= tp
        else:
            hit_sl, hit_tp = bar["high"] >= sl, bar["low"] <= tp
        if hit_sl and hit_tp:
            return {"be": "BE", "loss": "LOSS", "win": "WIN"}[ambiguous], j
        if hit_sl:
            return "LOSS", j
        if hit_tp:
            return "WIN", j
    return None, None


def backtest(df, account, risk_pct, ambiguous="loss"):
    n = len(df)
    balance = float(account)
    trades = []
    i = 0
    while i <= n - 5:
        found = detect_setup(df, i)
        if found is None:
            i += 1
            continue

        direction, sl = found
        c4_idx, c5 = i + 3, i + 4
        entry = df.iloc[c5]["open"]            # market entry once C4 closes

        # validate risk direction (skip if the open is already through the stop)
        if direction == "LONG":
            if entry <= sl:
                i += 1
                continue
            risk = entry - sl
            tp = entry + risk * RR_TARGET
        else:
            if entry >= sl:
                i += 1
                continue
            risk = sl - entry
            tp = entry - risk * RR_TARGET

        outcome, exit_idx = simulate_exit(df, direction, entry, sl, tp, c5, n, ambiguous)
        if outcome is None:
            i += 1
            continue

        r_mult = {"WIN": RR_TARGET, "LOSS": -1.0, "BE": 0.0}[outcome]
        risk_amt = balance * risk_pct / 100.0
        pnl = r_mult * risk_amt
        balance += pnl

        trades.append(dict(
            num=len(trades) + 1, direction=direction,
            setup_time=df.iloc[c4_idx]["datetime"], entry_time=df.iloc[c5]["datetime"],
            exit_time=df.iloc[exit_idx]["datetime"],
            entry=entry, sl=sl, tp=tp, outcome=outcome, r=r_mult,
            pnl=pnl, balance=balance, exit_idx=exit_idx,
        ))
        i = exit_idx + 1        # one trade at a time

    return pd.DataFrame(trades), balance


# --------------------------------------------------------------------------- #
# Metrics, formatting & charts
# --------------------------------------------------------------------------- #
def compute_metrics(trades, account, final):
    wins = int((trades["outcome"] == "WIN").sum())
    losses = int((trades["outcome"] == "LOSS").sum())
    be = int((trades["outcome"] == "BE").sum())
    total = len(trades)
    decided = wins + losses
    return dict(
        total=total, wins=wins, losses=losses, be=be,
        win_rate=(wins / decided * 100) if decided else 0.0,
        total_r=trades["r"].sum(), avg_r=trades["r"].mean() if total else 0.0,
        pnl=final - account, ret=(final / account - 1) * 100 if account else 0.0,
        final=final,
        max_dd=_max_dd(np.concatenate([[account], trades["balance"].to_numpy()])),
    )


def _max_dd(eq):
    peak = np.maximum.accumulate(eq)
    return ((eq - peak) / peak * 100).min()


def fmt_price(x):
    ax = abs(x)
    if ax >= 100:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def card(label, value, icon_key, value_color=INK, icon_color=BLUE, icon_bg="#eef2ff"):
    st.markdown(
        f'<div class="mcard"><div><div class="mcard-label">{label}</div>'
        f'<div class="mcard-value" style="color:{value_color}">{value}</div></div>'
        f'<div class="mcard-icon" style="background:{icon_bg};color:{icon_color}">{ICONS[icon_key]}</div>'
        f'</div>', unsafe_allow_html=True)


def _c(v):
    return GREEN if v >= 0 else RED


def _bg(v):
    return "#ecfdf5" if v >= 0 else "#fef2f2"


def equity_chart(trades, account):
    fig = go.Figure(go.Scatter(x=[0] + trades["num"].tolist(),
                               y=[account] + trades["balance"].tolist(),
                               mode="lines", line=dict(color=BLUE, width=2)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode=False,
                      font=dict(color=INK),
                      xaxis_title="Trade #", yaxis_title="Capital",
                      yaxis=dict(tickprefix="$", gridcolor="#eef1f6", fixedrange=True),
                      xaxis=dict(gridcolor="#f7f8fa", fixedrange=True))
    return fig


def drawdown_chart(trades, account):
    eq = np.concatenate([[account], trades["balance"].to_numpy()])
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
    fig = go.Figure(go.Scatter(x=list(range(len(dd))), y=dd, mode="lines", fill="tozeroy",
                               line=dict(color=RED, width=1.2), fillcolor="rgba(220,38,38,.18)"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode=False,
                      font=dict(color=INK),
                      xaxis_title="Trade #", yaxis_title="Drawdown %",
                      yaxis=dict(ticksuffix="%", gridcolor="#eef1f6", fixedrange=True),
                      xaxis=dict(gridcolor="#f7f8fa", fixedrange=True))
    return fig


def journal_table_html(trades):
    head = ("<tr><th>#</th><th>Side</th><th>Setup (C4)</th><th>Entry (C5)</th><th>Exit</th>"
            "<th>Entry</th><th>SL</th><th>TP</th><th>Outcome</th><th>R</th>"
            "<th>P&amp;L&nbsp;($)</th><th>Balance&nbsp;($)</th></tr>")
    rows = []
    for _, r in trades.iterrows():
        side = "b-long" if r["direction"] == "LONG" else "b-short"
        oc_cls = {"WIN": "b-win", "LOSS": "b-loss", "BE": "b-be"}[r["outcome"]]
        rcol = GREEN if r["r"] > 0 else (RED if r["r"] < 0 else MUTE)
        pcol = GREEN if r["pnl"] > 0 else (RED if r["pnl"] < 0 else MUTE)
        rows.append(
            "<tr>"
            f"<td>{int(r['num'])}</td>"
            f"<td><span class='badge {side}'>{r['direction']}</span></td>"
            f"<td>{r['setup_time']:%Y-%m-%d %H:%M}</td>"
            f"<td>{r['entry_time']:%Y-%m-%d %H:%M}</td>"
            f"<td>{r['exit_time']:%Y-%m-%d %H:%M}</td>"
            f"<td>{fmt_price(r['entry'])}</td>"
            f"<td>{fmt_price(r['sl'])}</td>"
            f"<td>{fmt_price(r['tp'])}</td>"
            f"<td><span class='pill {oc_cls}'>{r['outcome']}</span></td>"
            f"<td style='color:{rcol};font-weight:700'>{r['r']:.1f}</td>"
            f"<td style='color:{pcol};font-weight:600'>{r['pnl']:,.2f}</td>"
            f"<td>{r['balance']:,.2f}</td>"
            "</tr>")
    return f"<div class='tjwrap'><table class='tj'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"


# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload 60m OHLC CSV", type=["csv"])
symbol = st.sidebar.text_input(
    "Symbol (for TradingView)", value="XAUUSD",
    help="Just the ticker — e.g. XAUUSD, EURUSD, GBPJPY, NQ1!, ES1!, BTCUSD.").strip()

st.sidebar.subheader("Account")
account = st.sidebar.number_input("Account size ($)", min_value=1.0, value=100_000.0, step=1000.0)
risk_pct = st.sidebar.number_input("Risk per trade (%)", min_value=0.01, value=1.0, step=0.25)

with st.sidebar.expander("Advanced"):
    ambiguous = st.selectbox(
        "If one bar hits both TP and SL",
        options=["loss", "be", "win"],
        format_func=lambda v: {"loss": "Conservative (loss)",
                               "be": "Break-even (scratch)",
                               "win": "Optimistic (win)"}[v],
        help="OHLC bars don't reveal which level was touched first within a bar.")

# --------------------------------------------------------------------------- #
# Load + run
# --------------------------------------------------------------------------- #
if uploaded is not None:
    try:
        df = load_ohlc(uploaded)
        st.sidebar.success(f"Loaded {len(df):,} bars "
                           f"({df['datetime'].iloc[0]:%Y-%m-%d} -> {df['datetime'].iloc[-1]:%Y-%m-%d})")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    df = sample_data()
    st.info("No file uploaded - showing sample synthetic data. Upload a 60m OHLC CSV to backtest your own.")

trades, final_balance = backtest(df, account, risk_pct, ambiguous)
if trades.empty:
    st.warning("No trades were generated for this dataset / settings.")
    st.stop()
m = compute_metrics(trades, account, final_balance)

# --------------------------------------------------------------------------- #
# Performance Summary
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec-title">Performance Summary</div>', unsafe_allow_html=True)

r1 = st.columns(3)
with r1[0]:
    card("Total Trades", f"{m['total']}", "bar", BLUE, BLUE, "#eef2ff")
with r1[1]:
    card("Win Rate", f"{m['win_rate']:.1f}%", "target", GREEN, GREEN, "#ecfdf5")
with r1[2]:
    card("Total R", f"{m['total_r']:.2f}", "award", _c(m['total_r']), _c(m['total_r']), _bg(m['total_r']))

r2 = st.columns(3)
with r2[0]:
    card("Average R", f"{m['avg_r']:.2f}", "up", _c(m['avg_r']), _c(m['avg_r']), _bg(m['avg_r']))
with r2[1]:
    card("Total P&L", f"${m['pnl']:,.0f}", "dollar", _c(m['pnl']), _c(m['pnl']), _bg(m['pnl']))
with r2[2]:
    card("Return", f"{m['ret']:.2f}%", "up", _c(m['ret']), _c(m['ret']), _bg(m['ret']))

st.markdown(
    f'<div class="panel"><div class="panel-title">Trade Breakdown</div>'
    f'<div class="bk-row">'
    f'<div><div class="bk-label">Winning</div><div class="bk-value" style="color:{GREEN}">{m["wins"]}</div></div>'
    f'<div><div class="bk-label">Losing</div><div class="bk-value" style="color:{RED}">{m["losses"]}</div></div>'
    f'<div><div class="bk-label">Break Even</div><div class="bk-value" style="color:{MUTE}">{m["be"]}</div></div>'
    f'</div></div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="panel"><div class="twocol">'
    f'<div><div class="lab">Final Capital</div><div class="val">${m["final"]:,.2f}</div></div>'
    f'<div><div class="lab">Max Drawdown</div><div class="val" style="color:{RED}">{m["max_dd"]:.2f}%</div></div>'
    f'</div></div>', unsafe_allow_html=True)

ch = st.columns(2)
with ch[0]:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Equity Curve</div>', unsafe_allow_html=True)
        st.plotly_chart(equity_chart(trades, account), use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": False})
with ch[1]:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Drawdown Chart</div>', unsafe_allow_html=True)
        st.plotly_chart(drawdown_chart(trades, account), use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": False})

# --------------------------------------------------------------------------- #
# Trade Journal
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec-title">Trade Journal</div>', unsafe_allow_html=True)
st.markdown(journal_table_html(trades), unsafe_allow_html=True)

csv = trades[["num", "direction", "setup_time", "entry_time", "exit_time",
              "entry", "sl", "tp", "outcome", "r", "pnl", "balance"]].copy()
csv.columns = ["#", "Side", "Setup_C4", "Entry_C5", "Exit", "Entry", "SL", "TP",
               "Outcome", "R", "PnL", "Balance"]
st.download_button("Download trade journal (CSV)", data=csv.to_csv(index=False).encode(),
                   file_name="trade_journal.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Inspect a trade on TradingView
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec-title">Inspect a trade on TradingView</div>', unsafe_allow_html=True)

sel = st.selectbox(
    "Select trade", options=trades["num"].tolist(),
    format_func=lambda x: (f"#{x}  ·  {trades.loc[trades['num']==x,'direction'].iloc[0]}"
                           f"  ·  {trades.loc[trades['num']==x,'outcome'].iloc[0]}"
                           f"  ·  {trades.loc[trades['num']==x,'entry_time'].iloc[0]:%Y-%m-%d %H:%M}"))
t = trades[trades["num"] == sel].iloc[0]
side = "b-long" if t["direction"] == "LONG" else "b-short"
oc_cls = {"WIN": "b-win", "LOSS": "b-loss", "BE": "b-be"}[t["outcome"]]

st.markdown(
    f'<div class="panel">'
    f'<div style="display:flex;gap:10px;">'
    f'<span class="badge {side}">{t["direction"]}</span>'
    f'<span class="badge {oc_cls}">{t["outcome"]}</span></div>'
    f'<div class="dgrid">'
    f'<div><div class="lab">Entry</div><div class="val">{fmt_price(t["entry"])}</div></div>'
    f'<div><div class="lab">Stop Loss</div><div class="val">{fmt_price(t["sl"])}</div></div>'
    f'<div><div class="lab">Take Profit</div><div class="val">{fmt_price(t["tp"])}</div></div>'
    f'<div><div class="lab">R Multiple</div><div class="val" style="color:{_c(t["r"])}">{t["r"]:.1f}</div></div>'
    f'<div><div class="lab">Entry time</div><div class="val">{t["entry_time"]:%Y-%m-%d %H:%M}</div></div>'
    f'<div><div class="lab">Exit time</div><div class="val">{t["exit_time"]:%Y-%m-%d %H:%M}</div></div>'
    f'<div><div class="lab">P&amp;L</div><div class="val" style="color:{_c(t["pnl"])}">${t["pnl"]:,.2f}</div></div>'
    f'<div><div class="lab">Symbol</div><div class="val">{symbol}</div></div>'
    f'</div></div>', unsafe_allow_html=True)

tv_url = f"https://www.tradingview.com/chart/?symbol={symbol}&interval=60"
st.markdown(
    f'<a class="tvbtn" href="{tv_url}" target="_blank">Open {symbol} (1H) on TradingView ↗</a>'
    f'<span style="color:{MUTE};margin-left:12px;font-size:14px;">scroll to '
    f'<b style="color:{INK}">{t["entry_time"]:%Y-%m-%d %H:%M} UTC</b> to review this setup</span>',
    unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(f'<div class="panel-title">Live chart — {symbol} · 1H</div>', unsafe_allow_html=True)
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
            "allow_symbol_change": true, "hide_side_toolbar": false,
            "container_id": "tv_adv"
          }});
          </script>
        </div>
        """, height=540)

st.caption("Educational backtester. Entry: market at C5 open after C4 (EBP) close · stop at C4 extreme · "
           "target 1.5R · no break-even rule. Not financial advice; simulated results do not predict future returns.")

