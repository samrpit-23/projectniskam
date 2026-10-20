import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty 500 Swing Bot", layout="wide")

NSE_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
FALLBACK = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LT",
            "AXISBANK", "KOTAKBANK", "HINDUNILVR", "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE",
            "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL", "ADANIPORTS", "M&M", "ULTRACEMCO",
            "ASIANPAINT", "HCLTECH", "WIPRO", "ONGC", "COALINDIA", "DRREDDY", "CIPLA"]


# ----------------------------------------------------------------- data
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_universe():
    try:
        r = requests.get(NSE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))[["Symbol", "Industry"]]
        df.columns = ["symbol", "industry"]
        return df
    except Exception:
        return pd.DataFrame({"symbol": FALLBACK, "industry": "NA"})


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def download(symbols, years):
    out = {}
    for k in range(0, len(symbols), 50):
        chunk = list(symbols[k:k + 50])
        try:
            raw = yf.download([s + ".NS" for s in chunk], period=f"{years + 1}y", group_by="ticker",
                              auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        for s in chunk:
            try:
                d = raw[s + ".NS"][["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) > 260:
                    out[s] = d
            except Exception:
                pass
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def download_index(years):
    d = yf.download("^NSEI", period=f"{years + 1}y", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d[["Open", "High", "Low", "Close"]].dropna()


# ------------------------------------------------------------- strategy
def prep(df, p):
    d = df.copy()
    c = d.Close
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()
    tr = pd.concat([d.High - d.Low, (d.High - c.shift()).abs(), (d.Low - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    dl = c.diff()
    g = dl.clip(lower=0).rolling(14).mean()
    l = (-dl.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + g / l)
    d["hh"] = d.High.rolling(p["breakout"]).max().shift(1)
    d["avgvol"] = d.Volume.rolling(20).mean().shift(1)
    d["turnover"] = (c * d.Volume).rolling(20).mean()
    d["ret126"] = c.pct_change(126)
    d["signal"] = ((c > d.hh) & (c > d.sma200) & (d.sma50 > d.sma200) &
                   (d.Volume > p["volx"] * d.avgvol) & d.rsi.between(55, 78) &
                   (d.turnover > 5e7) & (d.ret126 > 0))
    return d


def size_trade(equity, cash, entry, atr, p):
    sl = entry - p["sl_atr"] * atr
    risk = entry - sl
    if risk <= 0:
        return None
    tp = entry + p["rr"] * risk
    qty = int(min(equity * p["risk"] / risk, equity * p["max_pct"] / entry, cash / (entry * (1 + p["cost"]))))
    return (sl, tp, qty) if qty >= 1 else None


# ------------------------------------------------------------- backtest
def backtest(data, idx, p):
    dates = idx.index
    n = len(dates)
    syms = list(data)
    m = len(syms)
    O = np.full((n, m), np.nan)
    H, L, C, A, R = O.copy(), O.copy(), O.copy(), O.copy(), O.copy()
    SIG = np.zeros((n, m), bool)
    for j, s in enumerate(syms):
        d = prep(data[s], p).reindex(dates)
        O[:, j], H[:, j], L[:, j], C[:, j] = d.Open, d.High, d.Low, d.Close
        A[:, j], R[:, j] = d.atr, d.ret126
        SIG[:, j] = d.signal.fillna(False).values.astype(bool)
    Cf = pd.DataFrame(C).ffill().values
    reg = (idx.Close > idx.Close.rolling(200).mean()).values
    ir = idx.Close.pct_change(126).values

    cash = float(p["capital"])
    pos, pend, trades = {}, [], []
    eq = np.full(n, np.nan)
    cur_month, month_start = None, cash

    def close_pos(j, q, px, i):
        nonlocal cash
        cash += q["qty"] * px * (1 - p["cost"])
        pnl = q["qty"] * (px * (1 - p["cost"]) - q["entry"] * (1 + p["cost"]))
        trades.append(dict(symbol=syms[j], entry_date=dates[q["i"]], exit_date=dates[i],
                           entry=q["entry"], exit=px, qty=q["qty"], pnl=pnl))

    for i in range(n):
        for j in list(pos):
            q = pos[j]
            if np.isnan(O[i, j]):
                continue
            o, h, l, c = O[i, j], H[i, j], L[i, j], C[i, j]
            px = None
            if o <= q["sl"]: px = o
            elif l <= q["sl"]: px = q["sl"]
            elif o >= q["tp"]: px = o
            elif h >= q["tp"]: px = q["tp"]
            elif i - q["i"] >= p["max_hold"]: px = c
            if px is not None:
                close_pos(j, q, px, i)
                del pos[j]

        prev = Cf[i - 1] if i > 0 else Cf[i]
        equity_now = cash + sum(q["qty"] * np.nan_to_num(prev[j]) for j, q in pos.items())
        mk = (dates[i].year, dates[i].month)
        if mk != cur_month:
            cur_month, month_start = mk, equity_now
        allow = equity_now >= month_start * (1 - p["month_stop"])

        if allow:
            for j, atr in pend:
                if len(pos) >= p["max_pos"]:
                    break
                o = O[i, j]
                if np.isnan(o) or j in pos or o > prev[j] * 1.03:
                    continue
                t = size_trade(equity_now, cash, o, atr, p)
                if not t:
                    continue
                sl, tp, qty = t
                cash -= qty * o * (1 + p["cost"])
                pos[j] = dict(entry=o, sl=sl, tp=tp, qty=qty, i=i)
                if L[i, j] <= sl:  # stopped out on entry day (conservative)
                    close_pos(j, pos[j], sl, i)
                    del pos[j]
        pend = []
        if reg[i] and allow:
            cand = np.where(SIG[i] & (R[i] > ir[i]))[0]
            cand = sorted(cand, key=lambda j: -R[i, j])
            pend = [(j, A[i, j]) for j in cand if j not in pos]
        eq[i] = cash + sum(q["qty"] * np.nan_to_num(Cf[i, j]) for j, q in pos.items())

    s = pd.Series(eq, index=dates).iloc[210:]
    return s, pd.DataFrame(trades)


def metrics(eq, tr):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    mon = eq.groupby(eq.index.to_period("M")).last().pct_change().dropna()
    out = {"CAGR": f"{cagr:.1%}", "Avg monthly return": f"{mon.mean():.2%}",
           "Months >= 5%": f"{(mon >= 0.05).mean():.0%}", "Losing months": f"{(mon < 0).mean():.0%}",
           "Worst month": f"{mon.min():.1%}", "Max drawdown": f"{dd:.1%}"}
    if len(tr):
        w = tr[tr.pnl > 0].pnl.sum()
        lo = -tr[tr.pnl <= 0].pnl.sum()
        out.update({"Trades": len(tr), "Win rate": f"{(tr.pnl > 0).mean():.0%}",
                    "Profit factor": f"{w / lo:.2f}" if lo else "n/a"})
    return out, mon


# ------------------------------------------------------------------- UI
st.title("Nifty 500 Swing Bot")
st.caption("Educational tool. Returns are never guaranteed. Not investment advice. "
           "Backtests use today's Nifty 500 list (survivorship bias) and can differ from live results.")

with st.sidebar:
    st.header("Capital & risk")
    P = {}
    P["capital"] = st.number_input("Capital (₹)", 10000, 100000000, 100000, 10000)
    P["risk"] = st.slider("Risk per trade (% of capital)", 0.25, 3.0, 1.0, 0.25) / 100
    P["max_pos"] = st.slider("Max open positions", 1, 10, 5)
    P["max_pct"] = st.slider("Max % of capital in one stock", 5, 40, 20) / 100
    P["month_stop"] = st.slider("Stop new trades if month drawdown (%)", 2.0, 15.0, 6.0) / 100
    st.header("Strategy")
    P["sl_atr"] = st.slider("Stop loss (x ATR)", 1.0, 4.0, 2.0, 0.25)
    P["rr"] = st.slider("Target (x risk)", 1.0, 4.0, 2.0, 0.25)
    P["breakout"] = st.slider("Breakout lookback (days)", 10, 60, 20)
    P["volx"] = st.slider("Volume multiple", 1.0, 3.0, 1.5, 0.1)
    P["max_hold"] = st.slider("Max holding days", 5, 60, 30)
    P["cost"] = st.slider("Cost + slippage per side (%)", 0.0, 0.5, 0.12, 0.01) / 100
    st.header("Data")
    N = st.slider("Stocks to scan (top of Nifty 500 list)", 30, 500, 200, 10)
    YEARS = st.slider("Backtest years", 2, 10, 10)

uni = load_universe().head(N)
sector_of = dict(zip(uni.symbol, uni.industry))
tab1, tab2 = st.tabs(["Today's scanner", "10-year backtest"])

with tab1:
    if st.button("Run scanner", type="primary"):
        with st.spinner("Downloading data..."):
            data = download(tuple(uni.symbol), 2)
            idx = download_index(2)
        reg_ok = bool(idx.Close.iloc[-1] > idx.Close.rolling(200).mean().iloc[-1])
        st.metric("Market regime (Nifty vs 200-DMA)", "BULLISH - trades allowed" if reg_ok else "BEARISH - stay in cash")
        irx = idx.Close.pct_change(126).iloc[-1]
        rows, sect = [], {}
        for s, d in data.items():
            d = prep(d, P)
            r = d.iloc[-1]
            sect.setdefault(sector_of.get(s, "NA"), []).append(d.Close.iloc[-1] / d.Close.iloc[-64] - 1)
            if r.signal and r.ret126 > irx:
                t = size_trade(P["capital"], P["capital"], r.Close, r.atr, P)
                if t:
                    sl, tp, qty = t
                    rows.append(dict(Symbol=s, Sector=sector_of.get(s, "NA"), Entry=round(r.Close, 2),
                                     StopLoss=round(sl, 2), Target=round(tp, 2), Qty=qty,
                                     Invest=round(qty * r.Close), RiskRs=round(qty * (r.Close - sl)),
                                     Strength6m=f"{r.ret126:.0%}", _s=r.ret126))
        st.subheader("Signals")
        if not reg_ok:
            st.warning("Market filter is off. The bot suggests no new trades.")
        if rows:
            df = pd.DataFrame(rows).sort_values("_s", ascending=False).drop(columns="_s").head(P["max_pos"])
            st.dataframe(df, use_container_width=True)
            st.caption("Buy only if price trades above the Entry on the next session and does not gap more than 3% higher. "
                       "Place the stop loss immediately.")
        else:
            st.info("No qualifying signals today.")
        st.subheader("Sector momentum (3-month average return)")
        sdf = pd.DataFrame({"Sector": list(sect), "3M return": [np.mean(v) for v in sect.values()],
                            "Stocks": [len(v) for v in sect.values()]}).sort_values("3M return", ascending=False)
        sdf["3M return"] = sdf["3M return"].map("{:.1%}".format)
        st.dataframe(sdf, use_container_width=True)

with tab2:
    st.write("Runs the same rules day by day over the past years with your current settings, "
             "including costs, gaps and stop-before-target on ambiguous days.")
    if st.button("Run backtest"):
        with st.spinner("Downloading data and simulating (can take a few minutes)..."):
            data = download(tuple(uni.symbol), YEARS)
            idx = download_index(YEARS)
            eq, tr = backtest(data, idx, P)
        m, mon = metrics(eq, tr)
        cols = st.columns(len(m))
        for c, (k, v) in zip(cols, m.items()):
            c.metric(k, v)
        st.line_chart(eq)
        pv = mon.to_frame("r")
        pv["y"], pv["m"] = pv.index.year, pv.index.month
        st.subheader("Monthly returns")
        st.dataframe(pv.pivot(index="y", columns="m", values="r").style.format("{:.1%}", na_rep=""),
                     use_container_width=True)
        if len(tr):
            with st.expander("All trades"):
                st.dataframe(tr, use_container_width=True)
        st.warning("If 'Months >= 5%' is well below 100%, that is the honest answer to the 5% monthly goal. "
                   "Avoid over-tuning the sliders to the past; that fits noise, not the future.")
