import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Alpha Terminal Pro | Quant-Engineer", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 5px; color: white; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #238636; }
    div[data-testid="stExpander"] { border: 1px solid #30363d; background-color: #0e1117; }
    .status-box { padding: 25px; border-radius: 12px; text-align: center; font-weight: bold; font-size: 26px; margin-bottom: 10px; border: 2px solid #30363d; }
    .intel-box { padding: 15px; border-radius: 8px; background-color: #1c2128; border-left: 5px solid #238636; margin-bottom: 20px; }
    .disclaimer { font-size: 12px; color: #8b949e; margin-top: 50px; border-top: 1px solid #30363d; padding-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE ENGINE FUNCTIONS ---

@st.cache_data(ttl=86400)
def get_company_name(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info.get('longName', ticker)
    except: return ticker

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        return [t.replace('.', '-') for t in pd.read_html(response.text)[0]['Symbol'].tolist()]
    except: return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

def calc_rs_stable(prices, bm_prices):
    combined = pd.concat([prices, bm_prices], axis=1).ffill().dropna()
    if combined.empty: return pd.Series()
    combined.columns = ['Asset', 'BM']
    ratio = combined['Asset'] / combined['BM']
    return (ratio / ratio.rolling(window=50).mean()) - 1

def calc_rsi(prices, window=14):
    if len(prices) < window: return pd.Series([50]*len(prices))
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).ffill()))

def get_market_intelligence(bm_prices):
    sma50 = bm_prices.rolling(50).mean().iloc[-1]
    sma200 = bm_prices.rolling(200).mean().iloc[-1]
    curr = bm_prices.iloc[-1]
    if curr > sma50 and sma50 > sma200:
        phase, advice = "PHASE 2: Aufwärtstrend 🚀", "Aggressives Momentum. Leader halten."
    elif curr < sma50 and curr > sma200:
        phase, advice = "PHASE 3: Top-Bildung / Korrektur ⚠️", "Vorsicht bei Neukäufen. Stops prüfen."
    elif curr < sma50 and curr < sma200:
        phase, advice = "PHASE 4: Abwärtstrend 🔴", "Kapital schützen. Cash halten."
    else:
        phase, advice = "PHASE 1: Bodenbildung 📈", "Geduld. Auf erste Ausbrüche warten."
    m_idx = datetime.now().month
    seasonal_map = {1: "Positiv", 2: "Neutral", 3: "Positiv", 4: "Sehr Positiv", 5: "Vorsicht", 6: "Neutral", 7: "Positiv", 8: "Schwach", 9: "Sehr Schwach", 10: "Bodenbildung", 11: "Sehr Stark", 12: "Sehr Stark"}
    return phase, advice, seasonal_map.get(m_idx, "N/A")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620611.png", width=60)
    st.title("Alpha Terminal v2.6")
    user_input = st.text_area("Watchlist Tickers:", value="GOOG, AAPL, AMZN, WMT, T, META, NVDA, TSLA, MSFT, LLY, GE, PYPL, SNAP, ASML, PLTR, ADBE, NKE, KO, PFE, UBER, MCD, SBUX, RIO, ORCL, ABNB, BTI, JNJ, PEP, BA, AAL", height=150)
    portfolio_list = [x.strip().upper() for x in user_input.split(",") if x.strip()]
    st.divider()
    start_year = st.number_input("Start Year", 2015, 2024, 2021)
    end_year = st.number_input("End Year", 2016, 2026, 2025)
    bt_univ = st.radio("Universe", ["Watchlist", "S&P 500"], horizontal=True)
    hold_period = st.select_slider("Holding Period (Months)", options=[1, 2, 3, 4, 6], value=1)
    sl_pct = st.slider("Stop Loss (%)", 5, 30, 15)

# --- 4. DATA ---
all_live_data = yf.download(portfolio_list + ["^GSPC"], period="3y", progress=False)['Close']
if isinstance(all_live_data.columns, pd.MultiIndex): all_live_data.columns = all_live_data.columns.get_level_values(0)
all_live_data = all_live_data.ffill()
benchmark_live = all_live_data["^GSPC"]
sma200_live = benchmark_live.rolling(200).mean()
market_bullish = benchmark_live.iloc[-1] > sma200_live.iloc[-1]

# --- 5. HEADER ---
phase, advice, seasonal = get_market_intelligence(benchmark_live)
if market_bullish:
    st.markdown(f'<div class="status-box" style="background-color: #238636; color: white;">MARKET STATUS: BULLISH 🟢</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-box" style="background-color: #da3633; color: white;">MARKET STATUS: BEARISH 🔴</div>', unsafe_allow_html=True)

c_i1, c_i2 = st.columns(2)
with c_i1: st.markdown(f'<div class="intel-box" style="border-color: #238636;"><b>Trend-Phase:</b> {phase}<br><i>{advice}</i></div>', unsafe_allow_html=True)
with c_i2: st.markdown(f'<div class="intel-box" style="border-color: #f1e05a;"><b>Saisonalität:</b> {seasonal}<br><i>Historische Tendenz.</i></div>', unsafe_allow_html=True)

# --- 6. TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Action", "🔭 Scanner", "📈 Charts", "🧪 Backtest", "📖 Info"])

with tab1:
    results = []
    for t in portfolio_list:
        if t in all_live_data.columns and t != "^GSPC":
            p = all_live_data[t].dropna()
            rs_s = calc_rs_stable(p, benchmark_live)
            if not rs_s.empty:
                rs_v, rsi_v, sma200_s = rs_s.iloc[-1], calc_rsi(p).iloc[-1], p.rolling(200).mean().iloc[-1]
                action = "🚨 TREND BREAK" if p.iloc[-1] < sma200_s else "🟢 HOLD" if rs_v > 0 else "🔴 SELL"
                results.append({"Ticker": t, "Name": get_company_name(t), "RS Score": rs_v, "RSI": rsi_v, "Action": action})
    st.dataframe(pd.DataFrame(results).sort_values(by="RS Score", ascending=False).style.map(lambda x: f'background-color: {"#238636" if "HOLD" in str(x) else "#da3633" if any(y in str(x) for y in ["SELL", "BREAK"]) else ""}; color: white', subset=['Action']).background_gradient(subset=['RS Score'], cmap='RdYlGn').format(subset=['RS Score', 'RSI'], formatter="{:.2f}"), width='stretch', hide_index=True)

with tab2:
    if st.button("🚀 Run S&P 500 Scan"):
        sp500 = get_sp500_tickers()
        sp_raw = yf.download(sp500 + ["^GSPC"], period="1y", progress=False)['Close']
        if isinstance(sp_raw.columns, pd.MultiIndex): sp_raw.columns = sp_raw.columns.get_level_values(0)
        bm_sp = sp_raw["^GSPC"].ffill()
        opps = []
        for t in sp500:
            p = sp_raw[t].ffill()
            if len(p) > 100:
                rs_s = calc_rs_stable(p, bm_sp)
                if not rs_s.empty and rs_s.iloc[-1] > 0.12: opps.append({"Ticker": t, "Name": get_company_name(t), "RS Score": rs_s.iloc[-1]})
        df_opps = pd.DataFrame(opps).sort_values(by="RS Score", ascending=False).head(15)
        st.table(df_opps)
        st.code(", ".join([t for t in df_opps['Ticker'].tolist() if t not in portfolio_list]))

with tab3:
    sel_t = st.selectbox("Select Asset:", portfolio_list)
    if sel_t:
        hist = yf.download(sel_t, period="1y", progress=False)
        if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), line=dict(color='red'), name="SMA 200"), row=1, col=1)
        rs_l = calc_rs_stable(hist['Close'], benchmark_live)
        fig.add_trace(go.Scatter(x=rs_l.index, y=rs_l, fill='tozeroy', line=dict(color='lime'), name="RS"), row=2, col=1)
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False); st.plotly_chart(fig, width='stretch')

with tab4:
    col_a, col_c = st.columns(2)
    with col_a: cap_start = st.number_input("Capital (USD):", value=10000)
    with col_c: n_stocks = st.slider("Stocks:", 3, 10, 3)
    if st.button("🧪 Run Backtest"):
        with st.spinner("Processing Strategy..."):
            bt_ticks = portfolio_list if bt_univ == "Watchlist" else get_sp500_tickers()
            s_d, e_d = f"{start_year}-01-01", f"{end_year}-12-31"
            s_p = (datetime.strptime(s_d, "%Y-%m-%d") - timedelta(days=450)).strftime("%Y-%m-%d")
            bm_f = yf.download("^GSPC", start=s_p, end=e_d, progress=False)['Close']
            if isinstance(bm_f, pd.DataFrame): bm_f = bm_f.iloc[:, 0]
            bm_s = bm_f.rolling(200).mean()
            bt_d = yf.download(bt_ticks + ["^GSPC"], start=s_p, end=e_d, progress=False)['Close']
            if isinstance(bt_d.columns, pd.MultiIndex): bt_d.columns = bt_d.columns.get_level_values(0)
            bt_d = bt_d.ffill()
            rs_h = pd.DataFrame(index=bt_d.index)
            for t in bt_ticks:
                if t in bt_d.columns and t != "^GSPC": rs_h[t] = calc_rs_stable(bt_d[t], bt_d["^GSPC"])
            f_c = f'{hold_period}ME'
            t_d = bt_d.loc[s_d:e_d].groupby(pd.Grouper(freq=f_c)).apply(lambda x: x.index[-1] if not x.empty else None).dropna()
            c, c_h, t_l, f_dt = cap_start, [], [], None
            for i in range(len(t_d)-1):
                cur, nxt = t_d[i], t_d[i+1]
                if cur not in rs_h.index or cur not in bm_s.index: continue
                is_bull = bm_f.loc[cur] > bm_s.loc[cur]
                exp = 1.0 if is_bull else 0.2
                rank = rs_h.loc[cur].dropna().sort_values(ascending=False)
                if len(rank) < n_stocks: continue
                if f_dt is None: f_dt = cur
                sel = rank.head(n_stocks).index.tolist(); cash_p_s = (c * exp / n_stocks) * 0.9988
                p_r, det = [], []
                for ticker in sel:
                    b_p = bt_d.loc[cur, ticker]; d_p = bt_d.loc[cur:nxt, ticker]
                    if d_p.min() <= b_p * (1-sl_pct/100): ex_p, stt = b_p * (1-sl_pct/100), "🚨SL"
                    else: ex_p, stt = d_p.iloc[-1], "OK"
                    p_r.append(cash_p_s * (ex_p / b_p))
                    # Compact Trade Details within the string
                    det.append(f"{ticker}(In:{b_p:.1f}/Out:{ex_p:.1f}/{ex_p/b_p-1:+.1%}/{stt})")
                
                c_pv = c; c = sum(p_r) + (c * (1-exp))
                s_pf, b_pf = (c/c_pv-1)*100, (bm_f.loc[nxt]/bm_f.loc[cur]-1)*100
                c_h.append({"Date": nxt, "Strategy": c})
                t_l.append({"Date": cur.date(), "Regime": "BULL" if is_bull else "BEAR", "Trades": " | ".join(det), "Strat%": s_pf, "S&P500%": b_pf, "Alpha%": s_pf-b_pf, "Value": c})
            
            if c_h:
                res = pd.DataFrame(c_h).set_index("Date")
                bm_curve = (bm_f.reindex(res.index)/bm_f.loc[f_dt])*cap_start
                m_dd = ((res['Strategy'] - res['Strategy'].cummax())/res['Strategy'].cummax()).min()*100
                st.metric("Final Value", f"{c:,.0f} USD", f"{(c/cap_start-1)*100:.1f}%")
                st.write(f"Max Drawdown: {m_dd:.2f}%")
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['Strategy'], name="Strategy", line=dict(color='lime', width=3)))
                fig_bt.add_trace(go.Scatter(x=res.index, y=bm_curve, name="S&P 500", line=dict(color='gray', dash='dash')))
                st.plotly_chart(fig_bt, width='stretch')
                
                # FINAL AUDIT LOG STYLE
                log_df = pd.DataFrame(t_l).sort_values(by="Date", ascending=False)
                st.dataframe(log_df.style.map(lambda x: 'background-color: #238636; color: white' if str(x) == "BULL" else 'background-color: #da3633; color: white' if str(x) == "BEAR" else '', subset=['Regime'])
                             .map(lambda x: 'color: #238636; font-weight: bold' if (isinstance(x, float) and x > 0) else 'color: #da3633' if (isinstance(x, float) and x < 0) else '', subset=['Alpha%'])
                             .format(subset=['Strat%', 'S&P500%', 'Alpha%'], formatter="{:+.2f}%").format(subset=['Value'], formatter="{:,.0f}"), width='stretch', hide_index=True)

with tab5:
    st.subheader("📖 System Intel")
    st.write("Alpha Terminal v2.6 | Combined Performance Metrics | Compact Trade Audit.")
    st.markdown('<div class="disclaimer">© 2026 Manuel Kössler | Final Production Build</div>', unsafe_allow_html=True)