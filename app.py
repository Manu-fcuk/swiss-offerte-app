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
    .intel-box { padding: 15px; border-radius: 8px; background-color: #1c2128; border-left: 5px solid #238636; margin-bottom: 20px; min-height: 120px; }
    .fomc-card { background-color: #1c2128; padding: 15px; border-radius: 10px; border-top: 3px solid #238636; margin-bottom: 10px; }
    .disclaimer { font-size: 13px; color: #8b949e; margin-top: 50px; border: 1px solid #da3633; padding: 20px; border-radius: 10px; background-color: #211111; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE ENGINE FUNCTIONS ---

@st.cache_data(ttl=86400)
def get_ticker_details(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {"Name": info.get('longName', ticker), "Sector": info.get('sector', 'N/A')}
    except: return {"Name": ticker, "Sector": "N/A"}

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
    if len(bm_prices) < 200: return "N/A", "Loading...", "N/A", "", "Neutral"
    sma50 = bm_prices.rolling(50).mean().iloc[-1]
    sma200 = bm_prices.rolling(200).mean().iloc[-1]
    curr = bm_prices.iloc[-1]
    
    # Phase Logic
    if curr > sma50 and sma50 > sma200: phase, advice = "PHASE 2: Markup 🚀", "Strong Uptrend. Risk-on."
    elif curr < sma50 and curr > sma200: phase, advice = "PHASE 3: Distribution ⚠️", "Topping/Correction. Tighten stops."
    elif curr < sma50 and curr < sma200: phase, advice = "PHASE 4: Markdown 🔴", "Downtrend. Protect capital."
    else: phase, advice = "PHASE 1: Accumulation 📈", "Basing. Watch RS breakouts."

    # Seasonality
    m_idx = datetime.now().month
    seasonal_data = {
        1: ("Januar Effekt", "New inflows."), 2: ("Feb Volatility", "Chop typical."),
        3: ("Q-End Drive", "Window dressing."), 4: ("April Gold", "Top performance month."),
        5: ("Sell in May", "Selective mode."), 9: ("Sept Blues", "Historically worst month."),
        11: ("Year-End Turbo", "Strongest seasonal wind."), 12: ("Santa Rally", "Bullish holiday bias.")
    }
    s_title, s_desc = seasonal_data.get(m_idx, ("Neutral Season", "Market following technicals."))
    
    # Fear & Greed Simulation based on RSI of Benchmark
    mkt_rsi = calc_rsi(bm_prices).iloc[-1]
    sentiment = "Greed 🔥" if mkt_rsi > 65 else "Fear 😨" if mkt_rsi < 35 else "Neutral ⚖️"
    
    return phase, advice, s_title, s_desc, sentiment

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620611.png", width=60)
    st.title("Alpha Terminal v5.0")
    user_input = st.text_area("Watchlist Tickers:", value="GOOG, AAPL, AMZN, WMT, T, META, NVDA, TSLA, MSFT, LLY, GE, PYPL, SNAP, ASML, PLTR, ADBE, NKE, KO, PFE, UBER, MCD, SBUX, RIO, ORCL, ABNB, BTI, JNJ, PEP, BA, AAL", height=150)
    portfolio_list = [x.strip().upper() for x in user_input.split(",") if x.strip()]
    st.divider()
    start_year = st.number_input("Backtest Start Year", 2015, 2024, 2021)
    end_year = st.number_input("Backtest End Year", 2016, 2026, 2025)
    bt_univ = st.radio("Backtest Universe", ["Watchlist", "S&P 500"], horizontal=True)
    hold_period = st.select_slider("Holding Period (Months)", options=[1, 2, 3, 4, 6], value=1)
    sl_pct = st.slider("Stop Loss (%)", 5, 30, 15)

# --- 4. DATA ---
@st.cache_data(ttl=3600)
def fetch_live_data(tickers):
    bm = yf.download("^GSPC", period="4y", progress=False)['Close']
    if isinstance(bm, pd.DataFrame): bm = bm.iloc[:, 0]
    df = yf.download(tickers, period="4y", progress=False)['Close']
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df.ffill(), bm.ffill()

all_data, benchmark = fetch_live_data(portfolio_list)

# --- 5. HEADER & STATUS ---
sma200_live = benchmark.rolling(200).mean()
market_bullish = benchmark.iloc[-1] > sma200_live.iloc[-1]
phase, advice, s_title, s_desc, sentiment = get_market_intelligence(benchmark)

if market_bullish:
    st.markdown(f'<div class="status-box" style="background-color: #238636; color: white;">MARKET STATUS: BULLISH 🟢 | SENTIMENT: {sentiment}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-box" style="background-color: #da3633; color: white;">MARKET STATUS: BEARISH 🔴 | SENTIMENT: {sentiment}</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="intel-box"><b>🧠 Phase:</b><br>{phase}<br><span style="font-size: 14px;">{advice}</span></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="intel-box" style="border-color: #f1e05a;"><b>📅 Saisonalität:</b><br>{s_title}<br><span style="font-size: 14px;">{s_desc}</span></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="intel-box" style="border-color: #388bfd;"><b>🌐 FOMC Target:</b><br>Mar 19, 2025<br><span style="font-size: 14px;">Market expects stability. Watch the Dot Plot!</span></div>', unsafe_allow_html=True)

# --- 6. TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Action Plan", "🔭 Scanner", "📈 Charts", "🧪 Backtest", "📖 Strategy & FOMC"])

with tab1:
    results = []
    for t in portfolio_list:
        if t in all_data.columns and t != "^GSPC":
            p = all_data[t].dropna()
            rs_s = calc_rs_stable(p, benchmark)
            if not rs_s.empty:
                det = get_ticker_details(t)
                rs_v, rsi_v = rs_s.iloc[-1], calc_rsi(p).iloc[-1]
                sma200_s = p.rolling(200).mean().iloc[-1]
                action = "🚨 SYSTEM EXIT" if p.iloc[-1] < sma200_s else "🟢 HOLD" if rs_v > 0 else "🔴 SELL"
                results.append({"Ticker": t, "Name": det["Name"], "Sector": det["Sector"], "RS Score": rs_v, "RSI": rsi_v, "Action": action})
    st.dataframe(pd.DataFrame(results).sort_values(by="RS Score", ascending=False).style.map(lambda x: f'background-color: {"#238636" if "HOLD" in str(x) else "#da3633" if any(y in str(x) for y in ["SELL", "EXIT"]) else ""}; color: white', subset=['Action']).background_gradient(subset=['RS Score'], cmap='RdYlGn').format(subset=['RS Score', 'RSI'], formatter="{:.2f}"), width='stretch', hide_index=True)

with tab2:
    if st.button("🚀 Run Universe Scanner"):
        sp500 = get_sp500_tickers(); sp_raw = yf.download(sp500, period="1y", progress=False)['Close']
        if isinstance(sp_raw.columns, pd.MultiIndex): sp_raw.columns = sp_raw.columns.get_level_values(0)
        opps = []
        for t in sp500:
            if t in sp_raw.columns and t not in portfolio_list:
                p = sp_raw[t].ffill()
                if len(p) < 100: continue
                rs_s = calc_rs_stable(p, benchmark.reindex(p.index).ffill())
                if not rs_s.empty and rs_s.iloc[-1] > 0.12:
                    det = get_ticker_details(t); opps.append({"Ticker": t, "Name": det["Name"], "Sector": det["Sector"], "RS Score": rs_s.iloc[-1]})
        df_opps = pd.DataFrame(opps).sort_values(by="RS Score", ascending=False).head(15); st.table(df_opps)
        st.code(", ".join([t for t in df_opps['Ticker'].tolist() if t not in portfolio_list]))

with tab3:
    sel_t = st.selectbox("Deep Dive:", portfolio_list)
    if sel_t and sel_t in all_data.columns:
        df_c = yf.download(sel_t, period="1y", progress=False)
        if isinstance(df_c.columns, pd.MultiIndex): df_c.columns = df_c.columns.get_level_values(0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c['Close'].rolling(200).mean(), line=dict(color='red'), name="SMA 200"), row=1, col=1)
        rs_l = calc_rs_stable(df_c['Close'], benchmark.reindex(df_c.index).ffill())
        fig.add_trace(go.Scatter(x=rs_l.index, y=rs_l, fill='tozeroy', line=dict(color='lime'), name="RS"), row=2, col=1)
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False); st.plotly_chart(fig, width='stretch')

with tab4:
    col_a, col_c = st.columns(2)
    with col_a: cap_start = st.number_input("Seed Capital (USD):", value=10000)
    with col_c: n_stocks = st.slider("Positions:", 3, 10, 5)
    if st.button("🧪 Execute Backtest"):
        with st.spinner("Calculating..."):
            bt_ticks = portfolio_list if bt_univ == "Watchlist" else get_sp500_tickers()
            s_d, e_d = f"{start_year}-01-01", f"{end_year}-12-31"
            s_p = (datetime.strptime(s_d, "%Y-%m-%d") - timedelta(days=450)).strftime("%Y-%m-%d")
            bm_f = yf.download("^GSPC", start=s_p, end=e_d, progress=False)['Close']
            if isinstance(bm_f, pd.DataFrame): bm_f = bm_f.iloc[:, 0]
            bt_d = yf.download(bt_ticks, start=s_p, end=e_d, progress=False)['Close']
            if isinstance(bt_d.columns, pd.MultiIndex): bt_d.columns = bt_d.columns.get_level_values(0)
            common = bt_d.index.intersection(bm_f.index); bt_d, bm_f = bt_d.loc[common].ffill(), bm_f.loc[common].ffill()
            rs_h = pd.DataFrame(index=bt_d.index)
            for t in bt_ticks:
                if t in bt_d.columns: rs_h[t] = calc_rs_stable(bt_d[t], bm_f)
            t_d = bt_d.loc[s_d:e_d].groupby(pd.Grouper(freq=f'{hold_period}ME')).apply(lambda x: x.index[-1] if not x.empty else None).dropna()
            c, c_h, t_l, f_dt = cap_start, [], [], None
            for i in range(len(t_d)-1):
                cur, nxt = t_d[i], t_d[i+1]; is_bull = bm_f.loc[cur] > bm_f.rolling(200).mean().loc[cur]
                exp = 1.0 if is_bull else 0.2; rank = rs_h.loc[cur].dropna().sort_values(ascending=False)
                if len(rank) < n_stocks: continue
                if f_dt is None: f_dt = cur
                sel = rank.head(n_stocks).index.tolist(); cash_p_s = (c * exp / n_stocks) * 0.9988
                p_r, det = [], []
                for tkr in sel:
                    b_p = bt_d.loc[cur, tkr]; d_p = bt_d.loc[cur:nxt, tkr]
                    if d_p.min() <= b_p * (1-sl_pct/100): ex_p, stt = b_p * (1-sl_pct/100), "🚨SL"
                    else: ex_p, stt = d_p.iloc[-1], "OK"
                    p_r.append(cash_p_s * (ex_p / b_p)); det.append(f"{tkr}(In:{b_p:.1f}/Out:{ex_p:.1f}/{ex_p/b_p-1:+.1%}/{stt})")
                c_pv = c; c = sum(p_r) + (c * (1-exp))
                s_pf, b_pf = (c/c_pv-1)*100, (bm_f.loc[nxt]/bm_f.loc[cur]-1)*100
                c_h.append({"Date": nxt, "Strategy": c}); t_l.append({"Date": cur.date(), "Regime": "BULL" if is_bull else "BEAR", "Trades": " | ".join(det), "Strat%": s_pf, "S&P500%": b_pf, "Alpha%": s_pf-b_pf, "Value": c})
            if c_h:
                res = pd.DataFrame(c_h).set_index("Date"); st.metric("Final Value", f"{c:,.0f} USD", f"{(c/cap_start-1)*100:.1f}%")
                fig_bt = go.Figure(); fig_bt.add_trace(go.Scatter(x=res.index, y=res['Strategy'], name="Strategy", line=dict(color='lime', width=3))); fig_bt.add_trace(go.Scatter(x=res.index, y=(bm_f.reindex(res.index)/bm_f.loc[f_dt])*cap_start, name="S&P 500", line=dict(color='gray', dash='dash')))
                st.plotly_chart(fig_bt, width='stretch'); log_df = pd.DataFrame(t_l).sort_values(by="Date", ascending=False)
                st.dataframe(log_df.style.map(lambda x: 'background-color: #238636; color: white' if str(x) == "BULL" else 'background-color: #da3633; color: white' if str(x) == "BEAR" else '', subset=['Regime']).map(lambda x: 'color: #238636; font-weight: bold' if (isinstance(x, float) and x > 0) else 'color: #da3633' if (isinstance(x, float) and x < 0) else '', subset=['Alpha%']).format(subset=['Strat%', 'S&P500%', 'Alpha%'], formatter="{:+.2f}%").format(subset=['Value'], formatter="{:,.0f}"), width='stretch', hide_index=True)

with tab5:
    st.header("📖 Institutional Strategy Guide")
    st.markdown("""
    ### 1. Relative Strength (RS) Engine
    We buy momentum leaders. The Mansfield RS Score identifies stocks outperforming the S&P 500. Smart money flows into these assets over weeks.
    ### 2. Market Regime & FOMC
    - **Bull Markets (🟢):** FED is neutral/supportive. S&P 500 > SMA 200. Maximize risk (100% Exposure).
    - **Bear Markets (🔴):** FED is aggressive/hostile. S&P 500 < SMA 200. Strategy shifts to 80% Cash.
    ### 3. Risk Protocols
    - **Sector Mix:** Max 2 stocks per industry group to avoid clustering risk.
    - **Stop Loss:** 15% automatic exit protects the principal.
    ### 4. 2025 FOMC Tracker
    - **March 18-19:** Watch for Dot Plot changes.
    - **May 6-7 / June 17-18:** Interest Rate Decisions. 
    """)
    st.markdown(f'<div class="disclaimer">⚠️ **LEGAL NOTICE:** Not financial advice. Investing involves risk. Past performance is no guarantee of results. © {datetime.now().year} Manuel Kössler.</div>', unsafe_allow_html=True)