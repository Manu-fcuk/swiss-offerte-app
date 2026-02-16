import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Quant-Engineer Pro | Alpha Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 5px; color: white; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #238636; }
    div[data-testid="stExpander"] { border: 1px solid #30363d; background-color: #0e1117; }
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
    except: return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]

def calc_rs(prices, bm_prices):
    ratio = prices / bm_prices
    return (ratio / ratio.rolling(window=50).mean()) - 1

def calc_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def fetch_data(tickers, period="2y"):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers + ["^GSPC"], period=period, interval="1d", progress=False)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.ffill()

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620611.png", width=60)
    st.title("Alpha Terminal v1.0")
    st.caption("Quant-Driven Portfolio Management")
    
    st.divider()
    st.header("📋 Global Setup")
    default_tickers = "GOOG, AAPL, AMZN, WMT, T, META, NVDA, TSLA, MSFT, LLY, GE, PYPL, SNAP, ASML, PLTR, ADBE, NKE, KO, PFE, UBER, MCD, SBUX, RIO, ORCL, ABNB, BTI, JNJ, PEP, BA, AAL"
    user_input = st.text_area("Watchlist Tickers (Max 30):", value=default_tickers, height=150)
    portfolio_list = [x.strip().upper() for x in user_input.split(",")]
    
    st.divider()
    st.header("🧪 Backtest Config")
    bt_years = st.select_slider("Simulation Horizon (Years)", options=[1, 3, 5, 10], value=3)
    bt_univ = st.radio("Strategy Universe", ["Watchlist", "S&P 500 Index"], horizontal=True)
    
    st.divider()
    st.caption("© 2024 Manuel Kössler | Professional Financial Tool")

# --- 4. DATA PROCESSING ---
market_data = fetch_data(portfolio_list)
benchmark = market_data["^GSPC"]

# --- 5. MAIN INTERFACE ---
st.title("💹 Quant-Engineer Alpha Terminal")
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Action Plan", "🔭 Opportunity Scanner", "📈 Technical Analysis", "🧪 Strategy Backtest"])

# TAB 1: PORTFOLIO ACTION
with tab1:
    st.subheader("Current Portfolio Health & Allocation Signals")
    results = []
    for t in portfolio_list:
        if t in market_data.columns and t != "^GSPC":
            p = market_data[t]; curr_p = p.iloc[-1]
            rs = calc_rs(p, benchmark).iloc[-1]
            rsi = calc_rsi(p).iloc[-1]
            sma200 = p.rolling(200).mean().iloc[-1]
            
            action = "🟢 HOLD" if rs > 0 else "🔴 SELL"
            if curr_p < sma200: action = "🚨 TREND BREAK"
            
            results.append({
                "Ticker": t, "Name": get_company_name(t), "RS Score": rs, 
                "RSI(14)": rsi, "Market Trend": "Bull" if curr_p > sma200 else "Bear", "RECOMMENDATION": action
            })
    
    res_df = pd.DataFrame(results).sort_values(by="RS Score", ascending=False)
    st.dataframe(res_df.style.background_gradient(subset=['RS Score'], cmap='RdYlGn'), use_container_width=True, hide_index=True)

# TAB 2: OPPORTUNITY SCANNER
with tab2:
    st.subheader("Global Leader Scan (S&P 500 Universe)")
    if st.button("🚀 Run Market-Wide Scan"):
        with st.spinner("Analyzing 500 tickers..."):
            sp500 = get_sp500_tickers()
            sp_data = yf.download(sp500 + ["^GSPC"], period="1y", progress=False)['Close']
            if isinstance(sp_data.columns, pd.MultiIndex): sp_data.columns = sp_data.columns.get_level_values(0)
            bm_sp = sp_data["^GSPC"].ffill()
            opps = []
            for t in sp500:
                if t in sp_data.columns and t not in portfolio_list:
                    p = sp_data[t].ffill()
                    if len(p) < 100: continue
                    rs = ((p / bm_sp) / (p / bm_sp).rolling(50).mean() - 1).iloc[-1]
                    if rs > 0.12: opps.append({"Ticker": t, "Name": get_company_name(t), "RS Score": rs})
            st.table(pd.DataFrame(opps).sort_values(by="RS Score", ascending=False).head(15))

# TAB 3: TECHNICAL ANALYSIS
with tab3:
    c1, c2 = st.columns([1, 4])
    with c1: sel_t = st.selectbox("Select Asset:", portfolio_list)
    if sel_t:
        hist = yf.download(sel_t, period="1y", progress=False)
        if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        hist['SMA200'] = hist['Close'].rolling(200).mean()
        rsi_val = calc_rsi(hist['Close']).iloc[-1]
        
        with c1: st.metric("RSI (14d)", f"{rsi_val:.2f}", "Overbought" if rsi_val > 70 else "Normal", delta_color="inverse" if rsi_val > 70 else "normal")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Candles"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange'), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='red'), name="SMA 200"), row=1, col=1)
        rs_l = (hist['Close'] / benchmark.reindex(hist.index).ffill()); rs_l = (rs_l / rs_l.rolling(50).mean()) - 1
        fig.add_trace(go.Scatter(x=rs_l.index, y=rs_l, fill='tozeroy', line=dict(color='lime'), name="Relative Strength"), row=2, col=1)
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: BACKTEST ENGINE (THE "SALES" FEATURE)
with tab4:
    st.subheader("🧪 Professional Strategy Simulation")
    col_a, col_b, col_c = st.columns(3)
    with col_a: cap_start = st.number_input("Seed Capital (USD):", value=10000)
    with col_b: freq = st.selectbox("Frequency:", ["Monthly", "Weekly"], index=0)
    with col_c: n_stocks = st.slider("Stocks in Portfolio:", 3, 10, 3)
    
    if st.button("🧪 Execute Simulation"):
        with st.spinner("Processing Alpha..."):
            bt_ticks = portfolio_list if bt_univ == "Watchlist" else get_sp500_tickers()
            s_date = (datetime.now() - timedelta(days=bt_years*365 + 100)).strftime('%Y-%m-%d')
            data_bt = yf.download(bt_ticks + ["^GSPC"], start=s_date, progress=False)['Close']
            if isinstance(data_bt.columns, pd.MultiIndex): data_bt.columns = data_bt.columns.get_level_values(0)
            data_bt = data_bt.ffill().dropna(axis=1, how='all')
            bm = data_bt["^GSPC"]
            
            rs_h = pd.DataFrame(index=data_bt.index)
            for t in bt_ticks:
                if t in data_bt.columns and t != "^GSPC":
                    rat = data_bt[t] / bm
                    rs_h[t] = (rat / rat.rolling(50).mean()) - 1
            
            f_code = 'ME' if freq == "Monthly" else 'W-MON'
            t_dates = data_bt.groupby(pd.Grouper(freq=f_code)).apply(lambda x: x.index[-1]).dropna()
            
            c = cap_start; c_history = []; t_log = []; f_date = None
            for i in range(len(t_dates)-1):
                cur, nxt = t_dates[i], t_dates[i+1]
                if cur not in rs_h.index: continue
                rank = rs_h.loc[cur].dropna().sort_values(ascending=False)
                if len(rank) < n_stocks: continue
                if f_date is None: f_date = cur
                
                sel = rank.head(n_stocks).index.tolist()
                cash_p_s = (c / n_stocks) * 0.998 # 0.2% Fee
                p_res = []
                details = []
                for ticker in sel:
                    b_p = data_bt.loc[cur, ticker]; d_p = data_bt.loc[cur:nxt, ticker]
                    low_p = d_p.min(); fin_p = d_p.iloc[-1]
                    if low_p <= b_p * 0.85: # 15% Stop Loss
                        exit_p = b_p * 0.85; stat = "🚨SL"
                    else: exit_p = fin_p; stat = "OK"
                    p_res.append(cash_p_s * (exit_p / b_p))
                    details.append(f"{ticker}(In:{b_p:.1f}/Out:{exit_p:.1f}/{stat})")
                
                c_prev = c; c = sum(p_res)
                s_perf = (c / c_prev - 1) * 100; b_perf = (bm.loc[nxt] / bm.loc[cur] - 1) * 100
                c_history.append({"Date": nxt, "Strategy": c})
                t_log.append({"Date": cur.date(), "Trades": " | ".join(details), "Strat%": f"{s_perf:+.1f}%", "BM%": f"{b_perf:+.1f}%", "Alpha": f"{s_perf-b_perf:+.1f}%", "Value": f"{c:,.0f}"})

            if c_history:
                res = pd.DataFrame(c_history).set_index("Date")
                bm_curve = (bm.reindex(res.index) / bm.loc[f_date]) * cap_start
                st.metric("Final Capital", f"{c:,.0f} USD", f"{(c/cap_start-1)*100:.1f}% Total Return")
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['Strategy'], name="Strategy (Stop-Loss)", line=dict(color='lime', width=3)))
                fig_bt.add_trace(go.Scatter(x=res.index, y=bm_curve, name="S&P 500 Benchmark", line=dict(color='gray', dash='dash')))
                fig_bt.update_layout(template="plotly_dark", height=500); st.plotly_chart(fig_bt, use_container_width=True)
                with st.expander("📜 Full Audit Log"): st.dataframe(pd.DataFrame(t_log).sort_values(by="Date", ascending=False), use_container_width=True)

st.divider()
st.caption("Quant-Engineer Alpha Terminal | Level: Institutional Grade | Created for Professional Portfolio Scaling")