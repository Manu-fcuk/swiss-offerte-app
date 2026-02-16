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
    .status-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 20px; }
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
    combined.columns = ['Asset', 'BM']
    ratio = combined['Asset'] / combined['BM']
    rs_series = (ratio / ratio.rolling(window=50).mean()) - 1
    return rs_series

def calc_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).ffill()))

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620611.png", width=60)
    st.title("Alpha Terminal v1.5")
    
    st.header("📋 Global Watchlist")
    default_tickers = "GOOG, AAPL, AMZN, WMT, T, META, NVDA, TSLA, MSFT, LLY, GE, PYPL, SNAP, ASML, PLTR, ADBE, NKE, KO, PFE, UBER, MCD, SBUX, RIO, ORCL, ABNB, BTI, JNJ, PEP, BA, AAL"
    user_input = st.text_area("Symbols (Max 30):", value=default_tickers, height=150)
    portfolio_list = [x.strip().upper() for x in user_input.split(",")]
    
    st.divider()
    st.header("🧪 Backtest Range")
    start_year = st.number_input("Start Year", 2015, 2024, 2021)
    end_year = st.number_input("End Year", 2016, 2026, 2024)
    bt_univ = st.radio("Universe", ["Watchlist", "S&P 500"], horizontal=True)

# --- 4. LIVE MARKET DATA ---
@st.cache_data(ttl=3600)
def fetch_live_data(tickers):
    data = yf.download(tickers + ["^GSPC"], period="3y", interval="1d", progress=False)['Close']
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    return data.ffill()

all_live_data = fetch_live_data(portfolio_list)
benchmark_live = all_live_data["^GSPC"]
market_bullish = benchmark_live.iloc[-1] > benchmark_live.rolling(200).mean().iloc[-1]

if market_bullish:
    st.markdown(f'<div class="status-box" style="background-color: #238636; color: white;">MARKET STATUS: BULLISH 🟢 (Full Risk Exposure)</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-box" style="background-color: #da3633; color: white;">MARKET STATUS: BEARISH 🔴 (Defensive Mode)</div>', unsafe_allow_html=True)

# --- 5. TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Action Plan", "🔭 Opportunity Scanner", "📈 Technical Analysis", "🧪 Strategy Backtest"])

# TAB 1: PORTFOLIO ACTION
with tab1:
    results = []
    for t in portfolio_list:
        if t in all_live_data.columns and t != "^GSPC":
            p = all_live_data[t]; curr_p = p.iloc[-1]
            rs = calc_rs_stable(p, benchmark_live).iloc[-1]
            rsi = calc_rsi(p).iloc[-1]
            sma200 = p.rolling(200).mean().iloc[-1]
            
            if curr_p < sma200: action = "🚨 TREND BREAK"
            elif rs > 0: action = "🟢 HOLD"
            else: action = "🔴 SELL"
            
            results.append({"Ticker": t, "Name": get_company_name(t), "RS Score": rs, "RSI": rsi, "Trend": "Bull" if curr_p > sma200 else "Bear", "Signal": action})
    
    df_tab1 = pd.DataFrame(results).sort_values(by="RS Score", ascending=False)
    
    # Styling Tab 1
    def style_signal(val):
        color = '#238636' if 'HOLD' in val else '#da3633' if 'SELL' in val or 'BREAK' in val else ''
        return f'background-color: {color}; color: white; font-weight: bold'

    st.dataframe(df_tab1.style.applymap(style_signal, subset=['Signal'])
                 .background_gradient(subset=['RS Score'], cmap='RdYlGn')
                 .format(subset=['RS Score', 'RSI'], formatter="{:.2f}"), 
                 use_container_width=True, hide_index=True)

# TAB 2: SCANNER (Mit Firmennamen)
with tab2:
    if st.button("🚀 Run S&P 500 Scan"):
        with st.spinner("Analyzing Market Leaders..."):
            sp500 = get_sp500_tickers()
            sp_raw = yf.download(sp500 + ["^GSPC"], period="1y", progress=False)['Close']
            if isinstance(sp_raw.columns, pd.MultiIndex): sp_raw.columns = sp_raw.columns.get_level_values(0)
            bm_sp = sp_raw["^GSPC"].ffill()
            opps = []
            for t in sp500:
                if t in sp_raw.columns and t not in portfolio_list:
                    p = sp_raw[t].ffill()
                    if len(p) < 100: continue
                    rs = calc_rs_stable(p, bm_sp).iloc[-1]
                    if rs > 0.12: opps.append({"Ticker": t, "Name": get_company_name(t), "RS Score": rs})
            st.table(pd.DataFrame(opps).sort_values(by="RS Score", ascending=False).head(15))

# TAB 3: ANALYSIS
with tab3:
    sel_t = st.selectbox("Select Asset:", portfolio_list)
    if sel_t:
        hist = yf.download(sel_t, period="1y", progress=False)
        if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), line=dict(color='orange'), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), line=dict(color='red'), name="SMA 200"), row=1, col=1)
        rs_line = calc_rs_stable(hist['Close'], benchmark_live)
        fig.add_trace(go.Scatter(x=rs_line.index, y=rs_line, fill='tozeroy', line=dict(color='lime'), name="RS Score"), row=2, col=1)
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: BACKTEST (Farben für Audit Log)
with tab4:
    col_a, col_b, col_c = st.columns(3)
    with col_a: cap_start = st.number_input("Seed Capital (USD):", value=10000)
    with col_b: freq = st.selectbox("Freq:", ["Monthly", "Weekly"], index=0)
    with col_c: n_stocks = st.slider("Positions:", 3, 10, 3)
    
    if st.button("🧪 Execute Custom Range Backtest"):
        with st.spinner(f"Simulating {start_year} to {end_year}..."):
            bt_ticks = portfolio_list if bt_univ == "Watchlist" else get_sp500_tickers()
            s_date, e_date = f"{start_year}-01-01", f"{end_year}-12-31"
            s_pre = (datetime.strptime(s_date, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
            
            bm_full = yf.download("^GSPC", start=s_pre, end=e_date, progress=False)['Close']
            if isinstance(bm_full, pd.DataFrame): bm_full = bm_full.iloc[:, 0]
            bm_sma = bm_full.rolling(200).mean()
            bt_data = yf.download(bt_ticks + ["^GSPC"], start=s_pre, end=e_date, progress=False)['Close']
            if isinstance(bt_data.columns, pd.MultiIndex): bt_data.columns = bt_data.columns.get_level_values(0)
            bt_data = bt_data.ffill()
            
            rs_h = pd.DataFrame(index=bt_data.index)
            for t in bt_ticks:
                if t in bt_data.columns and t != "^GSPC":
                    rs_h[t] = calc_rs_stable(bt_data[t], bt_data["^GSPC"])
            
            f_code = 'ME' if freq == "Monthly" else 'W-MON'
            t_dates = bt_data.loc[s_date:e_date].groupby(pd.Grouper(freq=f_code)).apply(lambda x: x.index[-1] if not x.empty else None).dropna()
            
            c = cap_start; c_history, t_log, f_date = [], [], None
            for i in range(len(t_dates)-1):
                cur, nxt = t_dates[i], t_dates[i+1]
                if cur not in rs_h.index or cur not in bm_sma.index: continue
                is_bull = bm_full.loc[cur] > bm_sma.loc[cur]
                exposure = 1.0 if is_bull else 0.2
                rank = rs_h.loc[cur].dropna().sort_values(ascending=False)
                if len(rank) < n_stocks: continue
                if f_date is None: f_date = cur
                sel = rank.head(n_stocks).index.tolist()
                cash_p_s = (c * exposure / n_stocks) * 0.998
                p_res, details = [], []
                for ticker in sel:
                    b_p = bt_data.loc[cur, ticker]; d_p = bt_data.loc[cur:nxt, ticker]
                    if d_p.min() <= b_p * 0.85: exit_p = b_p * 0.85; stat = "🚨SL"
                    else: exit_p = d_p.iloc[-1]; stat = "OK"
                    p_res.append(cash_p_s * (exit_p / b_p))
                    details.append(f"{ticker}({b_p:.1f}/{exit_p:.1f}/{stat})")
                
                c_prev = c; c = sum(p_res) + (c * (1-exposure))
                s_perf, b_perf = (c / c_prev - 1) * 100, (bm_full.loc[nxt] / bm_full.loc[cur] - 1) * 100
                c_history.append({"Date": nxt, "Strategy": c})
                t_log.append({"Date": cur.date(), "Regime": "BULL 🟢" if is_bull else "BEAR 🔴", "Trades": " | ".join(details), "Strat%": s_perf, "BM%": b_perf, "Alpha%": s_perf-b_perf, "Value": c})

            if c_history:
                res = pd.DataFrame(c_history).set_index("Date")
                bm_curve = (bm_full.reindex(res.index) / bm_full.loc[f_date]) * cap_start
                st.metric("Final Capital", f"{c:,.0f} USD", f"{(c/cap_start-1)*100:.1f}%")
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['Strategy'], name="Strategy", line=dict(color='lime', width=3)))
                fig_bt.add_trace(go.Scatter(x=res.index, y=bm_curve, name="S&P 500", line=dict(color='gray', dash='dash')))
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Styling Audit Log
                log_df = pd.DataFrame(t_log).sort_values(by="Date", ascending=False)
                def style_alpha(val):
                    return 'color: #238636; font-weight: bold' if val > 0 else 'color: #da3633'
                
                st.dataframe(log_df.style.applymap(style_alpha, subset=['Alpha%'])
                             .format(subset=['Strat%', 'BM%', 'Alpha%'], formatter="{:+.2f}%")
                             .format(subset=['Value'], formatter="{:,.0f} USD"), 
                             use_container_width=True, hide_index=True)

st.divider()
st.caption("Quant-Engineer Alpha Terminal | v1.5 Styled | © 2024 Manuel Kössler")