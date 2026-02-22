import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Alpha Terminal Pro v11.9.7", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 5px; color: white; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #238636; }
    div[data-testid="stExpander"] { border: 1px solid #30363d; background-color: #0e1117; }
    .status-box { padding: 25px; border-radius: 12px; text-align: center; font-weight: bold; font-size: 26px; margin-bottom: 10px; border: 2px solid #30363d; }
    .intel-box { padding: 15px; border-radius: 8px; background-color: #1c2128; border-left: 5px solid #238636; margin-bottom: 20px; min-height: 160px; }
    .outlook-card { background-color: #1c2128; padding: 20px; border-radius: 10px; border-top: 3px solid #388bfd; margin-top: 10px; line-height: 1.6; }
    .calendar-event { padding: 8px; border-bottom: 1px solid #30363d; font-size: 14px; }
    .disclaimer { font-size: 13px; color: #8b949e; margin-top: 50px; border: 1px solid #da3633; padding: 20px; border-radius: 10px; background-color: #211111; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE ENGINE FUNCTIONS ---

@st.cache_data(ttl=86400)
def get_company_static_info(ticker):
    try:
        t = yf.Ticker(ticker)
        inf = t.info
        return {"Name": inf.get('longName', ticker), "Sector": inf.get('sector', 'N/A')}
    except: return {"Name": ticker, "Sector": "N/A"}

def calc_rs_stable(prices, bm_prices):
    combined = pd.concat([prices, bm_prices], axis=1).ffill().dropna()
    if combined.empty: return pd.Series()
    combined.columns = ['Asset', 'BM']
    ratio = combined['Asset'] / combined['BM'].replace(0, np.nan)
    return (ratio / ratio.rolling(window=50).mean()) - 1

def calc_rsi(prices, window=14):
    if len(prices) < window: return pd.Series([50]*len(prices))
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).ffill()))

@st.cache_data(ttl=3600)
def get_sp500_list():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        df = pd.read_html(requests.get(url, headers=headers).text)[0]
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: return ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]

def get_market_intelligence(bm_prices):
    if len(bm_prices) < 200: return "N/A", "N/A", "N/A", "", "Neutral"
    sma50 = bm_prices.rolling(50).mean().iloc[-1]
    sma200 = bm_prices.rolling(200).mean().iloc[-1]
    curr = bm_prices.iloc[-1]
    if curr > sma50 and sma50 > sma200: ph, adv = "Markup 🚀 (Phase 2)", "Starker Bullen-Trend. Momentum-Fokus maximieren."
    elif curr < sma50 and curr > sma200: ph, adv = "Distribution ⚠️ (Phase 3)", "Vorsicht am Top. Volatilität steigt. Stops eng ziehen."
    elif curr < sma50 and curr < sma200: ph, adv = "Markdown 🔴 (Phase 4)", "Bärenmarkt. 80% Cash-Quote zum Kapitalschutz."
    else: ph, adv = "Accumulation 📈 (Phase 1)", "Bodenbildung. Auf RS-Ausbrüche neuer Leader achten."
    m = datetime.now().month
    s_map = {1: ("Januar", "Bullish"), 2: ("Februar", "Volatil"), 3: ("März", "Bullish"), 4: ("April", "Sehr Stark"), 5: ("Mai", "Vorsicht"), 9: ("September", "Schwach"), 11: ("November", "Sehr Stark"), 12: ("Dezember", "Bullish")}
    s_t, s_d = s_map.get(m, ("Neutral", "Folge dem System."))
    sent = "Greed 🔥" if calc_rsi(bm_prices).iloc[-1] > 65 else "Fear 😨" if calc_rsi(bm_prices).iloc[-1] < 35 else "Neutral ⚖️"
    return ph, adv, s_t, s_d, sent

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620611.png", width=60)
    st.title("Alpha Master v11.9.7")
    u_input = st.text_area("Watchlist Tickers:", value="GOOG, AAPL, AMZN, WMT, T, META, NVDA, TSLA, MSFT, LLY, GE, PYPL, SNAP, ASML, PLTR", height=150)
    portfolio_list = [x.strip().upper() for x in u_input.split(",") if x.strip()]
    st.divider()
    bt_s_year = st.number_input("Backtest Start", 2015, 2024, 2021)
    bt_e_year = st.number_input("Backtest End", 2016, 2026, 2025)
    bt_univ_choice = st.radio("Backtest Universe", ["Watchlist", "S&P 500 Index"])
    hold_mo_val = st.select_slider("Holding (Mo)", options=[1, 2, 3, 4, 6], value=1)
    sl_input_val = st.slider("Stop Loss %", 5, 30, 15)

# --- 4. DATA FETCH ---
with st.spinner("Synchronisiere Terminal-Daten..."):
    bm_prices_full = yf.download("^GSPC", period="5y", progress=False, threads=False, auto_adjust=True)['Close']
    if isinstance(bm_prices_full, pd.DataFrame): bm_prices_full = bm_prices_full.iloc[:, 0]
    live_port_prices = yf.download(portfolio_list, period="4y", progress=False, threads=False, auto_adjust=True)['Close']
    if isinstance(live_port_prices, pd.DataFrame) and isinstance(live_port_prices.columns, pd.MultiIndex): 
        live_port_prices.columns = live_port_prices.columns.get_level_values(0)
    live_port_prices = live_port_prices.ffill()

# --- 5. HEADER ---
m_ph, m_adv, s_t, s_d, m_sent = get_market_intelligence(bm_prices_full)
m_bull = bm_prices_full.iloc[-1] > bm_prices_full.rolling(200).mean().iloc[-1]

st.markdown(f'<div class="status-box" style="background-color: {"#238636" if m_bull else "#da3633"}; color: white;">MARKET STATUS: {"BULLISH 🟢" if m_bull else "BEARISH 🔴"} | SENTIMENT: {m_sent}</div>', unsafe_allow_html=True)

c_i1, c_i2, c_i3 = st.columns(3)
with c_i1: st.markdown(f'<div class="intel-box" style="border-color: #238636;"><b>🧠 Phase:</b><br>{m_ph}<br><small>{m_adv}</small></div>', unsafe_allow_html=True)
with c_i2: st.markdown(f'<div class="intel-box" style="border-color: #f1e05a;"><b>📅 Saisonalität:</b><br>{s_t}<br><small>{s_d}</small></div>', unsafe_allow_html=True)
with c_i3: st.markdown(f'<div class="intel-box" style="border-color: #388bfd;"><b>🌐 Makro-Radar:</b><br>Zins-Stabilität erwartet.<br><small>Fokus auf Inflation.</small></div>', unsafe_allow_html=True)

# --- 6. TABS ---
t1, t2, t3, t4, t5 = st.tabs(["🎯 Action Plan", "🔭 Scanner", "📈 Charts", "🧪 Backtest", "📖 Strategy & Calendar"])

with t1:
    res = []
    for t in portfolio_list:
        if t in live_port_prices.columns:
            p = live_port_prices[t].dropna()
            rs_series = calc_rs_stable(p, bm_prices_full.reindex(p.index).ffill())
            rsi_series = calc_rsi(p)
            if not rs_series.empty and not rsi_series.empty:
                rs = rs_series.iloc[-1]
                rsi_val = rsi_series.iloc[-1]
                d = get_company_static_info(t)
                res.append({"Ticker": t, "Name": d["Name"], "Sector": d["Sector"], "RS Score": rs, "RSI(14)": rsi_val, "Action": "🟢 HOLD" if rs > 0 else "🔴 SELL"})
    if res:
        st.dataframe(pd.DataFrame(res).sort_values(by="RS Score", ascending=False).style.background_gradient(subset=['RS Score'], cmap='RdYlGn').format(subset=['RS Score', 'RSI(14)'], formatter="{:.2f}"), width='stretch', hide_index=True)

with t2:
    if st.button("🚀 Run S&P 500 Scan"):
        with st.spinner("Scanning S&P 500 Leaders..."):
            sp_all = get_sp500_list(); sp_data = yf.download(sp_all, period="1y", progress=False, threads=False, auto_adjust=True)['Close']
            if isinstance(sp_data, pd.DataFrame) and isinstance(sp_data.columns, pd.MultiIndex): 
                sp_data.columns = sp_data.columns.get_level_values(0)
            opps = []
            for t in sp_data.columns:
                if t not in portfolio_list:
                    p = sp_data[t].ffill(); rs_series = calc_rs_stable(p, bm_prices_full.reindex(p.index).ffill())
                    if not rs_series.empty and rs_series.iloc[-1] > 0.12: 
                        d = get_company_static_info(t)
                        opps.append({"Ticker": t, "Name": d["Name"], "Sector": d["Sector"], "RS Score": rs_series.iloc[-1]})
            if opps:
                df_o = pd.DataFrame(opps).sort_values(by="RS Score", ascending=False).head(15); st.table(df_o)
                st.code(", ".join([t for t in pd.DataFrame(opps)['Ticker'].tolist() if t not in portfolio_list]), language="text")
            else:
                st.info("Keine neuen Leader gefunden.")

with t3:
    sel = st.selectbox("Deep Dive Asset:", portfolio_list)
    if sel:
        df_c = yf.download(sel, period="1y", progress=False, threads=False, auto_adjust=True)
        if isinstance(df_c, pd.DataFrame) and isinstance(df_c.columns, pd.MultiIndex): 
            df_c.columns = df_c.columns.get_level_values(0)
        df_c['SMA50'] = df_c['Close'].rolling(50).mean(); df_c['SMA200'] = df_c['Close'].rolling(200).mean()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA50'], line=dict(color='orange'), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA200'], line=dict(color='red'), name="SMA 200"), row=1, col=1)
        rs_l = calc_rs_stable(df_c['Close'], bm_prices_full.reindex(df_c.index).ffill())
        if not rs_l.empty:
            fig.add_trace(go.Scatter(x=rs_l.index, y=rs_l, fill='tozeroy', line=dict(color='lime'), name="RS Score"), row=2, col=1)
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False); st.plotly_chart(fig, width='stretch')

with t4:
    col_c1, col_c2 = st.columns(2)
    cap_in = col_c1.number_input("Backtest Capital (USD):", value=10000); n_st = col_c2.slider("Positions:", 3, 10, 5)
    if st.button("🧪 Execute Advanced Backtest"):
        with st.spinner("Processing History..."):
            ticks_bt = portfolio_list if bt_univ_choice == "Watchlist" else get_sp500_list()
            s_p = (datetime.strptime(f"{bt_s_year}-01-01", "%Y-%m-%d") - timedelta(days=450)).strftime("%Y-%m-%d")
            bt_d = yf.download([t for t in ticks_bt if t != "^GSPC"] + ["^GSPC"], start=s_p, end=f"{bt_e_year}-12-31", progress=False, threads=False, auto_adjust=True)['Close']
            if isinstance(bt_d, pd.DataFrame) and isinstance(bt_d.columns, pd.MultiIndex): 
                bt_d.columns = bt_d.columns.get_level_values(0)
            bt_d = bt_d.ffill(); bm_f = bt_d["^GSPC"]
            common_idx = bt_d.index.intersection(bm_f.index); bt_d, bm_f = bt_d.loc[common_idx], bm_f.loc[common_idx]
            
            rs_dict = {}
            for t in ticks_bt:
                if t in bt_d.columns and t != "^GSPC": 
                    rs_dict[t] = calc_rs_stable(bt_d[t], bm_f)
            rs_h = pd.concat(rs_dict, axis=1)
            
            t_d = bt_d.loc[f"{bt_s_year}-01-01":].groupby(pd.Grouper(freq=f'{hold_mo_val}ME')).apply(lambda x: x.index[-1] if not x.empty else None).dropna()
            c, c_h, t_l, f_dt = cap_in, [], [], None
            for i in range(len(t_d)-1):
                cur, nxt = t_d.iloc[i], t_d.iloc[i+1]; is_bul = bm_f.loc[cur] > bm_f.rolling(200).mean().loc[cur]
                exp = 1.0 if is_bul else 0.2; rank = rs_h.loc[cur].dropna().sort_values(ascending=False).head(n_st)
                if len(rank) < n_st: continue
                if f_dt is None: f_dt = cur
                sel_tk = rank.index.tolist(); p_r, det = [] , []
                for tk in sel_tk:
                    bp, dp = bt_d.loc[cur, tk], bt_d.loc[cur:nxt, tk]
                    if dp.min() <= bp * (1-sl_input_val/100): ex_p, stt = bp * (1-sl_input_val/100), "🚨SL"
                    else: ex_p, stt = dp.iloc[-1], "OK"
                    p_r.append((c*exp/n_st)*0.9988*(ex_p/bp))
                    
                    # --- FIXED PERCENTAGE FORMULA ---
                    stock_gain_raw = (ex_p / bp - 1) * 100
                    emoji = "🟢" if stock_gain_raw > 0 else "🔴"
                    det.append(f"{tk}({emoji}{stock_gain_raw:+.1f}%, In:{bp:.1f}/Out:{ex_p:.1f}, {stt})")
                
                c_pv = c; c = sum(p_r) + (c * (1-exp))
                s_pf, b_pf = (c/c_pv-1)*100, (bm_f.loc[nxt]/bm_f.loc[cur]-1)*100
                c_h.append({"Date": nxt, "Strategy": c, "Market": (bm_f.loc[nxt]/bm_f.loc[f_dt])*cap_in, "StratPerf": s_pf})
                t_l.append({"Date": cur.date(), "Regime": "BULL" if is_bul else "BEAR", "Trades": " | ".join(det), "Strat%": s_pf, "S&P500%": b_pf, "Alpha%": s_pf-b_pf, "Value": c})
            if c_h:
                res = pd.DataFrame(c_h).set_index("Date")
                def m_dd(s): return ((s - s.cummax()) / s.cummax()).min() * 100
                
                # Winning/Losing Streak Calculation
                perf_series = res['StratPerf']
                streak, max_streak = 0, 0
                for p in perf_series:
                    if p < 0: streak += 1
                    else: streak = 0
                    if streak > max_streak: max_streak = streak
                
                strat_total = (c / cap_in - 1) * 100
                index_total = (res['Market'].iloc[-1] / cap_in - 1) * 100
                dd_s, dd_i = m_dd(res['Strategy']), m_dd(res['Market'])

                m1, m2, m3 = st.columns(3)
                m1.metric("Final Capital", f"{c:,.0f} USD", f"{strat_total:+.1f}% Total")
                m2.metric("Alpha vs S&P 500", f"{strat_total - index_total:+.1f}%", f"Index: {index_total:+.1f}%")
                m3.metric("Max DD Strat", f"{dd_s:.1f}%", f"Index: {dd_i:.1f}%", delta_color="inverse")
                
                st.write(f"**Längste Pechsträhne (Losing Streak):** {max_streak} Monate in Folge mit Verlust.")
                
                fig_p = go.Figure(); fig_p.add_trace(go.Scatter(x=res.index, y=res['Strategy'], name="Strategy", line=dict(color='lime', width=3))); fig_p.add_trace(go.Scatter(x=res.index, y=res['Market'], name="S&P 500 Index", line=dict(color='gray', dash='dash')))
                st.plotly_chart(fig_p, width='stretch'); log_df = pd.DataFrame(t_l).sort_values(by="Date", ascending=False)
                st.dataframe(log_df.style.map(lambda x: 'background-color: #238636; color: white' if str(x) == "BULL" else 'background-color: #da3633; color: white' if str(x) == "BEAR" else '', subset=['Regime']).map(lambda x: 'color: #238636; font-weight: bold' if (isinstance(x, float) and x > 0) else 'color: #da3633' if (isinstance(x, float) and x < 0) else '', subset=['Alpha%']).format(subset=['Strat%', 'S&P500%', 'Alpha%'], formatter="{:+.2f}%").format(subset=['Value'], formatter="{:,.0f}"), width='stretch', hide_index=True)

with t5:
    st.header("📈 Market Intelligence Hub 2025")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Saisonaler Outlook")
        st.markdown(f"<div class='outlook-card'><b>Aktueller Monat: {s_t}</b><br>{s_d}<br><br><b>Roadmap:</b> Momentum-Trader profitieren historisch am meisten im Zeitraum Nov-April. Der Zeitraum Mai-Oktober erfordert engere Stops.</div>", unsafe_allow_html=True)
    with col2:
        st.subheader("Wichtige Termine 2025")
        events = [("März 12", "CPI Inflationsdaten"), ("März 19", "FOMC Zinsentscheid & Dot Plot"), ("April 04", "Arbeitsmarktbericht (NFP)"), ("Mai 01", "PCE Preisindex (FED Favorit)")]
        for date, event in events:
            st.markdown(f"<div class='calendar-event'><b>{date}:</b> {event}</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer">⚠️ LEGAL NOTICE: Not financial advice. Investing involves risk of loss. © {datetime.now().year} Manuel Kössler.</div>', unsafe_allow_html=True)