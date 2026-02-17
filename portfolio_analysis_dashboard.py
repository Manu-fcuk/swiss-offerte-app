import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Portfolio & Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #ffffff;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .buy-signal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .sentiment-positive {
        color: #38ef7d;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    .sentiment-negative {
        color: #f45c43;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    .sentiment-neutral {
        color: #f2c94c;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    
    .stSelectbox label, .stMultiSelect label {
        color: #ffffff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Stock tickers database
TICKERS = {
    'NVDA': 'NVIDIA Corp', 'AAPL': 'Apple Inc', 'GOOG': 'Alphabet Inc Class C',
    'MSFT': 'Microsoft Corp', 'AMZN': 'Amazon.com Inc', 'META': 'Meta Platforms Inc',
    'TSLA': 'Tesla Inc', 'WMT': 'Walmart Inc', 'LLY': 'Eli Lilly And Co',
    'JPM': 'JPMorgan Chase & Co', 'AZN': 'AstraZeneca plc', 'XOM': 'Exxon Mobil Corp',
    'V': 'Visa Inc', 'JNJ': 'Johnson & Johnson', 'MU': 'Micron Technology Inc',
    'MA': 'Mastercard Inc', 'ORCL': 'Oracle Corp', 'ASML': 'ASML Holding NV',
    'COST': 'Costco Wholesale Corp', 'VALE': 'Vale SA', 'HD': 'Home Depot Inc',
    'BAC': 'Bank of America Corp', 'PG': 'Procter & Gamble Co', 'CVX': 'Chevron Corp',
    'CAT': 'Caterpillar Inc', 'NOVN': 'Novartis', 'KO': 'Coca-Cola Co',
    'AMD': 'Advanced Micro Devices Inc', 'NFLX': 'Netflix Inc', 'PLTR': 'Palantir Technologies Inc',
    'CSCO': 'Cisco Systems Inc', 'MRK': 'Merck & Co Inc', 'PM': 'Philip Morris International Inc.',
    'HSBA': 'HSBC Holdings plc', 'AMAT': 'Applied Materials Inc', 'MS': 'Morgan Stanley',
    'NESN': 'Nestle SA', 'SAP': 'SAP SE', 'IBM': 'IBM Common Stock',
    'INTC': 'Intel Corp', 'MCD': "McDonald's Corp", 'AXP': 'American Express Co',
    'PEP': 'PepsiCo Inc', 'LIN': 'Linde PLC', 'VZ': 'Verizon Communications Inc',
    'T': 'AT&T Inc', 'AMGN': 'Amgen Inc', 'C': 'Citigroup Inc',
    'SIE': 'Siemens AG', 'BA': 'Boeing Co', 'DIS': 'Walt Disney Co',
    'CRM': 'Salesforce Inc', 'TTE': 'TotalEnergies SE', 'RIO': 'Rio Tinto plc ADR Common Stock',
    'BLK': 'BlackRock Inc', 'DE': 'Deere & Co', 'ULVR': 'Unilever plc',
    'PFE': 'Pfizer Inc', 'HON': 'Honeywell International Inc', 'LMT': 'Lockheed Martin Corp',
    'QCOM': 'Qualcomm Inc', 'SHOP': 'Shopify Inc', 'BUD': 'Anheuser-Busch Inbev SA',
    'UBER': 'Uber Technologies Inc', 'ACN': 'Accenture Plc', 'PANW': 'Palo Alto Networks Inc',
    'BKNG': 'Booking Holdings Inc', 'BHP': 'BHP Group Ltd', 'VIV': 'Telefonica Brasil SA',
    'BATS': 'BRITISH AMERICAN TOBACCO PLC ADS Common Stock', 'GSK': 'GSK plc', 'BN': 'Brookfield Corp',
    'CMCSA': 'Comcast Corp', 'SANO': 'Sanofi SA', 'INTU': 'Intuit Inc',
    'ADBE': 'Adobe Inc', 'CRWD': 'Crowdstrike Holdings Inc', 'SBUX': 'Starbucks Corp',
    'ZURN': 'Zurich Insurance Group AG', 'UPS': 'United Parcel Service Inc', 'BNP': 'BNP Paribas SA',
    'SPOT': 'Spotify Technology SA', 'MAR': 'Marriott International Inc', 'NKE': 'Nike Inc',
    'MMM': '3M Co', 'CDI': 'Christian Dior SE', 'FDX': 'FedEx Corp',
    'INGA': 'ING Groep NV', 'MDLZ': 'MONDELEZ INTERNATIONAL INC Common Stock', 'LLOY': 'Lloyds Banking Group PLC',
    'DELL': 'Dell Technologies Inc', 'ABNB': 'Airbnb Inc', 'GM': 'General Motors Co',
    'RACE': 'Ferrari NV', 'WBD': 'Warner Bros Discovery Inc', 'MUV2': 'Mnchnr Rckvrschrngs-Gsllschft n Mnchn AG',
    'NET': 'Cloudflare Inc', 'ENI': 'Eni SpA', 'DBK': 'Deutsche Bank AG',
    'ENGI': 'Engie SA', 'FTNT': 'Fortinet Inc', 'SNOW': 'Snowflake Inc',
    'O': 'Realty Income Corp', 'GLEN': 'Glencore PLC', 'VOW': 'Volkswagen AG',
    'F': 'Ford Motor Co', 'HMC': 'Honda Motor Co Ltd', 'BMW': 'Bayerische Motoren Werke AG',
    'TGT': 'Target Corp', 'HOLN': 'Holcim AG', 'EA': 'Electronic Arts Inc',
    'SCMN': 'Swisscom AG', 'DAL': 'Delta Air Lines Inc', 'RBLX': 'Roblox Corp',
    'COIN': 'Coinbase Global Inc', 'RWE': 'RWE AG', 'CCL': 'Carnival Corp',
    'K': 'Kinross Gold Corp', 'EL': 'Estee Lauder Companies Inc', 'SREN': 'Swiss Re AG',
    'EBAY': 'eBay Inc', 'PYPL': 'PayPal Holdings Inc', 'LONN': 'Lonza Group AG',
    'KMB': 'Kimberly-Clark Corp', 'VOD': 'Vodafone Group PLC', 'TTWO': 'TAKE-TWO INTERACTIVE SOFTWARE, INC Common Stock',
    'KER': 'Kering SA', 'SCHN': 'Schindler Holding AG', 'ALC': 'Alcon AG',
    'PHIA': 'Koninklijke Philips NV', 'DTE': 'DTE Energy Co', 'VIE': 'Veolia Environnement SA',
    'KHC': 'Kraft Heinz Co', 'TSCO': 'Tractor Supply Co', 'GIVN': 'Givaudan SA',
    'ZS': 'Zscaler Inc', 'LISN': 'Chocoladefabriken Lindt & Spruengli AG', 'BNTX': 'BioNTech SE - ADR',
    'ZM': 'Zoom Communications Inc', 'EXPE': 'Expedia Group Inc', 'GIS': 'General Mills Inc',
    'PGHN': 'Partners Group Holding AG', 'SIKA': 'Sika AG', 'SLHN': 'Swiss Life Holding AG',
    'HEIO': 'Heineken Holding NV', 'RIVN': 'Rivian Automotive Inc', 'GEBN': 'Geberit AG',
    'LULU': 'Lululemon Athletica Inc', 'KNIN': 'Kuehne und Nagel International AG', 'STLA': 'Stellantis NV',
    'P911': 'Dr Ing hc F Porsche AG', 'CHKP': 'Check Point Software Technologies Ltd', 'SGSN': 'SGS SA',
    'MRNA': 'Moderna Inc', 'VACN': 'VAT Group AG', 'HAS': 'Hasbro Inc',
    'MONC': 'Moncler SpA', 'ALO': 'Alstom SA', 'ROKU': 'Roku Inc',
    'LHA': 'Deutsche Lufthansa AG', 'CS': 'Capstone Copper Corp', 'SOON': 'Sonova Holding AG',
    'SUN': 'Sunoco LP', 'OR': 'OR Royalties Inc', 'PINS': 'Pinterest Inc',
    'UHR': 'The Swatch Group Ord Shs', 'NCLH': 'Norwegian Cruise Line Holdings Ltd', 'BCVN': 'Banque Cantonale Vaudoise',
    'ALV': 'Autoliv Inc', 'AAL': 'American Airlines Group Inc', 'CFR': 'Cullen/Frost Bankers Inc',
    'RMS': 'Ramelius Resources Ltd', 'MKS': 'Marks and Spencer Group Plc', 'SNAP': 'Snap Inc',
    'FHZN': 'Flughafen Zuerich AG', 'MTCH': 'Match Group Inc', 'ORA': 'Ormat Technologies, Inc.',
    'ACLN': 'Accelleron Industries AG', 'TKA': 'ThyssenKrupp AG', 'SQN': 'Swissquote Group Holding SA',
    'ACA': 'Arcosa Inc', 'ZAL': 'Zalando SE', 'MC': 'Moelis & Co',
    'TEMN': 'Temenos AG', 'ETSY': 'Etsy Inc', 'EMMN': 'Emmi AG',
    'BRBY': 'Burberry Group plc', 'TUI1': 'TUI AG', 'VONN': 'Vontobel Holding AG',
    'SGKN': 'St Galler Kantonalbank AG', 'PUM': 'Puma SE', 'LCID': 'Lucid Group Inc',
    'BEKN': 'Berner Kantonalbank AG', 'UAA': 'Under Armour Inc Class A', 'MANU': 'Manchester United PLC',
    'MSGE': 'Madison Square Garden Entertainment Corp', 'CLN': 'City of London Investment Trust plc', 'GT': 'Goodyear Tire & Rubber Co',
    'BOSS': 'Hugo Boss AG Common Stock', 'VATN': 'Valiant Holding AG', 'HOG': 'Harley-Davidson Inc',
    'SRAIL': 'Stadler Rail AG', 'PTON': 'Peloton Interactive Inc', 'AI': 'C3.ai Inc',
    'FL': 'Frontier Lithium Inc'
}

@st.cache_data(ttl=300)
def get_stock_history(ticker, period='1y'):
    """Fetch stock historical data with caching"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_stock_info(ticker):
    """Fetch stock info (not cached due to serialization issues)"""
    try:
        stock = yf.Ticker(ticker)
        return stock
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for trading signals"""
    if df is None or len(df) < 50:
        return df
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def generate_trading_signals(df):
    """Generate buy/sell signals based on multiple indicators"""
    if df is None or len(df) < 200:
        return "HOLD", []
    
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Golden Cross / Death Cross
    if latest['SMA_50'] > latest['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
        signals.append(("BUY", "Golden Cross detected (SMA 50 crossed above SMA 200)"))
    elif latest['SMA_50'] < latest['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
        signals.append(("SELL", "Death Cross detected (SMA 50 crossed below SMA 200)"))
    
    # RSI signals
    if latest['RSI'] < 30:
        signals.append(("BUY", f"RSI oversold ({latest['RSI']:.1f})"))
    elif latest['RSI'] > 70:
        signals.append(("SELL", f"RSI overbought ({latest['RSI']:.1f})"))
    
    # MACD signals
    if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        signals.append(("BUY", "MACD bullish crossover"))
    elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        signals.append(("SELL", "MACD bearish crossover"))
    
    # Bollinger Bands
    if latest['Close'] < latest['BB_Lower']:
        signals.append(("BUY", "Price below lower Bollinger Band"))
    elif latest['Close'] > latest['BB_Upper']:
        signals.append(("SELL", "Price above upper Bollinger Band"))
    
    # Volume confirmation
    if latest['Volume'] > latest['Volume_SMA'] * 1.5:
        signals.append(("INFO", "High volume detected"))
    
    # Determine overall signal
    buy_count = sum(1 for s in signals if s[0] == "BUY")
    sell_count = sum(1 for s in signals if s[0] == "SELL")
    
    if buy_count > sell_count and buy_count >= 2:
        overall = "BUY"
    elif sell_count > buy_count and sell_count >= 2:
        overall = "SELL"
    else:
        overall = "HOLD"
    
    return overall, signals

def get_fundamental_data(stock):
    """Extract fundamental data from stock info"""
    try:
        info = stock.info
        fundamentals = {
            'Market Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'Forward P/E': info.get('forwardPE', 'N/A'),
            'PEG Ratio': info.get('pegRatio', 'N/A'),
            'Price to Book': info.get('priceToBook', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Revenue': info.get('totalRevenue', 'N/A'),
            'Profit Margin': info.get('profitMargins', 'N/A'),
            'Operating Margin': info.get('operatingMargins', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'ROA': info.get('returnOnAssets', 'N/A'),
            'Debt to Equity': info.get('debtToEquity', 'N/A'),
            'Current Ratio': info.get('currentRatio', 'N/A'),
            'Free Cash Flow': info.get('freeCashflow', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'Analyst Target': info.get('targetMeanPrice', 'N/A'),
            'Recommendation': info.get('recommendationKey', 'N/A'),
        }
        return fundamentals
    except:
        return {}

def analyze_sentiment(stock, fundamentals, signals):
    """Analyze overall sentiment based on fundamentals and technicals"""
    sentiment_score = 0
    reasons = []
    
    # Technical sentiment
    signal_type = signals[0] if signals else "HOLD"
    if signal_type == "BUY":
        sentiment_score += 2
        reasons.append("✅ Technical indicators show buying opportunity")
    elif signal_type == "SELL":
        sentiment_score -= 2
        reasons.append("⚠️ Technical indicators show selling pressure")
    
    # Fundamental sentiment
    try:
        # P/E Ratio analysis
        pe = fundamentals.get('P/E Ratio', 'N/A')
        if pe != 'N/A' and pe is not None:
            if pe < 15:
                sentiment_score += 1
                reasons.append("✅ Attractive P/E ratio (undervalued)")
            elif pe > 30:
                sentiment_score -= 1
                reasons.append("⚠️ High P/E ratio (potentially overvalued)")
        
        # Profit Margin
        margin = fundamentals.get('Profit Margin', 'N/A')
        if margin != 'N/A' and margin is not None:
            if margin > 0.15:
                sentiment_score += 1
                reasons.append("✅ Strong profit margins")
            elif margin < 0:
                sentiment_score -= 1
                reasons.append("⚠️ Negative profit margins")
        
        # ROE
        roe = fundamentals.get('ROE', 'N/A')
        if roe != 'N/A' and roe is not None:
            if roe > 0.15:
                sentiment_score += 1
                reasons.append("✅ Strong return on equity")
            elif roe < 0:
                sentiment_score -= 1
                reasons.append("⚠️ Negative return on equity")
        
        # Debt to Equity
        dte = fundamentals.get('Debt to Equity', 'N/A')
        if dte != 'N/A' and dte is not None:
            if dte < 0.5:
                sentiment_score += 1
                reasons.append("✅ Low debt levels")
            elif dte > 2:
                sentiment_score -= 1
                reasons.append("⚠️ High debt levels")
        
        # Analyst Recommendation
        rec = fundamentals.get('Recommendation', 'N/A')
        if rec in ['buy', 'strong_buy']:
            sentiment_score += 1
            reasons.append("✅ Analysts recommend buying")
        elif rec in ['sell', 'strong_sell']:
            sentiment_score -= 1
            reasons.append("⚠️ Analysts recommend selling")
        
    except Exception as e:
        pass
    
    # Determine overall sentiment
    if sentiment_score >= 3:
        sentiment = "BULLISH"
        color = "positive"
    elif sentiment_score <= -3:
        sentiment = "BEARISH"
        color = "negative"
    else:
        sentiment = "NEUTRAL"
        color = "neutral"
    
    return sentiment, color, reasons, sentiment_score

def generate_pros_cons(fundamentals, df, signal):
    """Generate pros and cons for buying the stock"""
    pros = []
    cons = []
    
    try:
        # Technical pros/cons
        if signal == "BUY":
            pros.append("📈 Technical indicators suggest upward momentum")
        elif signal == "SELL":
            cons.append("📉 Technical indicators suggest downward pressure")
        
        latest = df.iloc[-1]
        
        # Price momentum
        if len(df) >= 20:
            price_change_20d = ((latest['Close'] - df.iloc[-20]['Close']) / df.iloc[-20]['Close']) * 100
            if price_change_20d > 5:
                pros.append(f"🚀 Strong 20-day momentum (+{price_change_20d:.1f}%)")
            elif price_change_20d < -5:
                cons.append(f"⬇️ Negative 20-day momentum ({price_change_20d:.1f}%)")
        
        # Fundamental pros/cons
        pe = fundamentals.get('P/E Ratio', 'N/A')
        if pe != 'N/A' and pe is not None:
            if pe < 15:
                pros.append(f"💰 Low P/E ratio ({pe:.2f}) - potentially undervalued")
            elif pe > 35:
                cons.append(f"💸 High P/E ratio ({pe:.2f}) - potentially overvalued")
        
        margin = fundamentals.get('Profit Margin', 'N/A')
        if margin != 'N/A' and margin is not None:
            if margin > 0.20:
                pros.append(f"💪 Excellent profit margin ({margin*100:.1f}%)")
            elif margin < 0.05:
                cons.append(f"⚠️ Low profit margin ({margin*100:.1f}%)")
        
        roe = fundamentals.get('ROE', 'N/A')
        if roe != 'N/A' and roe is not None:
            if roe > 0.20:
                pros.append(f"🎯 Strong ROE ({roe*100:.1f}%)")
            elif roe < 0.05:
                cons.append(f"📊 Weak ROE ({roe*100:.1f}%)")
        
        div_yield = fundamentals.get('Dividend Yield', 'N/A')
        if div_yield != 'N/A' and div_yield is not None and div_yield > 0:
            if div_yield > 0.03:
                pros.append(f"💵 Attractive dividend yield ({div_yield*100:.2f}%)")
        
        dte = fundamentals.get('Debt to Equity', 'N/A')
        if dte != 'N/A' and dte is not None:
            if dte < 0.5:
                pros.append(f"✅ Low debt-to-equity ({dte:.2f})")
            elif dte > 2:
                cons.append(f"⚠️ High debt-to-equity ({dte:.2f})")
        
        beta = fundamentals.get('Beta', 'N/A')
        if beta != 'N/A' and beta is not None:
            if beta < 1:
                pros.append(f"🛡️ Lower volatility than market (Beta: {beta:.2f})")
            elif beta > 1.5:
                cons.append(f"⚡ High volatility (Beta: {beta:.2f})")
        
        rec = fundamentals.get('Recommendation', 'N/A')
        if rec in ['buy', 'strong_buy']:
            pros.append(f"👍 Analyst recommendation: {rec.upper()}")
        elif rec in ['sell', 'strong_sell']:
            cons.append(f"👎 Analyst recommendation: {rec.upper()}")
        
        # 52-week position
        high_52 = fundamentals.get('52 Week High', 'N/A')
        low_52 = fundamentals.get('52 Week Low', 'N/A')
        if high_52 != 'N/A' and low_52 != 'N/A':
            current_price = latest['Close']
            range_position = ((current_price - low_52) / (high_52 - low_52)) * 100
            if range_position < 30:
                pros.append(f"📍 Near 52-week low ({range_position:.0f}% of range)")
            elif range_position > 90:
                cons.append(f"📍 Near 52-week high ({range_position:.0f}% of range)")
        
    except Exception as e:
        pass
    
    if not pros:
        pros.append("ℹ️ Limited positive indicators at this time")
    if not cons:
        cons.append("ℹ️ Limited negative indicators at this time")
    
    return pros, cons

def plot_stock_chart(df, ticker):
    """Create interactive stock chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{ticker} Price & Indicators', 'MACD', 'RSI')
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#38ef7d',
        decreasing_line_color='#f45c43'
    ), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                            line=dict(color='#667eea', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                            line=dict(color='#f2994a', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200',
                            line=dict(color='#eb3349', width=1.5)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                            line=dict(color='rgba(102, 126, 234, 0.3)', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                            line=dict(color='rgba(102, 126, 234, 0.3)', width=1, dash='dash'),
                            fill='tonexty', fillcolor='rgba(102, 126, 234, 0.1)'), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color='#667eea', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                            line=dict(color='#f2994a', width=2)), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
                        marker_color='rgba(102, 126, 234, 0.3)'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color='#667eea', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

# Main app
st.title("📈 Advanced Portfolio & Trading Dashboard")
st.markdown("### Real-time Analysis with Trading Signals, Sentiment & Fundamentals")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stocks.png", width=80)
    st.title("Settings")
    
    selected_ticker = st.selectbox(
        "Select Stock",
        options=list(TICKERS.keys()),
        format_func=lambda x: f"{x} - {TICKERS[x]}"
    )
    
    time_period = st.selectbox(
        "Time Period",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Main content
if selected_ticker:
    with st.spinner(f'Loading data for {selected_ticker}...'):
        df = get_stock_history(selected_ticker, time_period)
        stock = get_stock_info(selected_ticker)
        
        if df is not None and len(df) > 0 and stock is not None:
            df = calculate_technical_indicators(df)
            signal, signal_details = generate_trading_signals(df)
            fundamentals = get_fundamental_data(stock)
            sentiment, sentiment_color, sentiment_reasons, sentiment_score = analyze_sentiment(stock, fundamentals, (signal, signal_details))
            pros, cons = generate_pros_cons(fundamentals, df, signal)
            
            # Header metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            latest_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            with col1:
                st.metric("Current Price", f"${latest_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
            
            with col2:
                if signal == "BUY":
                    st.markdown('<div class="buy-signal">🚀 BUY SIGNAL</div>', unsafe_allow_html=True)
                elif signal == "SELL":
                    st.markdown('<div class="sell-signal">⚠️ SELL SIGNAL</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="hold-signal">⏸️ HOLD</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'<div class="sentiment-{sentiment_color}">{sentiment}</div>', unsafe_allow_html=True)
                st.caption(f"Sentiment Score: {sentiment_score}")
            
            with col4:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            
            with col5:
                market_cap = fundamentals.get('Market Cap', 'N/A')
                if market_cap != 'N/A':
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", "N/A")
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Chart & Signals",
                "💡 Buy Analysis",
                "📈 Fundamentals",
                "🎯 Sentiment",
                "📋 Portfolio"
            ])
            
            with tab1:
                st.plotly_chart(plot_stock_chart(df, selected_ticker), use_container_width=True)
                
                st.markdown("### 🎯 Active Trading Signals")
                if signal_details:
                    for sig_type, sig_desc in signal_details:
                        if sig_type == "BUY":
                            st.success(f"✅ {sig_desc}")
                        elif sig_type == "SELL":
                            st.error(f"⚠️ {sig_desc}")
                        else:
                            st.info(f"ℹ️ {sig_desc}")
                else:
                    st.info("No strong signals detected. Market is in consolidation.")
            
            with tab2:
                st.markdown("### 🎯 Should You Buy Right Now?")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ✅ PROS")
                    for pro in pros:
                        st.success(pro)
                
                with col2:
                    st.markdown("#### ⚠️ CONS")
                    for con in cons:
                        st.warning(con)
                
                # Overall recommendation
                st.markdown("---")
                st.markdown("### 🎓 Final Recommendation")
                
                pro_count = len(pros)
                con_count = len(cons)
                
                if pro_count > con_count * 1.5 and signal == "BUY":
                    st.success(f"""
                    ### 🚀 STRONG BUY
                    Based on {pro_count} positive factors vs {con_count} concerns, combined with technical buy signals,
                    this stock shows strong potential for investment. Consider your risk tolerance and portfolio allocation.
                    """)
                elif pro_count > con_count and signal != "SELL":
                    st.info(f"""
                    ### 👍 MODERATE BUY
                    With {pro_count} positive factors and {con_count} concerns, this stock shows promise but requires
                    careful consideration. Wait for stronger technical confirmation or better entry points.
                    """)
                elif con_count > pro_count and signal == "SELL":
                    st.error(f"""
                    ### ⚠️ AVOID / SELL
                    With {con_count} concerns outweighing {pro_count} positive factors, and technical sell signals,
                    this stock may not be ideal for investment at this time. Consider waiting for better conditions.
                    """)
                else:
                    st.warning(f"""
                    ### ⏸️ HOLD / WAIT
                    The analysis shows mixed signals ({pro_count} pros vs {con_count} cons). Consider waiting for
                    clearer market direction or better entry/exit points before making a decision.
                    """)
            
            with tab3:
                st.markdown("### 📊 Fundamental Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                metrics_list = list(fundamentals.items())
                third = len(metrics_list) // 3
                
                with col1:
                    for key, value in metrics_list[:third]:
                        if value != 'N/A' and value is not None:
                            if isinstance(value, (int, float)):
                                if key in ['Market Cap', 'Revenue', 'Free Cash Flow']:
                                    st.metric(key, f"${value/1e9:.2f}B")
                                elif key in ['Dividend Yield', 'Profit Margin', 'Operating Margin', 'ROE', 'ROA']:
                                    st.metric(key, f"{value*100:.2f}%")
                                else:
                                    st.metric(key, f"{value:.2f}")
                            else:
                                st.metric(key, str(value))
                        else:
                            st.metric(key, "N/A")
                
                with col2:
                    for key, value in metrics_list[third:2*third]:
                        if value != 'N/A' and value is not None:
                            if isinstance(value, (int, float)):
                                if key in ['Market Cap', 'Revenue', 'Free Cash Flow']:
                                    st.metric(key, f"${value/1e9:.2f}B")
                                elif key in ['Dividend Yield', 'Profit Margin', 'Operating Margin', 'ROE', 'ROA']:
                                    st.metric(key, f"{value*100:.2f}%")
                                else:
                                    st.metric(key, f"{value:.2f}")
                            else:
                                st.metric(key, str(value))
                        else:
                            st.metric(key, "N/A")
                
                with col3:
                    for key, value in metrics_list[2*third:]:
                        if value != 'N/A' and value is not None:
                            if isinstance(value, (int, float)):
                                if key in ['Market Cap', 'Revenue', 'Free Cash Flow']:
                                    st.metric(key, f"${value/1e9:.2f}B")
                                elif key in ['Dividend Yield', 'Profit Margin', 'Operating Margin', 'ROE', 'ROA']:
                                    st.metric(key, f"{value*100:.2f}%")
                                else:
                                    st.metric(key, f"{value:.2f}")
                            else:
                                st.metric(key, str(value))
                        else:
                            st.metric(key, "N/A")
            
            with tab4:
                st.markdown(f"### 🎯 Overall Sentiment: {sentiment}")
                st.markdown(f"**Sentiment Score:** {sentiment_score} / 10")
                
                st.markdown("#### Key Factors:")
                for reason in sentiment_reasons:
                    if "✅" in reason:
                        st.success(reason)
                    elif "⚠️" in reason:
                        st.warning(reason)
                    else:
                        st.info(reason)
                
                # Sentiment gauge
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=sentiment_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Sentiment Score"},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [-10, 10]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [-10, -3], 'color': "#f45c43"},
                            {'range': [-3, 3], 'color': "#f2c94c"},
                            {'range': [3, 10], 'color': "#38ef7d"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': sentiment_score
                        }
                    }
                ))
                
                gauge_fig.update_layout(
                    template='plotly_dark',
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with tab5:
                st.markdown("### 📋 Portfolio Tracker")
                st.info("💡 Add this stock to your portfolio to track performance")
                
                col1, col2 = st.columns(2)
                with col1:
                    shares = st.number_input("Number of Shares", min_value=0.0, value=10.0, step=1.0)
                with col2:
                    avg_price = st.number_input("Average Buy Price ($)", min_value=0.0, value=float(latest_price), step=0.01)
                
                if shares > 0:
                    total_investment = shares * avg_price
                    current_value = shares * latest_price
                    profit_loss = current_value - total_investment
                    profit_loss_pct = (profit_loss / total_investment) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Investment", f"${total_investment:,.2f}")
                    with col2:
                        st.metric("Current Value", f"${current_value:,.2f}")
                    with col3:
                        st.metric("Profit/Loss", f"${profit_loss:,.2f}", f"{profit_loss_pct:+.2f}%")
                    with col4:
                        st.metric("Shares", f"{shares:.0f}")
        else:
            st.error(f"Unable to load data for {selected_ticker}. Please try another ticker.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.5); padding: 20px;'>
    <p>📊 Advanced Portfolio & Trading Dashboard | Data provided by Yahoo Finance</p>
    <p>⚠️ This is for informational purposes only. Not financial advice. Always do your own research.</p>
</div>
""", unsafe_allow_html=True)
