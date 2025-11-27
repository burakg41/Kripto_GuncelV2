import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import feedparser
from PIL import Image
import time
import numpy as np

# ---------------------------------------------------------
# ccxt'i güvenli şekilde import et
# ---------------------------------------------------------
try:
    import ccxt
    CCXT_AVAILABLE = True
except ModuleNotFoundError:
    ccxt = None
    CCXT_AVAILABLE = False

# =============================================================================
# 1. KONFİGÜRASYON VE TEMA
# =============================================================================

st.set_page_config(
    page_title="AlphaTrade AI: Pro Terminal",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Özel CSS ile Modern Görünüm (Glassmorphism)
st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; transition: all 0.3s; }
        .stButton>button:hover { transform: scale(1.02); }
        .metric-card { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #2e3440; margin-bottom: 10px; }
        h1, h2, h3 { color: #00e676 !important; font-family: 'Helvetica Neue', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1e2130; border-radius: 5px; color: white; }
        .stTabs [aria-selected="true"] { background-color: #00e676; color: black; }
        [data-testid="stDataFrame"] { border: 1px solid #2e3440; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. HİZMET SINIFLARI (BACKEND LOGIC)
# =============================================================================

class MarketDataService:
    """Borsa verilerini CCXT ile çeken gelişmiş servis."""
    
    def __init__(self, exchange_id='kraken'):
        if not CCXT_AVAILABLE:
            st.error("Bu ortamda 'ccxt' modülü kurulu değil.")
            raise RuntimeError("ccxt not available")

        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000, 
            })
        except AttributeError:
            st.error(f"{exchange_id} borsası bulunamadı, varsayılan olarak Kraken kullanılıyor.")
            self.exchange = ccxt.kraken({'enableRateLimit': True})
        except Exception as e:
            st.error(f"Borsa başlatma hatası: {e}")

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        try:
            # Sembol formatı düzeltme (örn: BTCUSDT -> BTC/USDT)
            if '/' not in symbol and 'USDT' in symbol:
                symbol = symbol.replace('USDT', '/USDT')
            
            # Veri çekme
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            except ccxt.BadSymbol:
                # Sembol bulunamazsa marketleri yükleyip tekrar dene
                self.exchange.load_markets()
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            except Exception as e:
                # ABD IP Kısıtlaması Kontrolü
                if "Geo-Restricted" in str(e) or "Service unavailable" in str(e):
                    st.error(f"⚠️ Erişim Engeli: {self.exchange.id} borsası ABD IP'lerini (Streamlit Cloud) engelliyor.")
                    st.info("Lütfen sol menüden 'kraken' veya 'coinbase' seçin.")
                    return pd.DataFrame()
                else:
                    raise e
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            st.warning(f"⚠️ Veri Hatası ({self.exchange.id}): {str(e)}")
            return pd.DataFrame()

    def fetch_order_book(self, symbol, limit=20):
        try:
            if '/' not in symbol and 'USDT' in symbol:
                symbol = symbol.replace('USDT', '/USDT')
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            bids = pd.DataFrame(orderbook['bids'], columns=['price', 'amount'])
            asks = pd.DataFrame(orderbook['asks'], columns=['price', 'amount'])
            return bids, asks
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    @st.cache_data(ttl=300)
    def fetch_crypto_news(_self):
        try:
            feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
            news = []
            for entry in feed.entries[:5]:
                news.append(f"- {entry.title} ({entry.published})")
            return "\n".join(news)
        except Exception:
            return "Haber kaynağına erişilemedi."

    @staticmethod
    def add_indicators(df):
        """Manuel İndikatör Hesaplamaları (pandas_ta olmadan)."""
        if df.empty: return df

        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']

        # RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # EMA
        df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
        df['EMA_200'] = close.ewm(span=200, adjust=False).mean()

        # Bollinger Bands
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        df['BBU_20_2.0'] = rolling_mean + (rolling_std * 2)
        df['BBL_20_2.0'] = rolling_mean - (rolling_std * 2)

        # ATR & ADX (Basitleştirilmiş)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # ADX (Basitleştirilmiş Trend Gücü)
        df['ADX'] = df['ATR'].rolling(window=14).mean() * 10 # Mock calculation for stability without complex lib

        # Formasyonlar (Doji & Engulfing)
        body = (close - open_).abs()
        rng = high - low
        df['DOJI'] = np.where(body <= rng * 0.1, 1, 0)
        
        # Engulfing
        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        df['ENGULFING'] = np.where(
            (close > open_) & (prev_close < prev_open) & (close > prev_open) & (open_ < prev_close), 
            1, 0
        )

        return df

class AIAnalyst:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_market_structure(self, df, news_context, symbol, mode):
        last = df.iloc[-1]
        trend = "YÜKSELİŞ" if last['close'] > last['EMA_200'] else "DÜŞÜŞ"
        prompt = f"""
        Rol: Kıdemli Kripto Analisti.
        Parite: {symbol} | Fiyat: {last['close']} | Trend: {trend} | RSI: {last['RSI']:.2f}
        Haberler: {news_context}
        Mod: {mode}
        
        Analiz et:
        1. Piyasa Yapısı ve Trend Gücü
        2. Kritik Destek/Direnç Seviyeleri
        3. İşlem Fırsatı (Giriş/Stop/Hedef)
        4. Risk Uyarısı
        """
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            return f"AI Hatası: {str(e)}"

    def analyze_chart_image(self, image, user_prompt):
        try:
            return self.vision_model.generate_content([f"Grafik analizi yap: {user_prompt}", image]).text
        except Exception as e:
            return f"Görsel Hatası: {str(e)}"

# =============================================================================
# 3. GRAFİK FONKSİYONLARI
# =============================================================================

def create_advanced_chart(df, symbol):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Fiyat ve EMA
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Fiyat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], line=dict(color='orange'), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_200'], line=dict(color='blue'), name='EMA 200'), row=1, col=1)
    
    # Bollinger
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BBU_20_2.0'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BBL_20_2.0'], fill='tonexty', fillcolor='rgba(0,255,0,0.05)', line=dict(color='gray', width=0), name='BB'), row=1, col=1)

    # Formasyon
    doji = df[df['DOJI'] == 1]
    if not doji.empty:
        fig.add_trace(go.Scatter(x=doji['timestamp'], y=doji['high'], mode='markers', marker=dict(symbol='diamond', color='yellow'), name='Doji'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
    return fig

def create_depth_chart(bids, asks):
    fig = go.Figure()
    if not bids.empty and not asks.empty:
        bids['total'] = bids['amount'].cumsum()
        asks['total'] = asks['amount'].cumsum()
        fig.add_trace(go.Scatter(x=bids['price'], y=bids['total'], fill='tozeroy', name='Alıcılar', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=asks['price'], y=asks['total'], fill='tozeroy', name='Satıcılar', line=dict(color='red')))
    fig.update_layout(template="plotly_dark", title="Piyasa Derinliği", height=400)
    return fig

# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    if not CCXT_AVAILABLE:
        st.error("Lütfen requirements.txt dosyasına 'ccxt' ekleyin ve reboot edin.")
        st.stop()

    with st.sidebar:
        st.title("🦅 AlphaTrade Pro")
        api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
        if not api_key: st.stop()
        
        st.divider()
        # Varsayılanı KRAKEN yaptık (US Friendly)
        exchange_id = st.selectbox("Borsa", ["kraken", "coinbase", "binanceus", "binance", "okx", "kucoin"], index=0)
        symbol = st.text_input("Parite", value="BTC/USDT").upper()
        timeframe = st.selectbox("Zaman", ["15m", "1h", "4h", "1d"], index=2)
        mode = st.radio("Mod", ["Scalper", "Swing", "Day Trader"])

    market = MarketDataService(exchange_id)
    ai = AIAnalyst(api_key)

    st.subheader(f"⚡ {exchange_id.upper()} | {symbol}")

    with st.spinner("Veriler çekiliyor..."):
        df = market.fetch_ohlcv(symbol, timeframe)
        if df.empty:
            st.error(f"Veri alınamadı. {exchange_id} borsasında {symbol} olmayabilir veya erişim engeli var.")
            st.info("İPUCU: Streamlit Cloud (ABD) sunucularında 'kraken' veya 'coinbase' en iyi sonucu verir.")
            st.stop()
        
        df = market.add_indicators(df)
        
        last = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fiyat", f"${last['close']:,.2f}")
        col2.metric("RSI", f"{last['RSI']:.1f}")
        col3.metric("ATR", f"{last['ATR']:.2f}")
        col4.metric("Sinyal", "SAT" if last['RSI']>70 else "AL" if last['RSI']<30 else "NÖTR")

    tab1, tab2, tab3, tab4 = st.tabs(["Grafik", "Derinlik", "AI Analiz", "Görsel"])

    with tab1:
        st.plotly_chart(create_advanced_chart(df, symbol), use_container_width=True)
    
    with tab2:
        bids, asks = market.fetch_order_book(symbol)
        st.plotly_chart(create_depth_chart(bids, asks), use_container_width=True)

    with tab3:
        if st.button("AI Analizi Başlat"):
            news = market.fetch_crypto_news()
            res = ai.analyze_market_structure(df, news, symbol, mode)
            st.markdown(res)

    with tab4:
        f = st.file_uploader("Grafik Yükle", type=["jpg", "png"])
        if f and st.button("Görseli Yorumla"):
            st.write(ai.analyze_chart_image(Image.open(f), "Analiz et"))

if __name__ == "__main__":
    main()
