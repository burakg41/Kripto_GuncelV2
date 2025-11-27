import streamlit as st
import pandas as pd
import ccxt
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import feedparser
from PIL import Image
import time
import numpy as np

# =============================================================================
# 1. KONFİGÜRASYON VE TEMA
# =============================================================================

st.set_page_config(
    page_title="AlphaTrade AI: Pro Terminal",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

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
# 2. HİZMET SINIFLARI
# =============================================================================

class MarketDataService:
    """Borsa verilerini CCXT ile çeken servis."""
    
    def __init__(self, exchange_id='binance'):
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({'enableRateLimit': True})
        except AttributeError:
            st.error(f"{exchange_id} borsası bulunamadı, Binance kullanılıyor.")
            self.exchange = ccxt.binance({'enableRateLimit': True})

    @st.cache_data(ttl=30)
    def fetch_ohlcv(_self, symbol, timeframe, limit=200):
        try:
            ohlcv = _self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=15)
    def fetch_order_book(_self, symbol, limit=20):
        try:
            orderbook = _self.exchange.fetch_order_book(symbol, limit)
            bids = pd.DataFrame(orderbook['bids'], columns=['price', 'amount'])
            asks = pd.DataFrame(orderbook['asks'], columns=['price', 'amount'])
            bids['side'] = 'bid'
            asks['side'] = 'ask'
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
        """RSI, EMA, MACD, BB, ATR, ADX, Doji, Engulfing (pandas ile)."""
        if df.empty:
            return df

        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']

        # --- RSI (14) ---
        rsi_len = 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_len, min_periods=rsi_len).mean()
        avg_loss = loss.rolling(window=rsi_len, min_periods=rsi_len).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df['RSI'] = rsi

        # --- EMA 50 & EMA 200 ---
        df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
        df['EMA_200'] = close.ewm(span=200, adjust=False).mean()

        # --- MACD (12,26,9) ---
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        df['MACD_12_26_9'] = macd
        df['MACDs_12_26_9'] = signal
        df['MACDh_12_26_9'] = hist

        # --- Bollinger Bands (20, 2) ---
        bb_len = 20
        bb_std = 2
        mid = close.rolling(window=bb_len, min_periods=bb_len).mean()
        std = close.rolling(window=bb_len, min_periods=bb_len).std()
        upper = mid + bb_std * std
        lower = mid - bb_std * std
        df['BBM_20_2.0'] = mid
        df['BBU_20_2.0'] = upper
        df['BBL_20_2.0'] = lower

        # --- ATR (14) ---
        atr_len = 14
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_len, min_periods=atr_len).mean()
        df['ATR'] = atr

        # --- ADX (14) ---
        adx_len = 14
        plus_dm = high.diff()
        minus_dm = low.diff().mul(-1)
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr_smooth = tr.rolling(window=adx_len, min_periods=adx_len).sum()
        plus_di = 100 * (
            plus_dm.rolling(window=adx_len, min_periods=adx_len).sum()
            / tr_smooth.replace(0, np.nan)
        )
        minus_di = 100 * (
            minus_dm.rolling(window=adx_len, min_periods=adx_len).sum()
            / tr_smooth.replace(0, np.nan)
        )
        dx = ((plus_di - minus_di).abs()
              / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.rolling(window=adx_len, min_periods=adx_len).mean()
        df['ADX_14'] = adx
        df['ADX'] = adx

        # --- Mum Formasyonları ---
        body = (close - open_).abs()
        range_ = high - low
        df['DOJI'] = ((body <= range_ * 0.1) & (range_ > 0)).astype(int)

        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        bull_engulf = (
            (prev_close < prev_open) &
            (close > open_) &
            (open_ <= prev_close) &
            (close >= prev_open)
        )
        bear_engulf = (
            (prev_close > prev_open) &
            (close < open_) &
            (open_ >= prev_close) &
            (close <= prev_open)
        )
        df['ENGULFING'] = bull_engulf.astype(int) - bear_engulf.astype(int)

        return df

class AIAnalyst:
    """Google Gemini AI Analiz Motoru."""
    
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Burada sabit olarak 1.5-flash kullanıyoruz
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_market_structure(self, df, news_context, symbol, mode):
        last = df.iloc[-1]
        trend = "YÜKSELİŞ" if last['close'] > last['EMA_200'] else "DÜŞÜŞ"
        
        prompt = f"""
        Sen Wall Street seviyesinde bir Kripto Kantitatif Analistisin.
        
        **Piyasa Durumu ({symbol}):**
        - Fiyat: {last['close']}
        - Trend (EMA200): {trend}
        - RSI: {last['RSI']:.2f}
        - ADX (Trend Gücü): {last['ADX']:.2f}
        - Volatilite (ATR): {last['ATR']:.2f}
        
        **Haberler:**
        {news_context}
        
        **Kullanıcı Modu:** {mode}
        
        Lütfen şu bölümlerden oluşan stratejik bir rapor yaz:
        1. **Piyasa Yapısı:** Trendin gücü ve yönü. (ADX ve EMA verilerini kullan).
        2. **Likidite ve Tuzaklar:** Stop-loss avı (liquidation hunt) ihtimali var mı?
        3. **İşlem Kurulumu (Trade Setup):**
           - Giriş Bölgesi:
           - Geçersiz Kılma (Stop-Loss):
           - Hedef (Take-Profit):
        4. **Formasyon Analizi:** Mum yapılarında belirgin bir dönüş formasyonu var mı?
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # 429 dahil tüm hatayı olduğu gibi döndürür
            return f"AI Analiz Hatası: {str(e)}"

    def analyze_chart_image(self, image, user_prompt):
        prompt = (
            "Sen uzman bir grafikçisin. "
            f"Şu grafiğe bak: {user_prompt}. "
            "Fibonacci seviyelerini, Elliot Dalga sayımını ve Smart Money Concepts (SMC) yapılarını ara."
        )
        try:
            response = self.vision_model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"Görsel Analiz Hatası: {str(e)}"

# =============================================================================
# 3. GRAFİKLER
# =============================================================================

def create_advanced_chart(df, symbol):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=(
            f'{symbol} Fiyat & Formasyonlar',
            'Momentum (RSI)',
            'Trend Gücü (ADX)'
        ),
        row_heights=[0.6, 0.2, 0.2]
    )

    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Fiyat'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['EMA_50'],
            line=dict(color='#00e676', width=1),
            name='EMA 50'
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['EMA_200'],
            line=dict(color='#2962ff', width=2),
            name='EMA 200'
        ),
        row=1,
        col=1
    )
    
    doji_points = df[df['DOJI'] != 0]
    if not doji_points.empty:
        fig.add_trace(
            go.Scatter(
                x=doji_points['timestamp'],
                y=doji_points['high'],
                mode='markers',
                marker=dict(symbol='diamond', size=5, color='yellow'),
                name='Doji'
            ),
            row=1,
            col=1
        )

    bb_upper = df.columns[df.columns.str.contains('BBU')][0]
    bb_lower = df.columns[df.columns.str.contains('BBL')][0]
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df[bb_upper],
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=False
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df[bb_lower],
            fill='tonexty',
            fillcolor='rgba(0, 230, 118, 0.05)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='BB Alanı'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['RSI'],
            line=dict(color='#ab47bc', width=2),
            name='RSI'
        ),
        row=2,
        col=1
    )
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ADX'],
            line=dict(color='#ff9100', width=2),
            name='ADX'
        ),
        row=3,
        col=1
    )
    fig.add_hline(y=25, line_dash="solid", line_color="gray", annotation_text="Trend Sınırı", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=900,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

def create_depth_chart(bids, asks):
    fig = go.Figure()
    bids = bids.copy()
    asks = asks.copy()
    
    bids['total'] = bids['amount'].cumsum()
    asks['total'] = asks['amount'].cumsum()

    fig.add_trace(
        go.Scatter(
            x=bids['price'],
            y=bids['total'],
            fill='tozeroy',
            name='Alıcılar (Bids)',
            line=dict(color='#00e676')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=asks['price'],
            y=asks['total'],
            fill='tozeroy',
            name='Satıcılar (Asks)',
            line=dict(color='#ff1744')
        )
    )

    fig.update_layout(
        title="Piyasa Derinliği (Market Depth)",
        template="plotly_dark",
        height=400,
        xaxis_title="Fiyat",
        yaxis_title="Kümülatif Hacim",
        hovermode="x unified"
    )
    return fig

# =============================================================================
# 4. ANA UYGULAMA
# =============================================================================

def main():
    with st.sidebar:
        st.title("🦅 AlphaTrade Pro")
        
        api_key = st.secrets.get("GOOGLE_API_KEY", None)
        if not api_key:
            api_key = st.text_input("Gemini API Key", type="password")
            if not api_key:
                st.warning("Analiz motoru için API Key gereklidir.")
                st.stop()
        
        st.divider()
        
        exchange_id = st.selectbox("Borsa Seç", ["binance", "okx", "kraken", "kucoin"], index=0)
        symbol = st.text_input("Parite (Sembol)", value="BTC/USDT").upper()
        timeframe = st.selectbox("Zaman Dilimi", ["15m", "1h", "4h", "1d"], index=2)
        
        st.divider()
        trader_mode = st.radio("Yatırımcı Profili", ["Scalper (Dakikalık)", "Day Trader (Günlük)", "Swing (Haftalık)"])
        
    market_service = MarketDataService(exchange_id)
    ai_engine = AIAnalyst(api_key)

    st.subheader(f"⚡ {exchange_id.upper()} | {symbol} Terminali")

    with st.spinner(f'{symbol} verileri işleniyor...'):
        df = market_service.fetch_ohlcv(symbol, timeframe)
        
        if df.empty:
            st.error(f"Veri alınamadı. {symbol} paritesinin {exchange_id} borsasında olduğundan emin olun.")
            st.stop()
            
        df = market_service.add_indicators(df)
        
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        change = ((current_price - prev_close) / prev_close) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Fiyat", f"${current_price:,.2f}", f"{change:.2f}%")
        col2.metric("RSI (Momentum)", f"{df['RSI'].iloc[-1]:.1f}", delta=None)
        col3.metric("ADX (Trend Gücü)", f"{df['ADX'].iloc[-1]:.1f}", help="25 üzeri güçlü trend demektir.")
        col4.metric("ATR (Risk/Volatilite)", f"{df['ATR'].iloc[-1]:.2f}")
        
        signal = "NÖTR"
        if df['RSI'].iloc[-1] < 30:
            signal = "AŞIRI SATIM (Al Fırsatı?)"
        elif df['RSI'].iloc[-1] > 70:
            signal = "AŞIRI ALIM (Sat Fırsatı?)"
        col5.metric("Teknik Sinyal", signal)

    tab_chart, tab_depth, tab_ai, tab_vision = st.tabs(
        ["📊 Pro Grafik", "🌊 Derinlik (Depth)", "🤖 AI Raporu", "👁️ Görsel Analiz"]
    )

    with tab_chart:
        st.plotly_chart(create_advanced_chart(df, symbol), use_container_width=True)
        
        with st.expander("🔍 Tespit Edilen Mum Formasyonları"):
            last_candles = df.tail(5)
            found_patterns = []
            if (last_candles['DOJI'] != 0).any():
                found_patterns.append("Doji (Kararsızlık)")
            if (last_candles['ENGULFING'] != 0).any():
                found_patterns.append("Engulfing (Yutma/Dönüş)")
            
            if found_patterns:
                st.success(f"Son 5 mumda tespit edilenler: {', '.join(found_patterns)}")
            else:
                st.info("Son mumlarda belirgin bir formasyon yok.")

    with tab_depth:
        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            bids, asks = market_service.fetch_order_book(symbol)
            if not bids.empty:
                st.plotly_chart(create_depth_chart(bids, asks), use_container_width=True)
            else:
                st.warning("Bu parite için derinlik verisi çekilemedi.")
        with col_d2:
            st.markdown("##### Emir Defteri")
            if not asks.empty and not bids.empty:
                st.markdown("**🔴 Satıcılar (Asks)**")
                st.dataframe(asks.head(5).sort_values('price', ascending=False), hide_index=True)
                st.markdown("**🟢 Alıcılar (Bids)**")
                st.dataframe(bids.head(5), hide_index=True)

    with tab_ai:
        if st.button("✨ Detaylı AI Raporu Oluştur", type="primary"):
            news = market_service.fetch_crypto_news()
            with st.spinner("Piyasa yapısı, haber akışı ve teknik veriler analiz ediliyor..."):
                report = ai_engine.analyze_market_structure(df, news, symbol, trader_mode)
                st.markdown(report)

    with tab_vision:
        st.info("Grafik ekran görüntüsünü yükleyin, AI sizin için formasyonları çizsin.")
        upl = st.file_uploader("Dosya Yükle", type=['png', 'jpg'])
        if upl:
            img = Image.open(upl)
            st.image(img, width=400)
            if st.button("Görseli Tara"):
                res = ai_engine.analyze_chart_image(img, "Detaylı teknik analiz yap.")
                st.write(res)

if __name__ == "__main__":
    main()
