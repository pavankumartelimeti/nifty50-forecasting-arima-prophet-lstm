import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY Forecasting Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0a0e1a;
        color: #e8eaf0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f1628 !important;
        border-right: 1px solid #1e2a45;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] p {
        color: #8892a4 !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Metric cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1e2a45;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #3b82f6; }
    .metric-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
        font-family: 'Space Mono', monospace;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #f0f4ff;
        font-family: 'Space Mono', monospace;
    }
    .metric-sub {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }
    .metric-good  { color: #10b981; }
    .metric-warn  { color: #f59e0b; }
    .metric-bad   { color: #ef4444; }

    /* Section headers */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin: 32px 0 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2a45;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f1628 0%, #111827 100%);
        border: 1px solid #1e2a45;
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: #f0f4ff;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-size: 14px;
        color: #6b7280;
    }
    .hero-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        font-size: 11px;
        font-family: 'Space Mono', monospace;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 6px;
        margin-top: 12px;
        border: 1px solid #2563eb33;
    }

    /* Winner badge */
    .winner-badge {
        background: #052e16;
        border: 1px solid #16a34a;
        color: #4ade80;
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        padding: 4px 14px;
        border-radius: 20px;
    }

    /* Table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .styled-table th {
        background: #111827;
        color: #6b7280;
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid #1e2a45;
    }
    .styled-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #1a2236;
        color: #d1d5db;
        font-family: 'Space Mono', monospace;
        font-size: 13px;
    }
    .styled-table tr:hover td { background: #111827; }
    .styled-table .best { color: #10b981; font-weight: 700; }

    /* Insight box */
    .insight-box {
        background: #0c1929;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.7;
    }
    .insight-box strong { color: #60a5fa; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    div[data-testid="stButton"] button {
        background: #1e3a5f;
        color: #60a5fa;
        border: 1px solid #2563eb55;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        padding: 8px 20px;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] button:hover {
        background: #2563eb;
        color: white;
        border-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().dropna()
        return raw
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return None


def add_indicators(df):
    try:
        import ta
        d = df.copy()
        d['SMA_20']  = d['Close'].rolling(20).mean()
        d['SMA_50']  = d['Close'].rolling(50).mean()
        d['EMA_12']  = d['Close'].ewm(span=12).mean()
        d['EMA_26']  = d['Close'].ewm(span=26).mean()
        d['RSI_14']  = ta.momentum.RSIIndicator(d['Close'], 14).rsi()
        macd = ta.trend.MACD(d['Close'])
        d['MACD'] = macd.macd()
        d['MACD_Signal'] = macd.macd_signal()
        d['MACD_Hist']   = macd.macd_diff()
        bb = ta.volatility.BollingerBands(d['Close'], 20, 2)
        d['BB_Upper']  = bb.bollinger_hband()
        d['BB_Middle'] = bb.bollinger_mavg()
        d['BB_Lower']  = bb.bollinger_lband()
        d['BB_Pct']    = bb.bollinger_pband()
        d['Daily_Return'] = d['Close'].pct_change() * 100
        return d.dropna()
    except ImportError:
        st.warning("Install `ta` library for technical indicators: pip install ta")
        return df


def run_arima(train, test_size):
    try:
        import pmdarima as pm
        from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
        model = pm.auto_arima(train, d=1, seasonal=False, stepwise=True,
                              suppress_warnings=True, error_action='ignore', trace=False)
        order = model.order
        history = list(train)
        preds = []
        for i in range(test_size):
            m = ARIMAModel(history, order=order).fit()
            preds.append(m.forecast(1)[0])
            history.append(train.iloc[-(test_size - i)])
        return np.array(preds), order
    except Exception as e:
        st.error(f"ARIMA error: {e}")
        return None, None


def run_prophet(train_df, test_dates):
    try:
        from prophet import Prophet
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    seasonality_mode='multiplicative', interval_width=0.95)
        m.fit(train_df)
        future = pd.DataFrame({'ds': test_dates})
        fc = m.predict(future)
        return fc['yhat'].values, fc['yhat_lower'].values, fc['yhat_upper'].values
    except Exception as e:
        st.error(f"Prophet error: {e}")
        return None, None, None


def run_lstm(df, test_size, seq_len=30):
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        tf.random.set_seed(42)
        np.random.seed(42)

        features = ['Close', 'RSI_14', 'MACD', 'BB_Pct', 'Daily_Return']
        available = [f for f in features if f in df.columns]
        data = df[available].values

        train_data = data[:-(test_size)]
        test_data  = data[-(test_size + seq_len):]

        scaler = MinMaxScaler()
        train_sc = scaler.fit_transform(train_data)
        test_sc  = scaler.transform(test_data)

        def make_seq(d, sl):
            X, y = [], []
            for i in range(sl, len(d)):
                X.append(d[i-sl:i])
                y.append(d[i, 0])
            return np.array(X), np.array(y)

        X_tr, y_tr = make_seq(train_sc, seq_len)
        X_te, y_te = make_seq(test_sc,  seq_len)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True,
                                 input_shape=(seq_len, len(available))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        model.fit(X_tr, y_tr, epochs=30, batch_size=32,
                  validation_split=0.1, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,
                             restore_best_weights=True)])

        pred_sc = model.predict(X_te, verbose=0)
        dummy = np.zeros((len(pred_sc), len(available)))
        dummy[:, 0] = pred_sc.flatten()
        preds = scaler.inverse_transform(dummy)[:, 0]
        return preds
    except Exception as e:
        st.error(f"LSTM error: {e}")
        return None


def compute_metrics(actual, predicted):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mask = ~np.isnan(predicted)
    a, p = actual[mask], predicted[mask]
    rmse = np.sqrt(mean_squared_error(a, p))
    mae  = mean_absolute_error(a, p)
    mape = np.mean(np.abs((a - p) / a)) * 100
    dirs_a = np.sign(np.diff(a))
    dirs_p = np.sign(np.diff(p))
    dir_acc = (dirs_a == dirs_p).mean() * 100
    return rmse, mae, mape, dir_acc


def backtest(actual, predictions, cost=0.001):
     # Align lengths
    min_len = min(len(actual), len(predictions))
    actual      = actual[:min_len]
    predictions = predictions[:min_len]
    
    daily_ret = np.diff(actual) / actual[:-1]
    signals   = np.array([1 if predictions[i+1] > actual[i] else 0
                          for i in range(len(actual)-1)])
    trades    = np.abs(np.diff(np.concatenate([[0], signals])))
    strat_ret = signals * daily_ret - trades[:-1] * cost
    cum_strat = (1 + strat_ret).cumprod()
    cum_bh    = (1 + daily_ret).cumprod()
    sharpe    = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 0 else 0
    max_dd    = ((cum_strat / cum_strat.cummax()) - 1).min() * 100
    n_trades  = int(trades.sum())
    return cum_strat, cum_bh, sharpe, max_dd, n_trades


# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='#0a0e1a',
    plot_bgcolor='#0f1628',
    font=dict(family='DM Sans', color='#8892a4', size=12),
    xaxis=dict(gridcolor='#1e2a45', showgrid=True, zeroline=False,
               tickfont=dict(color='#6b7280')),
    yaxis=dict(gridcolor='#1e2a45', showgrid=True, zeroline=False,
               tickfont=dict(color='#6b7280')),
    legend=dict(bgcolor='#111827', bordercolor='#1e2a45', borderwidth=1,
                font=dict(color='#94a3b8', size=11)),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode='x unified',
)

MODEL_COLORS = {
    'ARIMA'  : '#e94560',
    'Prophet': '#10b981',
    'LSTM'   : '#a78bfa',
}

TICKER_OPTIONS = {
    'NIFTY 50'        : '^NSEI',
    'BANKNIFTY'       : '^NSEBANK',
    'RELIANCE'        : 'RELIANCE.NS',
    'HDFCBANK'        : 'HDFCBANK.NS',
    'TCS'             : 'TCS.NS',
    'INFOSYS'         : 'INFY.NS',
    'ICICIBANK'       : 'ICICIBANK.NS',
    'WIPRO'           : 'WIPRO.NS',
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:18px;
                color:#f0f4ff;font-weight:700;margin-bottom:4px'>
        NIFTY LAB
    </div>
    <div style='font-size:11px;color:#4b5563;margin-bottom:28px'>
        Forecasting Dashboard
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**INSTRUMENT**")
    ticker_name = st.selectbox("", list(TICKER_OPTIONS.keys()), label_visibility='collapsed')
    ticker = TICKER_OPTIONS[ticker_name]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("**DATE RANGE**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.to_datetime("2020-01-01"),
                                   label_visibility='collapsed')
    with col2:
        end_date = st.date_input("To", value=pd.to_datetime("2024-12-31"),
                                 label_visibility='collapsed')

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("**MODELS**")
    models_selected = st.multiselect("",
        ['ARIMA', 'Prophet', 'LSTM'],
        default=['ARIMA'],
        label_visibility='collapsed'
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("**TEST PERIOD (DAYS)**")
    test_size = st.slider("", 30, 120, 60, label_visibility='collapsed')

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    run_btn = st.button("RUN FORECAST", use_container_width=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#374151;line-height:1.8'>
        Built with yfinance · statsmodels<br>
        pmdarima · Prophet · TensorFlow<br>
        ta · scikit-learn · Plotly
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='hero'>
    <div class='hero-title'>NIFTY 50 Forecasting Lab</div>
    <div class='hero-sub'>Multi-model time series forecasting — ARIMA · Prophet · LSTM</div>
    <span class='hero-badge'>ARIMA</span>
    <span class='hero-badge'>Prophet</span>
    <span class='hero-badge'>LSTM</span>
    <span class='hero-badge'>Technical Indicators</span>
    <span class='hero-badge'>Backtest</span>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div class='insight-box'>
        <strong>How to use:</strong> Select an instrument and date range from the sidebar,
        choose one or more models, set your test period, then click <strong>RUN FORECAST</strong>.
        The app will download live data, compute technical indicators, run selected models,
        and show forecast charts, metrics, and a trading backtest.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Downloading {ticker_name} data..."):
    df_raw = load_data(ticker, str(start_date), str(end_date))

if df_raw is None or len(df_raw) < 100:
    st.error("Not enough data. Try a wider date range.")
    st.stop()

with st.spinner("Computing technical indicators..."):
    df = add_indicators(df_raw)

close      = df['Close']
train      = close.iloc[:-test_size]
test       = close.iloc[-test_size:]
test_dates = test.index
actual     = test.values

# ── Run models ────────────────────────────────────────────────────────────────
results = {}

if 'ARIMA' in models_selected:
    with st.spinner("Running ARIMA (walk-forward)..."):
        arima_pred, arima_order = run_arima(train, test_size)
        if arima_pred is not None:
            results['ARIMA'] = arima_pred

if 'Prophet' in models_selected:
    with st.spinner("Running Prophet..."):
        train_prophet = pd.DataFrame({'ds': train.index, 'y': train.values})
        p_pred, p_lower, p_upper = run_prophet(train_prophet, test_dates)
        if p_pred is not None:
            results['Prophet'] = p_pred

if 'LSTM' in models_selected:
    with st.spinner("Training LSTM (30 epochs)..."):
        lstm_pred = run_lstm(df.iloc[:-test_size + len(df)], test_size)
        if lstm_pred is not None and len(lstm_pred) == test_size:
            results['LSTM'] = lstm_pred

if not results:
    st.error("No models ran successfully.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>01 — Performance Metrics</div>",
            unsafe_allow_html=True)

metrics_data = {}
for name, pred in results.items():
    rmse, mae, mape, dir_acc = compute_metrics(actual, pred)
    metrics_data[name] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'DirAcc': dir_acc}

best_mape_model = min(metrics_data, key=lambda x: metrics_data[x]['MAPE'])

cols = st.columns(len(results) * 2)
col_idx = 0
for name, m in metrics_data.items():
    is_best = name == best_mape_model
    badge = "<span class='winner-badge'>WINNER</span>" if is_best else ""
    with cols[col_idx]:
        mape_class = 'metric-good' if m['MAPE'] < 1 else ('metric-warn' if m['MAPE'] < 5 else 'metric-bad')
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{name} {badge}</div>
            <div class='metric-value {mape_class}'>{m['MAPE']:.2f}%</div>
            <div class='metric-sub'>MAPE</div>
        </div>
        """, unsafe_allow_html=True)
    col_idx += 1
    with cols[col_idx]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Direction Acc</div>
            <div class='metric-value'>{m['DirAcc']:.1f}%</div>
            <div class='metric-sub'>Up/Down correct</div>
        </div>
        """, unsafe_allow_html=True)
    col_idx += 1

# Metrics table
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
table_rows = ""
for name, m in metrics_data.items():
    is_best = name == best_mape_model
    cls = "best" if is_best else ""
    table_rows += f"""
    <tr>
        <td class='{cls}'>{name} {'🏆' if is_best else ''}</td>
        <td class='{cls}'>{m['RMSE']:,.1f}</td>
        <td class='{cls}'>{m['MAE']:,.1f}</td>
        <td class='{cls}'>{m['MAPE']:.2f}%</td>
        <td class='{cls}'>{m['DirAcc']:.1f}%</td>
    </tr>"""

st.markdown(f"""
<table class='styled-table'>
    <tr>
        <th>Model</th><th>RMSE (pts)</th><th>MAE (pts)</th>
        <th>MAPE</th><th>Direction Acc</th>
    </tr>
    {table_rows}
</table>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FORECAST CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>02 — Forecast vs Actual</div>",
            unsafe_allow_html=True)

fig = go.Figure()

# Train context (last 90 days)
train_tail = train.iloc[-90:]
fig.add_trace(go.Scatter(
    x=train_tail.index, y=train_tail.values,
    name='Train', line=dict(color='#374151', width=1),
    showlegend=True
))

# Actual
fig.add_trace(go.Scatter(
    x=test_dates, y=actual,
    name='Actual', line=dict(color='#f0f4ff', width=2),
    showlegend=True
))

# Model forecasts
for name, pred in results.items():
    fig.add_trace(go.Scatter(
        x=test_dates, y=pred,
        name=f'{name} (MAPE: {metrics_data[name]["MAPE"]:.2f}%)',
        line=dict(color=MODEL_COLORS[name], width=1.5, dash='dash'),
        showlegend=True
    ))

# Prophet confidence interval
if 'Prophet' in results and p_lower is not None:
    fig.add_trace(go.Scatter(
        x=list(test_dates) + list(test_dates[::-1]),
        y=list(p_upper) + list(p_lower[::-1]),
        fill='toself', fillcolor='rgba(16,185,129,0.06)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Prophet 95% CI', showlegend=True
    ))

# Forecast start line
fig.add_vline(x=str(test_dates[0]), line_dash='dot',
              line_color='#374151', line_width=1)

fig.update_layout(**PLOT_LAYOUT,
    title=dict(text=f'{ticker_name} — Forecast vs Actual',
               font=dict(color='#f0f4ff', size=14)),
    height=420,
    yaxis_title='Index Level (INR)',
)
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>03 — Technical Indicators</div>",
            unsafe_allow_html=True)

display_df = df.iloc[-180:]   # last 6 months

fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                     row_heights=[0.5, 0.25, 0.25],
                     vertical_spacing=0.04)

# Panel 1: Price + MAs + BB
fig2.add_trace(go.Scatter(x=display_df.index, y=display_df['Close'],
    name='Close', line=dict(color='#f0f4ff', width=1.5)), row=1, col=1)
if 'SMA_20' in display_df.columns:
    fig2.add_trace(go.Scatter(x=display_df.index, y=display_df['SMA_20'],
        name='SMA 20', line=dict(color='#e94560', width=1, dash='dot')), row=1, col=1)
if 'SMA_50' in display_df.columns:
    fig2.add_trace(go.Scatter(x=display_df.index, y=display_df['SMA_50'],
        name='SMA 50', line=dict(color='#3b82f6', width=1, dash='dot')), row=1, col=1)
if 'BB_Upper' in display_df.columns:
    fig2.add_trace(go.Scatter(x=display_df.index, y=display_df['BB_Upper'],
        name='BB Upper', line=dict(color='#4b5563', width=0.8, dash='dash')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=display_df.index, y=display_df['BB_Lower'],
        name='BB Lower', line=dict(color='#4b5563', width=0.8, dash='dash'),
        fill='tonexty', fillcolor='rgba(75,85,99,0.05)'), row=1, col=1)

# Panel 2: RSI
if 'RSI_14' in display_df.columns:
    fig2.add_trace(go.Scatter(x=display_df.index, y=display_df['RSI_14'],
        name='RSI 14', line=dict(color='#f59e0b', width=1.2)), row=2, col=1)
    fig2.add_hline(y=70, line_dash='dot', line_color='#ef4444',
                   line_width=0.8, row=2, col=1)
    fig2.add_hline(y=30, line_dash='dot', line_color='#10b981',
                   line_width=0.8, row=2, col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.05)',
                   line_width=0, row=2, col=1)
    fig2.add_hrect(y0=0, y1=30, fillcolor='rgba(16,185,129,0.05)',
                   line_width=0, row=2, col=1)

# Panel 3: MACD
if 'MACD' in display_df.columns:
    colors_hist = ['#10b981' if v >= 0 else '#ef4444'
                   for v in display_df['MACD_Hist']]
    fig2.add_trace(go.Bar(x=display_df.index, y=display_df['MACD_Hist'],
        name='MACD Hist', marker_color=colors_hist, opacity=0.6), row=3, col=1)
    fig2.add_trace(go.Scatter(x=display_df.index, y=display_df['MACD'],
        name='MACD', line=dict(color='#3b82f6', width=1.2)), row=3, col=1)
    fig2.add_trace(go.Scatter(x=display_df.index, y=display_df['MACD_Signal'],
        name='Signal', line=dict(color='#e94560', width=1, dash='dot')), row=3, col=1)

fig2.update_layout(**PLOT_LAYOUT,
    title=dict(text=f'{ticker_name} — Technical Indicators (Last 6 months)',
               font=dict(color='#f0f4ff', size=14)),
    height=560,
    showlegend=True,
)
fig2.update_yaxes(title_text='Price (INR)', row=1, col=1,
                  title_font=dict(color='#6b7280', size=11))
fig2.update_yaxes(title_text='RSI', row=2, col=1,
                  range=[0, 100], title_font=dict(color='#6b7280', size=11))
fig2.update_yaxes(title_text='MACD', row=3, col=1,
                  title_font=dict(color='#6b7280', size=11))

st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>04 — Trading Backtest</div>",
            unsafe_allow_html=True)

st.markdown("""
<div class='insight-box'>
    <strong>Strategy:</strong> If model predicts tomorrow's price &gt; today → BUY.
    Otherwise → stay CASH. Transaction cost: <strong>0.1% per trade</strong> (realistic for NSE F&O).
    Compared against passive <strong>Buy &amp; Hold</strong>.
</div>
""", unsafe_allow_html=True)

fig3 = go.Figure()
bh_plotted = False
bt_metrics = {}

for name, pred in results.items():
    cum_s, cum_bh, sharpe, max_dd, n_trades = backtest(actual, pred)
    bt_metrics[name] = {'Sharpe': sharpe, 'MaxDD': max_dd, 'Trades': n_trades,
                        'Return': (cum_s[-1] - 1) * 100}

    if not bh_plotted:
        fig3.add_trace(go.Scatter(
            x=test_dates[1:], y=cum_bh,
            name='Buy & Hold', line=dict(color='#6b7280', width=1.5),
        ))
        bh_plotted = True

    fig3.add_trace(go.Scatter(
        x=test_dates[1:], y=cum_s,
        name=f'{name} Strategy',
        line=dict(color=MODEL_COLORS[name], width=1.5, dash='dash'),
    ))

fig3.add_hline(y=1.0, line_dash='dot', line_color='#374151', line_width=1)
fig3.update_layout(**PLOT_LAYOUT,
    title=dict(text='Cumulative Returns — Strategy vs Buy & Hold',
               font=dict(color='#f0f4ff', size=14)),
    height=360,
    yaxis_title='Cumulative Return',
)
st.plotly_chart(fig3, use_container_width=True)

# Backtest metrics table
bt_rows = ""
bh_ret = (cum_bh[-1] - 1) * 100
for name, m in bt_metrics.items():
    beat = "✓" if m['Return'] > bh_ret else "✗"
    color = "#10b981" if m['Return'] > bh_ret else "#ef4444"
    bt_rows += f"""
    <tr>
        <td>{name}</td>
        <td style='color:{color}'>{m['Return']:+.2f}%</td>
        <td style='color:#94a3b8'>{bh_ret:+.2f}%</td>
        <td>{m['Sharpe']:.2f}</td>
        <td>{m['MaxDD']:.2f}%</td>
        <td>{m['Trades']}</td>
        <td style='color:{color}'>{beat} Buy&Hold</td>
    </tr>"""

st.markdown(f"""
<table class='styled-table'>
    <tr>
        <th>Model</th><th>Strategy Return</th><th>Buy&Hold</th>
        <th>Sharpe</th><th>Max Drawdown</th><th>Trades</th><th>Result</th>
    </tr>
    {bt_rows}
</table>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>05 — Key Insights</div>",
            unsafe_allow_html=True)

winner = min(metrics_data, key=lambda x: metrics_data[x]['MAPE'])
winner_mape = metrics_data[winner]['MAPE']

st.markdown(f"""
<div class='insight-box'>
    <strong>Best model by MAPE:</strong> {winner} at {winner_mape:.2f}% —
    meaning average daily forecast error is {winner_mape:.2f}% of the actual index level.
</div>
<div class='insight-box'>
    <strong>Why ARIMA often wins short-term:</strong> Daily price autocorrelation (lag-1)
    is the strongest signal in a 60-day window. LSTM needs longer sequences and more data
    to capture non-linear patterns that outperform this simple dependency.
</div>
<div class='insight-box'>
    <strong>Why Prophet is valuable despite higher MAPE:</strong> Its components chart
    (trend + weekly + yearly + holidays) gives <strong>interpretable insights</strong>
    that ARIMA and LSTM cannot provide — which days of week are strongest,
    which months historically rally, holiday effects on NIFTY.
</div>
<div class='insight-box'>
    <strong>Directional accuracy vs MAPE:</strong> For trading, getting the
    <strong>direction right</strong> (up vs down) matters more than minimising point error.
    A model with higher MAPE but better directional accuracy may generate better trading signals.
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;font-family:Space Mono,monospace;
            font-size:11px;color:#374151;padding:20px'>
    NIFTY 50 Forecasting Lab · Built with Streamlit · Data via Yahoo Finance
</div>
""", unsafe_allow_html=True)