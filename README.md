# nifty50-forecasting-arima-prophet-lstm
NIFTY 50 multi-model time series forecasting — ARIMA vs Prophet vs LSTM
# NIFTY 50 Multi-Model Time Series Forecasting
> ARIMA vs Prophet vs LSTM with Technical Indicators & Trading Backtest

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview
A complete end-to-end time series forecasting system for the **NIFTY 50 index** comparing three model families — classical statistics (ARIMA), decomposition-based (Prophet), and deep learning (LSTM) — with a live **Streamlit web app**.

---

## Results

| Model | MAPE | RMSE | Winner |
|-------|------|------|--------|
| ARIMA(1,1,0) | **0.66%** | 203.50 | 🏆 |
| Prophet | 3.34% | — | |
| LSTM | 7.41% | — | |

**Key finding:** ARIMA dominates short-term (60-day) forecasting because
price autocorrelation (lag-1) is the strongest signal. LSTM requires
longer sequences and more data to outperform on financial time series.

---

## Live App
```bash
streamlit run nifty_app.py
```
**Features:**
- Select ticker (NIFTY 50, BANKNIFTY, RELIANCE, HDFCBANK, TCS...)
- Choose model (ARIMA / Prophet / LSTM)
- Interactive forecast chart
- Technical indicators (RSI, MACD, Bollinger Bands)
- Trading backtest vs Buy & Hold
- Metrics comparison table

---

## Project Structure
```
nifty50-forecasting/
├── notebooks/
│   ├── nifty_step1_data_collection.ipynb  # Data + indicators
│   ├── nifty_step2_arima.ipynb            # ARIMA modelling
│   ├── nifty_step3_prophet.ipynb          # Prophet modelling
│   ├── nifty_step4_lstm.ipynb             # LSTM modelling
│   └── nifty_step5_comparison.ipynb       # Final comparison
├── data/
│   └── nifty50_with_indicators.csv
├── charts/
├── nifty_app.py                           # Streamlit app
├── requirements.txt
└── README.md
```

---

## Technical Indicators Used
| Indicator | Type | Purpose |
|-----------|------|---------|
| RSI (14) | Momentum | Overbought/oversold signals |
| MACD (12/26/9) | Trend | Trend direction & strength |
| Bollinger Bands (20) | Volatility | Price channel & squeeze |
| SMA 20/50/200 | Trend | Moving average crossovers |
| ATR (14) | Volatility | Average true range |

---

## Installation
```bash
git clone https://github.com/pavankumartelimeti/nifty50-forecasting-arima-prophet-lstm.git
cd nifty50-forecasting-arima-prophet-lstm
pip install -r requirements.txt
streamlit run nifty_app.py
```

---

## Tools & Libraries
`yfinance` `pandas` `numpy` `statsmodels` `pmdarima` `prophet`
`tensorflow` `scikit-learn` `ta` `streamlit` `plotly` `matplotlib`

---

## Key Learnings
1. **ARIMA wins short-term** — lag-1 autocorrelation is the dominant signal in 60-day windows
2. **Prophet excels at decomposition** — trend + weekly + yearly + holiday components are interpretable
3. **LSTM needs more data** — 5 years of daily data is insufficient; 10+ years would help
4. **Directional accuracy matters** — for trading, up/down prediction beats minimising point error

---

*Built as part of a data science portfolio — Pavan Kumar*
