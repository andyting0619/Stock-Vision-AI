import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import LSTM

st.set_page_config(page_title="Stock Vision AI",
                   page_icon="🤖", layout="wide")
tab1, tab2 = st.tabs(["Prediction", "Dashboard"])
info_multi = '''📈This is an AI-powered stock forecasting and analysis web application using real-time stock data.   
💻This app blends cutting-edge deep learning with intuitive tools to bring you actionable insights in stock prediction.'''

with tab1:
    st.header('🤖Stock Vision AI Web Application🌐')
    st.info(info_multi)
    st.write(' ')


class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)


apple_model = load_model('./models/Apple-Model.h5',
                         custom_objects={"LSTM": CustomLSTM}, compile=False)
google_model = load_model('./models/Google-Model.h5',
                          custom_objects={"LSTM": CustomLSTM}, compile=False)
tesla_model = load_model('./models/Tesla-Model.h5',
                         custom_objects={"LSTM": CustomLSTM}, compile=False)
amazon_model = load_model('./models/Amazon-Model.h5',
                          custom_objects={"LSTM": CustomLSTM}, compile=False)
intel_model = load_model('./models/Intel-Model.h5',
                         custom_objects={"LSTM": CustomLSTM}, compile=False)
meta_model = load_model('./models/Meta-Model.h5',
                        custom_objects={"LSTM": CustomLSTM}, compile=False)
microsoft_model = load_model('./models/Microsoft-Model.h5',
                             custom_objects={"LSTM": CustomLSTM}, compile=False)

ticker_list = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'META', 'MSFT', 'TSLA']


def On_Balance_Volume(Close, Volume):
    change = Close.diff()
    OBV = np.cumsum(np.where(change > 0, Volume,
                    np.where(change < 0, -Volume, 0)))
    return OBV


@st.cache_data
def df_process(ticker):
    end = datetime.now()
    start = end - relativedelta(months=3)

    df = yf.download(ticker, start=start, end=end)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.columns.name = None

    column_dict = {'Open': 'open', 'High': 'high', 'Low': 'low',
                   'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}

    df = df.rename(columns=column_dict)

    df['garman_klass_volatility'] = ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - (
        2 * np.log(2) - 1) * ((np.log(df['close']) - np.log(df['open'])) ** 2)

    df['dollar_volume'] = (df['close'] * df['volume']) / 1e6

    df['ema'] = df['close'].ewm(span=14, adjust=False).mean()

    df['obv'] = On_Balance_Volume(df['close'], df['volume'])

    df['macd'] = df['close'].ewm(span=12, adjust=False).mean(
    ) - df['close'].ewm(span=26, adjust=False).mean()

    df['ma_3_days'] = df['close'].rolling(3).mean()
    return df


apple_df_processed = df_process(ticker_list[0])
amazon_df_processed = df_process(ticker_list[1])
google_df_processed = df_process(ticker_list[2])
intel_df_processed = df_process(ticker_list[3])
meta_df_processed = df_process(ticker_list[4])
microsoft_df_processed = df_process(ticker_list[5])
tesla_df_processed = df_process(ticker_list[6])

apple_features = ['close', 'garman_klass_volatility',
                  'dollar_volume', 'obv', 'ma_3_days']
amazon_features = ['close', 'volume', 'dollar_volume', 'obv', 'ema']
google_features = ['close', 'volume',
                   'dollar_volume', 'obv', 'ma_3_days', 'macd']
intel_features = ['close', 'garman_klass_volatility',
                  'dollar_volume', 'obv', 'ma_3_days']
meta_features = ['close', 'volume', 'dollar_volume', 'obv', 'ema']
microsoft_features = ['close', 'volume',
                      'garman_klass_volatility', 'dollar_volume', 'obv', 'ma_3_days']
tesla_features = ['close', 'dollar_volume', 'obv', 'ema', 'ma_3_days']


def create_feed_dset(df_processed, feature_list, n_past, model):
    dset = df_processed.filter(feature_list)
    dset.dropna(axis=0, inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(dset)

    dataX = []
    dataY = []
    for i in range(n_past, len(df_scaled)):
        dataX.append(df_scaled[i - n_past:i, 0:df_scaled.shape[1]])
        dataY.append(df_scaled[i, 0])
    dataX = np.array(dataX)

    prediction = model.predict(dataX)
    return prediction, scaler


apple_prediction_init, scaler1 = create_feed_dset(
    apple_df_processed, apple_features, 21, apple_model)
amazon_prediction_init, scaler2 = create_feed_dset(
    amazon_df_processed, amazon_features, 15, amazon_model)
google_prediction_init, scaler3 = create_feed_dset(
    google_df_processed, google_features, 21, google_model)
intel_prediction_init, scaler4 = create_feed_dset(
    intel_df_processed, intel_features, 25, intel_model)
meta_prediction_init, scaler5 = create_feed_dset(
    meta_df_processed, meta_features, 20, meta_model)
microsoft_prediction_init, scaler6 = create_feed_dset(
    microsoft_df_processed, microsoft_features, 20, microsoft_model)
tesla_prediction_init, scaler7 = create_feed_dset(
    tesla_df_processed, tesla_features, 15, tesla_model)


def inverse_transform_predictions1(prediction_init, scaler):
    prediction_array = np.repeat(prediction_init, 5, axis=-1)

    pred = scaler.inverse_transform(np.reshape(
        prediction_array, (len(prediction_init), 5)))[:5, 0]
    return pred


def inverse_transform_predictions2(prediction_init, scaler):
    prediction_array = np.repeat(prediction_init, 6, axis=-1)
    pred = scaler.inverse_transform(np.reshape(
        prediction_array, (len(prediction_init), 6)))[:5, 0]
    return pred


apple_pred_list = inverse_transform_predictions1(
    apple_prediction_init, scaler1).tolist()
amazon_pred_list = inverse_transform_predictions1(
    amazon_prediction_init, scaler2).tolist()
intel_pred_list = inverse_transform_predictions1(
    intel_prediction_init, scaler4).tolist()
meta_pred_list = inverse_transform_predictions1(
    meta_prediction_init, scaler5).tolist()
tesla_pred_list = inverse_transform_predictions1(
    tesla_prediction_init, scaler7).tolist()
google_pred_list = inverse_transform_predictions2(
    google_prediction_init, scaler3).tolist()
microsoft_pred_list = inverse_transform_predictions2(
    microsoft_prediction_init, scaler6).tolist()


def prediction_table(pred_list):
    pred_df = pd.DataFrame({'Time (Day)': ['Tomorrow', '2nd Day', '3rd Day', '4th Day', '5th Day'],
                            'Prediction of Adjusted Closing Price ($)': ['%.2f' % elem for elem in pred_list]})

    pred_df.set_index('Time (Day)', inplace=True)
    return pred_df


def generate_insight(df_processed, pred_list):
    actual_values = df_processed['close'].values.tolist()

    if actual_values and pred_list:
        last_actual_price = actual_values[-1]
        next_predicted_price = pred_list[0]

        percent_change = (next_predicted_price -
                          last_actual_price) / last_actual_price * 100

        insight = f"""
        <div style="font-family: Inter, sans-serif; font-size: 16px; line-height: 1.6;">
            <strong>The predicted stock price of the next day is:</strong> 
            <span style="color: #2ECC71;">${next_predicted_price:.2f}</span><br>
            <strong>The actual stock price of the last day:</strong> 
            <span style="color: #B71C1C;">${last_actual_price:.2f}</span><br>
            <strong>Predicted stock return:</strong> 
            <span style="color: {'#2ECC71' if percent_change >= 0 else '#B71C1C'};">
                {percent_change:+.2f}%
            </span>
        </div>
        """
    else:
        insight = """<div style="font-family: Inter, sans-serif; font-size: 16px; color: #E0E0E0; text-align: center;">
    Not enough data to generate insights.
</div>
"""
    return insight


stock_selection = tab1.selectbox("🔍Selected Company:",
                                 options=["Apple", "Amazon", "Google", "Intel", "Meta", "Microsoft", "Tesla"])

if stock_selection == "Apple":
    selected_pred_list = apple_pred_list
    selected_df_processed = apple_df_processed
elif stock_selection == "Amazon":
    selected_pred_list = amazon_pred_list
    selected_df_processed = amazon_df_processed
elif stock_selection == "Google":
    selected_pred_list = google_pred_list
    selected_df_processed = google_df_processed
elif stock_selection == "Intel":
    selected_pred_list = intel_pred_list
    selected_df_processed = intel_df_processed
elif stock_selection == "Meta":
    selected_pred_list = meta_pred_list
    selected_df_processed = meta_df_processed
elif stock_selection == "Microsoft":
    selected_pred_list = microsoft_pred_list
    selected_df_processed = microsoft_df_processed
elif stock_selection == "Tesla":
    selected_pred_list = tesla_pred_list
    selected_df_processed = tesla_df_processed

pred_df = prediction_table(selected_pred_list)
insight = generate_insight(selected_df_processed, selected_pred_list)


tab1.col1, tab1.col2 = tab1.columns(2)
with tab1.col1:
    st.markdown(f"""<div style="font-family: Inter, sans-serif; font-size: 18px; line-height: 1.6;"> 
    <strong>{stock_selection}</strong><be> 
    <strong>Stock Predictions for the Next 5 Days</strong>
    </div>""", unsafe_allow_html=True)
    st.dataframe(pred_df)

with tab1.col2:
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.markdown(insight, unsafe_allow_html=True)


dedication = """<div style="font-family: Inter, sans-serif; font-size: 16px; line-height: 1.6;">🛠️The Stock Vision AI model is trained by Andy Ting Zhi Wei."""
with tab1.container(border=True):
    st.markdown(dedication, unsafe_allow_html=True)
    st.markdown(
        '''👨‍💻This web application is deployed by :blue-background[Andy Ting Zhi Wei].''')

tab1.warning('Disclaimer: This project is only for research and educational purposes, and it is not intended for financial or investment advice!', icon="⚠️")

with tab2:
    st.header("🖥️Stock Data Interactive Dashboard📉")
    st.markdown(
        '''<div style="background-color:#1E1E1E; padding:10px; border-radius:5px; font-family: Inter, sans-serif; color:#E0E0E0;">
        📊<strong>Technical Analysis:</strong> Explore trends with indicators like <span style="color:#007BFF;">SMA</span>, <span style="color:#FFD700;">EMA</span>, <span style="color:#2ECC71;">RSI</span>, and <span style="color:#FF5733;">OBV</span> using interactive charts.
        </div>''', unsafe_allow_html=True
    )
    st.write("\n")

obv_text = '''

**Meaning:** Tracks the flow of stock volume to predict stock price changes.  

**Purpose:** Identifies buying/selling pressure based on stock volume. A rising OBV suggests accumulation (buying), while a falling OBV suggests distribution (selling).  

**Usage:** Combines with price trends to confirm breakout patterns or reversals.  

'''

ma_text = '''

**Meaning:** Moving averages smooth out price data to identify trends over a period.  
Simple Moving Average (SMA) -- Average of closing prices over a fixed period.   
Exponential Moving Average (EMA) -- Similar to SMA but gives more weight to recent prices for faster responsiveness.  

**Purpose:**
SMA -- Tracks long-term trend (e.g., 50-day and 200-day SMA).   
EMA -- Tracks short-term momentum (e.g., 15-day and 30-day EMA).

**Usage:**
Bullish signal -- Short-term MA crosses above long-term MA ("Golden Cross").  
Bearish signal -- Short-term MA crosses below long-term MA ("Death Cross").

'''

rsi_text = '''

**Meaning:** RSI measures price momentum to identify overbought/oversold conditions, and compares average gains and losses over 14 days to generate a score between 0-100.   
RSI > 70 -- Overbought (may signal a sell opportunity).  
RSI < 30 -- Oversold (may signal a buy opportunity).   

**Purpose:** Indicates potential reversals or continuation in price trends.  

**Usage:** Combine with other indicators to confirm breakout or correction signals.

'''

tab2.col1, tab2.col2, tab2.col3 = tab2.columns([1, 1, 1])

st.markdown("""
<style>
.stPop {
    background: rgba(30, 30, 30, 0.85); /* Semi-transparent dark */
    border: 1px solid #00D4FF; /* Cyber Blue accent */
    border-radius: 8px; /* Rounded edges for smooth UI */
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

with tab2.col1:
    with st.popover("📈On-Balance Volume (OBV)"):
        st.markdown(
            f"<div style='font-family: Inter, sans-serif; color:#E0E0E0;'>{obv_text}</div>", unsafe_allow_html=True)

with tab2.col2:
    with st.popover("🔄Moving Averages (SMA/EMA)"):
        st.markdown(
            f"<div style='font-family: Inter, sans-serif; color:#E0E0E0;'>{ma_text}</div>", unsafe_allow_html=True)

with tab2.col3:
    with st.popover("💪Relative Strength Index (RSI)"):
        st.markdown(
            f"<div style='font-family: Inter, sans-serif; color:#E0E0E0;'>{rsi_text}</div>", unsafe_allow_html=True)


def load_data(ticker, start_date):
    stock_data = yf.download(ticker, start=start_date)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    stock_data.reset_index(inplace=True)
    return stock_data


def calculate_indicators(data):
    data['OBV'] = (data['Volume'] *
                   ((data['Close'] > data['Close'].shift(1)) * 2 - 1)).cumsum()

    data['SMA_50'] = data['Close'].rolling(
        window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(
        span=200, adjust=False).mean()

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Mid'] + \
        (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower'] = data['BB_Mid'] - \
        (data['Close'].rolling(window=20).std())

    return data


def plot_line_chart(data, x_col, y_cols, title):
    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=data[x_col], y=data[col], mode='lines', name=col))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Value",
        template="plotly_white")
    return fig


START_DATE = "2015-01-01"

ticker_list = ['AAPL', 'AMZN ', 'AMD', 'GOOGL',
               'INTC', 'META', 'MSFT', 'NVDA', 'TSLA']

selected_stock = tab2.selectbox('🏢Selected Company:', ticker_list)

technical_indicator = tab2.selectbox(
    '💡Selected Technical Indicator:',
    [
        'Open-High',
        'Low-Close',
        'Stock Volume',
        'OBV (On-Balance Volume)',
        'SMA/EMA',
        'RSI (Relative Strength Index)'])

data = load_data(selected_stock, START_DATE)
data = calculate_indicators(data)

if technical_indicator == 'Open-High':
    fig = plot_line_chart(data, 'Date', [
                          'Open', 'High'], f"Opening vs. Highest Prices for {selected_stock}")
elif technical_indicator == 'Low-Close':
    fig = plot_line_chart(data, 'Date', [
                          'Low', 'Close'], f"Lowest vs. Closing Prices for {selected_stock}")
elif technical_indicator == 'Stock Volume':
    fig = plot_line_chart(
        data, 'Date', ['Volume'], f"Stock Volume for {selected_stock}")
elif technical_indicator == 'OBV (On-Balance Volume)':
    fig = plot_line_chart(data, 'Date', ['OBV'], f"OBV for {selected_stock}")
elif technical_indicator == 'SMA/EMA':
    fig = plot_line_chart(data, 'Date', [
                          'Close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200'], f"SMA/EMA for {selected_stock}")
elif technical_indicator == 'RSI (Relative Strength Index)':
    fig = plot_line_chart(data, 'Date', ['RSI'], f"RSI for {selected_stock}")

with tab2:
    st.plotly_chart(fig)
