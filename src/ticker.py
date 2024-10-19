import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import time
import yfinance as yf
from datetime import datetime
from scipy import stats
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from sklearn.metrics import precision_score, recall_score, f1_score

from src.helper import *


def model_selection(df, stock_ticker):
    
    st.title("Model Selection")


    with st.spinner("CACULATION IS GOING ON -----"):


        z_df = df.copy()
        z_df['Close_Zscore'] = stats.zscore(z_df['Close'])

        threshold = 1.5
        z_df['Anomaly'] = np.where(np.abs(z_df['Close_Zscore']) > threshold, 'YES', "NO")


        # Isoation Forest
        # Create iso_df by copying the specified columns from new_df

        iso_df = df[['Close','Price Change', 'Volatility (7 days)',
                    'SMA_10', 'EMA_10', 'RSI', 'Upper_Band','Lower_Band', 'MACD', 'Signal_Line']].copy()

        iso_df.dropna(inplace=True)

        scaler=StandardScaler()
        X = scaler.fit_transform(iso_df)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)

        iso_df['Anomaly_iso'] = iso_forest.fit_predict(X)

        anomaly_count = (iso_df['Anomaly_iso'] == -1).sum()


        # ONE CLASS SVM ----------------
        # Create svm_df by selecting the relevant columns

        svm_df = df[['Close', 'Price Change', 'Volatility (7 days)', 'SMA_10', 'EMA_10', 'RSI', 'Upper_Band','Lower_Band']].copy()

        svm_df.dropna(inplace=True)
        # st.write(svm_df.shape)

        svm_model = OneClassSVM(nu= 0.05, kernel='rbf')
        svm_model.fit(svm_df)
        svm_df['SVM_Anomaly_Pred'] = svm_model.predict(svm_df)

        svm_anomaly_count = (svm_df['SVM_Anomaly_Pred'] == -1).sum()


        # DBSCAN -------------
        # Create dbscan_df by selecting the relevant columns

        dbscan_df = df[['Close', 'Price Change', 'Volatility (7 days)', 'SMA_10', 'EMA_10', 'RSI', 'Upper_Band', 'Lower_Band']].copy()
        dbscan_df.dropna(inplace=True)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        X2 = scaler.fit_transform(dbscan_df)
        dbscan_df['Cluster'] = dbscan.fit_predict(X2)
        dbscan_df['Anomaly_Detected'] = dbscan_df['Cluster'] == -1

        anomaly_count = (dbscan_df['Cluster'] == -1).sum()


        # LSTM ------------------
        # Create lstm_df by selecting relevant columns for LSTM
        lstm_df = df[['Close', 'Price Change', 'Volatility (7 days)', 'SMA_10', 'EMA_10', 'RSI', 'Upper_Band', 
                        'Lower_Band']].copy()

        lstm_df.dropna(inplace=True)
        
        lstm_df['Close'] = lstm_df['Close'].astype(float)
        prices = lstm_df['Close'].values.reshape(-1, 1)

        lstm_scalers = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = lstm_scalers.fit_transform(prices)

        time_step_lstm = 10
        X, y = create_sequences(scaled_prices, time_step_lstm)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X, y, epochs=100, batch_size=32, verbose = 0)

        lstm_predicted = model.predict(X)   # Generate predictions

        # Inverse transform to get lstm_actual values
        lstm_predicted = lstm_scalers.inverse_transform(lstm_predicted)
        lstm_actual = lstm_scalers.inverse_transform(y.reshape(-1, 1))
        
        lstm_error = lstm_actual - lstm_predicted # Calculate errors
        threshold = 1.5 * np.std(lstm_error) # Define a threshold 
        anomalies_lstm = np.where(np.abs(lstm_error) > threshold)[0] # Identify anomalies
        anomalies_lstm_len = len(anomalies_lstm) # Identify anomalies


        # Auto Encoders ----------------------

        # Create lstm_df by selecting relevant columns for LSTM
        autoenc_df = df[['Close', 'Price Change', 'Volatility (7 days)', 'SMA_10', 'EMA_10', 'RSI', 'Upper_Band', 'Lower_Band']].copy()

        autoenc_df.dropna(inplace=True)

        autoenc_df['Close'] = autoenc_df['Close'].astype(float)
        auto_prices = autoenc_df['Close'].values.reshape(-1, 1)
        # Scale the data
        scaler_auto = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler_auto.fit_transform(auto_prices)

        time_step_auto = 10
        X = create_sequences_for_AutoEnc(scaled_prices, time_step_auto)
        X = X.reshape(X.shape[0], X.shape[1])

        input_dim = X.shape[1]

        # Define the Autoencoder architecture
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(X, X, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, verbose = 0)

        # Get the reconstructed output
        reconstructed = autoencoder.predict(X)
        # Calculate the reconstruction error
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)

        threshold = 0.01     # Define a threshold
        anomalies_auto = np.where(mse > threshold)[0]
        anomalies_auto_len = len(anomalies_auto)


    # COLUMNS DEFINE ----------
    col1, col2 = st.columns(2)

    with col1.expander("Z Score"):
        plt.figure(figsize=(10, 5))
        plt.plot(z_df.index, z_df['Close'], label='Close Price', color='blue')
        plt.plot(z_df.index, z_df['Close_Zscore'], label='Close Z-Score', color='orange')
        plt.title(f'{stock_ticker} Close Price Z-Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('Z-Score')
        plt.grid(True)
        plt.legend()
        plt.show()
        st.pyplot(plt)


        plt.figure(figsize=(10, 5))
        plt.plot(z_df.index, z_df['Close_Zscore'], label='Close Z-Score', color='orange')
        plt.title(f'{stock_ticker} Close Price Z-Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('Z-Score')
        plt.grid(True)
        plt.legend()
        plt.show()
        st.pyplot(plt)


        plt.figure(figsize=(12, 6))
        plt.plot(z_df.index, z_df['Close'], label='Close Price', color='orange')
        plt.plot(z_df[z_df['Anomaly'] == 'YES'].index, z_df[z_df['Anomaly'] == 'YES']['Close'], 
                'ro', label='Anomalies (Z-Score)')
        plt.title(f'{stock_ticker} Close Price and Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)

    with col1.expander("ISO"):

        st.write(f"Number of anomalies detected by Isolation Forest: {anomaly_count}")

        plt.figure(figsize=(12, 6))
        plt.plot(iso_df.index, iso_df['Close'], label='Close (iso_df)', color='orange')
        plt.plot(iso_df[iso_df['Anomaly_iso'] == -1].index, iso_df[iso_df['Anomaly_iso'] == -1]['Close'], 'ro', label='Anomalies (iso_df)')

        plt.title(f'{stock_ticker} Close Price and Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)
    
    with col1.expander("SVM"):
        st.write(f"Number of anomalies detected by One-Class SVM: {svm_anomaly_count}")
        
        plt.figure(figsize=(12, 6))

        plt.plot(svm_df.index, svm_df['Close'], label='Close (svm_df)', color='orange')

        plt.plot(svm_df[svm_df['SVM_Anomaly_Pred'] == -1].index, svm_df[svm_df['SVM_Anomaly_Pred'] == -1]['Close'], 'ro', label='Anomalies (svm_df)')

        plt.title(f'{stock_ticker} Close Z-Score and Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.legend()
        plt.grid(True)
        plt.show()

        st.pyplot(plt)

    with col2.expander("DBSCAN"):
        st.write(f"Number of anomalies detected by DBSCAN: {anomaly_count}")


        plt.figure(figsize=(12, 6))
        # Plotting the clusters
        plt.plot(dbscan_df.index, dbscan_df['Close'], label='Close (dbscan_df)', color='orange')

        # Plot anomalies
        plt.scatter(dbscan_df[dbscan_df['Anomaly_Detected'] == False].index,
                    dbscan_df[dbscan_df['Anomaly_Detected'] == False]['Close'],
                    color='red', label='Anomalies', marker='o')

        plt.title('DBSCAN Anomaly Detection')
        plt.xlabel('Date')
        plt.ylabel('Close Z-Score')
        plt.legend()
        plt.grid(True)
        plt.show()

        st.pyplot(plt)

    with col2.expander("LSTM"):
        st.write(f"Number of anomalies detected by LSTM: {anomalies_lstm_len}")

        plt.figure(figsize=(14, 7))
        plt.plot(lstm_df.index[:len(lstm_actual)], lstm_actual, label='Actual Prices', color='blue')
        plt.plot(lstm_df.index[:len(lstm_predicted)], lstm_predicted, label='Predicted Prices', color='orange')
        plt.scatter(lstm_df.iloc[anomalies_lstm].index, lstm_actual[anomalies_lstm], color='red', label='Anomalies', zorder=5)
        plt.xlabel('Date')
        plt.ylabel('Stock Close')
        plt.title('Stock Price with LSTM Anomaly Detection')
        plt.legend()
        plt.show()

        st.pyplot(plt)

    with col2.expander("Auto Encoder"):
        st.write(f"Number of anomalies detected by Auto Encoder: {anomalies_auto_len}")

        plt.figure(figsize=(12, 6))
        plt.plot(autoenc_df.index, autoenc_df['Close'], label='Stock Close')
        plt.scatter(autoenc_df.iloc[anomalies_auto].index, autoenc_df.iloc[anomalies_auto]['Close'], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Autoencoder')
        plt.xlabel('Date')
        plt.ylabel('Stock Close')
        plt.legend()
        plt.show()

        st.pyplot(plt)

    st.divider()

    st.title("Model Evaluation")

    col3, col4 = st.columns(2)

    with st.spinner("Wait"):
 
        final_df = df.copy()

        final_df['Combined_Anomaly'] = (
        (abs(z_df['Close_Zscore']) > 3) |
        (iso_df['Anomaly_iso'] == -1) |
        (dbscan_df['Cluster'] == -1) |
        (np.isin(final_df.index, lstm_df.iloc[anomalies_lstm].index)) |
        (np.isin(final_df.index, autoenc_df.iloc[anomalies_auto].index)) |
        (svm_df['SVM_Anomaly_Pred'] == -1)).astype(int)


        final_df['Combined_Anomaly'].dropna(inplace = True)

        results = []

        methods = ['Z_Score', 'Isolation Forest', 'One Class SVM', 'DBSCAN']

        for method in methods:
            if method == 'Z_Score':
                predictions = (abs(z_df['Close_Zscore']) > 1.5).astype(int)

            elif method == 'Isolation Forest':
                predictions = (iso_df['Anomaly_iso'] == -1).astype(int)

            elif method == 'DBSCAN':
                predictions = (dbscan_df['Cluster'] == -1).astype(int)

            else:
                predictions = (svm_df['SVM_Anomaly_Pred'] == -1).astype(int)
                # print('Something Else')

            precision = precision_score(final_df['Combined_Anomaly'][:len(predictions)], predictions)
            recall = recall_score(final_df['Combined_Anomaly'][:len(predictions)], predictions)
            f1 = f1_score(final_df['Combined_Anomaly'][:len(predictions)], predictions)

            results.append({
                'Method': method,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })

        model_eva_df = pd.DataFrame(results)

        # -------

        lstm_predictions = [1 if x in anomalies_lstm else 0 for x in range(len(lstm_actual))]

        # Dimenstions are diff. SO..
        precision_lstm = precision_score(final_df['Combined_Anomaly'][:len(lstm_predictions)], lstm_predictions)
        recall_lstm = recall_score(final_df['Combined_Anomaly'][:len(lstm_predictions)], lstm_predictions)
        f1_lstm = f1_score(final_df['Combined_Anomaly'][:len(lstm_predictions)], lstm_predictions)

        results.append({
                'Method': 'LSTM',
                'Precision': precision_lstm,
                'Recall': recall_lstm,
                'F1-Score': f1_lstm
            })

        model_eva_df = pd.DataFrame(results)

        # ------

        autoencoder_predictions = [1 if i in anomalies_auto else 0 for i in range(len(autoenc_df))]

        anomaly_subset = final_df['Combined_Anomaly'][:len(autoencoder_predictions)]

        precision_auto = precision_score(anomaly_subset, autoencoder_predictions)
        recall_auto = recall_score(anomaly_subset, autoencoder_predictions)
        f1_auto = f1_score(anomaly_subset, autoencoder_predictions)

        results.append({
                'Method': 'AutoEncoder',
                'Precision': precision_auto,
                'Recall': recall_auto,
                'F1-Score': f1_auto
            })

        model_eva_df = pd.DataFrame(results)

    with col3.expander("Dataframe"):
        st.dataframe(model_eva_df)
        # Comparing all the precision, recall, f1 score using bar plot

    with col4.expander("Comparison"):
        plt.figure(figsize=(10, 6))
        bar_width = 0.2

        # Plotting Precision
        plt.bar(np.arange(len(model_eva_df['Method'])) - bar_width, model_eva_df['Precision'], width=bar_width, label='Precision')

        # Plotting Recall
        plt.bar(np.arange(len(model_eva_df['Method'])), model_eva_df['Recall'], width=bar_width, label='Recall')

        # Plotting F1-Score
        plt.bar(np.arange(len(model_eva_df['Method'])) + bar_width, model_eva_df['F1-Score'], width=bar_width, label='F1-Score')

        plt.xlabel('Anomaly Detection Method')
        plt.ylabel('Score')
        plt.title('Comparison of Precision, Recall, and F1-Score for Anomaly Detection Methods')
        plt.xticks(np.arange(len(model_eva_df['Method'])), model_eva_df['Method'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

        st.pyplot(plt)


    with st.expander("Plotly"):
        fig = make_subplots()

        # Plot the Close price
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))

        # Z-Score anomalies
        fig.add_trace(go.Scatter(
            x=z_df[z_df['Anomaly'] == 'YES'].index,
            y=z_df[z_df['Anomaly'] == 'YES']['Close'],
            mode='markers',
            name='Anomalies (Z-Score)',
            marker=dict(color='red', size=10, symbol='circle')
        ))

        # Isolation Forest anomalies
        fig.add_trace(go.Scatter(
            x=iso_df[iso_df['Anomaly_iso'] == -1].index,
            y=iso_df[iso_df['Anomaly_iso'] == -1]['Close'],
            mode='markers',
            name='Anomalies (Isolation Forest)',
            marker=dict(color='blue', size=10, symbol='x')
        ))

        # One-Class SVM anomalies
        fig.add_trace(go.Scatter(
            x=svm_df[svm_df['SVM_Anomaly_Pred'] == -1].index,
            y=svm_df[svm_df['SVM_Anomaly_Pred'] == -1]['Close'],
            mode='markers',
            name='Anomalies (One-Class SVM)',
            marker=dict(color='green', size=10, symbol='star')
        ))

        # AutoEncoder anomalies
        fig.add_trace(go.Scatter(
            x=autoenc_df.iloc[anomalies_auto].index,
            y=autoenc_df['Close'].iloc[anomalies_auto],
            mode='markers',
            name='Anomalies Autoencoder',
            marker=dict(color='purple', size=10, symbol='circle-open')
        ))

        # LSTM anomalies
        fig.add_trace(go.Scatter(
            x=lstm_df.iloc[anomalies_lstm].index,
            y=lstm_df['Close'].iloc[anomalies_lstm],
            mode='markers',
            name='Anomalies LSTM',
            marker=dict(color='#b6825a', size=10, symbol='diamond')
        ))

        # Get the minimum date and add one year to it
        # start_date = df.index.min() + pd.DateOffset(years=1)

        # Update layout
        fig.update_layout(
            title='Anomalies Detected by All Methods',
            xaxis_title='Date',
            yaxis_title='Close Price',
            xaxis=dict(range=[df.index.min(), df.index.max()]),
            showlegend=True,
            template='plotly'
        )

        st.plotly_chart(fig)



def visualizations(df, stock_ticker):

    st.title("Visualizations")

    # Calculations ------------
    df['Price Change'] = df['Close'].diff()

    # Calculate simple moving average (SMA) and exponential moving average (EMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # df = df[['Open', 'Close', 'Price Change']]
    for window in [7, 30, 90, 365]:  # 1 week, 1 month, 3 months, 1 year
        df[f'Volatility ({window} days)'] = df['Price Change'].rolling(window=window).std()


    df['RSI'] = calculate_rsi(df['Close'])

    df['Upper_Band'] = df['SMA_20'] + (df['SMA_20'].std() * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['SMA_20'].std() * 2)
    
    # MACD Calculation
    short_window = 12
    long_window = 26
    signal_window = 9

    df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()

    df['MACD'] = df['EMA_Short'] - df['EMA_Long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()


    # Plotting ------------
    col1, col2 = st.columns(2)

    with col1.expander("Basic"):
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'])
        plt.title(f'{stock_ticker} Closing Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.grid(True)
        plt.show()
        st.pyplot(plt)


        
        plt.figure(figsize=(12, 6))
        plt.plot(df['Price Change'])
        plt.title(f'{stock_ticker} Closing Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.grid(True)
        plt.show()

        st.pyplot(plt)

    with col1.expander("SMA EMA"):

        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Close')
        plt.plot(df['SMA_10'], label='SMA 10')
        plt.plot(df['SMA_20'], label='SMA 20')
        plt.title(f'{stock_ticker} Close, SMA 10, SMA 20 Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

        st.pyplot(plt)

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Close')
        plt.plot(df.index, df['EMA_10'], label='EMA 10')
        plt.plot(df.index, df['EMA_20'], label='EMA 20')
        plt.title(f'{stock_ticker} Close, EMA 10, EMA 20 Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

        st.pyplot(plt)

    with col2.expander("Volatility"):

        for column in df.columns:
            if 'Volatility' in column:
                plt.figure(figsize=(10, 5))
                plt.plot(df.index, df[column])
                plt.title(f'{stock_ticker} {column} Over Time')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.grid(True)
                st.pyplot(plt) 

    with col2.expander("Advanced"):
        # Create a new DataFrame for visualizations

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Close')
        plt.plot(df.index, df['Upper_Band'], label='Upper_Band')
        plt.plot(df.index, df['Lower_Band'], label='Lower_Band')
        plt.plot(df.index, df['SMA_20'], label='SMA 20')
        plt.title(f'{stock_ticker} Close, SMA 20, Upper Band, Lower Band Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)


        # Plotting MACD and Signal Line
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['MACD'], label='MACD')
        plt.plot(df.index, df['Signal_Line'], label='Signal Line')
        plt.title(f'{stock_ticker} MACD and Signal Line')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)

        # Histogram for MACD
        plt.figure(figsize=(10, 5))
        plt.hist(df['MACD'], bins=30)
        plt.title(f'{stock_ticker} MACD Histogram')
        plt.xlabel('MACD')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        st.pyplot(plt)

    # with st.expander("Final DataFrame"):
    #     st.dataframe(df.head())

    st.divider()

    
    model_selection(df, stock_ticker)
        
def ticker():
    st.title("Stock Price Viewer")

    # Predefined list of stock symbols
    symbols = [
        'NVDA', 'LCID', 'NIO', 'NOK', 'OKLO', 'TSLA', 'MARA', 'DNN', 'CLSK', 'CVS','AAPL', 'AAL', 'GRAB', 'BBD', 'AMZN', 'SMCI', 'SOFI', 'PLTR', 'PTON', 'INTC','T', 'DJT', 'BAC', 'IAG', 'F', 'RIOT', 'VALE', 'MSTR', 'GOLD', 'AMD','ERIC', 'IONQ', 'PFE', 'BABA', 'WFC', 'WBA', 'WULF', 'SLB', 'AG', 'JBLU','TSM', 'BTG', 'WOLF', 'RIVN', 'PSLV', 'HBAN', 'WBD', 'CCL', 'CSX', 'GOOGL','XPEV', 'CDE', 'IQ', 'NFLX', 'MU', 'JD', 'KGC', 'ALTM', 'NU', 'SNAP','LRCX', 'KMI', 'UEC', 'AVGO', 'BTE', 'BEKE', 'PTEN', 'XOM', 'HL', 'HAL','ABEV', 'CSCO', 'UBER', 'AGNC', 'COIN', 'MSFT', 'LUMN', 'ITUB', 'C','KVUE', 'RF', 'MPW', 'LYFT', 'RIG', 'ET', 'PDD', 'KEY', 'NGD', 'CMCSA','INFY', 'HPE', 'PBR', 'GOOG', 'VZ', 'APH', 'HOOD', 'UMC', 'USB', 'WMT','RKLB', 'OXY', 'BCS', 'CORZ', 'YMM', 'KO', 'AFRM', 'IOVA', 'CLF', 'GT','GGB', 'GERN', 'SBSW', 'ZIM', 'DAL', 'ALLY', 'NEE', 'LW', 'BILI', 'AXP','FITB', 'ASTS', 'PAAS', 'UAL', 'CELH', 'PR', 'PYPL', 'SCHW', 'KDP', 'LUV','MRK', 'QS', 'DKNG', 'ZETA', 'NEM', 'PG', 'PCG', 'PINXF', 'NXE', 'HST','COTY', 'DIS', 'TFC', 'FCX', 'ED', 'LI', 'U', 'JOBY', 'OWL', 'META','ACHC', 'TOST', 'MDT', 'UPST', 'DVN', 'FUTU', 'GM', 'MS', 'FHN', 'HIMS','NKE', 'RLX', 'MO', 'CCJ', 'UAA', 'BRK-B', 'EQX', 'NCLH', 'AMCR', 'BA','BMY', 'MRNA', 'AUR', 'HMY', 'ACI', 'VST', 'EQT', 'V', 'JPM', 'BSX','STLA', 'RDDT', 'SIRI', 'LBRT', 'SW', 'CNH', 'KSS', 'SBUX', 'DELL']


    col1, col2 = st.columns(2)
    # Dropdown for stock selection from predefined symbols
    selected_ticker = col1.selectbox("Select a stock ticker:", symbols)

    # Date input for start and end dates
    col1.write("Select date range for historical stock prices:")
    st.info("Please select atleast A Year Gap for Better Visualization and Understanding")
    
    # Set default start and end dates
    default_start_date = datetime.today().replace(day=1)  # Start of the current month
    default_end_date = datetime.today()  # Today's date

    start_date = col1.date_input("Start Date", default_start_date)
    end_date = col1.date_input("End Date", default_end_date)

    if st.button("Submit"):
    # Ensure that end_date is always after start_date
        if end_date < start_date:
            col2.error("End date must be after start date.")
        else:
            # Display selected ticker
            col2.write(f"You selected: **{selected_ticker}**")
            
            # Fetch and display additional information about the selected ticker
            stock = yf.Ticker(selected_ticker)
            info = stock.info
            
            # Display basic information
            col2.write(f"**Symbol:** {selected_ticker}")
            col2.write(f"**Name:** {info.get('shortName', 'N/A')}")
        
            # Fetch historical data for the specified date range
            historical_data = stock.history(start=start_date, end=end_date)

            with col2.expander("View"):
                st.write("Historical Stock Prices:")
                st.dataframe(historical_data.head())

            st.divider()

            visualizations(historical_data, selected_ticker)
