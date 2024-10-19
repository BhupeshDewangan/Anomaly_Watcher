import numpy as np


def calculate_rsi(data, window=14):
  delta = data.diff()
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)

  avg_gain = gain.rolling(window=window, center=False).mean()
  avg_loss = loss.rolling(window=window, center=False).mean()

  rs = avg_gain / avg_loss
  rsi = 100 - (100 / (1 + rs))

  return rsi

# Create sequences
def create_sequences(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


# Create sequences for training (e.g., using time steps of 10)
def create_sequences_for_AutoEnc(data, time_step=1):
    sequences = []
    for i in range(len(data) - time_step):
        sequences.append(data[i:(i + time_step), 0])
    return np.array(sequences)


def create_plot():
    fig = make_subplots()

    # Plot the Close price
    fig.add_trace(go.Scatter(x=new_df.index, y=new_df['Close'], mode='lines', name='Close Price'))

    # Z-Score anomalies
    fig.add_trace(go.Scatter(
        x=new_df[new_df['Anomaly'] == 'YES'].index,
        y=new_df[new_df['Anomaly'] == 'YES']['Close'],
        mode='markers',
        name='Anomalies (Z-Score)',
        marker=dict(color='red', size=10, symbol='circle')
    ))

    # Isolation Forest anomalies
    fig.add_trace(go.Scatter(
        x=iso_df[iso_df['Anomaly_Pred'] == -1].index,
        y=iso_df[iso_df['Anomaly_Pred'] == -1]['Close'],
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
        x=autoenc_df.iloc[anomalies].index,
        y=autoenc_df['Close'].iloc[anomalies],
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
    start_date = new_df.index.min() + pd.DateOffset(years=1)

    # Update layout
    fig.update_layout(
        title='Anomalies Detected by All Methods',
        xaxis_title='Date',
        yaxis_title='Close Price',
        xaxis=dict(range=[start_date, new_df.index.max()]),
        showlegend=True,
        template='plotly'
    )

    return fig
