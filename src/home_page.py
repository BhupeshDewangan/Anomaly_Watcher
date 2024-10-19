import streamlit as st

# Home Page
def home_page():
    st.title("Anomaly Detection in Stock Prices")
    
    st.subheader("Overview of the Project")
    st.write(
        "This project focuses on detecting anomalies in stock prices using various machine learning models. "
        "We aim to identify unusual price movements that may indicate potential market opportunities or risks."
    )

    col1, col2 = st.columns(2)

    col1.subheader("Introduction to Anomaly Detection")
    col1.write(
        "Anomaly detection is a crucial aspect of data analysis, particularly in the context of financial markets. "
        "It involves identifying outliers or unexpected patterns in time series data, such as stock prices. "
        "These anomalies can signify significant market events, like sudden price drops or spikes, which may indicate underlying issues or opportunities. "
        "By employing various statistical and machine learning techniques, anomaly detection helps in uncovering these irregularities, allowing traders and investors to respond promptly. "
        "For instance, detecting an unusual increase in trading volume may signal potential insider trading or news events affecting a stock's price. "
        "Moreover, effective anomaly detection can enhance risk management by alerting users to potential losses or market shifts, enabling informed decision-making. "
        "As financial markets grow increasingly complex, the ability to swiftly identify and analyze anomalies becomes vital for maintaining a competitive edge."
    )


    col2.subheader("Types of Anomalies")

    # Type 1: Point Anomalies
    col2.write("1. Point Anomalies --- "
        "These are individual data points that significantly deviate from the dataset's overall pattern. "
        "In stock prices, this could appear as a sudden spike or drop, indicating potential market manipulation or significant news events. "
        "Identifying these anomalies helps traders spot immediate risks or opportunities."
    )

    # Type 2: Contextual Anomalies
    col2.write("2. Contextual Anomalies --- "
        "These are normal in some contexts but abnormal in others. "
        "For example, a stock's price surge during earnings season may be expected, but a similar surge outside this context might signal unusual activity. "
        "Recognizing these anomalies is crucial for accurate analysis."
    )

    # Type 3: Collective Anomalies
    col2.write("3. Collective Anomalies --- "
        "These occur when a group of data points shows an abnormal pattern, even if individual points seem normal. "
        "For instance, a sudden price movement across several stocks in the same sector can indicate a market-wide event. "
        "Detecting these patterns aids in understanding broader market dynamics."
    )

    st.divider()

    col3, col4 = st.columns(2)

    col3.subheader("Importance in Stock Price Analysis")
    col3.write(
        "1. **Risk Management:** By identifying anomalies, investors can manage risks better and make timely decisions.\n"
        "2. **Market Trends:** Anomalies can reveal shifts in market trends that might not be immediately apparent.\n"
        "3. **Fraud Detection:** Spotting unusual patterns can help in detecting fraudulent activities.\n"
        "4. **Investment Opportunities:** Identifying anomalies can uncover undervalued or overvalued stocks."
    )

    col4.subheader("Future Work")
    col4.write(
        "1. **Integration with Real-Time Data:** Developing the capability to process and analyze stock price data in real-time to enable immediate anomaly detection and alerting for traders.\n"
        "2. **Exploration of Advanced Models:** Investigating and implementing more complex models such as Autoencoders or Variational Autoencoders (VAEs) for improved anomaly detection performance.\n"
        "3. **User Interface Enhancements:** Creating a more interactive user interface that allows users to customize parameters, visualize results more dynamically, and compare different models more effectively.\n"
        "4. **Backtesting Strategies:** Implementing backtesting capabilities to evaluate the effectiveness of the anomaly detection models in historical trading strategies and their impact on portfolio performance.\n"
    )

