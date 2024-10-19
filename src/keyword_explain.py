import streamlit as st

# Keywords Explanation Page
def keywords_explanation_page():
    st.title("Keywords Explanation Page")

    # Yfinance Expander
    with st.expander("Yfinance Library"):
        st.write("The **Yfinance** library is a powerful tool in Python that allows users to easily access financial data from Yahoo Finance. "
                "It provides a simple interface for downloading historical market data, including stock prices, dividends, and splits. "
                "With Yfinance, you can retrieve data for individual stocks, indices, or even entire sectors, making it an essential resource for financial analysis and research.")
        
        st.write("Key features of Yfinance include:")
        
        st.write("• **Historical Data Retrieval:** Access daily, weekly, or monthly historical price data for various time frames.")
        st.write("• **Real-time Data:** Fetch current market prices and key statistics for stocks.")
        st.write("• **Financial Statements:** Obtain comprehensive financial data, including balance sheets, income statements, and cash flow statements.")
        st.write("• **Ticker Module:** Utilize the Ticker module to gather detailed information about specific companies, including news, analyst ratings, and market cap.")
        
        st.write("Yfinance is particularly popular among data analysts and developers due to its ease of use and integration with popular data science libraries like Pandas. "
                "By leveraging this library, users can automate the retrieval and analysis of financial data, enhancing their ability to make informed investment decisions.")

    # Terms used in this project
    st.header("Terms Used in This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Moving Averages (SMA, EMA)"):
            st.write(
                "Moving Averages are widely used indicators in financial markets to smooth out price data and identify trends over a specific period. "
                "The Simple Moving Average (SMA) calculates the average of a set of prices over a specified time frame. For example, a 20-day SMA takes the average of the closing prices of the last 20 days. "
                "Traders often use the SMA to spot trends; a rising SMA indicates an upward trend, while a declining SMA suggests a downward trend.\n\n"
                "On the other hand, the Exponential Moving Average (EMA) gives more weight to recent prices, making it more responsive to new information compared to the SMA. "
                "This characteristic allows traders to react more quickly to price changes. EMAs are often preferred for shorter time frames as they can help capture quicker price movements. "
                "For instance, the 9-day EMA is frequently used in day trading to identify short-term trends. Both SMA and EMA are commonly used in conjunction with other indicators to confirm signals or identify potential reversals. "
                "Understanding the differences between these moving averages helps traders make informed decisions based on market conditions."
            )

        with st.expander("Returns"):
            st.write(
            "Returns are a crucial concept in finance that measures the daily percentage change in closing prices of an asset, providing insights into its performance. "
                "Calculating returns allows investors to assess how much an investment has increased or decreased in value over time, helping to inform their decision-making. "
                "The formula for daily returns is simple: (Current Price - Previous Price) / Previous Price * 100. This calculation provides a percentage that indicates the asset's price movement relative to its previous closing price.\n\n"
                "Returns can be positive or negative, with positive returns indicating gains and negative returns signaling losses. Daily returns are often analyzed over longer periods to compute metrics such as average returns and cumulative returns. "
                "Moreover, analyzing the volatility of returns helps investors understand the risk associated with an asset. High volatility suggests greater uncertainty in price movements, which may influence investment strategies.\n\n"
                "In addition to daily returns, investors often look at other return metrics, such as annualized returns or risk-adjusted returns (like the Sharpe Ratio), to gauge the overall performance of their investments. Understanding returns is fundamental for any investor, as it directly impacts portfolio performance and risk assessment."
            )

        with st.expander("Volatility Indicator"):
            st.write(
                "The Volatility Indicator is a crucial metric in financial analysis that measures the degree of variation in trading prices over time, specifically focusing on the standard deviation of daily returns. "
                    "Volatility reflects the level of risk associated with an asset; higher volatility indicates a larger price range and greater uncertainty, which can either present opportunities or risks for investors.\n\n"
                    "Traders and investors use volatility indicators to assess market conditions. For example, a high volatility level might suggest that a stock is experiencing significant price swings, prompting traders to be cautious or to capitalize on potential price movements through strategies like options trading.\n\n"
                    "There are several methods to measure volatility, with standard deviation being one of the most common. Standard deviation quantifies how much prices deviate from the mean price over a specified period. Additionally, the Average True Range (ATR) is another popular indicator that measures market volatility by considering price gaps and absolute price movement.\n\n"
                    "Understanding volatility helps investors to make informed decisions about their portfolios, such as adjusting their asset allocations or using hedging strategies to mitigate risk. Moreover, traders often combine volatility indicators with other tools to develop comprehensive trading strategies that account for both risk and potential reward."
                )

    with col2:
        with st.expander("Relative Strength Index (RSI)"):
            st.write(
                "The Relative Strength Index (RSI) is a popular momentum oscillator used to measure the speed and change of price movements. "
                    "Developed by J. Welles Wilder, the RSI ranges from 0 to 100 and is typically plotted on a scale with 70 indicating an overbought condition and 30 indicating an oversold condition. "
                    "It is calculated using the average gains and losses over a specified period, usually 14 days, providing a ratio that reflects recent price performance.\n\n"
                    "Traders often use RSI to identify potential reversal points. For instance, when the RSI exceeds 70, it suggests that an asset may be overbought, indicating a potential price correction. Conversely, an RSI below 30 may signal that an asset is oversold, possibly indicating a buying opportunity. "
                    "However, RSI can remain in overbought or oversold territories for extended periods, making it essential for traders to combine it with other technical analysis tools.\n\n"
                    "Additionally, traders may look for divergence between RSI and price action. For example, if prices are making new highs but RSI is not, this divergence can indicate weakening momentum and a potential trend reversal. Understanding the RSI's implications enables traders to make informed decisions in their trading strategies."
                )

        with st.expander("Bollinger Bands"):
            st.write(
                "Bollinger Bands are a technical analysis tool that consists of a middle band (a Simple Moving Average) and two outer bands (standard deviations from the middle band). "
                    "Developed by John Bollinger, these bands help traders visualize volatility and identify potential price reversals in an asset.\n\n"
                    "The middle band typically represents a 20-day SMA, providing a baseline trend. The upper and lower bands are calculated based on the standard deviation, which adjusts to price volatility. "
                    "When prices approach the upper band, it indicates that the asset may be overbought, while touching the lower band suggests an oversold condition. "
                    "This characteristic makes Bollinger Bands effective for identifying potential trading opportunities.\n\n"
                    "Traders also use Bollinger Bands to gauge market volatility. When the bands contract, it indicates low volatility, often preceding significant price movements. Conversely, when the bands widen, it suggests increased volatility. "
                    "Combining Bollinger Bands with other indicators, such as RSI or MACD, allows traders to refine their strategies and enhance decision-making. Overall, Bollinger Bands are a versatile tool for analyzing price action and market behavior."
                )

        with st.expander("MACD (Moving Average Convergence Divergence)"):
            st.write(
                "MACD is a widely-used trend-following momentum indicator that illustrates the relationship between two moving averages of a security’s price. "
                    "It consists of three components: the MACD line, the signal line, and the histogram. The MACD line is calculated by subtracting the 26-day Exponential Moving Average (EMA) from the 12-day EMA, creating a line that fluctuates above and below zero.\n\n"
                    "The signal line, usually a 9-day EMA of the MACD line, is used to identify potential buy or sell signals. When the MACD line crosses above the signal line, it generates a bullish signal, suggesting it may be a good time to buy. Conversely, when the MACD line crosses below the signal line, it indicates a bearish signal, suggesting it may be a good time to sell.\n\n"
                    "The histogram represents the difference between the MACD line and the signal line. A growing histogram indicates increasing momentum in the direction of the MACD line, while a shrinking histogram signals a decrease in momentum. Traders often look for divergences between the MACD and price action, as these can provide insights into potential trend reversals. Overall, MACD is a powerful tool for traders to assess market momentum and make informed trading decisions."
                )


    # Model Explanations
    st.header("Model Explanations")

    # Anomaly Detection Techniques
    col3, col4 = st.columns(2)

    # with col3:
    #     with st.expander("Statistical Methods"):
    #         st.write("Statistical methods identify anomalies based on statistical properties of the data.")
    #         st.write("**Z-Score Method:** Flags any point where the price exceeds a certain number of standard deviations from the mean.")

    #     with st.expander("Machine Learning Approaches"):
    #         st.write("Machine learning approaches utilize algorithms to detect patterns and anomalies.")
    #         st.write("**Isolation Forest:** Detects outliers by isolating points that appear different from others in the dataset.")
    #         st.write("**One-Class SVM:** Learns a decision boundary to separate normal from anomalous data.")
    #         st.write("**K-means Clustering:** Identifies data points that don't fit well into clusters.")

    # with col4:
    #     with st.expander("Deep Learning Methods"):
    #         st.write("Deep learning methods leverage neural networks for complex pattern recognition.")
    #         st.write("**LSTM (Long Short-Term Memory):** Good for time-series data, it detects sequences with unusually high error as anomalies.")
    #         st.write("**Autoencoders:** Train on regular patterns to compress and reconstruct data. A high reconstruction error suggests an anomaly.")
    
    
    with col3:
        with st.expander("Statistical Methods"):
            st.write("Statistical methods identify anomalies based on statistical properties of the data.")
            st.write("**Z-Score Method:** Flags any point where the price exceeds a certain number of standard deviations from the mean.")
            st.write("""
            Statistical methods are widely used for anomaly detection due to their simplicity and efficiency. The Z-Score method, in particular, is effective in identifying outliers in normally distributed data. By calculating the mean and standard deviation of the dataset, the Z-Score method determines how many standard deviations away from the mean each data point lies. This allows for the identification of data points that are significantly different from the rest of the data.

            The Z-Score method is often used in financial applications, such as detecting fraudulent transactions or identifying unusual stock price movements. It is also used in quality control to detect defects in manufacturing processes. The method's simplicity and speed make it an attractive choice for real-time anomaly detection.

            In addition to the Z-Score method, other statistical methods for anomaly detection include the Modified Z-Score method, which is more robust to outliers, and the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) method, which can handle clusters of varying densities.
            """)


        with st.expander("Machine Learning Approaches"):
            st.write("Machine learning approaches utilize algorithms to detect patterns and anomalies.")
            st.write("**Isolation Forest:** Detects outliers by isolating points that appear different from others in the dataset.")
            st.write("**One-Class SVM:** Learns a decision boundary to separate normal from anomalous data.")
            st.write("**K-means Clustering:** Identifies data points that don't fit well into clusters.")
            st.write("""
            Machine learning approaches to anomaly detection have gained popularity due to their ability to handle complex and high-dimensional data. Isolation Forest, for instance, is an ensemble method that combines multiple decision trees to identify outliers. By randomly selecting features and splitting data into subsets, Isolation Forest can effectively identify points that are anomalous.

            One-Class SVM is another popular machine learning approach that learns a decision boundary to separate normal data from anomalies. This method is particularly effective in cases where the normal data class is well-defined, but the anomalous class is unknown or diverse.

            K-means Clustering is also widely used for anomaly detection, particularly in cases where the data has a clustering structure. By identifying data points that don't fit well into clusters, K-means can detect anomalies that may not be apparent through other methods.
            """)


    with col4:
        with st.expander("Deep Learning Methods"):
            st.write("Deep learning methods leverage neural networks for complex pattern recognition.")
            st.write("**LSTM (Long Short-Term Memory):** Good for time-series data, it detects sequences with unusually high error as anomalies.")
            st.write("**Autoencoders:** Train on regular patterns to compress and reconstruct data. A high reconstruction error suggests an anomaly.")
            st.write("""
            Deep learning methods have revolutionized anomaly detection in recent years, particularly in applications involving complex patterns and relationships. LSTMs, for example, are well-suited for time-series data, where they can detect sequences with unusually high error as anomalies.

            Autoencoders are another powerful deep learning tool for anomaly detection. By training on regular patterns, autoencoders learn to compress and reconstruct data effectively. When confronted with anomalous data, autoencoders exhibit high reconstruction error, indicating the presence of an anomaly.

            Deep learning methods can also be combined with other approaches to improve anomaly detection performance. For instance, hybrid models combining LSTM and autoencoder architectures have shown promising results in detecting complex anomalies.
            """)