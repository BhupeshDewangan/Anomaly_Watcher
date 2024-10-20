
# Anomaly Watcher - Anomaly Detection in Stock Prices
[Live Preview 🔗](https://anomaly-watcher-stocks.streamlit.app/)

This project focuses on detecting anomalies in stock prices using various machine learning models. and aim to identify unusual price movements that may indicate potential market opportunities or risks.

## Features

* Data retrieval using yfinance
* Comprehensive Exploratory Data Analysis (EDA)
* Implementation of multiple anomaly detection techniques:
    * Z-Score
    * One Class SVM
    * Isolation Forest
    * DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    * LSTM (Long Short-Term Memory) 
    * Autoencoder
* Performance comparison of different methods
* Interactive Streamlit app for result visualization


## Usage

### Jupyter Notebook
To run the Jupyter Notebook for detailed analysis, execute the following command in your terminal:

```bash
jupyter notebook Stock_Anomaly_Detection.ipynb
``` 
## Streamlit App
To launch the Streamlit app, run:

```bash
streamlit run app.py
```

## Project Structure
- **app.py**: Streamlit app for interactive visualization.
- **requirements.txt**: List of required Python packages.
- **jupyter_file/**: Directory containing the main Jupyter Notebook with the analysis.
- **src/**: Directory for all other Python files.


## Methodology

### 1. Data Collection
- **Retrieve stock data with option to start and end date **: Utilize the `yfinance` library to collect historical stock data for Stock Data, also its infomation like Stock Symbol and Name of the Company.

### 2. Preprocessing
- **Clean data**: Remove any irrelevant or corrupted entries.
- **Handle missing values**: Fill or interpolate missing data points as necessary.
- **Calculate additional features**:
  - Returns
  - Volatility
  - SMA / EMA
  - RSI
  - MACD
  - Upper Band abd Lower Band

#### All the Terms Are Explained in The - [Website 🔗](https://anomaly-watcher-stocks.streamlit.app/) !!!

### 3. Exploratory Data Analysis
- **Visualize stock price trends**: Plot historical stock prices to identify overall trends.
- **Analyze volume**: Examine trading volume alongside price changes.
- **Visualize returns and volatility**: Assess the relationship between returns and volatility over time.
- **Simple Moving Average (SMA) / Exponential Moving Average (EMA)**: Calculate and plot moving averages to identify trends and potential reversal points.
- **Relative Strength Index (RSI)**: Use RSI to measure the speed and change of price movements, indicating overbought or oversold conditions.
- **Moving Average Convergence Divergence (MACD)**: Analyze the MACD for trend-following and momentum signals.
- **Upper Band and Lower Band**: Utilize Bollinger Bands to assess volatility and potential price breakout points.


### 4. Anomaly Detection Methods
- **Z-Score**: 
  - Identify outliers based on standard deviations from the mean.
- **Isolation Forest**: 
  - Detect anomalies using isolation in the feature space.
- **One-Class SVM**: 
  - Model the normal data distribution to identify deviations as anomalies.
- **DBSCAN**: 
  - Cluster data points and identify outliers.
- **LSTM (Long Short-Term Memory)**: 
  - Predict stock prices and flag significant deviations as anomalies.
- **Autoencoder**: 
  - Learn normal patterns and detect anomalies based on reconstruction error.

### 5. Model Comparison
- **Evaluate performance**: Compare the effectiveness of each method using:
  - Precision
  - Recall
  - F1-score

### 6. Visualization
- **Create interactive plots**: Display detected anomalies and compare results across different methods.
- **Dynamic visualizations**: Allow users to explore different time frames and detection methods interactively.
- **Highlight anomalies**: Clearly mark identified anomalies on the plots for easy identification.
- **Comparative charts**: Provide side-by-side comparisons of the results from various anomaly detection techniques.


## Results

The project provides insights into:

- **Periods of unusual activity in stocks**: Identification of significant fluctuations in stock prices.
  
- **Effectiveness of different anomaly detection techniques**: Analysis of how various methods perform on stock market data.
  
- **Comparative analysis of model performances**: Evaluation of precision, recall, and F1-score for each anomaly detection method.

#### Detailed results and visualizations are available in the [Website 🔗](https://anomaly-watcher-stocks.streamlit.app/) !!!


## Streamlit App Features

The Streamlit app offers an interactive interface for exploring the anomaly detection results:

- **Multipage Interface**: Includes pages for Home, Keywords Explanation, Models, and About the Developer.

- **Stock Data Input and Date Range Selection**: Users can upload their own stock data and specify the date range for analysis.

- **Interactive EDA Visualizations**: Visualize exploratory data analysis findings, including:
  - Price trends
  - Preprocessing metrics like Returns
  - Volatility
  - Simple Moving Average (SMA) / Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Upper Band and Lower Band

- **Individual Plots for Each Anomaly Detection Method**: View results from different techniques separately for detailed analysis.

- **Performance Metrics Comparison**: Display precision, recall, and F1-score for each anomaly detection technique side by side.

- **User Interactivity**: Utilize Streamlit features such as Expander, Buttons, and Option Menu for a more engaging user experience.

- **Dynamic Plotting**: Leverage Matplotlib’s `pyplot` to create responsive and interactive visualizations.


## Future Work

- **Integration with Real-Time Data**: 
  - Develop the capability to process and analyze stock price data in real-time, enabling immediate anomaly detection and alerting for traders.

- **Exploration of Advanced Models**: 
  - Investigate and implement more complex models, such as Autoencoders or Variational Autoencoders (VAEs), to enhance anomaly detection performance.

- **User Interface Enhancements**: 
  - Create a more interactive user interface that allows users to customize parameters, visualize results dynamically, and compare different models more effectively.

- **Backtesting Strategies**: 
  - Implement backtesting capabilities to evaluate the effectiveness of anomaly detection models in historical trading strategies and assess their impact on portfolio performance.


## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

### Contributors
- Shubh Shrishrimal [Github](https://github.com/shubh123a3) - [LinkedIn Profile](https://www.linkedin.com/in/shubh-shrishrimal-a02636225/)


## Acknowledgments

- **yfinance**: For providing easy access to Yahoo Finance data.
- **Streamlit**: For enabling interactive data visualization.
- **The open-source community**: For the various machine learning libraries used in this project.

## Contact

For any queries or discussions related to this project, please open an issue in the GitHub repository.

- **Email**: [bhupeshdewangan2003@gmail.com](mailto:bhupeshdewangan2003@gmail.com)
- **Phone**: 8319341550
- **LinkedIn**: [Bhupesh Dewangan](https://www.linkedin.com/in/bhupesh-dewangan-7121851ba/)
- **GitHub**: [BhupeshDewangan](https://github.com/BhupeshDewangan)
