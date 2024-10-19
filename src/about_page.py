import streamlit as st

# About Page
def about_page():
    st.title("About Page")

    # Mission
    st.header("Mission")
    st.write("""
    Our mission is to empower investors and traders by providing actionable insights into stock price movements. 
    By leveraging advanced analytics and machine learning techniques, we strive to help users make informed decisions 
    based on real-time data and historical analysis.
    """)

    # Development Tech and Tools
    st.header("Development Tech and Tools")
    st.write("""
    - **Python, Streamlit, Jupyter**: All the code for Analysis, Model Building are written in Python, Streamlit For building the web application, Jypter for Step by Step Process.
    - **Libraries**: Such as Scikit-learn and TensorFlow for implementing anomaly detection models, Financial Data APIsTo gather real-time stock price data for analysis, Pandas and NumPy For data manipulation and analysis, Matplotlib, Plotly for Visualization.
    - **Techniques for Anomaly Detection** : Statistical Method like Z Score, Machine Learning - Isolation Forest, DBSCAN, One Class SVM, Deep Learning - LSTM, Auto Encoders.
    """)

    # Contact Information
    st.header("Contact Me")
    st.write("""
    I welcome your feedback and suggestions. If you have any questions or encounter any issues while using the Anomaly Detection in Stock Prices application, please feel free to reach out:
    - **Email**: bhupeshdewangan2003@gmail.com
    - **Phone**: 8319341550
    - **LinkedIn**: [Bhupesh Dewangan](https://www.linkedin.com/in/bhupesh-dewangan-7121851ba/)
    - **GitHub**: [BhupeshDewangan](https://github.com/BhupeshDewangan)
    """)

    st.success("""
    Thank you for exploring Anomaly Detection in Stock Prices. Your input helps us improve the application and provide a better experience for all users!
    """)
