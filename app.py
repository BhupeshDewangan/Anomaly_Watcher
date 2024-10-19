import streamlit as st

from src.keyword_explain import *
from src.home_page import *
from src.ticker import *
from src.sidebar import *


st.set_page_config(
        page_title="🎥 Anomaly Watcher✨",
        page_icon="✨",
        layout="wide"
    )
# Streamlit application

def main():
    # Render the sidebar
    sidebar()

    
# Run the Streamlit app
if __name__ == "__main__":
    main()


