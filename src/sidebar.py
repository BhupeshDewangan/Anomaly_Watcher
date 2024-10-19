import streamlit as st
from streamlit_option_menu import option_menu
from src.home_page import *
from src.keyword_explain import *
from src.ticker import *
from src.about_page import *

# Sidebar
# def siderbar():
#     # Sidebar
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Select a page:", ["Home", "Keywords", "Models", "About"])

#     # Home Page
#     if page == "Home":
#         home_page()

#     elif page == "Keywords":
#         keywords_explanation_page()

#     # Models Page
#     elif page == "Models":
#         ticker()

#     # About Page
#     elif page == "About":
#         about_page()

def sidebar():

    with st.sidebar:
        page = option_menu("Main Menu", ["Home", 'Keywords', 'Models', 'About'], 
            icons=['house-check', 'search-heart-fill', 'file-person-fill', 'gear'], menu_icon="list", default_index=0,
            styles={
                # "icon": {"color": "white"}, 
                # "container": {"padding": "0!important", "background-color": "#fafafa"},
                # "nav-link": {"text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#ff4b4b"},
            })
        
    # Home Page
    if page == "Home":
        home_page()

    elif page == "Keywords":
        keywords_explanation_page()

    # Models Page
    elif page == "Models":
        ticker()

    # About Page
    elif page == "About":
        about_page()