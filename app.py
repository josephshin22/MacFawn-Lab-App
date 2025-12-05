import streamlit as st
from streamlit_option_menu import option_menu
import home, graphs, about

# Set page configuration
st.set_page_config(
    page_title="MacFawn Lab",
    page_icon="🧬",
    layout="wide",
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Graphs", "About"],
    icons=["house", "bar-chart", "info-circle"],
    orientation="horizontal",
)

if selected == "Home":
    home.show()
elif selected == "Graphs":
    graphs.show()
elif selected == "About":
    about.show()

