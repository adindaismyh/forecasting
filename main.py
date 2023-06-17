import streamlit as st
from streamlit_option_menu import option_menu
from minyak import minyak
from beras import beras
from gula import gula
from cabai import cabai
from daging import daging

st.set_page_config(layout='wide', initial_sidebar_state='expanded',page_title="Forecasting", page_icon="🧊")
with open("style.css") as gaya:
        st.markdown(f'<style>{gaya.read()}</style>', unsafe_allow_html=True)
with st.sidebar:
    st.sidebar.header('Main `DASBOARD`')
    selected = option_menu(
        menu_title = "Komoditi Sembako",
        options = ["Minyak Goreng", "Beras", "Gula Pasir", "Cabai Rawit", "Daging Sapi"],
        icons =["reception-4", "reception-4", "reception-4", "reception-4", "reception-4"],
        menu_icon = "cast",
        default_index = 0,
    )

if selected == "Minyak Goreng":
    minyak()
if selected == "Beras":
    beras()
if selected == "Gula Pasir":
    gula()
if selected == "Cabai Rawit":
    cabai()
if selected == "Daging Sapi":
    daging()
