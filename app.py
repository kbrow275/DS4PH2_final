import streamlit as st
import folium
from streamlit_folium import st_folium  
import pandas as pd
import matplotlib.pyplot as plt

st.title("Olympic Medal Prediction") 
tab_titles = ["1", "2", "3"]
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.header("Olympic Medal Prediction by Country")
    st.subheader("Enter the country name:")
    st.text_input("Enter the Country Name")


with tabs[1]:
    st.header("Olympic Medal Prediction by Country and Event")
    st.subheader("Enter the Event and country name:")
    st.text_input("Enter the Event")
    st.text_input("Enter Country name")



