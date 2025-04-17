import streamlit as st
import folium
from streamlit_folium import st_folium  
import pandas as pd
import matplotlib.pyplot as plt

from predictor import predict_olympic_medals_detailed

df = pd.read_csv("features.csv")
country_list = df["country_name"].unique().tolist()

st.title("Olympic Medal Prediction") 
st.write("This app predicts the number of medals a country will win in the next Olympics based on historical data.")
st.write("Please enter the country name to get the prediction.")
selected_country = st.selectbox("Select a country:", country_list)

if selected_country:
    try:
        predictions = predict_olympic_medals_detailed(selected_country)

        st.subheader(f"Predicted Medals for {predictions['country']} in next Olympics")
        st.markdown("### Summer Olympics")
        st.write(f"🥇 Individual Medals: {predictions['summer']['individual_medals']}")
        st.write(f"🥈 Doubles Medals: {predictions['summer']['doubles_medals']}")
        st.write(f"🥉 Team Medals: {predictions['summer']['team_medals']}")
        st.write(f"🏆 Total Summer Medals: {predictions['summer']['total_medals']}")

        st.markdown("### Winter Olympics")
        st.write(f"❄️ Individual Medals: {predictions['winter']['individual_medals']}")
        st.write(f"⛷️ Doubles Medals: {predictions['winter']['doubles_medals']}")
        st.write(f"🏒 Team Medals: {predictions['winter']['team_medals']}")
        st.write(f"🏆 Total Winter Medals: {predictions['winter']['total_medals']}")

    except ValueError as e:
        st.error(str(e))