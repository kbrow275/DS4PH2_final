import streamlit as st
import folium
from streamlit_folium import st_folium  
import pandas as pd
import matplotlib.pyplot as plt

from predictor import predict_olympic_medals, predict_olympic_medals_detailed
df = pd.read_csv("features.csv")
country_list = df["country_name"].unique().tolist()

tab1, tab2, tab3 = st.tabs(["Overall Prediction", "Detailed Medal Breakdown", "Event Podium Prediction"])

with tab1:
    st.title("Olympic Medal Prediction") 
    st.write("This app predicts the number of medals a country will win in the next Olympics based on historical data.")
    selected_country = st.selectbox("Select a country:", country_list, key = "overall_country")

    if selected_country:
        try:
            predictions = predict_olympic_medals(selected_country)

            st.subheader(f"Predicted Medals for {predictions['country']} in the Next Olympics")
            st.write(f"🌊 **Summer Olympic Medals:** {predictions['next_summer_medals_pred']}")
            st.write(f"❄️ **Winter Olympic Medals:** {predictions['next_winter_medals_pred']}")
            st.write(f"**Probability of being in Top 10 (Summer):** {predictions['summer_top10_prob']}")
            st.write(f"**Probability of being in Top 10 (Winter):** {predictions['winter_top10_prob']}")

        except ValueError as e:
            st.error(str(e))


with tab2:
    country_list = df["country_name"].unique().tolist()
    st.title("Detailed Medal Breakdown") 
    st.write("See predicted counts of individual, doubles, and team medals by Olympic season.")
    selected_country = st.selectbox("Select a country:", country_list, key = "detailed_country")

    if selected_country:
        try:
            predictions = predict_olympic_medals_detailed(selected_country)
    
            st.subheader(f"Detailed Medal Breakdown for {predictions['country']}")

            st.markdown("### Summer Olympics")
            st.write(f"🏄 **Individual Medals:** {predictions['summer']['individual_medals']}")
            st.write(f"🧑‍🤝‍🧑 **Doubles Medals:** {predictions['summer']['doubles_medals']}")
            st.write(f"🏅 **Team Medals:** {predictions['summer']['team_medals']}")
            st.write(f"🏆 **Total Summer Medals:** {predictions['summer']['total_medals']}")

            st.markdown("### Winter Olympics")
            st.write(f"🏂 **Individual Medals:** {predictions['winter']['individual_medals']}")
            st.write(f"🧑‍🤝‍🧑 **Doubles Medals:** {predictions['winter']['doubles_medals']}")
            st.write(f"🏅 **Team Medals:** {predictions['winter']['team_medals']}")
            st.write(f"🏆 **Total Winter Medals:** {predictions['winter']['total_medals']}")

        except ValueError as e:
            st.error(str(e))
            
       
with tab3:
    st.title("Podium Prediction by Event")
    st.write("Select a sport discipline and event to see the predicted podium finish.")

    all_disciplines = sorted(df['discipline_title'].dropna().unique())
    discipline = st.selectbox("Select a discipline:", all_disciplines)

    if discipline:
        event_subset = df[df['discipline_title'] == discipline]['event_title'].dropna().unique()
        event_title = st.selectbox("Select an event:", sorted(event_subset))

        if event_title:
            try:
                podium = predict_event_podium(discipline, event_title)
                st.subheader(f"Predicted Podium for {discipline} - {event_title}")
                for medal, info in podium['podium'].items():
                    st.markdown(f"**{medal.capitalize()}**: {info['country']}  ")
                    st.write(f"- Probability: {info['probability']}")
                    st.write(f"- Historical Medals: {info['historical_golds']} Gold, {info['historical_silvers']} Silver, {info['historical_bronzes']} Bronze")
                    st.write(f"- Total Appearances: {info['total_appearances']}")

            except ValueError as e:
                st.error(str(e))
