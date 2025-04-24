import streamlit as st
import folium
from streamlit_folium import st_folium  
import pandas as pd
import matplotlib.pyplot as plt


from predictor import predict_olympic_medals, predict_olympic_medals_detailed, predict_event_podium
df = pd.read_csv("features.csv")
year_data = pd.read_csv("data_clean.csv")
results = pd.read_csv("olympic_results.csv")
country_list = df["country_name"].unique().tolist()

st.title("Olympic Medal Prediction App 🏅")

tab1, tab2, tab3 = st.tabs(["Overall Prediction", "Event Podium Prediction", "Country Medal History"])



with tab1:
    country_list = df["country_name"].unique().tolist()
    st.title("Detailed Medal Breakdown") 
    st.write("See predicted counts of individual, doubles, and team medals for the next Olympic games.")
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
            
       
with tab2:
    st.title("Podium Prediction by Event")
    st.write("Select a sport discipline and event to see the predicted podium finish for next olympics.")

    all_disciplines = sorted(results['discipline_title'].dropna().unique())
    discipline = st.selectbox("Select a discipline:", all_disciplines)

    if discipline:
        event_subset = results[results['discipline_title'] == discipline]['event_title'].dropna().unique()
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

with tab3:
    st.title("Country Medal History")
    st.write("Select a country to see its historical medal counts in the Summer and Winter Olympics.")
    selected_country = st.selectbox("Select a country:", country_list, key = "country_history")

    
