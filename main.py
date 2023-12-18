import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
#from openai import OpenAI

client = openai.OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
# 1. Player Identification and Demographics
player_identification_demographics = ['player', 'Position', 'Name', 'Team', 'League', 'Season', 'Age', 'Height', 
                                      'Weight', 'Birthdate', 'tabb', 'Minutes']

# 2. Playing Time and Performance Metrics
playing_time_performance_metrics = ['Minutes', 'avgawayplaytime', 'avghomeplaytime', 'homeoxg95min', 'awayoxg95min', 
                                    'homexga95', 'awayxga95', 'homexcs95', 'awayxcs95', 'xGFAR', 'xGAAR', 'xGDAR', 
                                    'goal', 'goalpershot', 'xgpershot', 'pctxgpass', 'pctxgrecv', 'smgpi', 'smspi', 
                                    'stxgpi']

# 3. Attacking Skills and Style Ratings
attacking_skills_style_ratings = ['Attack', 'Forward', 'Carry', 'Receive', 'Shoot', 'Set', 'Dribble', 'Attack_SC', 
                                  'Attack_BP', 'Attack_OP_SC', 'Attack_OP_BP', 'Attack_DB_SC', 'Attack_DB_BP', 
                                  'xG_Receive_BP', 'xG_Pass_BP', 'xG_Indv_BP', 'xG_Indv_Takeon_BP', 'xG_Cntr_SC', 
                                  'xG_Cross_SC', 'xG_Shots', 'xG_Open_SC', 'xG_Dead_SC', 'xG_Open_BP', 'xG_Dead_BP']

# 4. Defensive Skills and Style Ratings
defensive_skills_style_ratings = ['DefQual', 'DefQuant', 'BallRet', 'Disrupt', 'Recover', 'Aerial', 'Tackle', 
                                  'xG_Indv_Tackle_BP', 'xG_Indv_Inter_BP', 'xG_Indv_Aerial_BP']

# 5. Play Style and Specialties
play_style_specialties = ['Link', 'Pass1', 'Pass2', 'Pass3', 'Cross', 'Open_Foot', 'Open_Head', 'Dead_Foot', 
                          'Dead_Head', 'Direct', 'xG_Indv_Carry_BP', 'xG_Indv_Loose_BP']

# 6. Age-Related Traits and Trends
age_related_traits_trends = ['a_Hot', 'a_Cold', 'a_Breakout', 'a_Underused', 'a_Prospect']

# Load your data
# Load data
@st.cache_data  # This will cache the data and won't reload unless the file changes.
def get_players_data(df, analyze_features, league, position, season_min, season_max, age_min, age_max):
    df_pct = df.merge(df[analyze_features].rank(pct=True), left_index=True, right_index=True, suffixes=('', '_pct'))
    columns_ending_with_pct = df_pct.filter(regex='_pct$').columns

    league_df = df[df['League'] == league]

    position_df = league_df[league_df['Position'] == position]

    age_df = position_df[position_df['Age'].between(age_min, age_max, 'both')]
    season_df = age_df[age_df['Season'].between(season_min, season_max, 'both')]

    final_players = (season_df[analyze_features]
                    [season_df[analyze_features] > season_df[analyze_features]
                    .describe().loc['min']].dropna())

    merged_df = final_players.merge(season_df[player_identification_demographics], left_index=True, right_index=True)

    merged_pct = merged_df.merge(df_pct[columns_ending_with_pct], left_index=True, right_index=True)

    return merged_pct

def get_completion(prompt, model="gpt-4-1106-preview", temperature=0):
    messages = [{"role": "user", "content":prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature)
        #stream=True)
       
    #for chunk in response:
     #       if chunk.choices[0].delta.content is not None:
     #            st.write(chunk.choices[0].delta.content, end="")
    return response.choices[0].message.content

def analyze_with_chatgpt(data):
    # Create a prompt based on the current state
    prompt = f"Analyze the following data in the context of soccer: {data.to_json()}"
    return get_completion(prompt)

@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/ENG1_GPT2_clean.csv', index_col=0)
    return df

df = load_data()
#### Getting the data, make initial forcast and build a front end web-app with Taipy GUI
#features = attacking_skills_style_ratings

features = st.selectbox('Select Features', options=['Attacking Skills and Style Ratings', 'Defensive Skills and Style Ratings'])

if features == 'Attacking Skills and Style Ratings':
    features = attacking_skills_style_ratings   
elif features == 'Defensive Skills and Style Ratings':
    features = defensive_skills_style_ratings

league = st.selectbox('Select League', options=df['League'].unique())
position = st.selectbox('Select Position', options=df['Position'].unique())
season_min = st.slider('Select Season Min', min_value=2015, max_value=2025, value=2015)
season_max = st.slider('Select Season Max', min_value=2015, max_value=2025, value=2025)
age_min = st.slider('Select Age Min', min_value=15, max_value=40, value=15)
age_max = st.slider('Select Age Max', min_value=15, max_value=40, value=40)


data = get_players_data(df, features, league, position, season_min, season_max, age_min, age_max)

################# Plot the graph #####################

# Assuming 'merged_pct' is your DataFrame and it has columns for each category ending with '_pct'
categories = [col for col in data.columns if col.endswith('_pct')]
# Assuming the first category is at the top (12 o'clock position) and we go clockwise
categories = [*categories, categories[0]]

# Create a figure for the plot
fig = go.Figure()

# Add a trace for each player
for player in data['Name'].unique():
    player_data = data[data['Name'] == player]
    values = player_data[categories[:-1]].values.flatten().tolist()
    # Complete the loop for the radar chart
    values = [*values, values[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=player
    ))

# Layout settings
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]  # Assuming the values are normalized
        ),
    ),
    showlegend=True,
    title=dict(
        text='Spider Plot of Attacking Skills and Style Ratings',
        xanchor='center',
        y=0.9,
        x=0.5
    )
)

# Plot!
st.plotly_chart(fig, use_container_width=True)

import sys

if st.button('Analyze with chatgpt'):
    st.write(analyze_with_chatgpt(data))
    

