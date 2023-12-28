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

########## new filters ##########
model_ratings = {
    'Attack': ['Attack', 'Forward', 'Carry', 'Receive', 'Shoot', 'Set', 'Dribble', 
               'Attack_SC', 'Attack_BP', 'Attack_OP_SC', 'Attack_OP_BP', 'Attack_DB_SC', 
               'Attack_DB_BP'],
    'Defending Quality': ['DefQual', 'Tackle', 'xG_Indv_Tackle_BP', 'xG_Indv_Inter_BP'],
    'Defending Quantity': ['DefQuant', 'Aerial', 'xG_Indv_Aerial_BP'],
    'Ball Retention': ['BallRet', 'Pass1', 'Pass2', 'Pass3', 'Link']
}

skill_ratings = {
    'Aerial Duels in Open Play': ['Aerial', 'xG_Open_SC', 'xG_Indv_Aerial_BP'],
    'Aerial Duels from Dead Balls': ['Dead_Head', 'xG_Dead_SC'],
    'Ground Duels in Possession': ['Carry', 'xG_Indv_Carry_BP'],
    'Ground Duels Out of Possession': ['Tackle', 'xG_Indv_Tackle_BP', 'xG_Indv_Inter_BP']
}

style_ratings = {
    'Disrupting Opposition Moves': ['Disrupt', 'xG_Indv_Tackle_BP', 'xG_Indv_Inter_BP'],
    'Recovering a Moving Ball': ['Recover', 'xG_Indv_Loose_BP'],
    'Aerial Duels': ['Aerial', 'xG_Indv_Aerial_BP'],
    'Link-up Passing': ['Link', 'xG_Pass_BP'],
    'Passing toward Goal': ['Pass1', 'Pass2', 'Pass3', 'Direct'],
    'Dribbling': ['Dribble'],
    'Receiving in the Box': ['Receive', 'xG_Receive_BP'],
    'Shooting': ['Shoot', 'xG_Shots']
}

st.sidebar.title('Select the type of comparison')
within_league = st.sidebar.checkbox('Compare only within the league', value=False)
within_pos = st.sidebar.checkbox('Compare only within the position', value=False)
within_age = st.sidebar.checkbox('Compare only within the age', value=False)
within_season = st.sidebar.checkbox('Compare only within the season', value=False)

# Load your data
# Load data
@st.cache_data  # This will cache the data and won't reload unless the file changes.
def get_players_data(df, league, position, season_min, season_max, age_min, age_max,
                     model_ratings_features, model_ratings_values, 
                     model_skill_ratings_features, model_skill_ratings_values, 
                     model_style_ratings_features, model_style_ratings_values,
                     within_league, within_pos, within_age, within_season):
    
    # Decide when to filter
    if within_league:
        df = df[df['League'].isin(league)]
    if within_pos:
        df = df[df['Position'] == position]
    if within_age:
        df = df[df['Age'].between(age_min, age_max, 'both')]
    if within_season:
        df = df[df['Season'].between(season_min, season_max, 'both')]

    # After filtering, convert to percentiles
    df = convert_to_percentiles(df, model_ratings_features)
    df = convert_to_percentiles(df, model_skill_ratings_features)
    df = convert_to_percentiles(df, model_style_ratings_features)

    # Apply filters after converting to percentiles if the corresponding 'within_' flags are False
    if not within_league:
        df = df[df['League'].isin(league)]
    if not within_pos:
        df = df[df['Position'] == position]
    if not within_age:
        df = df[df['Age'].between(age_min, age_max, 'both')]
    if not within_season:
        df = df[df['Season'].between(season_min, season_max, 'both')]

    model_rating_df = model_rating_filter(df, model_ratings_features, model_ratings_values)

    skill_rating_df = model_skill_ratings(model_rating_df, model_skill_ratings_features, model_skill_ratings_values)

    style_rating_df = model_style_ratings(skill_rating_df, model_style_ratings_features, model_style_ratings_values)


    return style_rating_df


def convert_to_percentiles(df, features_list):
    for features in features_list:
        for feature in features:
            df[feature] = df[feature].rank(pct=True)
    return df


def get_completion(prompt, model="gpt-4-1106-preview", temperature=0):
    messages = [{"role": "user", "content":prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature)

    return response.choices[0].message.content

def analyze_with_chatgpt(data):
    # Create a prompt based on the current state
    prompt = f"Analyze the following data in the context of soccer, provide a description of the players and analyze their strengths and weakness,\
          evaluate their playing style and provide suggestions for the player to improve their game and the best fit team style for that player: {data.to_json()}"
    return get_completion(prompt)

@st.cache_data
def load_data():
    df = pd.read_feather('compressed_data.feather')
    return df

df = load_data()
#### Getting the data, make initial forcast and build a front end web-app with Taipy GUI
#features = attacking_skills_style_ratings

features = st.sidebar.selectbox('Select Features to compare', options=['Attacking Skills and Style Ratings',
                                                                        'Defensive Skills and Style Ratings',
                                                                         'Play Style and Specialties',
                                                                         'Playing Time and Performance Metrics'
                                                                        ])

if features == 'Attacking Skills and Style Ratings':
    features = attacking_skills_style_ratings   
elif features == 'Defensive Skills and Style Ratings':
    features = defensive_skills_style_ratings
elif features == 'Play Style and Specialties':
    features = play_style_specialties
elif features == 'Playing Time and Performance Metrics':
    features = playing_time_performance_metrics

league = st.sidebar.multiselect('Select League', options=df['League'].unique())
#league = st.sidebar.selectbox('Select League', options=df['League'].unique())
position = st.sidebar.selectbox('Select Position', options=df['Position'].unique())
season_min = st.sidebar.slider('Select Season Min', min_value=2015, max_value=2025, value=2015)
season_max = st.sidebar.slider('Select Season Max', min_value=2015, max_value=2025, value=2025)
age_min = st.sidebar.slider('Select Age Min', min_value=15, max_value=40, value=15)
age_max = st.sidebar.slider('Select Age Max', min_value=15, max_value=40, value=40)


st.sidebar.header('Select the percentile for each rating')
st.sidebar.subheader('Model Ratings')
attack_slider = st.sidebar.slider('Attack', 0, 100, 0)
defending_quality_slider = st.sidebar.slider('Defending Quality', 0, 100, 0)
defending_quantity_slider = st.sidebar.slider('Defending Quantity', 0, 100, 0)
ball_retention_slider = st.sidebar.slider('Ball Retention', 0, 100, 0)


model_ratings_values = [attack_slider, defending_quality_slider, defending_quantity_slider, ball_retention_slider]

model_ratings_features = [model_ratings['Attack'], model_ratings['Defending Quality'],
                           model_ratings['Defending Quantity'], model_ratings['Ball Retention']]

def model_rating_filter(df, model_ratings_features, model_ratings_values):
    df_filtered = df.copy()

    for features_list, val in zip(model_ratings_features, model_ratings_values):
        val = val/100
        df_ranks = pd.DataFrame()
        for feature in features_list:
            # check if the value is greater than 0.5, if so, use the pct column so we can display only the important features for the user
            df_ranks[feature] = df[feature].rank(pct=True)
        # Determine if all percentile ranks in the current group are above the threshold
        df_filtered = df_filtered[df_ranks.gt(val).all(axis=1)]

        # If only one row is left, return the dataframe
        if len(df_filtered) == 1:
            return df_filtered

    return df_filtered


st.sidebar.subheader('Skill Ratings')
st.sidebar.subheader('Skill Ratings')
aerial_duels_open_play_slider = st.sidebar.slider('Aerial Duels in Open Play', 0, 100, 0)
aerial_duels_dead_balls_slider = st.sidebar.slider('Aerial Duels from Dead Balls', 0, 100, 0)
ground_duels_possession_slider = st.sidebar.slider('Ground Duels in Possession', 0, 100, 0)
ground_duels_out_possession_slider = st.sidebar.slider('Ground Duels Out of Possession', 0, 100, 0)

skill_ratings_values = [aerial_duels_open_play_slider, aerial_duels_dead_balls_slider, ground_duels_possession_slider, ground_duels_out_possession_slider]

skill_ratings_features = [skill_ratings['Aerial Duels in Open Play'], skill_ratings['Aerial Duels from Dead Balls'],
                           skill_ratings['Ground Duels in Possession'], skill_ratings['Ground Duels Out of Possession']]

def model_skill_ratings(df, skill_ratings_features, skill_ratings_values):
    df_filtered = df.copy()

    for features_list, val in zip(skill_ratings_features, skill_ratings_values):
        val = val/100
        df_ranks = pd.DataFrame()
        for feature in features_list:
            # check if the value is greater than 0.5, if so, use the pct column so we can display only the important features for the user
            df_ranks[feature] = df[feature].rank(pct=True)
        # Determine if all percentile ranks in the current group are above the threshold
        df_filtered = df_filtered[df_ranks.gt(val).all(axis=1)]

        # If only one row is left, return the dataframe
        if len(df_filtered) == 1:
            return df_filtered

    return df_filtered


st.sidebar.subheader('Style Ratings')
disrupting_moves_slider = st.sidebar.slider('Disrupting Opposition Moves', 0, 100, 0)
recovering_ball_slider = st.sidebar.slider('Recovering a Moving Ball', 0, 100, 0)
aerial_duels_slider = st.sidebar.slider('Aerial Duels', 0, 100, 0)
link_up_passing_slider = st.sidebar.slider('Link-up Passing', 0, 100, 0)
passing_toward_goal_slider = st.sidebar.slider('Passing toward Goal', 0, 100, 0)
dribbling_slider = st.sidebar.slider('Dribbling', 0, 100, 0)
receiving_in_the_box = st.sidebar.slider('Receiving in the Box', 0, 100, 0)
shooting = st.sidebar.slider('Shooting', 0, 100, 0)


style_ratings_values = [disrupting_moves_slider, recovering_ball_slider, aerial_duels_slider, link_up_passing_slider, 
                        passing_toward_goal_slider, dribbling_slider, receiving_in_the_box, shooting]

style_ratings_features = [style_ratings['Disrupting Opposition Moves'], style_ratings['Recovering a Moving Ball'],
                           style_ratings['Aerial Duels'], style_ratings['Link-up Passing'], style_ratings['Passing toward Goal'], 
                           style_ratings['Dribbling'], style_ratings['Receiving in the Box'], style_ratings['Shooting']]


def model_style_ratings(df, style_ratings_features, style_ratings_values):
    df_filtered = df.copy()

    for features_list, val in zip(style_ratings_features, style_ratings_values):
        val = val/100
        df_ranks = pd.DataFrame()
        for feature in features_list:
            # check if the value is greater than 0.5, if so, use the pct column so we can display only the important features for the user
            df_ranks[feature] = df[feature].rank(pct=True)
        # Determine if all percentile ranks in the current group are above the threshold
        df_filtered = df_filtered[df_ranks.gt(val).all(axis=1)]

        # If only one row is left, return the dataframe
        if len(df_filtered) == 1:
            return df_filtered

    return df_filtered


data = get_players_data(df, league, position, season_min, season_max, age_min, age_max,
                         model_ratings_features, model_ratings_values, 
                         skill_ratings_features, skill_ratings_values,
                           style_ratings_features, style_ratings_values,
                           within_league, within_pos, within_age, within_season)

################# Plot the graph #####################

# Assuming 'merged_pct' is your DataFrame and it has columns for each category ending with '_pct'
#categories = [col for col in data.columns if col.endswith('_pct')]
categories = features
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

#import sys
#st.write(sys.executable)

if st.button('Analyze with chatgpt'):
    st.write(analyze_with_chatgpt(data))


