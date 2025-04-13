import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set wide layout
st.set_page_config(layout="wide", page_title="IPL Win Predictor", page_icon="🏏")

# Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
delivery_df = pd.read_csv("delivery_df.csv")
teams = sorted([
    'Royal Challengers Bengaluru', 'Mumbai Indians', 'Kolkata Knight Riders', 'Rajasthan Royals',
    'Chennai Super Kings', 'Sunrisers Hyderabad', 'Delhi Capitals', 'Punjab Kings',
    'Lucknow Super Giants', 'Gujarat Titans'
])
cities = sorted([
    'Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur', 'Hyderabad', 'Chennai', 'Cape Town',
    'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
    'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Bengaluru', 'Indore', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow', 'Guwahati', 'Mohali'
])

# Add custom style
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        h1, h2, h3 {color: #004080;}

        /* Reduce spacing around buttons and inputs */
        .stButton > button, .stSelectbox, .stNumberInput {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 6px;
            padding-bottom: 6px;
        }

        .stButton > button {
            background-color: #004080;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }

        /* Fix layout padding */
        .css-1kyxreq, .css-1uixxvy, .css-1v3fvcr {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        /* Make input boxes cleaner */
        .stSelectbox > div, .stNumberInput > div {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)


# Match progression helper
def match_progression(x_df, match_id, pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[match['ball'] == 6]
    temp_df = match[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left',
                     'total_runs_x', 'crr', 'rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]

    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)

    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0, target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)

    wickets = list(temp_df['wickets_left'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0, 10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]

    return temp_df, target


# App Title
st.title('🏏 IPL Win Predictor')

# Sidebar Navigation
page = st.sidebar.radio("📁 Navigate", ["🏆 Prediction", "📊 Visualization"])

# Prediction Page
if page == "🏆 Prediction":
    st.header("💡 Match Win Probability")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('🏏 Batting Team', teams)
    with col2:
        bowling_team = st.selectbox('🎯 Bowling Team', teams)

    selected_city = st.selectbox('📍 Match Location', cities)
    target = st.number_input('🎯 Target Score', min_value=1)

    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('📊 Current Score', min_value=0)
    with col4:
        overs = st.number_input('⏱️ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
    with col5:
        wickets_out = st.number_input('🚶‍♂️ Wickets Fallen', min_value=0, max_value=10)

    if st.button("Predict Probability"):
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets = 10 - wickets_out
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
            'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets],
            'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]
        })

        with st.spinner("🏃‍♂️ Calculating..."):
            result = pipe.predict_proba(input_df)
            win = round(result[0][1] * 100, 2)
            loss = round(result[0][0] * 100, 2)

        st.success(f"🎉 **{batting_team} Win Probability:** `{win}%`")
        st.info(f"🛡️ **{bowling_team} Win Probability:** `{loss}%`")

# Visualization Page
elif page == "📊 Visualization":
    st.header("📈 Match Progression Over Time")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox('Select Team 1', teams)
    with col2:
        team2 = st.selectbox('Select Team 2', teams)

    match_dates = delivery_df[
        ((delivery_df['batting_team'] == team1) & (delivery_df['bowling_team'] == team2)) |
        ((delivery_df['batting_team'] == team2) & (delivery_df['bowling_team'] == team1))
    ]['date'].unique()

    selected_date = st.selectbox("📅 Match Date", sorted(match_dates))

    available_matches = delivery_df[
        (((delivery_df['batting_team'] == team1) & (delivery_df['bowling_team'] == team2)) |
         ((delivery_df['batting_team'] == team2) & (delivery_df['bowling_team'] == team1))) &
        (delivery_df['date'] == selected_date)
    ]['match_id'].unique()

    selected_match_id = st.selectbox("🎮 Select Match ID", available_matches)

    if st.button("📊 Show Match Trend"):
        temp_df, target = match_progression(delivery_df, selected_match_id, pipe)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

        # Win Probability
        ax1.plot(temp_df['end_of_over'], temp_df['win'], color='green', linewidth=3, label=f'{team1} Win %')
        ax1.plot(temp_df['end_of_over'], temp_df['lose'], color='red', linewidth=3, label=f'{team2} Win %')
        ax1.set_title(f'🏏 Win Probability Over Innings (Target: {target})', fontsize=16)
        ax1.set_xlabel("Overs", fontsize=12)
        ax1.set_ylabel("Probability (%)", fontsize=12)
        ax1.legend()
        ax1.grid(True)

        # Runs & Wickets
        ax2.bar(temp_df['end_of_over'], temp_df['runs_after_over'], color='skyblue', label='Runs in Over')
        ax2.plot(temp_df['end_of_over'], temp_df['wickets_in_over'], color='orange', marker='o', linewidth=3,
                 label='Wickets in Over')
        ax2.set_title('🎯 Runs and 🧹 Wickets per Over', fontsize=16)
        ax2.set_xlabel("Overs", fontsize=12)
        ax2.set_ylabel("Runs / Wickets", fontsize=12)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)
