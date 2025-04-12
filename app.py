import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load necessary data and model
pipe = pickle.load(open('pipe.pkl', 'rb'))
delivery_df = pd.read_csv("delivery_df.csv")
teams = ['Royal Challengers Bengaluru', 'Mumbai Indians', 'Kolkata Knight Riders', 'Rajasthan Royals',
         'Chennai Super Kings', 'Sunrisers Hyderabad', 'Delhi Capitals', 'Punjab Kings', 'Lucknow Super Giants',
         'Gujarat Titans']
cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur', 'Hyderabad', 'Chennai', 'Cape Town',
          'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
          'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Bengaluru', 'Indore', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow', 'Guwahati', 'Mohali']


# Match Progression Function for Visualization Page
def match_progression(x_df, match_id, pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]  # Get end-of-over data
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


# Streamlit UI
st.title('IPL Win Predictor')

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Visualization"])

if page == "Prediction":
    # Prediction page code
    st.header("Predict Match Outcome")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select the batting team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select the bowling team', sorted(teams))

    selected_city = st.selectbox('Select host city', sorted(cities))

    target = st.number_input('Target')

    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Score')
    with col4:
        overs = st.number_input('Overs completed')
    with col5:
        wickets = st.number_input('Wickets out')

    if st.button('Predict Probability'):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame(
            {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets],
             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(f"{batting_team} - {round(win * 100)}%")
        st.header(f"{bowling_team} - {round(loss * 100)}%")

elif page == "Visualization":
    # Visualization page code
    st.header("Visualize Match Progression")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox('Select Team 1', sorted(teams))
    with col2:
        team2 = st.selectbox('Select Team 2', sorted(teams))

    # Get available dates for selected teams
    match_dates = delivery_df[
        ((delivery_df['batting_team'] == team1) & (delivery_df['bowling_team'] == team2)) |
        ((delivery_df['batting_team'] == team2) & (delivery_df['bowling_team'] == team1))
        ]['date'].unique()

    selected_date = st.selectbox("Select Match Date", sorted(match_dates))

    # Get match IDs for selected teams and date
    available_matches = delivery_df[
        (((delivery_df['batting_team'] == team1) & (delivery_df['bowling_team'] == team2)) |
         ((delivery_df['batting_team'] == team2) & (delivery_df['bowling_team'] == team1))) &
        (delivery_df['date'] == selected_date)
        ]['match_id'].unique()

    selected_match_id = st.selectbox("Select Match", available_matches)

    if st.button("Show Match Trend"):
        temp_df, target = match_progression(delivery_df, selected_match_id, pipe)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

        # Win/Loss Probability Plot
        ax1.plot(temp_df['end_of_over'], temp_df['win'], color='green', linewidth=3, label=f'{team1} Win %')
        ax1.plot(temp_df['end_of_over'], temp_df['lose'], color='red', linewidth=3, label=f'{team2} Win %')
        ax1.set_title(f'Win Probability (Target: {target})')
        ax1.set_xlabel("Overs")
        ax1.set_ylabel("Probability (%)")
        ax1.legend()
        ax1.grid(True)

        # Runs and Wickets Plot
        ax2.bar(temp_df['end_of_over'], temp_df['runs_after_over'], color='blue', alpha=0.6, label='Runs in Over')
        ax2.plot(temp_df['end_of_over'], temp_df['wickets_in_over'], color='orange', marker='o',
                 linewidth=3, label='Wickets in Over')
        ax2.set_title('Runs and Wickets per Over')
        ax2.set_xlabel("Overs")
        ax2.set_ylabel("Runs/Wickets")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)
