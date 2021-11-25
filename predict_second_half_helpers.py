import pandas as pd
import csv

# QUESTIONS
# Do something about 'assume its normally distributed'?
# Incorporate stdevs?

# Utilize average team stats for seasons with average opposing team stats to predict stats related to second half games.
def predict_second_half():
    columns = None
    # Get column names from first half stats to use as column names for CSV of predicted second half stats
    with open('cleaned_data_v5/Logistic_Regression_GamebyGame/first_half_matchups.csv', newline='') as csv_to_read:
        reader = csv.reader(csv_to_read)
        columns = next(reader)
        columns.remove('Home Win')

    # Make the new CSV of predicted second half stats
    with open('cleaned_data_v5/Logistic_Regression_GamebyGame/stats_second_half_matchups.csv', 'w') as csv_to_write:
        writer = csv.writer(csv_to_write, delimiter=',')
        writer.writerow(columns)

        # Second half matchups to attach predicted game data to in new CSV
        second_half_matchups = pd.read_csv("cleaned_data_v5/Logistic_Regression_GamebyGame/second_half_matchups.csv")

        # First half data of teams' average season stats and average seasons stats of all opposing teams of a team
        teams_seasons_avgs = pd.read_csv("cleaned_data_v5/Logistic_Regression_GamebyGame/stat_generation_avg.csv")
        opposing_teams_seasons_avgs = pd.read_csv("cleaned_data_v5/Logistic_Regression_GamebyGame/stat_generation_avg_op.csv")

        # Columns of numerical data for first half CSVs
        team_numerical_cols = ['Shots', 'Blocked Shots', 'Power Play Opportunities', 'PIM', 'Player Hits', 'Giveaways', 'Takeaways', 'Injured Players']
        opposing_numerical_cols = ['Shots', 'Shots Blocked', 'Power Play Opportunities', 'PIM', 'Player Hits', 'Giveaways', 'Takeaways', 'Injured Players']

        for index, row in second_half_matchups.iterrows():
            season, game_id, home_team_id, away_team_id = row['Season'], row['game_id'], row['home_team_id'], row['away_team_id']

            # Get home team's average stats for the season
            home_team_season_avgs = teams_seasons_avgs.loc[teams_seasons_avgs['team_id'].eq(home_team_id) & teams_seasons_avgs['Season'].eq(season)]
            home_team_season_avgs_nums = home_team_season_avgs[team_numerical_cols]

            # Get average stats for the season of all opposing teams of the away team
            opposing_of_away_team_season_avgs = opposing_teams_seasons_avgs.loc[opposing_teams_seasons_avgs['team_id'].eq(away_team_id) & opposing_teams_seasons_avgs['Season'].eq(season)]
            opposing_of_away_team_season_avgs_nums = opposing_of_away_team_season_avgs[opposing_numerical_cols]

            # Get away team's average stats for the season
            away_team_season_avgs = teams_seasons_avgs.loc[teams_seasons_avgs['team_id'].eq(away_team_id) & teams_seasons_avgs['Season'].eq(season)]
            away_team_season_avgs_nums = away_team_season_avgs[team_numerical_cols]

            # Get average stats for the season of all opposing teams of the home team
            opposing_of_home_team_season_avgs = opposing_teams_seasons_avgs.loc[opposing_teams_seasons_avgs['team_id'].eq(home_team_id) & opposing_teams_seasons_avgs['Season'].eq(season)]
            opposing_of_home_team_season_avgs_nums = opposing_of_home_team_season_avgs[opposing_numerical_cols]

            predicted_game_stats_row =[season, game_id, home_team_id, home_team_season_avgs['Team'].values[0], away_team_id, away_team_season_avgs['Team'].values[0]]
            avg_home_team_stats = (home_team_season_avgs_nums.to_numpy() + opposing_of_away_team_season_avgs_nums.to_numpy()) / 2
            avg_away_team_stats = (away_team_season_avgs_nums.to_numpy() + opposing_of_home_team_season_avgs_nums.to_numpy()) / 2
            predicted_game_stats_row = predicted_game_stats_row + avg_home_team_stats[0].tolist() + avg_away_team_stats[0].tolist()
            writer.writerow(predicted_game_stats_row)

# After running PCA and Logistic Regression on predicted second half stats created above, use the predicted
# second half game outcomes to update the teams' seasons' win totals.
def get_season_win_totals(second_half_predictions):
    # Game matchups of the second half
    matchups = pd.read_csv('cleaned_data_v5/Logistic_Regression_GamebyGame/second_half_matchups.csv')
    # Win totals for each teams' season by the half way point of the season
    wins = pd.read_csv('cleaned_data_v5/Logistic_Regression_GamebyGame/first_half_win_totals.csv')
    # Update win totals with Logistic Regression predicted second half game outcomes
    for index, row in matchups.iterrows():
        season, home_team_id, away_team_id = row['Season'], row['home_team_id'], row['away_team_id']
        # Home team win
        if second_half_predictions[index] == 1:
            wins.loc[wins['team_id'].eq(home_team_id) & wins['Season'].eq(season), 'Total Wins'] += 1
        # Away team win
        elif second_half_predictions[index] == 0:
            wins.loc[wins['team_id'].eq(away_team_id) & wins['Season'].eq(season), 'Total Wins'] += 1
    wins.to_csv('cleaned_data_v5/Logistic_Regression_GamebyGame/win_totals.csv', index=False)
    return wins

# TODO
# Utilize updated teams' seasons' win totals to predict the 16 teams that make it to the playoffs each season
def get_playoff_teams(wins):
    # Eastern Conference
    atlantic = ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Montreal Canadiens (MTL)', 'Florida Panthers (FLA)', 'Ottawa Senators (OTT)', 'Tampa Bay Lightning (TBL)', 'Toronto Maple Leafs (TOR)']
    metropolitan = ['Philadelphia Flyers (PHI)', 'NY Rangers Rangers (NYR)', 'New Jersey Devils (NJD)', 'Pittsburgh Penguins (PIT)', 'Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'NY Islanders Islanders (NYI)', 'Washington Capitals (WSH)', 'Atlanta Thrashers (ATL)']

    # Western Conference
    central = ['Winnipeg Jets (WPG)', 'Phoenix Coyotes (PHX)', 'St Louis Blues (STL)', 'Colorado Avalanche (COL)', 'Nashville Predators (NSH)', 'Dallas Stars (DAL)', 'Arizona Coyotes (ARI)', 'Chicago Blackhawks (CHI)', 'Minnesota Wild (MIN)']
    pacific = ['San Jose Sharks (SJS)', 'Vancouver Canucks (VAN)', 'Edmonton Oilers (EDM)', 'Anaheim Ducks (ANA)', 'Los Angeles Kings (LAK)', 'Calgary Flames (CGY)', 'Vegas Golden Knights (VGK)']

    # Handle teams that no longer exist, young teams, and division & conference changes across the seasons

    # Take top 3 in each division plus top 2 by points in conference



# Incorporate stdevs?

# teams_seasons_stdevs = pd.read_csv("cleaned_data_v5/Logistic_Regression_GamebyGame/stat_generation_avg.csv")
# opposing_teams_seasons_stdevs = pd.read_csv("cleaned_data_v5/Logistic_Regression_GamebyGame/stat_generation_avg_op.csv")

# # Get home team's stdev stats for the season
# team_seasons_stdevs = teams_seasons_stdevs.loc[teams_seasons_stdevs['team_id'] == home_team_id]
# team_season_stdevs = team_seasons_stdevs.loc[team_seasons_stdevs['Season'] == season]

# # Get opposing teams' stdev stats for the seasonn against the home team
# opposing_team_seasons_stdevs = opposing_teams_seasons_stdevs.loc[opposing_teams_seasons_stdevs['team_id'] == home_team_id]
# opposing_team_season_stdevs = opposing_team_seasons_stdevs.loc[opposing_team_seasons_stdevs['Season'] == season]