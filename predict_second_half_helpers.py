import pandas as pd
import csv

# Utilize average team stats for seasons with average opposing team stats to predict stats related to second half games.
def predict_second_half():
    columns = None
    # Get column names from first half stats to use as column names for CSV of predicted second half stats
    with open('cleaned_data_v6/Logistic Regression - GamebyGame/first_half_matchups.csv', newline='') as csv_to_read:
        reader = csv.reader(csv_to_read)
        columns = next(reader)

    # Make the new CSV of predicted second half stats
    with open('cleaned_data_v6/Logistic Regression - GamebyGame/stats_second_half_matchups.csv', 'w') as csv_to_write:
        writer = csv.writer(csv_to_write, delimiter=',')
        writer.writerow(columns)

        # Second half matchups to attach predicted game data to in new CSV
        second_half_matchups = pd.read_csv("cleaned_data_v6/Logistic Regression - GamebyGame/second_half_matchups.csv")

        # First half data of teams' average season stats and average seasons stats of all opposing teams of a team
        teams_seasons_avgs = pd.read_csv("cleaned_data_v6/Logistic Regression - GamebyGame/stat_generation_avg.csv")
        opposing_teams_seasons_avgs = pd.read_csv("cleaned_data_v6/Logistic Regression - GamebyGame/stat_generation_avg_op.csv")

        # Columns of numerical data for first half CSVs
        team_numerical_cols = ['Shots', 'Blocked Shots', 'Power Play Opportunities', 'PIM', 'Player Hits', 'Giveaways', 'Takeaways', 'Injured Players']
        opposing_numerical_cols = ['Shots', 'Shots Blocked', 'Power Play Opportunities', 'PIM', 'Player Hits', 'Giveaways', 'Takeaways', 'Injured Players']

        for index, row in second_half_matchups.iterrows(): # 8657 games
            season, game_id, home_team_id, away_team_id, home_team_win = row['Season'], row['game_id'], row['home_team_id'], row['away_team_id'], row['Home Win']

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
            predicted_game_stats_row.append(home_team_win)
            writer.writerow(predicted_game_stats_row)

# After running PCA and Logistic Regression on predicted second half stats created above, use the predicted
# second half game outcomes to update the teams' seasons' win totals.
def get_season_win_totals(second_half_predictions):
    # Game matchups of the second half
    matchups = pd.read_csv('cleaned_data_v6/Logistic Regression - GamebyGame/second_half_matchups.csv')

    # Win totals for each teams' season by the half way point of the season
    wins = pd.read_csv('cleaned_data_v6/Logistic Regression - GamebyGame/first_half_win_totals.csv')

    # Update win totals with Logistic Regression predicted second half game outcomes
    for index, row in matchups.iterrows():
        season, home_team_id, away_team_id = row['Season'], row['home_team_id'], row['away_team_id']
        # Home team win
        if second_half_predictions[index] == 1:
            wins.loc[wins['team_id'].eq(home_team_id) & wins['Season'].eq(season), 'Total Wins'] += 1
        # Away team win
        elif second_half_predictions[index] == 0:
            wins.loc[wins['team_id'].eq(away_team_id) & wins['Season'].eq(season), 'Total Wins'] += 1
    wins.to_csv('cleaned_data_v6/Logistic Regression - GamebyGame/win_totals.csv', index=False)
    return wins

actual_playoffs_2003 = (['NY Rangers Rangers (NYR)', 'Pittsburgh Penguins (PIT)', 'Buffalo Sabres (BUF)', 'Atlanta Thrashers (ATL)', 'Carolina Hurricanes (CAR)', 'Florida Panthers (FLA)', 'Washington Capitals (WSH)', 'Chicago Blackhawks (CHI)', 'Columbus Blue Jackets (CBJ)', 'Edmonton Oilers (EDM)', 'Minnesota Wild (MIN)', 'Anaheim Ducks (ANA)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)'],
                        ['Tampa Bay Lightning (TBL)', 'NY Islanders Islanders (NYI)', 'Boston Bruins (BOS)', 'Montreal Canadiens (MTL)', 'Philadelphia Flyers (PHI)', 'New Jersey Devils (NJD)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'San Jose Sharks (SJS)', 'Calgary Flames (CGY)', 'Vancouver Canucks (VAN)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)'])

actual_playoffs_2005 = (['NY Rangers Rangers (NYR)', 'Buffalo Sabres (BUF)', 'Ottawa Senators (OTT)', 'New Jersey Devils (NJD)', 'Philadelphia Flyers (PHI)', 'Nashville Predators (NSH)', 'Montreal Canadiens (MTL)', 'Calgary Flames (CGY)', 'San Jose Sharks (SJS)', 'Carolina Hurricanes (CAR)', 'Tampa Bay Lightning (TBL)', 'Dallas Stars (DAL)', 'Detroit Red Wings (DET)', 'Colorado Avalanche (COL)', 'Anaheim Ducks (ANA)', 'Edmonton Oilers (EDM)'],
                        ['Pittsburgh Penguins (PIT)', 'Atlanta Thrashers (ATL)', 'Florida Panthers (FLA)', 'Chicago Blackhawks (CHI)', 'Columbus Blue Jackets (CBJ)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'Toronto Maple Leafs (TOR)', 'Washington Capitals (WSH)', 'Boston Bruins (BOS)', 'NY Islanders Islanders (NYI)', 'Vancouver Canucks (VAN)', 'St Louis Blues (STL)', 'Minnesota Wild (MIN)'])

actual_playoffs_2006 = (['Buffalo Sabres (BUF)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'Tampa Bay Lightning (TBL)', 'Atlanta Thrashers (ATL)', 'NY Rangers Rangers (NYR)', 'Pittsburgh Penguins (PIT)', 'Ottawa Senators (OTT)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)', 'Calgary Flames (CGY)', 'Minnesota Wild (MIN)', 'Anaheim Ducks (ANA)', 'Dallas Stars (DAL)', 'Vancouver Canucks (VAN)', 'San Jose Sharks (SJS)'],
                        ['Philadelphia Flyers (PHI)', 'Boston Bruins (BOS)', 'Montreal Canadiens (MTL)', 'Toronto Maple Leafs (TOR)', 'Carolina Hurricanes (CAR)', 'Florida Panthers (FLA)', 'Washington Capitals (WSH)', 'Chicago Blackhawks (CHI)', 'Columbus Blue Jackets (CBJ)', 'St Louis Blues (STL)', 'Colorado Avalanche (COL)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)'])

actual_playoffs_2008 = (['New Jersey Devils (NJD)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Montreal Canadiens (MTL)', 'Carolina Hurricanes (CAR)', 'Washington Capitals (WSH)', 'Chicago Blackhawks (CHI)', 'Columbus Blue Jackets (CBJ)', 'Detroit Red Wings (DET)', 'St Louis Blues (STL)', 'Calgary Flames (CGY)', 'Vancouver Canucks (VAN)', 'Anaheim Ducks (ANA)', 'San Jose Sharks (SJS)'],
                        ['NY Islanders Islanders (NYI)', 'Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)', 'Atlanta Thrashers (ATL)', 'Florida Panthers (FLA)', 'Tampa Bay Lightning (TBL)', 'Nashville Predators (NSH)', 'Colorado Avalanche (COL)', 'Edmonton Oilers (EDM)', 'Minnesota Wild (MIN)', 'Dallas Stars (DAL)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)'])

actual_playoffs_2009 = (['New Jersey Devils (NJD)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Washington Capitals (WSH)', 'Chicago Blackhawks (CHI)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)', 'Colorado Avalanche (COL)', 'Minnesota Wild (MIN)', 'Vancouver Canucks (VAN)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'San Jose Sharks (SJS)'],
                        ['NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Toronto Maple Leafs (TOR)', 'Atlanta Thrashers (ATL)', 'Carolina Hurricanes (CAR)', 'Florida Panthers (FLA)', 'Tampa Bay Lightning (TBL)', 'Columbus Blue Jackets (CBJ)', 'St Louis Blues (STL)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Anaheim Ducks (ANA)', 'Dallas Stars (DAL)'])

actual_playoffs_2010 = (['NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Montreal Canadiens (MTL)', 'Tampa Bay Lightning (TBL)', 'Washington Capitals (WSH)', 'Chicago Blackhawks (CHI)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)', 'Vancouver Canucks (VAN)', 'Anaheim Ducks (ANA)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'San Jose Sharks (SJS)'],
                        ['New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)', 'Carolina Hurricanes (CAR)', 'Florida Panthers (FLA)', 'Winnipeg Jets (WPG)', 'Columbus Blue Jackets (CBJ)', 'St Louis Blues (STL)', 'Calgary Flames (CGY)', 'Colorado Avalanche (COL)', 'Edmonton Oilers (EDM)', 'Minnesota Wild (MIN)', 'Dallas Stars (DAL)'])

actual_playoffs_2011 = (['NY Rangers Rangers (NYR)', 'Ottawa Senators (OTT)', 'Boston Bruins (BOS)', 'Washington Capitals (WSH)', 'New Jersey Devils (NJD)', 'Pittsburgh Penguins (PIT)', 'Florida Panthers (FLA)', 'Philadelphia Flyers (PHI)', 'Vancouver Canucks (VAN)', 'Los Angeles Kings (LAK)', 'San Jose Sharks (SJS)', 'St Louis Blues (STL)', 'Chicago Blackhawks (CHI)', 'Phoenix Coyotes (PHX)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)'],
                        ['NY Islanders Islanders (NYI)', 'Buffalo Sabres (BUF)', 'Montreal Canadiens (MTL)', 'Toronto Maple Leafs (TOR)', 'Carolina Hurricanes (CAR)', 'Tampa Bay Lightning (TBL)', 'Winnipeg Jets (WPG)', 'Columbus Blue Jackets (CBJ)', 'Calgary Flames (CGY)', 'Colorado Avalanche (COL)', 'Edmonton Oilers (EDM)', 'Minnesota Wild (MIN)', 'Anaheim Ducks (ANA)', 'Dallas Stars (DAL)'])

actual_playoffs_2013 = (['Boston Bruins (BOS)', 'Detroit Red Wings (DET)', 'Tampa Bay Lightning (TBL)', 'Montreal Canadiens (MTL)', 'Columbus Blue Jackets (CBJ)', 'Pittsburgh Penguins (PIT)', 'Philadelphia Flyers (PHI)', 'NY Rangers Rangers (NYR)', 'Colorado Avalanche (COL)', 'Minnesota Wild (MIN)', 'St Louis Blues (STL)', 'Chicago Blackhawks (CHI)', 'Anaheim Ducks (ANA)', 'Dallas Stars (DAL)', 'San Jose Sharks (SJS)', 'Los Angeles Kings (LAK)'],
                        ['Buffalo Sabres (BUF)', 'Florida Panthers (FLA)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)', 'Carolina Hurricanes (CAR)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'Washington Capitals (WSH)', 'Nashville Predators (NSH)', 'Winnipeg Jets (WPG)', 'Arizona Coyotes (ARI)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Vancouver Canucks (VAN)'])

actual_playoffs_2014 = (['Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Tampa Bay Lightning (TBL)', 'Detroit Red Wings (DET)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Pittsburgh Penguins (PIT)', 'Washington Capitals (WSH)', 'Minnesota Wild (MIN)', 'St Louis Blues (STL)', 'Nashville Predators (NSH)', 'Chicago Blackhawks (CHI)', 'Anaheim Ducks (ANA)', 'Winnipeg Jets (WPG)', 'Vancouver Canucks (VAN)', 'Calgary Flames (CGY)'],
                        ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Florida Panthers (FLA)', 'Toronto Maple Leafs (TOR)', 'Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'Philadelphia Flyers (PHI)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Arizona Coyotes (ARI)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'San Jose Sharks (SJS)'])

actual_playoffs_2015 = (['Florida Panthers (FLA)', 'NY Islanders Islanders (NYI)', 'Tampa Bay Lightning (TBL)', 'Detroit Red Wings (DET)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Washington Capitals (WSH)', 'Dallas Stars (DAL)', 'Minnesota Wild (MIN)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Chicago Blackhawks (CHI)', 'Anaheim Ducks (ANA)', 'Los Angeles Kings (LAK)', 'San Jose Sharks (SJS)'],
                        ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)', 'Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'Colorado Avalanche (COL)', 'Winnipeg Jets (WPG)', 'Arizona Coyotes (ARI)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Vancouver Canucks (VAN)'])

actual_playoffs_2016 = (['Montreal Canadiens (MTL)', 'NY Rangers Rangers (NYR)', 'Boston Bruins (BOS)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)', 'Washington Capitals (WSH)', 'Pittsburgh Penguins (PIT)', 'Columbus Blue Jackets (CBJ)', 'Chicago Blackhawks (CHI)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Minnesota Wild (MIN)', 'Anaheim Ducks (ANA)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'San Jose Sharks (SJS)'],
                        ['Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Tampa Bay Lightning (TBL)', 'Carolina Hurricanes (CAR)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'Philadelphia Flyers (PHI)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Winnipeg Jets (WPG)', 'Arizona Coyotes (ARI)', 'Los Angeles Kings (LAK)', 'Vancouver Canucks (VAN)', 'Vegas Golden Knights (VGK)'])

actual_playoffs_2017 = (['Tampa Bay Lightning (TBL)', 'Toronto Maple Leafs (TOR)', 'Boston Bruins (BOS)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'Washington Capitals (WSH)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Colorado Avalanche (COL)', 'Minnesota Wild (MIN)', 'Nashville Predators (NSH)', 'Winnipeg Jets (WPG)', 'Los Angeles Kings (LAK)', 'Vegas Golden Knights (VGK)', 'San Jose Sharks (SJS)', 'Anaheim Ducks (ANA)'],
                        ['Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Carolina Hurricanes (CAR)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Chicago Blackhawks (CHI)', 'Dallas Stars (DAL)', 'St Louis Blues (STL)', 'Arizona Coyotes (ARI)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Vancouver Canucks (VAN)'])

actual_playoffs_2018 = (['Tampa Bay Lightning (TBL)', 'Columbus Blue Jackets (CBJ)', 'Toronto Maple Leafs (TOR)', 'Boston Bruins (BOS)', 'Carolina Hurricanes (CAR)', 'Washington Capitals (WSH)', 'Pittsburgh Penguins (PIT)', 'NY Islanders Islanders (NYI)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Winnipeg Jets (WPG)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Calgary Flames (CGY)', 'San Jose Sharks (SJS)', 'Vegas Golden Knights (VGK)'],
                        ['Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'New Jersey Devils (NJD)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Chicago Blackhawks (CHI)', 'Minnesota Wild (MIN)', 'Anaheim Ducks (ANA)', 'Arizona Coyotes (ARI)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'Vancouver Canucks (VAN)'])

# Utilize updated teams' seasons' win totals to predict the 16 teams ranked that make it to the playoffs each season
# Handle teams new teams plus division & conference changes across the seasons
# Seasons in data set (year season started like 2007 for the 2007-2008 season): 2003, 2005-2006, 2008-2011, 2013-2018
def get_playoff_teams(wins):
    wins_2003 = []
    wins_2005 = []
    wins_2006 = []
    wins_2008 = []
    wins_2009 = []
    wins_2010 = []
    wins_2011 = []
    wins_2013 = []
    wins_2014 = []
    wins_2015 = []
    wins_2016 = []
    wins_2017 = []
    wins_2018 = []

    for index, row in wins.iterrows():
        season, team_name, season_wins = row['Season'], row['Team'], row['Total Wins']
        if season == 2003:
            wins_2003.append((team_name, season_wins))
        elif season == 2005:
            wins_2005.append((team_name, season_wins))
        elif season == 2006:
            wins_2006.append((team_name, season_wins))
        elif season == 2008:
            wins_2008.append((team_name, season_wins))
        elif season == 2009:
            wins_2009.append((team_name, season_wins))
        elif season == 2010:
            wins_2010.append((team_name, season_wins))
        elif season == 2011:
            wins_2011.append((team_name, season_wins))
        elif season == 2013:
            wins_2013.append((team_name, season_wins))
        elif season == 2014:
            wins_2014.append((team_name, season_wins))
        elif season == 2015:
            wins_2015.append((team_name, season_wins))
        elif season == 2016:
            wins_2016.append((team_name, season_wins))
        elif season == 2017:
            wins_2017.append((team_name, season_wins))
        elif season == 2018:
            wins_2018.append((team_name, season_wins))

    wins_2003.sort(key = lambda x: x[1], reverse = True)
    wins_2005.sort(key = lambda x: x[1], reverse = True)
    wins_2006.sort(key = lambda x: x[1], reverse = True)
    wins_2008.sort(key = lambda x: x[1], reverse = True)
    wins_2009.sort(key = lambda x: x[1], reverse = True)
    wins_2010.sort(key = lambda x: x[1], reverse = True)
    wins_2011.sort(key = lambda x: x[1], reverse = True)
    wins_2013.sort(key = lambda x: x[1], reverse = True)
    wins_2014.sort(key = lambda x: x[1], reverse = True)
    wins_2015.sort(key = lambda x: x[1], reverse = True)
    wins_2016.sort(key = lambda x: x[1], reverse = True)
    wins_2017.sort(key = lambda x: x[1], reverse = True)
    wins_2018.sort(key = lambda x: x[1], reverse = True)

    # Limitation: Not handling ties of same win count in deciding the last few teams of the 16
    predicted_playoffs_2003 = ([x[0] for i, x in enumerate(wins_2003) if i < 16],[x[0] for i, x in enumerate(wins_2003) if i > 15])
    predicted_playoffs_2005 = ([x[0] for i, x in enumerate(wins_2005) if i < 16],[x[0] for i, x in enumerate(wins_2005) if i > 15])
    predicted_playoffs_2006 = ([x[0] for i, x in enumerate(wins_2006) if i < 16],[x[0] for i, x in enumerate(wins_2006) if i > 15])
    predicted_playoffs_2008 = ([x[0] for i, x in enumerate(wins_2008) if i < 16],[x[0] for i, x in enumerate(wins_2008) if i > 15])
    predicted_playoffs_2009 = ([x[0] for i, x in enumerate(wins_2009) if i < 16],[x[0] for i, x in enumerate(wins_2009) if i > 15])
    predicted_playoffs_2010 = ([x[0] for i, x in enumerate(wins_2010) if i < 16],[x[0] for i, x in enumerate(wins_2010) if i > 15])
    predicted_playoffs_2011 = ([x[0] for i, x in enumerate(wins_2011) if i < 16],[x[0] for i, x in enumerate(wins_2011) if i > 15])
    predicted_playoffs_2013 = ([x[0] for i, x in enumerate(wins_2013) if i < 16],[x[0] for i, x in enumerate(wins_2013) if i > 15])
    predicted_playoffs_2014 = ([x[0] for i, x in enumerate(wins_2014) if i < 16],[x[0] for i, x in enumerate(wins_2014) if i > 15])
    predicted_playoffs_2015 = ([x[0] for i, x in enumerate(wins_2015) if i < 16],[x[0] for i, x in enumerate(wins_2015) if i > 15])
    predicted_playoffs_2016 = ([x[0] for i, x in enumerate(wins_2016) if i < 16],[x[0] for i, x in enumerate(wins_2016) if i > 15])
    predicted_playoffs_2017 = ([x[0] for i, x in enumerate(wins_2017) if i < 16],[x[0] for i, x in enumerate(wins_2017) if i > 15])
    predicted_playoffs_2018 = ([x[0] for i, x in enumerate(wins_2018) if i < 16],[x[0] for i, x in enumerate(wins_2018) if i > 15])

    print_accuracy(2003, actual_playoffs_2003, predicted_playoffs_2003)
    print_accuracy(2005, actual_playoffs_2005, predicted_playoffs_2005)
    print_accuracy(2006, actual_playoffs_2006, predicted_playoffs_2006)
    print_accuracy(2008, actual_playoffs_2008, predicted_playoffs_2008)
    print_accuracy(2009, actual_playoffs_2009, predicted_playoffs_2009)
    print_accuracy(2010, actual_playoffs_2010, predicted_playoffs_2010)
    print_accuracy(2011, actual_playoffs_2011, predicted_playoffs_2011)
    print_accuracy(2013, actual_playoffs_2013, predicted_playoffs_2013)
    print_accuracy(2014, actual_playoffs_2014, predicted_playoffs_2014)
    print_accuracy(2015, actual_playoffs_2015, predicted_playoffs_2015)
    print_accuracy(2016, actual_playoffs_2016, predicted_playoffs_2016)
    print_accuracy(2017, actual_playoffs_2017, predicted_playoffs_2017)
    print_accuracy(2018, actual_playoffs_2018, predicted_playoffs_2018)

def print_accuracy(season, actual_playoffs, predicted_playoffs):
    made_playoffs_actual, not_playoffs_actual = actual_playoffs
    made_playoffs_predicted, not_playoffs_predicted = predicted_playoffs
    true_positives_made_playoffs = len(set(made_playoffs_actual).intersection(set(made_playoffs_predicted)))
    true_negatives_not_playoffs = len(set(not_playoffs_actual).intersection(set(not_playoffs_predicted)))
    team_count = len(made_playoffs_predicted) + len(not_playoffs_predicted)
    accuracy = (true_positives_made_playoffs + true_negatives_not_playoffs) / team_count
    # print('{} season has {} playoff teams, {} not playoff teams, and {} teams.'.format(season, len(made_playoffs_predicted), len(not_playoffs_predicted), team_count))
    print('{} season accuracy: {}'.format(season, accuracy))

################################## DIVISIONS #########################################
# # 30 teams.
# # In 2006, Mighty Ducks of Anaheim changed their name to the Anaheim Ducks, but the data uses the same label so no change needed.
# # Unused from data set: 'Winnipeg Jets (WPG)', 'Arizona Coyotes (ARI)', 'Vegas Golden Knights (VGK)'
# divisions_2001_2010 = {
#     "Atlantic": ['New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)'],
#     "Northeast": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)'],
#     "Southeast": ['Atlanta Thrashers (ATL)', 'Carolina Hurricanes (CAR)', 'Florida Panthers (FLA)', 'Tampa Bay Lightning (TBL)', 'Washington Capitals (WSH)'], 
#     "Central": ['Chicago Blackhawks (CHI)', 'Columbus Blue Jackets (CBJ)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)'],
#     "Northwest": ['Calgary Flames (CGY)', 'Colorado Avalanche (COL)', 'Edmonton Oilers (EDM)', 'Minnesota Wild (MIN)', 'Vancouver Canucks (VAN)'],
#     "Pacific": ['Anaheim Ducks (ANA)', 'Dallas Stars (DAL)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'San Jose Sharks (SJS)']
# }

# # Atlanta Thrashers moved to Canada and became Winnipeg Jets
# # Unused from data set: 'Atlanta Thrashers (ATL)', 'Arizona Coyotes (ARI)', 'Vegas Golden Knights (VGK)'
# divisions_2011_2012 = {
#     "Atlantic": ['New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)'],
#     "Northeast": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)'],
#     "Southeast": ['Carolina Hurricanes (CAR)', 'Florida Panthers (FLA)', 'Tampa Bay Lightning (TBL)', 'Washington Capitals (WSH)', 'Winnipeg Jets (WPG)'], 
#     "Central": ['Chicago Blackhawks (CHI)', 'Columbus Blue Jackets (CBJ)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)'],
#     "Northwest": ['Calgary Flames (CGY)', 'Colorado Avalanche (COL)', 'Edmonton Oilers (EDM)', 'Minnesota Wild (MIN)', 'Vancouver Canucks (VAN)'],
#     "Pacific": ['Anaheim Ducks (ANA)', 'Dallas Stars (DAL)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'San Jose Sharks (SJS)']
# }

# # Realignment from 6 to 4 divisions
# # Unused from data set: 'Atlanta Thrashers (ATL)', 'Arizona Coyotes (ARI)', 'Vegas Golden Knights (VGK)', 
# divisions_2013 = {
#     # Eastern Conference
#     "Atlantic": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Tampa Bay Lightning (TBL)', 'Toronto Maple Leafs (TOR)'],
#     "Metropolitan": ['Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Washington Capitals (WSH)'],

#     # Western Conference
#     "Central": ['Chicago Blackhawks (CHI)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Minnesota Wild (MIN)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Winnipeg Jets (WPG)'],
#     "Pacific": ['Anaheim Ducks (ANA)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'San Jose Sharks (SJS)', 'Vancouver Canucks (VAN)']
# }

# # Phoenix Coyotes change name to Arizona Coyotes
# # Unused from data set: 'Atlanta Thrashers (ATL)', 'Vegas Golden Knights (VGK)', 'Phoenix Coyotes (PHX)', 
# divisions_2014_2016 = {
#     # Eastern Conference
#     "Atlantic": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Tampa Bay Lightning (TBL)', 'Toronto Maple Leafs (TOR)'],
#     "Metropolitan": ['Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Washington Capitals (WSH)'],

#     # Western Conference
#     "Central": ['Chicago Blackhawks (CHI)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Minnesota Wild (MIN)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Winnipeg Jets (WPG)'],
#     "Pacific": ['Anaheim Ducks (ANA)', 'Arizona Coyotes (ARI)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'San Jose Sharks (SJS)', 'Vancouver Canucks (VAN)']
# }

# # New team Vegas Golden Knights. 31 teams.
# # Unused from data set: 'Atlanta Thrashers (ATL)', 'Phoenix Coyotes (PHX)', 
# divisions_2017_2019 = {
#     # Eastern Conference
#     "Atlantic": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Tampa Bay Lightning (TBL)', 'Toronto Maple Leafs (TOR)'],
#     "Metropolitan": ['Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Washington Capitals (WSH)'],

#     # Western Conference
#     "Central": ['Chicago Blackhawks (CHI)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Minnesota Wild (MIN)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Winnipeg Jets (WPG)'],
#     "Pacific": ['Anaheim Ducks (ANA)', 'Arizona Coyotes (ARI)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'San Jose Sharks (SJS)', 'Vancouver Canucks (VAN)', 'Vegas Golden Knights (VGK)']
# }