import pandas as pd
import csv

# QUESTIONS
# Incorporate stdevs?

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

# IN PROGRESS
# Utilize updated teams' seasons' win totals to predict the 16 teams that make it to the playoffs each season
# Handle teams new teams plus division & conference changes across the seasons
# Seasons in data set: 2003, 2005-2006, 2008-2018
def get_playoff_teams(wins):
    # In 2006, Mighty Ducks of Anaheim changed their name to the Anaheim Ducks, but the data uses the same label so no change needed.
    # Unused from data set: 'Winnipeg Jets (WPG)', 'Arizona Coyotes (ARI)', 'Vegas Golden Knights (VGK)'
    divisions_2001_2010 = {
        "Atlantic": ['New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)'],
        "Northeast": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)'],
        "Southeast": ['Atlanta Thrashers (ATL)', 'Carolina Hurricanes (CAR)', 'Florida Panthers (FLA)', 'Tampa Bay Lightning (TBL)', 'Washington Capitals (WSH)'], 
        "Central": ['Chicago Blackhawks (CHI)', 'Columbus Blue Jackets (CBJ)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)'],
        "Northwest": ['Calgary Flames (CGY)', 'Colorado Avalanche (COL)', 'Edmonton Oilers (EDM)', 'Minnesota Wild (MIN)', 'Vancouver Canucks (VAN)'],
        "Pacific": ['Anaheim Ducks (ANA)', 'Dallas Stars (DAL)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'San Jose Sharks (SJS)']
    }

    # Atlanta Thrashers moved to Canada and became Winnipeg Jets
    # Unused from data set: 'Atlanta Thrashers (ATL)', 'Arizona Coyotes (ARI)', 'Vegas Golden Knights (VGK)'
    divisions_2011_2012 = {
        "Atlantic": ['New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)'],
        "Northeast": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Toronto Maple Leafs (TOR)'],
        "Southeast": ['Carolina Hurricanes (CAR)', 'Florida Panthers (FLA)', 'Tampa Bay Lightning (TBL)', 'Washington Capitals (WSH)', 'Winnipeg Jets (WPG)'], 
        "Central": ['Chicago Blackhawks (CHI)', 'Columbus Blue Jackets (CBJ)', 'Detroit Red Wings (DET)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)'],
        "Northwest": ['Calgary Flames (CGY)', 'Colorado Avalanche (COL)', 'Edmonton Oilers (EDM)', 'Minnesota Wild (MIN)', 'Vancouver Canucks (VAN)'],
        "Pacific": ['Anaheim Ducks (ANA)', 'Dallas Stars (DAL)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'San Jose Sharks (SJS)']
    }

    # Realignment from 6 to 4 divisions
    # Unused from data set: 'Atlanta Thrashers (ATL)', 'Arizona Coyotes (ARI)', 'Vegas Golden Knights (VGK)', 
    divisions_2013 = {
        # Eastern Conference
        "Atlantic": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Tampa Bay Lightning (TBL)', 'Toronto Maple Leafs (TOR)'],
        "Metropolitan": ['Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Washington Capitals (WSH)'],

        # Western Conference
        "Central": ['Chicago Blackhawks (CHI)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Minnesota Wild (MIN)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Winnipeg Jets (WPG)'],
        "Pacific": ['Anaheim Ducks (ANA)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'Phoenix Coyotes (PHX)', 'San Jose Sharks (SJS)', 'Vancouver Canucks (VAN)']
    }

    # Phoenix Coyotes change name to Arizona Coyotes
    # Unused from data set: 'Atlanta Thrashers (ATL)', 'Vegas Golden Knights (VGK)', 'Phoenix Coyotes (PHX)', 
    divisions_2014_2016 = {
        # Eastern Conference
        "Atlantic": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Tampa Bay Lightning (TBL)', 'Toronto Maple Leafs (TOR)'],
        "Metropolitan": ['Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Washington Capitals (WSH)'],

        # Western Conference
        "Central": ['Chicago Blackhawks (CHI)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Minnesota Wild (MIN)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Winnipeg Jets (WPG)'],
        "Pacific": ['Anaheim Ducks (ANA)', 'Arizona Coyotes (ARI)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'San Jose Sharks (SJS)', 'Vancouver Canucks (VAN)']
    }

    # New team Vegas Golden Knights
    # Unused from data set: 'Atlanta Thrashers (ATL)', 'Phoenix Coyotes (PHX)', 
    divisions_2017_2019 = {
        # Eastern Conference
        "Atlantic": ['Boston Bruins (BOS)', 'Buffalo Sabres (BUF)', 'Detroit Red Wings (DET)', 'Florida Panthers (FLA)', 'Montreal Canadiens (MTL)', 'Ottawa Senators (OTT)', 'Tampa Bay Lightning (TBL)', 'Toronto Maple Leafs (TOR)'],
        "Metropolitan": ['Carolina Hurricanes (CAR)', 'Columbus Blue Jackets (CBJ)', 'New Jersey Devils (NJD)', 'NY Islanders Islanders (NYI)', 'NY Rangers Rangers (NYR)', 'Philadelphia Flyers (PHI)', 'Pittsburgh Penguins (PIT)', 'Washington Capitals (WSH)'],

        # Western Conference
        "Central": ['Chicago Blackhawks (CHI)', 'Colorado Avalanche (COL)', 'Dallas Stars (DAL)', 'Minnesota Wild (MIN)', 'Nashville Predators (NSH)', 'St Louis Blues (STL)', 'Winnipeg Jets (WPG)'],
        "Pacific": ['Anaheim Ducks (ANA)', 'Arizona Coyotes (ARI)', 'Calgary Flames (CGY)', 'Edmonton Oilers (EDM)', 'Los Angeles Kings (LAK)', 'San Jose Sharks (SJS)', 'Vancouver Canucks (VAN)', 'Vegas Golden Knights (VGK)']
    }

    # Take top 3 in each division plus top 2 by points in conference
    atlantic_2001_2010, northeast_2001_2010, southeast_2001_2010, central_2001_2010, northwest_2001_2010, pacific_2001_2010 = {}, {}, {}, {}
    atlantic_2011_2012, northeast_2011_2012, southeast_2011_2012, central_2011_2012, northwest_2011_2012, pacific_2011_2012 = {}, {}, {}, {}
    atlantic_2013, metropolitan_2013, central_2013, pacific_2013 = {}, {}, {}, {}
    atlantic_2014_2016, metropolitan_2014_2016, central_2014_2016, pacific_2014_2016 = {}, {}, {}, {}
    atlantic_2017_2019, metropolitan_2017_2019, central_2017_2019, pacific_2017_2019 = {}, {}, {}, {}

    _2017_2019 = []
    for index, row in wins.iterrows():
        season, team_name, season_wins = row['Season'], row['Team'], row['Total Wins']
        if season >= 2003 and season <= 2010:
            if team_name in divisions_2001_2010['Atlantic']:
                atlantic_2001_2010[team_name] = season_wins
            elif team_name in divisions_2001_2010['Northeast']:
                northeast_2001_2010[team_name] = season_wins
            elif team_name in divisions_2001_2010['Southeast']:
                southeast_2001_2010[team_name] = season_wins
            elif team_name in divisions_2001_2010['Central']:
                central_2001_2010[team_name] = season_wins
            elif team_name in divisions_2001_2010['Northwest']:
                northwest_2001_2010[team_name] = season_wins
            elif team_name in divisions_2001_2010['Pacific']:
                pacific_2001_2010[team_name] = season_wins
        if season == 2011 and season == 2012:
            if team_name in divisions_2011_2012['Atlantic']:
                atlantic_2011_2012[team_name] = season_wins
            elif team_name in divisions_2011_2012['Northeast']:
                northeast_2011_2012[team_name] = season_wins
            elif team_name in divisions_2011_2012['Southeast']:
                southeast_2011_2012[team_name] = season_wins
            elif team_name in divisions_2011_2012['Central']:
                central_2011_2012[team_name] = season_wins
            elif team_name in divisions_2011_2012['Northwest']:
                northwest_2011_2012[team_name] = season_wins
            elif team_name in divisions_2011_2012['Pacific']:
                pacific_2011_2012[team_name] = season_wins
        if season == 2013:
            if team_name in divisions_2013['Atlantic']:
                atlantic_2013[team_name] = season_wins
            elif team_name in divisions_2013['Metropolitan']:
                metropolitan_2013[team_name] = season_wins
            elif team_name in divisions_2013['Central']:
                central_2013[team_name] = season_wins
            elif team_name in divisions_2013['Pacific']:
                pacific_2013[team_name] = season_wins
        if season >= 2014 and season <= 2016:
            if team_name in divisions_2014_2016['Atlantic']:
                atlantic_2014_2016[team_name] = season_wins
            elif team_name in divisions_2014_2016['Metropolitan']:
                metropolitan_2014_2016[team_name] = season_wins
            elif team_name in divisions_2014_2016['Central']:
                central_2014_2016[team_name] = season_wins
            elif team_name in divisions_2014_2016['Pacific']:
                pacific_2014_2016[team_name] = season_wins
        if season >= 2017 and season <= 2019:
            if team_name in divisions_2017_2019['Atlantic']:
                atlantic_2017_2019[team_name] = season_wins
            elif team_name in divisions_2017_2019['Metropolitan']:
                metropolitan_2017_2019[team_name] = season_wins
            elif team_name in divisions_2017_2019['Central']:
                central_2017_2019[team_name] = season_wins
            elif team_name in divisions_2017_2019['Pacific']:
                pacific_2017_2019[team_name] = season_wins

    # Sort by value largest first. List of tuples?
    atlantic_2001_2010.sort()
    northeast_2001_2010.sort()
    central_2001_2010.sort()
    pacific_2001_2010.sort()
    
    atlantic_2011_2012.sort()
    northeast_2011_2012.sort()
    central_2011_2012.sort()
    pacific_2011_2012.sort()
    
    atlantic_2013.sort()
    metropolitan_2013.sort()
    central_2013.sort()
    pacific_2013.sort()
    
    atlantic_2014_2016.sort()
    metropolitan_2014_2016.sort()
    central_2014_2016.sort()
    pacific_2014_2016.sort()
    
    atlantic_2017_2019.sort()
    metropolitan_2017_2019.sort()
    central_2017_2019.sort()
    pacific_2017_2019.sort()

    # Playoff system  with 6 divisions
    
    atlantic_2013_top3 = atlantic_2013[0:3]
    metropolitan_2013_top3 = metropolitan_2013[0:3]
    central_2013_top3 = central_2013[0:3]
    pacific_2013_top3 = pacific_2013[0:3]
    east_2013_wild_card_teams = (atlantic_2013[3:5] + metropolitan_2013[3:5]).sort()[0:2]
    west_2013_wild_card_teams = (central_2013[3:5] + pacific_2013[3:5]).sort()[0:2]

    atlantic_2014_2016_top3 = atlantic_2014_2016[0:3]
    metropolitan_2014_2016_top3 = metropolitan_2014_2016[0:3]
    central_2014_2016_top3 = central_2014_2016[0:3]
    pacific_2014_2016_top3 = pacific_2014_2016[0:3]
    east_2014_2016_wild_card_teams = (atlantic_2014_2016[3:5] + metropolitan_2014_2016[3:5]).sort()[0:2]
    west_2014_2016_wild_card_teams = (central_2014_2016[3:5] + pacific_2014_2016[3:5]).sort()[0:2]

    atlantic_2017_2019_top3 = atlantic_2017_2019[0:3]
    metropolitan_2017_2019_top3 = metropolitan_2017_2019[0:3]
    central_2017_2019_top3 = central_2017_2019[0:3]
    pacific_2017_2019_top3 = pacific_2017_2019[0:3]
    east_2017_2019_wild_card_teams = (atlantic_2017_2019[3:5] + metropolitan_2017_2019[3:5]).sort()[0:2]
    west_2017_2019_wild_card_teams = (central_2017_2019[3:5] + pacific_2017_2019[3:5]).sort()[0:2]



    # team_name_to_id = {
    #     'Atlanta Thrashers (ATL)': ,
    #     'Phoenix Coyotes (PHX)': ,
    #     'Boston Bruins (BOS)': ,
    #     'Buffalo Sabres (BUF)': ,
    #     'Detroit Red Wings (DET)': ,
    #     'Florida Panthers (FLA)': ,
    #     'Montreal Canadiens (MTL)': ,
    #     'Ottawa Senators (OTT)': ,
    #     'Tampa Bay Lightning (TBL)': ,
    #     'Toronto Maple Leafs (TOR)']: ,
    #     'Carolina Hurricanes (CAR)': ,
    #     'Columbus Blue Jackets (CBJ)': ,
    #     'New Jersey Devils (NJD)': ,
    #     'NY Islanders Islanders (NYI)': ,
    #     'NY Rangers Rangers (NYR)': ,
    #     'Philadelphia Flyers (PHI)': ,
    #     'Pittsburgh Penguins (PIT)': ,
    #     'Washington Capitals (WSH)': ,
    #     'Chicago Blackhawks (CHI)': ,
    #     'Colorado Avalanche (COL)': ,
    #     'Dallas Stars (DAL)': ,
    #     'Minnesota Wild (MIN)': ,
    #     'Nashville Predators (NSH)': ,
    #     'St Louis Blues (STL)': ,
    #     'Winnipeg Jets (WPG)': ,
    #     'Anaheim Ducks (ANA)': ,
    #     'Arizona Coyotes (ARI)': ,
    #     'Calgary Flames (CGY)': ,
    #     'Edmonton Oilers (EDM)': ,
    #     'Los Angeles Kings (LAK)': ,
    #     'San Jose Sharks (SJS)': ,
    #     'Vancouver Canucks (VAN)': ,
    #     'Vegas Golden Knights (VGK)': 
    # }


# Incorporate stdevs?

# teams_seasons_stdevs = pd.read_csv("cleaned_data_v6/Logistic Regression - GamebyGame/stat_generation_avg.csv")
# opposing_teams_seasons_stdevs = pd.read_csv("cleaned_data_v6/Logistic Regression - GamebyGame/stat_generation_avg_op.csv")

# # Get home team's stdev stats for the season
# team_seasons_stdevs = teams_seasons_stdevs.loc[teams_seasons_stdevs['team_id'] == home_team_id]
# team_season_stdevs = team_seasons_stdevs.loc[team_seasons_stdevs['Season'] == season]

# # Get opposing teams' stdev stats for the seasonn against the home team
# opposing_team_seasons_stdevs = opposing_teams_seasons_stdevs.loc[opposing_teams_seasons_stdevs['team_id'] == home_team_id]
# opposing_team_season_stdevs = opposing_team_seasons_stdevs.loc[opposing_team_seasons_stdevs['Season'] == season]