import pandas as pd

class Team(object):
    def __init__(self, name):
        self.name = name
        self.wins = 0.0
        self.home_wins = 0.0
        self.visitor_wins = 0.0
        self.point_differential = 0.0
        self.games = 0.0
        self.visitor_games = 0.0
        self.home_games = 0.0

    def update_features(self, win, home, point_dif):
        """ Updates all of this teams feature stats based on the last game they
        played """
        #new game played
        self.games += 1
        self.visitor_games = self.visitor_games + 1 if not home else self.visitor_games
        self.home_games = self.home_games + 1 if home else self.home_games
        #wins are increased if won
        self.wins = self.wins + 1 if win else self.wins
        #home wins increased if won and home game
        self.home_wins = self.home_wins + 1 if (win and home) else self.home_wins
        #visitor wins increased if won and not a home game
        self.visitor_wins = self.visitor_wins + 1 if (win and not home) else self.visitor_wins
        #point differential average is calculated by multiplying old PD by
        #games - 1 and adding current game PD and dividing by total number of games
        self.point_differential = (point_dif + self.point_differential * (self.games - 1))/self.games

    def get_WL(self):
        """ Gets this teams win-loss percentage """
        return self.wins/self.games

    def get_PD(self):
        """ Gets this teams point differential """
        return self.point_differential

    def get_VWL(self):
        """ Gets this teams win-loss percentage as a visitor """
        return self.visitor_wins / self.visitor_games

    def get_HWL(self):
        """ Gets this teams home game win-loss percentage """
        return self.home_wins / self.home_games

def parse_data(teams, filename):
    """ Reads CSV file and populates the features and outcomes lists and builds
    the teams dictionary for future prediction based on team stats """
    features = [] #contains a 6 element feature vector for each game
    outcomes = [] #contains a 1 or 0 for each game, 1 for home-team win, else 0
    box_scores = pd.read_csv(filename)

    for row in box_scores.itertuples():
        visitor = row[1]
        visitor_score = float(row[2])
        home = row[3]
        home_score = float(row[4])
        visitor_win = visitor_score > home_score
        home_win = not visitor_win

        #add and update team features
        if visitor not in teams:
            teams[visitor] = Team(visitor)
        teams[visitor].update_features(visitor_win, False, visitor_score - home_score)
        if home not in teams:
            teams[home] = Team(home)
        teams[home].update_features(home_win, True, home_score - visitor_score)

        #build feature vector
        v = teams[visitor]
        h = teams[home]
        vector = [v.get_WL(), h.get_WL(), v.get_PD(), h.get_PD(), v.get_VWL(), h.get_HWL()]

        features.append(vector)
        outcomes.append([0 if home_win else 1,1 if home_win else 0])
    return (features, outcomes)
