# -*- coding: utf-8 -*-

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

env = gym.make('FrozenLake-v0')


# helper variable, functions and classes
class Strategy():

    def __init__(self, pick_action, update_strategy):
        self.pick_action = pick_action
        self.update_strategy = update_strategy

def rand_pick_action(current_state, current_game):
    return env.action_space.sample()

def rand_update_strategy(current_state, action, next_state, reward, game_end):
    pass

rand_strategy = Strategy(rand_pick_action, rand_update_strategy)


def add_state_row_column(df, state_col_names):
    for state_col_name in state_col_names:
        df[state_col_name + "_row"] = df[state_col_name].map(lambda x: int(x / 4))
        df[state_col_name + "_col"] = df[state_col_name].map(lambda x: x % 4)

def run_n_games(num_games, strategy, current_info_dict = False):
    states_info_dict = {
      "current_game": [],
      "current_game_action_num": [],
      "current_state": [],
      "action": [],
      "next_state": [],
      "reward": [],
      "end_game": []
    } if current_info_dict == False else current_info_dict

    prev_games_played = 0 if current_info_dict == False else len(set(current_info_dict["current_game"]))

    # run the games
    for game_num in tqdm(range(1, num_games + 1)):
        env.reset()
        game_end = False
        current_state = 0
        current_game_action_num = 1
        while not game_end:
            # move
            action = strategy.pick_action(current_state, game_num + prev_games_played)
            next_state, reward, game_end, _ = env.step(action)
            strategy.update_strategy(current_state, action, next_state, reward, game_end)

            # add info to dict
            states_info_dict["current_game"].append(game_num + prev_games_played)
            states_info_dict["current_game_action_num"].append(current_game_action_num)
            states_info_dict["current_state"].append(current_state)
            states_info_dict["action"].append(action)
            states_info_dict["next_state"].append(next_state)
            states_info_dict["reward"].append(reward)
            states_info_dict["end_game"].append(game_end)

            # update info
            current_state = next_state
            current_game_action_num += 1

    game_df = pd.DataFrame(states_info_dict)
    add_state_row_column(game_df, ["current_state", "next_state"])
    return states_info_dict, game_df

def print_game_stats(game_df):
    num_games = max(game_df.current_game)

    # total number of moves?
    print("Total number of moves: {}\nNumber of moves per game: {}".format(
            len(game_df),
            1.0 * len(game_df) / num_games))

    # winning percentage?
    num_games_won = game_df[game_df.end_game].reward.sum()
    print("Total number of games won: {}\nPercentage of games won: {}".format(num_games_won, 100.0 * num_games_won / num_games))

def plot_state_dist(game_df):

    fig, ax = plt.subplots(1, 2, figsize = (20, 6))

    sns.distplot(game_df.current_state, kde = False, bins = None, ax = ax[0], hist_kws = {"alpha": 1})

    sns.distplot(game_df.next_state, kde = False, bins = None, ax = ax[1], hist_kws = {"alpha": 1})

def plot_winning_percentage(game_df, gap = 1000):

    games_df = game_df.groupby("current_game", as_index = False).sum()

    games_df["cum_games_won"] = games_df.reward.cumsum()

    games_df["winning_percentage"] = games_df.apply(lambda row: 100.0 * row.cum_games_won / row.current_game, axis = 1)

    games_plot_df = games_df[games_df.current_game % gap == 0]

    fig, ax = plt.subplots(figsize = (20, 6))

    ax.plot(games_plot_df.current_game, games_plot_df.winning_percentage)

def plot_game_length(game_df, window = 10):
    game_length_rolling_avg = game_df.groupby("current_game").count()[["action"]].rolling(window).mean()

    fig, ax = plt.subplots(figsize = (20, 6))

    ax.plot(game_length_rolling_avg.index, game_length_rolling_avg.action)
