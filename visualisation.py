import matplotlib.pyplot as plt
import pandas as pd

def visualise_cumulative_reward(experiments, environment, skip_first_rows=0):
    exp_dfs = []
    cols = dict()
    for exp in experiments:
        exp_df = pd.read_csv("experiments/" + exp[0] + "/timesteps.csv")
        exp_dfs.append(exp_df)
        cols[exp[1]] = exp_df["cumulative_reward"]
    df = pd.DataFrame(cols).iloc[skip_first_rows:]

    plt.figure(dpi=300)
    df.plot()
    plt.title(environment + ': cumulative reward in time')
    plt.xlabel("Time step")
    plt.ylabel("Cumulative reward")
    plt.show()

def visualise_episode_reward(experiments, environment, skip_first_rows=0):
    exp_dfs = []
    cols = dict()
    for exp in experiments:
        exp_df = pd.read_csv("experiments/" + exp[0] + "/episodes.csv")
        exp_dfs.append(exp_df)
        cols[exp[1]] = exp_df["reward"]
    df = pd.DataFrame(cols).iloc[skip_first_rows:]

    plt.figure(dpi=300)
    df.plot()
    plt.title(environment + ': episode reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

def visualise_episode_duration(experiments, environment, skip_first_rows=0):
    exp_dfs = []
    cols = dict()
    for exp in experiments:
        exp_df = pd.read_csv("experiments/" + exp[0] + "/episodes.csv")
        exp_dfs.append(exp_df)
        cols[exp[1]] = exp_df["duration"]
    df = pd.DataFrame(cols).iloc[skip_first_rows:]

    plt.figure(dpi=300)
    df.plot()
    plt.title(environment + ': episode duration')
    plt.xlabel("Episode")
    plt.ylabel("Time steps")
    plt.show()
