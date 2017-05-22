import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time


def visualise_results(experiments, file_name, column_name, title, xlabel, ylabel, skip_first_rows=0, show_bounds=True, bounds_alpha=0.15):
    exp_dfs = []
    cols = dict()
    cmap = matplotlib.cm.get_cmap("brg")
    cmap2 = matplotlib.cm.get_cmap("brg")

    for exp in experiments:
        if len(exp) == 2:
            exp_df = pd.read_csv("experiments/" + exp[1] + "/" + file_name)
            exp_dfs.append(None)
            cols[exp[0]] = exp_df[column_name]
        else:
            exp_results = pd.DataFrame()
            for e in exp[1:]:
                e_df = pd.read_csv("experiments/" + e + "/" + file_name)
                exp_results[e] = e_df[column_name]
            cols[exp[0]] = exp_results[list(exp[1:])].mean(axis=1)
            exp_results["lower"] = exp_results[list(exp[1:])].min(axis=1)
            exp_results["upper"] = exp_results[list(exp[1:])].max(axis=1)
            exp_dfs.append(exp_results)

    df = pd.DataFrame(cols).iloc[skip_first_rows:]
    axes = df.plot(colormap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show_bounds:
        for exp_id, exp in enumerate(experiments):
            exp_results = exp_dfs[exp_id]
            if exp_results is not None:
                axes.fill_between(exp_results.index, exp_results.upper, exp_results.lower, where=exp_results.upper > exp_results.lower, facecolor=cmap2(exp_id / float(len(experiments)-1)), alpha=bounds_alpha, interpolate=True)

    plt.savefig("figures/figure_" + str(int(time.time())) + ".png", dpi=400, bbox_inches='tight')
    plt.show()


def visualise_cumulative_reward(experiments, environment, skip_first_rows=0, show_bounds=True, bounds_alpha=0.15):
    visualise_results(experiments, "timesteps.csv", "cumulative_reward", environment + ": cumulative reward in time", "Time step", "Cumulative reward", skip_first_rows, show_bounds, bounds_alpha)


def visualise_episode_reward(experiments, environment, skip_first_rows=0, show_bounds=True, bounds_alpha=0.15):
    visualise_results(experiments, "episodes.csv", "reward", environment + ": episode reward", "Episode", "Reward", skip_first_rows, show_bounds, bounds_alpha)


def visualise_episode_duration(experiments, environment, skip_first_rows=0, show_bounds=True, bounds_alpha=0.15):
    visualise_results(experiments, "episodes.csv", "duration", environment + ": episode duration", "Episode", "Time steps", skip_first_rows, show_bounds, bounds_alpha)
