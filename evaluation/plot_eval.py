"""
This script plots evaluation graphs from a saved evaluation_summary.json
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
SUMMARY_PATH = os.path.join(
    ROOT_DIR, "results", "eval_summary.json"
)
PLOT_DIR = os.path.join(
    ROOT_DIR, "results", "plots"
)

to_plot = ["Random", "DQN Custom", "SB3 DQN", "DQN Double", "DQN Double+PER"]

AGENT_COLORS = {
    "DQN Custom": "#9C003C",
    "SB3 DQN":     "#FFA8A8",
    "DQN Double":                   "#975053",
    "DQN PER":                      "#BA919B",
    "DQN Double+PER":               "#F5405E",
}


def load_data():
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"File not found: {SUMMARY_PATH}")
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data(raw_data):
    global_records = []
    seed_records = []

    for agent in raw_data:
        if agent["name"] not in to_plot:
            continue

        global_records.append({
            "Agent":            agent["name"],
            "Reward":           agent["mean_reward"],
            "Std Reward":       agent["std_reward"],
            "SE Reward":        agent.get("se_reward", agent["std_reward"] / np.sqrt(max(agent.get("num_episodes", 1) * len(agent.get("seeds", [1])), 1))),
            "Success Rate (%)": agent["success_rate"] * 100,
            "Episode Length":   agent["mean_length"],
            "SE Length":        agent.get("se_length", 0.0),
            "Speed":            agent["mean_speed"],
        })

        for s in agent.get("per_seed", []):
            seed_records.append({
                "Agent":            agent["name"],
                "Seed":             str(s["seed"]),
                "Reward":           s["mean_reward"],
                "SE Reward":        s.get("se_reward", 0.0),
                "Success Rate (%)": s["success_rate"] * 100,
            })

    return pd.DataFrame(global_records), pd.DataFrame(seed_records)


def build_palette(agents):
    default_colors = sns.color_palette("tab10", n_colors=10).as_hex()
    palette = {}
    for i, agent in enumerate(agents):
        if agent in AGENT_COLORS:
            palette[agent] = AGENT_COLORS[agent]
        else:
            palette[agent] = default_colors[i % len(default_colors)]
            print(f"[Warning] No color defined for '{agent}', using default.")
    return palette


def split_baseline(df_global):
    df_random = df_global[df_global["Agent"] == "Random"]
    df_trained = df_global[df_global["Agent"] != "Random"]

    baselines = {}
    if not df_random.empty:
        row = df_random.iloc[0]
        baselines = {
            "Reward":           row["Reward"],
            "Success Rate (%)": row["Success Rate (%)"],
            "Episode Length":   row["Episode Length"],
            "Speed":            row["Speed"],
        }

    palette = build_palette(df_trained["Agent"].unique())
    return df_trained, baselines, palette


# ---------------------------------------------------------------------------
# Existing plots — enhanced with SE error bars
# ---------------------------------------------------------------------------

def plot_global_metrics(df_trained, baselines, palette):
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Agent Comparison (3×50 episodes)", fontsize=24, fontweight="bold", y=1.15)

    metrics = [
        ("Reward",           "SE Reward",  "Mean Reward",       axes[0]),
        ("Success Rate (%)", None,         "Survival Rate (%)", axes[1]),
        ("Episode Length",   "SE Length",  "Survival (Steps)",  axes[2]),
        ("Speed",            None,         "Mean Speed",        axes[3]),
    ]

    for metric, se_col, title, ax in metrics:
        bar_container = sns.barplot(
            data=df_trained, x="Agent", y=metric, ax=ax, palette=palette
        )

        if se_col and se_col in df_trained.columns:
            agents_ordered = [p.get_x() + p.get_width() / 2 for p in ax.patches]
            for patch, (_, row) in zip(ax.patches, df_trained.iterrows()):
                cx = patch.get_x() + patch.get_width() / 2
                cy = patch.get_height()
                se = row[se_col]
                ax.errorbar(cx, cy, yerr=se, fmt="none", color="black",
                            capsize=5, capthick=1.5, linewidth=1.5, zorder=5)

        ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)

        if metric in baselines:
            ax.axhline(baselines[metric], color="red", linestyle="--", linewidth=2)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=2))
    labels.append("Baseline (Random)")

    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels), fontsize=14, frameon=False,
    )

    axes[1].set_ylim(0, 110)
    axes[2].set_ylim(0, 30)

    plt.savefig(os.path.join(PLOT_DIR, "global_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_seed_stability(df_seed, baselines, palette):
    df_trained = df_seed[df_seed["Agent"] != "Random"]
    df_random = df_seed[df_seed["Agent"] == "Random"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Agent Stability per Seed", fontsize=16, fontweight="bold")

    for ax, metric in zip(axes, ["Reward", "Success Rate (%)"]):
        sns.barplot(
            data=df_trained, x="Seed", y=metric, hue="Agent", ax=ax, palette=palette
        )

        if "SE Reward" in df_trained.columns and metric == "Reward":
            for i, (_, row) in enumerate(df_trained.iterrows()):
                patches = [p for p in ax.patches if not np.isnan(p.get_height())]
                if i < len(patches):
                    cx = patches[i].get_x() + patches[i].get_width() / 2
                    cy = patches[i].get_height()
                    ax.errorbar(cx, cy, yerr=row["SE Reward"], fmt="none",
                                color="black", capsize=4, linewidth=1.2, zorder=5)

        if not df_random.empty:
            sns.lineplot(
                data=df_random, x="Seed", y=metric, ax=ax,
                color="red", linestyle="--", marker="o",
                markersize=8, linewidth=2, label="Baseline (Random)",
            )
        ax.legend()
        ax.set_title(f"{'Mean Reward' if metric == 'Reward' else metric} per Seed", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "stability_per_seed.png"), dpi=300)
    plt.close()


def plot_speed_vs_safety(df_trained, baselines, palette):
    sns.set_theme(style="darkgrid", context="talk")
    plt.figure(figsize=(12, 7))

    ax = sns.scatterplot(
        data=df_trained, x="Speed", y="Success Rate (%)",
        hue="Agent", size="Reward",
        sizes=(200, 800), palette=palette,
        alpha=0.9, edgecolor="black",
    )

    handles, labels = ax.get_legend_handles_labels()
    agent_handles = [(h, l) for h, l in zip(handles, labels) if l in palette]
    ax.legend(
        *zip(*agent_handles),
        title="Agents", bbox_to_anchor=(1.05, 1),
        loc="upper left", borderaxespad=0.0,
    )

    if "Speed" in baselines and "Success Rate (%)" in baselines:
        plt.axvline(baselines["Speed"], color="gray", linestyle=":", alpha=0.7)
        plt.axhline(baselines["Success Rate (%)"], color="gray", linestyle=":", alpha=0.7)
        plt.text(
            baselines["Speed"], baselines["Success Rate (%)"] + 2,
            " Random level", color="red", fontstyle="italic", fontsize=12,
        )

    plt.title("Speed / Safety Trade-off", fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("Mean Speed", fontsize=16)
    plt.ylabel("Survival Rate (%)", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "speed_vs_safety.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# New plots
# ---------------------------------------------------------------------------

def plot_reward_distribution(raw_data, palette):
    """
    Violin + swarm plot of per-episode rewards for each agent.
    Reveals whether the reward distribution is bimodal (crash vs. survive),
    unimodal, or has heavy tails — information that the mean alone hides.
    """
    records = []
    for agent in raw_data:
        if agent["name"] not in to_plot or agent["name"] == "Random":
            continue
        for r in agent.get("raw_rewards", []):
            records.append({"Agent": agent["name"], "Reward": r})

    if not records:
        return

    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    violin = sns.violinplot(
        data=df, x="Agent", y="Reward",
        palette=palette, inner=None,
        cut=0, linewidth=1.5, ax=ax,
    )
    sns.stripplot(
        data=df, x="Agent", y="Reward",
        color="black", alpha=0.15, size=3, jitter=True, ax=ax,
    )

    # overlay mean ± SE
    for i, agent_name in enumerate(df["Agent"].unique()):
        vals = df.loc[df["Agent"] == agent_name, "Reward"].values
        mu, se = vals.mean(), vals.std() / np.sqrt(len(vals))
        ax.errorbar(i, mu, yerr=se, fmt="D", color="white",
                    markeredgecolor="black", capsize=6, capthick=2,
                    linewidth=2, markersize=8, zorder=10, label="mean ± SE" if i == 0 else "")

    ax.set_title("Reward Distribution per Agent", fontsize=18, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Episode Reward", fontsize=14)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "reward_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_speed_distribution(raw_data, palette):
    """
    KDE + histogram of per-step speeds for each agent.
    Shows whether the agent prefers a particular speed band, how much
    it varies, and whether it ever hits the reward_speed_range [22, 30].
    """
    REWARD_SPEED_LOW, REWARD_SPEED_HIGH = 22, 30

    records = []
    for agent in raw_data:
        if agent["name"] not in to_plot:
            continue
        for sp in agent.get("raw_speeds", []):
            records.append({"Agent": agent["name"], "Speed": sp})

    if not records:
        print("[Warning] No raw_speeds in summary — re-run evaluate_agents.py first.")
        return

    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid", context="talk")
    agents = [a for a in to_plot if a in df["Agent"].unique()]
    n_agents = len(agents)

    fig, axes = plt.subplots(1, n_agents, figsize=(6 * n_agents, 5), sharey=False)
    if n_agents == 1:
        axes = [axes]

    fig.suptitle("Per-Step Speed Distribution", fontsize=18, fontweight="bold")

    for ax, agent_name in zip(axes, agents):
        color = palette.get(agent_name, "steelblue") if agent_name != "Random" else "red"
        speeds = df.loc[df["Agent"] == agent_name, "Speed"].values

        ax.hist(speeds, bins=30, density=True, alpha=0.35, color=color, edgecolor="white")
        kde_x = np.linspace(speeds.min() - 1, speeds.max() + 1, 300)
        kde = stats.gaussian_kde(speeds, bw_method="scott")
        ax.plot(kde_x, kde(kde_x), color=color, linewidth=2.5)

        ax.axvspan(REWARD_SPEED_LOW, REWARD_SPEED_HIGH, alpha=0.12,
                   color="green", label="reward speed range")
        ax.axvline(speeds.mean(), color=color, linestyle="--",
                   linewidth=1.5, label=f"mean={speeds.mean():.1f}")

        ax.set_title(agent_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Speed (m/s)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "speed_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_survival_curve(raw_data, palette):
    """
    Empirical survival function: P(episode length > t) for each agent.
    A drop at early steps = early crashes.  A flat curve = the agent
    mostly survives to the time limit.
    """
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent in raw_data:
        if agent["name"] not in to_plot:
            continue

        lengths = np.array(agent.get("raw_lengths", []))
        crashed = np.array(agent.get("raw_crashed", []))
        if len(lengths) == 0:
            continue

        t_max = lengths.max()
        t_values = np.arange(0, t_max + 1)
        # P(length > t)  — only episodes that actually crashed contribute
        # to the "death" at their length; surviving episodes are right-censored
        survival = np.array([np.mean(lengths > t) for t in t_values])

        color = palette.get(agent["name"], "steelblue") if agent["name"] != "Random" else "red"
        ls = "--" if agent["name"] == "Random" else "-"
        ax.plot(t_values, survival * 100, color=color, linewidth=2,
                linestyle=ls, label=agent["name"])

    ax.set_title("Survival Curve — P(Episode Length > t)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Step t", fontsize=14)
    ax.set_ylabel("% Episodes Still Running", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "survival_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_crash_step_distribution(raw_data, palette):
    """
    Box + strip plot of the step at which each crash occurred.
    Only includes crashed episodes.  Reveals whether crashes are
    clustered early (bad policy) or spread late (unlucky encounters).
    """
    records = []
    for agent in raw_data:
        if agent["name"] not in to_plot:
            continue
        for cs in agent.get("raw_crash_steps", []):
            records.append({"Agent": agent["name"], "Crash Step": cs})

    if not records:
        return

    df = pd.DataFrame(records)

    # percentage of episodes that crashed (for annotation)
    crash_rates = {}
    for agent in raw_data:
        if agent["name"] not in to_plot:
            continue
        crashed = agent.get("raw_crashed", [])
        if crashed:
            crash_rates[agent["name"]] = np.mean(crashed) * 100

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    trained_agents = [a for a in df["Agent"].unique() if a != "Random"]
    df_plot = df[df["Agent"].isin(trained_agents)]
    plot_palette = {a: palette.get(a, "steelblue") for a in trained_agents}

    sns.boxplot(data=df_plot, x="Agent", y="Crash Step",
                palette=plot_palette, width=0.5, ax=ax)
    sns.stripplot(data=df_plot, x="Agent", y="Crash Step",
                  palette=plot_palette, alpha=0.4, size=4, jitter=True, ax=ax)

    for i, agent_name in enumerate(trained_agents):
        rate = crash_rates.get(agent_name, 0)
        ax.text(i, ax.get_ylim()[1] * 0.97,
                f"crash\nrate\n{rate:.0f}%",
                ha="center", va="top", fontsize=10, color="dimgray")

    ax.set_title("Crash-Step Distribution (crashed episodes only)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Step at Crash", fontsize=14)
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "crash_step_distribution.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_reward_per_episode(raw_data, palette):
    """
    Rolling mean (window=10) of the reward over the 50 evaluation episodes,
    aggregated across seeds.  Shows whether performance drifts or is stable
    throughout the evaluation window — a proxy for policy consistency.
    """
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 5))

    for agent in raw_data:
        if agent["name"] not in to_plot or agent["name"] == "Random":
            continue

        rewards = np.array(agent.get("raw_rewards", []))
        if len(rewards) == 0:
            continue

        color = palette.get(agent["name"], "steelblue")
        ax.plot(rewards, alpha=0.2, color=color, linewidth=0.8)

        window = 10
        if len(rewards) >= window:
            rolled = pd.Series(rewards).rolling(window, center=True).mean()
            ax.plot(rolled, color=color, linewidth=2.5, label=agent["name"])

    ax.set_title("Reward per Episode (rolling mean, window=10)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Episode index (across all seeds)", fontsize=13)
    ax.set_ylabel("Episode Reward", fontsize=13)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "reward_per_episode.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    raw_data = load_data()
    df_global, df_seed = prepare_data(raw_data)
    df_trained, baselines, palette = split_baseline(df_global)

    # existing plots (now with SE bars)
    plot_global_metrics(df_trained, baselines, palette)
    plot_per_seed_stability(df_seed, baselines, palette)
    plot_speed_vs_safety(df_trained, baselines, palette)

    # new behavioural comparison plots
    plot_reward_distribution(raw_data, palette)
    plot_speed_distribution(raw_data, palette)
    plot_survival_curve(raw_data, palette)
    plot_crash_step_distribution(raw_data, palette)
    plot_reward_per_episode(raw_data, palette)

    print(f"Plots saved to: {PLOT_DIR}/")