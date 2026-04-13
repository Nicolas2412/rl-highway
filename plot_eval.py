import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

SUMMARY_PATH = "results/eval_summary.json"
PLOT_DIR = "results/plots/extension/all"

to_plot = ["Random", "DQN Custom", "SB3 DQN", "DQN Double", "DQN Double+PER"]

AGENT_COLORS = {
    "DQN Custom":     "#9C003C",
    "SB3 DQN":        "#FFA8A8",
    "DQN Double":     "#975053",
    "DQN PER":        "#BA919B",
    "DQN Double+PER": "#F5405E",
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
            "Success Rate (%)": agent["success_rate"] * 100,
            "Episode Length":   agent["mean_length"],
            "Speed":            agent["mean_speed"],
        })

        for s in agent.get("per_seed", []):
            seed_records.append({
                "Agent":            agent["name"],
                "Seed":             str(s["seed"]),
                "Reward":           s["mean_reward"],
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


def plot_global_metrics(df_trained, baselines, palette):
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Agent Comparison (3x50 episodes)",
                fontsize=24, fontweight="bold", y=1.15)

    metrics = [
        ("Reward",           "Mean Reward",       axes[0]),
        ("Success Rate (%)", "Survival Rate (%)", axes[1]),
        ("Episode Length",   "Survival (Steps)",  axes[2]),
        ("Speed",            "Mean Speed",        axes[3]),
    ]

    for metric, title, ax in metrics:
        sns.barplot(data=df_trained, x="Agent",
                    y=metric, ax=ax, palette=palette)
        ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)

        if metric in baselines:
            ax.axhline(baselines[metric], color="red",
                    linestyle="--", linewidth=2)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=2))
    labels.append("Baseline (Random)")

    fig.legend(handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.05),
            ncol=len(labels), fontsize=14, frameon=False)

    axes[1].set_ylim(0, 110)
    axes[2].set_ylim(0, 30)

    plt.savefig(os.path.join(PLOT_DIR, "global_metrics.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_seed_stability(df_seed, baselines, palette):
    df_trained = df_seed[df_seed["Agent"] != "Random"]
    df_random = df_seed[df_seed["Agent"] == "Random"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Agent Stability per Seed", fontsize=16, fontweight="bold")

    sns.barplot(data=df_trained, x="Seed", y="Reward",
                hue="Agent", ax=axes[0], palette=palette)
    axes[0].set_title("Mean Reward per Seed", fontsize=16)

    if not df_random.empty:
        sns.lineplot(data=df_random, x="Seed", y="Reward", ax=axes[0],
                    color="red", linestyle="--", marker="o",
                    markersize=8, linewidth=2, label="Baseline (Random)")
        axes[0].legend()

    sns.barplot(data=df_trained, x="Seed", y="Success Rate (%)",
                hue="Agent", ax=axes[1], palette=palette)
    axes[1].set_title("Success Rate per Seed", fontsize=16)

    if not df_random.empty:
        sns.lineplot(data=df_random, x="Seed", y="Success Rate (%)", ax=axes[1],
                    color="red", linestyle="--", marker="o",
                    markersize=8, linewidth=2, label="Baseline (Random)")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "stability_per_seed.png"), dpi=300)
    plt.close()


def plot_speed_vs_safety(df_trained, baselines, palette):
    sns.set_theme(style="darkgrid", context="talk")
    plt.figure(figsize=(12, 7))

    ax = sns.scatterplot(
        data=df_trained,
        x="Speed",
        y="Success Rate (%)",
        hue="Agent",
        size="Reward",
        sizes=(200, 800),
        palette=palette,
        alpha=0.9,
        edgecolor="black",
    )

    handles, labels = ax.get_legend_handles_labels()
    agent_handles = [(h, l) for h, l in zip(handles, labels) if l in palette]
    ax.legend(
        *zip(*agent_handles),
        title="Agents",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    if "Speed" in baselines and "Success Rate (%)" in baselines:
        plt.axvline(baselines["Speed"], color="gray", linestyle=":", alpha=0.7)
        plt.axhline(baselines["Success Rate (%)"],
                    color="gray", linestyle=":", alpha=0.7)
        plt.text(baselines["Speed"], baselines["Success Rate (%)"] + 2,
                " Random level", color="red", fontstyle="italic", fontsize=12)

    plt.title("Speed / Safety Trade-off",
            fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("Mean Speed", fontsize=16)
    plt.ylabel("Survival Rate (%)", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "speed_vs_safety.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    raw_data = load_data()
    df_global, df_seed = prepare_data(raw_data)
    df_trained, baselines, palette = split_baseline(df_global)

    plot_global_metrics(df_trained, baselines, palette)
    plot_per_seed_stability(df_seed, baselines, palette)
    plot_speed_vs_safety(df_trained, baselines, palette)

    print(f"Plots saved to: {PLOT_DIR}/")
