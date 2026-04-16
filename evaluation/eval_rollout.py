"""
eval_rollout.py  --  State-conditional action analysis for Highway-v0 agents.

Two figures are produced:

  1. action_by_speed.png
       Stacked-bar chart: action distribution conditioned on ego speed.
       Answers: does the agent accelerate when slow and brake when fast?

  2. action_by_headway.png
       Stacked-bar chart: action distribution conditioned on the normalised
       relative-x distance to the nearest detected vehicle ahead.
       Answers: does the agent brake / lane-change when a vehicle is close?

"""

import os
import sys
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from evaluation.run_eval import _load_agent, _make_env

import gymnasium as gym
import highway_env  # noqa: F401  registers the envs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

ACTION_COLORS = {
    "LANE_LEFT":  "#4C72B0",
    "IDLE":       "#8172B3",
    "LANE_RIGHT": "#64B5CD",
    "FASTER":     "#55A868",
    "SLOWER":     "#C44E52",
}

AGENT_COLORS = {
    "DQN Custom": "#9C003C",
    "SB3 DQN":     "#FFA8A8",
    "DQN Double":                   "#975053",
    "DQN PER":                      "#BA919B",
    "DQN Double+PER":               "#F5405E",
}

EVAL_REGISTRY = [
    {
        "name":       "Random",
        "agent_type": "random",
        "checkpoint": None,
    },
    {
        "name":       "DQN Custom",
        "agent_type": "dqn_custom",
        "checkpoint": "checkpoints/dqn_custom_20260413-082750/model_dqn_custom.pt",
    },
    {
        "name":       "SB3 DQN",
        "agent_type": "sb3",
        "checkpoint": "checkpoints/sb3_dqn/model_dqn_sb3.zip",
    },
    {
        "name":       "DQN Double",
        "agent_type": "dqn_custom",
        "checkpoint": "checkpoints/dqn_20260411-135652/20260413-063222_dqn_highway_final.pt",
        "double_dqn": True,
    },
    # {
    #     "name":       "DQN PER",
    #     "agent_type": "dqn_per",
    #     "checkpoint": "checkpoints/per_dqn_20260411-191026/20260412-021940_per_dqn_final.pt",
    # },
    # {
    #     "name":       "DQN Double+PER",
    #     "agent_type": "dqn_per",
    #     "checkpoint": "checkpoints/20260412-084516_per_double_dqn/20260412-084516_per_double_dqn_final.pt",
    #     "double_dqn": True,
    # },
]

N_EPISODES   = 50
SEED         = 9
N_SPEED_BINS = 5
N_HEAD_BINS  = 5
MIN_SAMPLES  = 10   # bins with fewer steps are dropped (unreliable)

PLOT_DIR = os.path.join(
    ROOT_DIR, "results", "rollout", "plots"
)

# Observation feature indices -- must match SHARED_CORE_CONFIG features order
_FEAT_PRESENCE = 0
_FEAT_X        = 1
_FEAT_VX       = 3

_ROW_EGO     = 0
_ROW_NEAREST = 1


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _run_rollout_episode(agent, env: gym.Env, seed: int | None) -> list[dict]:
    """
    Run one episode and return a list of per-step records.

    The observation used to derive state features is the one BEFORE the step,
    i.e. the state that actually caused the action to be chosen.

    Each record:
        action    (int)   -- discrete action taken
        speed     (float) -- raw ego speed from info["speed"]
        rel_x     (float) -- normalised relative-x of nearest vehicle;
                             NaN when presence == 0
        rel_vx    (float) -- normalised relative-vx of nearest vehicle;
                             NaN when presence == 0
        presence  (float) -- obs[1, 0] presence flag
    """
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    records: list[dict] = []

    while not (done or truncated):
        action = agent.act(obs, epsilon=0.0)

        nearest_presence = float(obs[_ROW_NEAREST, _FEAT_PRESENCE])
        detected         = nearest_presence > 0.5
        rel_x  = float(obs[_ROW_NEAREST, _FEAT_X])  if detected else float("nan")
        rel_vx = float(obs[_ROW_NEAREST, _FEAT_VX]) if detected else float("nan")

        obs, _reward, done, truncated, info = env.step(action)

        records.append({
            "action":   int(action),
            "speed":    float(info.get("speed", float("nan"))),
            "rel_x":    rel_x,
            "rel_vx":   rel_vx,
            "presence": nearest_presence,
        })

    return records


def collect_rollouts(entry: dict, n_episodes: int, seed: int) -> pd.DataFrame:
    """
    Run `n_episodes` for the agent described by `entry` and return a
    DataFrame with one row per environment step.
    """
    env   = _make_env()
    agent = _load_agent(entry, env)

    all_records: list[dict] = []
    with tqdm(total=n_episodes, desc=entry["name"], unit="ep") as pbar:
        for ep_idx in range(n_episodes):
            ep_seed = seed if ep_idx == 0 else None
            ep_records = _run_rollout_episode(agent, env, seed=ep_seed)
            all_records.extend(ep_records)
            pbar.update(1)
            pbar.set_postfix(steps=len(all_records))

    env.close()
    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Shared computation helper
# ---------------------------------------------------------------------------

def _action_dist_by_bin(
    df: pd.DataFrame,
    feature_col: str,
    n_bins: int,
) -> pd.DataFrame:
    """
    Bin `feature_col` into `n_bins` equal-frequency quantile bins,
    compute the per-bin action proportion matrix, and drop bins whose
    sample count is below MIN_SAMPLES.

    Returns a DataFrame:
        index   = readable bin label strings  (e.g. "22.1-24.5")
        columns = action name strings  (subset of ACTION_NAMES.values())
    """
    clean = df.dropna(subset=[feature_col]).copy()
    if clean.empty:
        return pd.DataFrame()

    clean["_bin"] = pd.qcut(clean[feature_col], q=n_bins, duplicates="drop")

    counts = (
        clean.groupby(["_bin", "action"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    counts = counts[counts.sum(axis=1) >= MIN_SAMPLES]
    if counts.empty:
        return pd.DataFrame()

    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions = proportions.rename(columns=ACTION_NAMES)

    # enforce consistent column ordering
    ordered = [ACTION_NAMES[i] for i in sorted(ACTION_NAMES) if ACTION_NAMES[i] in proportions.columns]
    proportions = proportions[ordered]

    proportions.index = [
        f"{iv.left:.1f}-{iv.right:.1f}" for iv in proportions.index
    ]
    return proportions


# ---------------------------------------------------------------------------
# Shared drawing helper
# ---------------------------------------------------------------------------

def _draw_stacked_bar(ax: plt.Axes, dist: pd.DataFrame, title: str, xlabel: str) -> None:
    """
    Draw a stacked bar chart of action proportions onto `ax`.
    Each bar = one bin, stack segments = actions with consistent colours.
    """
    colors = [ACTION_COLORS[col] for col in dist.columns]
    dist.plot(
        kind="bar", stacked=True, ax=ax,
        color=colors, edgecolor="white", linewidth=0.5,
        legend=False, width=0.75,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Action proportion", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)


def _shared_legend(fig: plt.Figure) -> None:
    patches = [
        plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[a], label=a)
        for a in ACTION_NAMES.values()
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(ACTION_NAMES),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )


# ---------------------------------------------------------------------------
# Figure 1 -- Action distribution conditioned on ego speed
# ---------------------------------------------------------------------------

def plot_action_by_speed(dfs: dict[str, pd.DataFrame]) -> None:
    """
    For each agent: bin ego speed (from info["speed"]) into N_SPEED_BINS
    equal-frequency quantile bins, then plot the stacked action proportions.

    Hypothesis to test: a well-trained agent should predominantly choose
    FASTER at low speeds (target speed not yet reached) and SLOWER / lane
    changes at high speeds (risk of collision increases).
    """
    agents = list(dfs.keys())
    n      = len(agents)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Action distribution conditioned on ego speed",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, name in zip(axes, agents):
        dist = _action_dist_by_bin(dfs[name], "speed", N_SPEED_BINS)
        if dist.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(name, fontsize=11)
            continue
        _draw_stacked_bar(ax, dist, title=name, xlabel="Ego speed bin (m/s)")

    _shared_legend(fig)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "action_by_speed.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 -- Action distribution conditioned on headway to nearest vehicle
# ---------------------------------------------------------------------------

def plot_action_by_headway(dfs: dict[str, pd.DataFrame]) -> None:
    """
    For each agent: restrict to timesteps where a vehicle was detected ahead
    (presence=1, rel_x > 0), bin the normalised relative-x into N_HEAD_BINS
    equal-frequency quantile bins, then plot stacked action proportions.

    The leftmost bin = closest vehicle (highest collision risk).
    The rightmost bin = farthest detected vehicle (low immediate risk).

    Hypothesis to test: a safe agent should show elevated SLOWER / lane-change
    rates when a vehicle is close (left bins) and higher FASTER / IDLE rates
    when the road ahead is clear (right bins).
    """
    agents = list(dfs.keys())
    n      = len(agents)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Action distribution conditioned on headway to nearest vehicle\n"
        "(steps where a vehicle was detected ahead only)",
        fontsize=13, fontweight="bold", y=1.04,
    )

    for ax, name in zip(axes, agents):
        df_ahead = dfs[name][(dfs[name]["presence"] > 0.5) & (dfs[name]["rel_x"] > 0)].copy()
        dist = _action_dist_by_bin(df_ahead, "rel_x", N_HEAD_BINS)

        if dist.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(name, fontsize=11)
            continue

        _draw_stacked_bar(
            ax, dist,
            title=name,
            xlabel="Normalised relative distance to nearest vehicle\n(left = close,  right = far)",
        )

    _shared_legend(fig)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "action_by_headway.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}

    for entry in EVAL_REGISTRY:
        name = entry["name"]
        dfs[name] = collect_rollouts(entry, n_episodes=N_EPISODES, seed=SEED)
        print(f"  -> {len(dfs[name])} steps collected")

    plot_action_by_speed(dfs)
    plot_action_by_headway(dfs)

    print(f"\nAll plots saved to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()