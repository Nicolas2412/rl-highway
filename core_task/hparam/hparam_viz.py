"""
Visualisation des résultats de la recherche d'hyperparamètres.

Usage :
    python core_task/hparam/hparam_viz.py --study-db hparam_results/optuna_study.db
    python core_task/hparam/hparam_viz.py --study-db hparam_results/optuna_study.db --save

    rl-highway\core_task\hparam\hparam_viz.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import optuna


STUDY_NAME  = "highway_dqn"
RESULTS_DIR = "hparam_results"

CONTINUOUS_PARAMS  = ["lr", "gamma", "eps_decay", "target_upd"]
CATEGORICAL_PARAMS = ["batch_size", "buffer_cap", "hidden", "n_layers", "double_dqn"]

# ── Palette bordeaux ──────────────────────────────────────────────────────────
PRIMARY   = "#9C003C"   # rgba(156, 0, 60)
SECONDARY = "#C8476E"   # rose moyen
LIGHT     = "#E8A0B4"   # rose clair
DARK      = "#5C0020"   # bordeaux foncé

CMAP_MONO = LinearSegmentedColormap.from_list("bordeaux", ["#FAE0EA", PRIMARY, DARK])

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       "#EEEEEE",
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "legend.framealpha": 0.9,
})


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_study(db_path: str) -> optuna.Study:
    storage = f"sqlite:///{db_path}"
    return optuna.load_study(study_name=STUDY_NAME, storage=storage)


def get_completed(study: optuna.Study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def get_pruned(study: optuna.Study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]


def _new_slide_fig(title: str):
    """Figure 16:9 avec fond blanc et titre centré."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=15, fontweight="bold", color=DARK, y=0.97)
    return fig


# ─── Figure 1 : Historique + Importances ─────────────────────────────────────

def plot_optimization_history(study: optuna.Study, ax: plt.Axes) -> None:
    completed = get_completed(study)
    if not completed:
        ax.set_title("No completed trial")
        return

    trial_nums  = [t.number for t in completed]
    values      = [t.value  for t in completed]
    best_so_far = np.maximum.accumulate(values)

    pruned = get_pruned(study)
    if pruned:
        pruned_x, pruned_y = [], []
        for t in pruned:
            if t.intermediate_values:
                pruned_x.append(t.number)
                pruned_y.append(t.intermediate_values[max(t.intermediate_values)])
        if pruned_x:
            ax.scatter(pruned_x, pruned_y,
                       marker="x", color=LIGHT, s=50,
                       label="Pruned (last value)", zorder=3)

    ax.scatter(trial_nums, values, color=SECONDARY, s=35, alpha=0.75,
               label="Completed trial", zorder=4)
    ax.plot(trial_nums, best_so_far, color=PRIMARY, linewidth=2.5,
            label="Cumulative best")

    best_idx = int(np.argmax(values))
    ax.axvline(trial_nums[best_idx], color=DARK, linestyle="--", alpha=0.6, linewidth=1.5)
    ax.scatter([trial_nums[best_idx]], [values[best_idx]],
               color=DARK, s=120, zorder=5, edgecolors="white", linewidths=1.5,
               label=f"Best (#{completed[best_idx].number})")

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Average reward (last 20 episodes)")
    ax.set_title("Optimization history")
    ax.legend()


def plot_param_importances(study: optuna.Study, ax: plt.Axes) -> None:
    completed = get_completed(study)
    if len(completed) < 5:
        ax.set_title("Not enough trials (min 5)")
        return

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        ax.set_title(f"Importance error: {e}")
        return

    params = list(importances.keys())
    values = list(importances.values())
    max_v  = max(values) if values else 1.0

    norm_vals = [v / max_v for v in values]
    colors = [plt.matplotlib.colors.to_hex(CMAP_MONO(n)) for n in norm_vals]

    bars = ax.barh(params[::-1], values[::-1], color=colors[::-1], edgecolor="white",
                   linewidth=0.5, height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10, color=DARK)
    ax.set_xlabel("Relative importance (fANOVA)")
    ax.set_title("Hyperparameter importances")
    ax.set_xlim(0, max_v * 1.25)


def make_slide1(study: optuna.Study) -> plt.Figure:
    fig = _new_slide_fig("Hyperparameter Search — Overview")
    ax_hist = fig.add_subplot(1, 2, 1)
    ax_imp  = fig.add_subplot(1, 2, 2)
    plot_optimization_history(study, ax_hist)
    plot_param_importances(study, ax_imp)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─── Figure 2 : Scatter paramètres continus ───────────────────────────────────

def make_slide2(study: optuna.Study) -> plt.Figure:
    fig = _new_slide_fig("Continuous Parameters vs Reward")
    completed = get_completed(study)
    if not completed:
        return fig

    values = np.array([t.value for t in completed])
    norm   = plt.Normalize(values.min(), values.max())

    for i, param in enumerate(CONTINUOUS_PARAMS):
        ax = fig.add_subplot(2, 2, i + 1)

        param_vals, trial_vals = [], []
        for t in completed:
            if param in t.params:
                param_vals.append(t.params[param])
                trial_vals.append(t.value)

        if not param_vals:
            ax.set_visible(False)
            continue

        sc = ax.scatter(param_vals, trial_vals,
                        c=trial_vals, cmap=CMAP_MONO, norm=norm,
                        s=50, alpha=0.85, edgecolors="white", linewidths=0.5)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Reward", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        if param == "lr":
            ax.set_xscale("log")

        if len(param_vals) > 3:
            x  = np.log10(param_vals) if param == "lr" else np.array(param_vals)
            z  = np.polyfit(x, trial_vals, 1)
            p  = np.poly1d(z)
            xs = np.linspace(min(x), max(x), 100)
            xs_orig = 10**xs if param == "lr" else xs
            ax.plot(xs_orig, p(xs), color=DARK, linestyle="--", alpha=0.7,
                    linewidth=1.5, label="Trend")
            ax.legend(fontsize=9)

        ax.set_xlabel(param)
        ax.set_ylabel("Reward")
        ax.set_title(f"Reward vs {param}")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─── Figure 3 : Boxplots catégoriels + meilleure config ───────────────────────

def make_slide3(study: optuna.Study) -> plt.Figure:
    fig = _new_slide_fig("Categorical Parameters & Best Configuration")
    completed = get_completed(study)
    if not completed:
        return fig

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    n_cat = min(3, len(CATEGORICAL_PARAMS))
    positions = [(0, 0), (0, 1), (0, 2)]

    for pos, param in zip(positions[:n_cat], CATEGORICAL_PARAMS[:n_cat]):
        ax = fig.add_subplot(gs[pos])
        groups: dict[str, list[float]] = {}
        for t in completed:
            if param in t.params:
                key = str(t.params[param])
                groups.setdefault(key, []).append(t.value)

        if not groups:
            ax.set_visible(False)
            continue

        labels  = sorted(groups.keys(), key=lambda x: (x == "False", x == "True", x))
        data    = [groups[k] for k in labels]
        medians = [np.median(d) for d in data]
        max_med = max(medians) if max(medians) > 0 else 1.0
        norm_med = np.array(medians) / max_med
        colors   = [plt.matplotlib.colors.to_hex(CMAP_MONO(n)) for n in norm_med]

        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                        medianprops=dict(color=DARK, linewidth=2),
                        whiskerprops=dict(color="#888888"),
                        capprops=dict(color="#888888"),
                        flierprops=dict(marker="o", markerfacecolor=LIGHT,
                                        markersize=4, alpha=0.6))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
            patch.set_edgecolor(DARK)

        ax.set_xlabel(param)
        ax.set_ylabel("Reward")
        ax.set_title(f"Distribution by {param}")

    ax_best = fig.add_subplot(gs[1, :])
    ax_best.axis("off")

    best = max(completed, key=lambda t: t.value)
    rows = [
        [k, f"{v:.5g}" if isinstance(v, float) else str(v)]
        for k, v in sorted(best.params.items())
    ]

    table = ax_best.table(
        cellText=rows,
        colLabels=["Hyperparameter", "Optimal value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(0.6, 1.8)

    for j in range(2):
        table[(0, j)].set_facecolor(PRIMARY)
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor("#FDF0F4" if i % 2 == 0 else "white")
            table[(i, j)].set_text_props(color="#333333")

    ax_best.set_title(
        f"Best trial #{best.number} — R̄ = {best.value:.4f}",
        fontsize=12, fontweight="bold", color=PRIMARY, pad=10,
    )

    return fig


# ─── Entrée principale ────────────────────────────────────────────────────────

def visualize(db_path: str, save: bool = False, out_dir: str = RESULTS_DIR) -> None:
    study     = load_study(db_path)
    completed = get_completed(study)
    pruned    = get_pruned(study)
    print(f"Trials complétés : {len(completed)} | Élagués : {len(pruned)}")

    if not completed:
        print("Aucun trial complété — rien à visualiser.")
        return

    slides = [
        ("slide1_overview",          make_slide1(study)),
        ("slide2_continuous_params", make_slide2(study)),
        ("slide3_categorical_best",  make_slide3(study)),
    ]

    if save:
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in slides:
            path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Sauvegardé → {path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-db", type=str,
        default=f"{RESULTS_DIR}/optuna_study.db",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Sauvegarde les figures en PNG (une par slide)",
    )
    args = parser.parse_args()
    visualize(args.study_db, save=args.save)