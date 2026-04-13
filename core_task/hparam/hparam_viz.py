"""
Visualisation des résultats de la recherche d'hyperparamètres.

Usage :
    python hparam_viz.py --study-db core_task/hparam_results/optuna_study.db
    python hparam_viz.py --study-db core_task/hparam_results/optuna_study.db --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import optuna


STUDY_NAME  = "highway_dqn"
RESULTS_DIR = "core_task/hparam_results"

# Noms de paramètres tels qu'enregistrés par l'objectif dans hparam_search.py
CONTINUOUS_PARAMS  = ["lr", "gamma", "eps_decay", "target_upd"]

CATEGORICAL_PARAMS = ["batch_size", "buffer_cap", "hidden", "n_layers", "double_dqn"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_study(db_path: str) -> optuna.Study:
    storage = f"sqlite:///{db_path}"
    return optuna.load_study(study_name=STUDY_NAME, storage=storage)


def get_completed(study: optuna.Study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def get_pruned(study: optuna.Study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]


# ─── Figures ─────────────────────────────────────────────────────────────────

def plot_optimization_history(study: optuna.Study, ax: plt.Axes) -> None:
    """Valeur objective par trial + meilleur cumulatif."""
    completed = get_completed(study)
    if not completed:
        ax.set_title("Aucun trial complété")
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
                       marker="x", color="lightcoral", s=40,
                       label="Élagué (dernière valeur)", zorder=3)

    ax.scatter(trial_nums, values, color="steelblue", s=30, alpha=0.7,
               label="Trial complété", zorder=4)
    ax.plot(trial_nums, best_so_far, color="darkorange", linewidth=2,
            label="Meilleur cumulatif")

    best_idx = int(np.argmax(values))
    ax.axvline(trial_nums[best_idx], color="green", linestyle="--", alpha=0.5)
    ax.scatter([trial_nums[best_idx]], [values[best_idx]],
               color="green", s=100, zorder=5,
               label=f"Meilleur (#{completed[best_idx].number})")

    ax.set_xlabel("Numéro de trial")
    ax.set_ylabel("Récompense moyenne (20 derniers épisodes)")
    ax.set_title("Historique d'optimisation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_param_importances(study: optuna.Study, ax: plt.Axes) -> None:
    """Importance des hyperparamètres via fANOVA (calculée par Optuna)."""
    completed = get_completed(study)
    if len(completed) < 5:
        ax.set_title("Pas assez de trials pour estimer les importances (min 5)")
        return

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        ax.set_title(f"Erreur importance : {e}")
        return

    params = list(importances.keys())
    values = list(importances.values())
    max_v  = max(values) if values else 1.0
    colors = cm.RdYlGn([v / max_v for v in values])

    bars = ax.barh(params[::-1], values[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Importance relative (fANOVA)")
    ax.set_title("Importance des hyperparamètres")
    ax.set_xlim(0, max_v * 1.2)
    ax.grid(axis="x", alpha=0.3)


def plot_scatter_matrix(study: optuna.Study, axes) -> None:
    """Scatter plots : chaque paramètre continu vs récompense."""
    completed = get_completed(study)
    if not completed:
        return

    values = np.array([t.value for t in completed])
    norm   = plt.Normalize(values.min(), values.max())
    cmap   = cm.viridis

    for ax, param in zip(axes, CONTINUOUS_PARAMS):
        param_vals, trial_vals = [], []
        for t in completed:
            if param in t.params:
                param_vals.append(t.params[param])
                trial_vals.append(t.value)

        if not param_vals:
            ax.set_visible(False)
            continue

        sc = ax.scatter(param_vals, trial_vals,
                        c=trial_vals, cmap=cmap, norm=norm,
                        s=40, alpha=0.8, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Récompense", pad=0.02)

        if param == "lr":
            ax.set_xscale("log")

        if len(param_vals) > 3:
            x  = np.log10(param_vals) if param == "lr" else np.array(param_vals)
            z  = np.polyfit(x, trial_vals, 1)
            p  = np.poly1d(z)
            xs = np.linspace(min(x), max(x), 100)
            xs_orig = 10**xs if param == "lr" else xs
            ax.plot(xs_orig, p(xs), "r--", alpha=0.6, linewidth=1)

        ax.set_xlabel(param)
        ax.set_ylabel("Récompense")
        ax.set_title(f"Récompense vs {param}")
        ax.grid(alpha=0.3)


def plot_categorical_boxplots(study: optuna.Study, axes) -> None:
    """Boxplots par valeur de paramètre catégoriel."""
    completed = get_completed(study)
    if not completed:
        return

    for ax, param in zip(axes, CATEGORICAL_PARAMS):
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
        max_med = max(medians)

        norm_medians = (
            np.array(medians) / max_med if max_med > 0
            else np.ones(len(medians))
        )
        colors = cm.RdYlGn(norm_medians)

        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_xlabel(param)
        ax.set_ylabel("Récompense")
        ax.set_title(f"Distribution par {param}")
        ax.grid(axis="y", alpha=0.3)


def plot_best_config(study: optuna.Study, ax: plt.Axes) -> None:
    """Affiche la configuration du meilleur trial sous forme de tableau."""
    ax.axis("off")
    completed = get_completed(study)
    if not completed:
        ax.text(0.5, 0.5, "Aucun trial complété", ha="center", va="center")
        return

    best = max(completed, key=lambda t: t.value)
    rows = [
        [k, f"{v:.5g}" if isinstance(v, float) else str(v)]
        for k, v in sorted(best.params.items())
    ]

    table = ax.table(
        cellText=rows,
        colLabels=["Hyperparamètre", "Valeur optimale"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for j in range(2):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Meilleur trial #{best.number} — R̄ = {best.value:.4f}",
        fontsize=11, fontweight="bold", pad=12,
    )


# ─── Entrée principale ────────────────────────────────────────────────────────

def visualize(db_path: str, save: bool = False, out_dir: str = RESULTS_DIR) -> None:
    study     = load_study(db_path)
    completed = get_completed(study)
    pruned    = get_pruned(study)
    print(f"Trials complétés : {len(completed)} | Élagués : {len(pruned)}")

    if not completed:
        print("Aucun trial complété — rien à visualiser.")
        return

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        "Résultats de la recherche d'hyperparamètres — DQN Highway",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Ligne 1 : historique + importances
    ax_hist = fig.add_subplot(4, 2, 1)
    ax_imp  = fig.add_subplot(4, 2, 2)
    plot_optimization_history(study, ax_hist)
    plot_param_importances(study, ax_imp)

    # Lignes 2-3 : scatter params continus (2×2)
    scatter_axes = [fig.add_subplot(4, 2, 3 + i) for i in range(len(CONTINUOUS_PARAMS))]
    plot_scatter_matrix(study, scatter_axes)

    # Ligne 4 : boxplots catégoriels (3 premiers) + meilleure config
    n_cat_shown = min(3, len(CATEGORICAL_PARAMS))
    cat_axes    = [fig.add_subplot(4, 4, 13 + i) for i in range(n_cat_shown)]
    plot_categorical_boxplots(study, cat_axes)

    ax_best = fig.add_subplot(4, 4, 16)
    plot_best_config(study, ax_best)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "hparam_results.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée → {path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-db", type=str,
        default=f"{RESULTS_DIR}/optuna_study.db",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Sauvegarde la figure en PNG",
    )
    args = parser.parse_args()
    visualize(args.study_db, save=args.save)