import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
SUMMARY_PATH = "results/eval_summary.json"
# PLOT_DIR = "results/plots/core"
# to_plot = ["Random", "DQN Custom", "SB3 DQN"]
# PLOT_DIR = "results/plots/extension/double"
# to_plot = ["Random", "DQN Custom", "SB3 DQN", "DQN Double"]
# PLOT_DIR = "results/plots/extension/per"
# to_plot = ["Random", "DQN Custom", "SB3 DQN", "DQN Double", "DQN PER"]
PLOT_DIR = "results/plots/extension/all"
to_plot = ["Random", "DQN Custom", "SB3 DQN", "DQN Double", "DQN Double+PER"]

AGENT_COLORS = {
    "DQN Custom": "#9C003C",
    "SB3 DQN":    "#FFA8A8",
    "DQN Double": "#975053",
    "DQN PER":    "#BA919B",
    "DQN Double+PER": "#F5405E"
}



def load_data():
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"Fichier introuvable : {SUMMARY_PATH}")
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data(raw_data):
    global_records = []
    seed_records = []

    for agent in raw_data:
        if agent["name"] in to_plot:
            global_records.append({
                "Agent": agent["name"],
                "Reward": agent["mean_reward"],
                "Std Reward": agent["std_reward"],
                "Success Rate (%)": agent["success_rate"] * 100,
                "Episode Length": agent["mean_length"],
                "Speed": agent["mean_speed"]
            })

            if "per_seed" in agent:
                for s in agent["per_seed"]:
                    seed_records.append({
                        "Agent": agent["name"],
                        "Seed": str(s["seed"]),
                        "Reward": s["mean_reward"],
                        "Success Rate (%)": s["success_rate"] * 100
                    })

    return pd.DataFrame(global_records), pd.DataFrame(seed_records)


def get_baselines_and_palette(df_global):
    """Extrait les métriques du Random et génère la palette sur-mesure."""
    df_random = df_global[df_global["Agent"] == "Random"]
    df_trained = df_global[df_global["Agent"] != "Random"]

    # Dictionnaire des baselines
    baselines = {}
    if not df_random.empty:
        baselines = {
            "Reward": df_random["Reward"].values[0],
            "Success Rate (%)": df_random["Success Rate (%)"].values[0],
            "Episode Length": df_random["Episode Length"].values[0],
            "Speed": df_random["Speed"].values[0],
        }

    # Création de la palette stricte basée sur ton dictionnaire AGENT_COLORS
    agents = df_trained["Agent"].unique()
    agent_palette = {}
    default_colors = sns.color_palette("tab10", n_colors=10).as_hex()

    for i, agent in enumerate(agents):
        if agent in AGENT_COLORS:
            agent_palette[agent] = AGENT_COLORS[agent]
        else:
            # Sécurité : couleur par défaut si tu as oublié de le mettre dans AGENT_COLORS
            agent_palette[agent] = default_colors[i % len(default_colors)]
            print(
                f"[Attention] Pas de couleur définie pour '{agent}', utilisation par défaut.")

    return df_trained, baselines, agent_palette


def plot_global_metrics(df_trained, baselines, agent_palette):
    sns.set_theme(style="whitegrid", context="talk")

    # On garde le format panoramique
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Comparaison Globale des Agents (sur 3x50 épisodes)",
                 # Augmenté y pour laisser de la place à la légende
                 fontsize=24, fontweight='bold', y=1.15)

    metrics = [
        ("Reward", "Récompense Moyenne", axes[0]),
        ("Success Rate (%)", "Taux de Survie (%)", axes[1]),
        ("Episode Length", "Survie (Steps)", axes[2]),
        ("Speed", "Vitesse Moyenne", axes[3])
    ]

    for metric, title, ax in metrics:
        sns.barplot(data=df_trained, x="Agent", y=metric,
                    ax=ax, palette=agent_palette)

        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("")  # Nettoyage pour éviter la redondance avec le titre
        ax.tick_params(axis='x', rotation=25)

        # Ajout de la ligne de baseline (sans label ici pour ne pas polluer)
        if metric in baselines:
            ax.axhline(baselines[metric], color='red',
                       linestyle='--', linewidth=2)

    # --- LÉGENDE AU DESSUS ---
    # On récupère les handles d'un seul graphique (ils sont identiques pour tous)
    handles, labels = axes[0].get_legend_handles_labels()

    # On ajoute manuellement la ligne de la baseline à la légende si besoin
    from matplotlib.lines import Line2D
    line = Line2D([0], [0], color='red', linestyle='--',
                  linewidth=2, label='Baseline (Random)')
    handles.append(line)
    labels.append('Baseline (Random)')

    fig.legend(
        handles,
        labels,
        loc='upper center',
        # Centre x=0.5, au-dessus de la figure y=1.05
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),           # Met tous les éléments sur une ligne
        fontsize=14,
        frameon=False               # Enlever le cadre pour plus de légèreté
    )

    axes[1].set_ylim(0, 110)
    axes[2].set_ylim(0, 30)

    # On utilise bbox_inches='tight' car la légende dépasse de la zone standard
    plt.savefig(os.path.join(PLOT_DIR, "global_metrics.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_seed_stability(df_seed, baselines, agent_palette):
    # On sépare les agents entraînés de la baseline Random
    df_seed_trained = df_seed[df_seed["Agent"] != "Random"]
    df_seed_random = df_seed[df_seed["Agent"] == "Random"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Stabilité des Agents par Seed",
                 fontsize=16, fontweight='bold')

    # --- 1. Graphique Reward ---
    sns.barplot(data=df_seed_trained, x="Seed", y="Reward",
                hue="Agent", ax=axes[0], palette=agent_palette)
    axes[0].set_title("Récompense moyenne selon la Seed", fontsize=16)

    # Ajout de la ligne Baseline qui varie par seed
    if not df_seed_random.empty:
        # lineplot se superpose parfaitement aux catégories (les seeds) en X
        sns.lineplot(data=df_seed_random, x="Seed", y="Reward", ax=axes[0],
                     color='red', linestyle='--', marker='o', markersize=8, linewidth=2,
                     label='Baseline (Random)')
        axes[0].legend()

    # --- 2. Graphique Taux de Succès ---
    sns.barplot(data=df_seed_trained, x="Seed", y="Success Rate (%)",
                hue="Agent", ax=axes[1], palette=agent_palette)
    axes[1].set_title("Taux de succès selon la Seed",fontsize=16)

    if not df_seed_random.empty:
        sns.lineplot(data=df_seed_random, x="Seed", y="Success Rate (%)", ax=axes[1],
                     color='red', linestyle='--', marker='o', markersize=8, linewidth=2,
                     label='Baseline (Random)')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "stability_per_seed.png"), dpi=300)
    plt.close()


def plot_speed_vs_safety(df_trained, baselines, agent_palette):
    plt.figure(figsize=(12, 7))  # Un peu plus large pour la légende
    sns.set_theme(style="darkgrid", context="talk")

    ax = sns.scatterplot(
        data=df_trained,
        x="Speed",
        y="Success Rate (%)",
        hue="Agent",
        size="Reward",
        sizes=(200, 800),  # Bulles légèrement plus grosses pour les slides
        palette=agent_palette,
        alpha=0.9,
        edgecolor="black"
    )

    # --- NETTOYAGE DE LA LÉGENDE ---
    # On récupère les handles (objets) et labels (textes) créés par Seaborn
    handles, labels = ax.get_legend_handles_labels()

    # On ne garde que les éléments dont le label est un nom d'agent
    # (Seaborn insère "Agent" et "Reward" comme titres dans la liste, on les filtre)
    final_handles = []
    final_labels = []
    for h, l in zip(handles, labels):
        if l in agent_palette.keys():
            final_handles.append(h)
            final_labels.append(l)

    # On réaffiche la légende proprement
    ax.legend(
        final_handles,
        final_labels,
        title="Agents",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )
    # -------------------------------

    if "Speed" in baselines and "Success Rate (%)" in baselines:
        plt.axvline(baselines["Speed"], color='gray', linestyle=':', alpha=0.7)
        plt.axhline(baselines["Success Rate (%)"],
                    color='gray', linestyle=':', alpha=0.7)
        plt.text(baselines["Speed"], baselines["Success Rate (%)"] + 2,
                 ' Niveau Random', color='red', fontstyle='italic', fontsize=12)

    plt.title("Compromis Vitesse / Sécurité",
              fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Vitesse Moyenne", fontsize=16)
    plt.ylabel("Taux de Survie (%)", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "speed_vs_safety.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    raw_data = load_data()
    df_global, df_seed = prepare_data(raw_data)

    df_trained, baselines, agent_palette = get_baselines_and_palette(df_global)

    plot_global_metrics(df_trained, baselines, agent_palette)
    plot_per_seed_stability(df_seed, baselines, agent_palette)
    plot_speed_vs_safety(df_trained, baselines, agent_palette)

    print(f"[OK] Graphiques générés dans le dossier : {PLOT_DIR}/")
