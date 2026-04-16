import numpy as np
import matplotlib.pyplot as plt


rewards  = np.load(r"C:\Users\Asturiel\Documents\Cours\CentraleSupelec_COURS\Third_Year\RL-Reinforcement Learning\projet\rl-highway\checkpoints\dqn_20260411-135652\episode_rewards.npy")
losses   = np.load(r"C:\Users\Asturiel\Documents\Cours\CentraleSupelec_COURS\Third_Year\RL-Reinforcement Learning\projet\rl-highway\checkpoints\dqn_20260411-135652\losses.npy")
lengths  = np.load(r"C:\Users\Asturiel\Documents\Cours\CentraleSupelec_COURS\Third_Year\RL-Reinforcement Learning\projet\rl-highway\checkpoints\dqn_20260411-135652\ep_lengths.npy")


def afficher_stats(nom, data):
    print(f"\n{'─'*40}")
    print(f"  {nom}")
    print(f"{'─'*40}")
    print(f"  Taille    : {data.shape}")
    print(f"  Min       : {data.min():.4f}")
    print(f"  Max       : {data.max():.4f}")
    print(f"  Moyenne   : {data.mean():.4f}")
    print(f"  Écart-type: {data.std():.4f}")

afficher_stats("Récompenses par épisode (episode_rewards)", rewards)
afficher_stats("Pertes / Loss (losses)",                    losses)
afficher_stats("Longueurs d'épisodes (ep_lengths)",         lengths)

# --- Tracé des courbes ---
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
fig.suptitle("Métriques d'entraînement RL", fontsize=14, fontweight="bold")

# Récompenses cumulées par épisode
axes[0].plot(rewards, color="steelblue", linewidth=0.8, alpha=0.6, label="brut")
# Moyenne glissante pour lisser la courbe
window = max(1, len(rewards) // 20)
axes[0].plot(np.convolve(rewards, np.ones(window)/window, mode="valid"),
             color="steelblue", linewidth=2, label=f"moy. glissante ({window})")
axes[0].set_title("Récompenses par épisode")
axes[0].set_xlabel("Épisode")
axes[0].set_ylabel("Récompense")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Évolution de la loss
axes[1].plot(losses, color="tomato", linewidth=0.8, alpha=0.6, label="brut")
window_l = max(1, len(losses) // 20)
axes[1].plot(np.convolve(losses, np.ones(window_l)/window_l, mode="valid"),
             color="tomato", linewidth=2, label=f"moy. glissante ({window_l})")
axes[1].set_title("Loss au cours de l'entraînement")
axes[1].set_xlabel("Étape")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Longueur des épisodes
axes[2].plot(lengths, color="seagreen", linewidth=0.8, alpha=0.6, label="brut")
window_e = max(1, len(lengths) // 20)
axes[2].plot(np.convolve(lengths, np.ones(window_e)/window_e, mode="valid"),
             color="seagreen", linewidth=2, label=f"moy. glissante ({window_e})")
axes[2].set_title("Longueur des épisodes")
axes[2].set_xlabel("Épisode")
axes[2].set_ylabel("Nombre de pas")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\Asturiel\Documents\Cours\CentraleSupelec_COURS\Third_Year\RL-Reinforcement Learning\projet\rl-highway\checkpoints\dqn_20260411-135652\training_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n Graphique sauvegardé : training_metrics.png")