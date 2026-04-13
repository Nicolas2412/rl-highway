#!/bin/bash



#SBATCH --job-name=highway_sb3
#SBATCH --output=results/logs/dce/sb3/slurm_%j.out
#SBATCH --error=results/logs/dce/sb3/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_prod_long

# ==============================================================================
# PRÉPARATION DE L'ENVIRONNEMENT
# ==============================================================================

module load anaconda3/2022.10/gcc-13.1.0
source activate rl_hw_env

# ==============================================================================
# LANCEMENT DE L'ENTRAÎNEMENT
# ==============================================================================
echo "Démarrage de l'entraînement à $(date)"

# On utilise les paramètres que nous avons validés ensemble
python training/train_sb3.py "$@"

echo "Entraînement terminé à $(date)"