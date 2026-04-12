import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, Any

# Ajout du chemin pour les imports locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, DQN
from src.env_utils import create_ice_hockey_env
from src.ppo_addendum import PartialPPO
from src.per_dqn import PERDQN

# Configuration des algorithmes

def get_model_class(algo_name: str):
    mapping = {
        "ppo": PPO,
        "partial-ppo": PartialPPO,
        "dqn": DQN,
        "per-dqn": PERDQN
    }
    algo_key = algo_name.lower()
    if algo_key not in mapping:
        raise ValueError(f"Algorithme {algo_name} non reconnu. Choix : {list(mapping.keys())}")
    return mapping[algo_key]

def load_model(model_path: str, algo_name: str):
    """Charge le modèle avec les objets personnalisés nécessaires."""
    model_class = get_model_class(algo_name)
    
    # Objets pour éviter les erreurs de chargement des schedules ou buffers complexes
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.1,
        "exploration_schedule": lambda _: 0.0,
    }
    
    if not os.path.exists(model_path):
        # Essayer d'ajouter .zip si oublié
        if not model_path.endswith(".zip"):
            model_path += ".zip"
            
    print(f"Chargement du modèle : {model_path} ({algo_name})")
    return model_class.load(model_path, custom_objects=custom_objects)

# Logique d'éval

def run_evaluation(model, env, num_matches: int, visual: bool = False):
    """Exécute les matchs et calcule les statistiques."""
    stats = {"wins": 0, "goals_scored": 0, "goals_conceded": 0}
    
    for match in range(num_matches):
        obs = env.reset()
        done = False
        match_scored = 0
        match_conceded = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)
            
            # Filtrage des récompenses (Shaping vs Goals)
            # Les buts valent +/- 1.0, les micro-récompenses sont < 0.1
            reward = rewards[0]
            if reward > 0.5: match_scored += 1
            elif reward < -0.5: match_conceded += 1

            if visual:
                time.sleep(0.01) # Ralentir pour l'observation humaine
            
            if dones[0]:
                done = True
        
        # Mise à jour des stats globales
        stats["goals_scored"] += match_scored
        stats["goals_conceded"] += match_conceded
        if match_scored > match_conceded:
            stats["wins"] += 1
            
        if visual:
            print(f"Match {match+1} fini : {match_scored} - {match_conceded}")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation unifiée des agents Ice Hockey")
    parser.add_argument("--model", type=str, required=True, help="Nom ou chemin du modèle .zip")
    parser.add_argument("--algo", type=str, default="ppo", help="ppo, partial-ppo, dqn, per-dqn")
    parser.add_argument("--n-matches", type=int, default=1, help="Nombre de matchs à jouer")
    parser.add_argument("--visual", action="store_true", help="Activer le rendu graphique")
    
    args = parser.parse_args()

    # chemin
    model_path = args.model
    if not os.path.sep in model_path:
        model_path = os.path.join("models", model_path)

    # Environnement
    render_mode = "human" if args.visual else None
    env = create_ice_hockey_env(n_envs=1, render_mode=render_mode, seed=42)

    try:
        model = load_model(model_path, args.algo)

        print(f"\nDébut de l'évaluation sur {args.n_matches} matchs...")
        results = run_evaluation(model, env, args.n_matches, visual=args.visual)

        print("\n" + "="*40)
        print(f" RÉSULTATS : {os.path.basename(model_path)} ")
        print("="*40)
        print(f"Victoires : {results['wins']}/{args.n_matches}")
        print(f"Moy. Buts Marqués : {results['goals_scored']/args.n_matches:.2f}")
        print(f"Moy. Buts Encaissés : {results['goals_conceded']/args.n_matches:.2f}")
        print("="*40)

    except Exception as e:
        print(f"Erreur lors de l'évaluation : {e}")
    finally:
        env.close()