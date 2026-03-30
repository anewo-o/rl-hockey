from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

def create_ice_hockey_env(n_envs=1, render_mode=None, n_stack=4, seed=42):
    """
    Crée et configure l'environnement vectorisé pour ALE/IceHockey-v5.
    
    Args:
        n_envs (int): Nombre d'environnements à exécuter en parallèle.
        render_mode (str): Mode de rendu (ex: "human" pour le test, None pour l'entraînement).
        n_stack (int): Nombre de frames consécutives à empiler.
        seed (int): Graine aléatoire pour la reproductibilité.
        
    Returns:
        VecFrameStack: L'environnement vectorisé et configuré (4 frames empilées).
    """
    env_id = "ALE/IceHockey-v5"

    # Création de l'environnement avec make_atari_env
    env = make_atari_env(
        env_id, 
        n_envs=n_envs, 
        seed=seed, 
        env_kwargs={"render_mode": render_mode}
    )

    # Empilement de 4 frames pour capturer le mouvement 
    env = VecFrameStack(env, n_stack=n_stack)

    # Transposition pour pytorch (fix d'un warning)
    env = VecTransposeImage(env)

    return env