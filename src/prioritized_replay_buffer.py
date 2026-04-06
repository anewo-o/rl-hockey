import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, alpha=0.6, beta=0.4, beta_increment=1e-6, **kwargs):
        super().__init__(*args, **kwargs)

        # alpha : contrôle à quel point on privilégie les transitions importantes
        # alpha = 0 → échantillonnage uniforme (comme DQN classique)
        # alpha = 1 → totalement basé sur la priorité
        self.alpha = alpha

        # beta : corrige le biais introduit par l'échantillonnage non uniforme
        self.beta = beta
        self.beta_increment = beta_increment

        # Tableau des priorités (une par transition)
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

        # Petite constante pour éviter les priorités nulles
        self.eps = 1e-6

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)

        # Index réel de la transition ajoutée (gestion du buffer circulaire)
        idx = (self.pos - 1) % self.buffer_size

        # Nouvelle transition → priorité maximale pour être vue au moins une fois
        max_prio = self.priorities.max() if self.pos > 0 else 1.0
        if max_prio == 0:
            max_prio = 1.0

        self.priorities[idx] = max_prio

    def sample(self, batch_size, env=None):
        # On récupère les priorités valides
        if self.full:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # Cas initial : toutes les priorités sont nulles → fallback uniforme
        if len(prios) == 0 or prios.sum() == 0:
            probs = np.ones_like(prios) / len(prios)
        else:
            probs = prios ** self.alpha
            probs_sum = probs.sum()

            if probs_sum == 0:
                probs = np.ones_like(prios) / len(prios)
            else:
                probs /= probs_sum

        # Échantillonnage selon les probabilités
        indices = np.random.choice(len(probs), batch_size, p=probs)

        samples = self._get_samples(indices, env)

        # Calcul des poids d'importance pour la correction du biais
        total = len(probs)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Incrémentation de beta pour réduire progressivement le biais
        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        # Mise à jour des priorités selon les TD errors
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.eps

