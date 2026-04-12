from stable_baselines3 import DQN as SB3DQN
import torch as th
import numpy as np
from torch.nn import functional as F

class PERDQN(SB3DQN):
    def train(self, gradient_steps: int, batch_size: int = 100):

        # Mode entraînement
        self.policy.set_training_mode(True)

        # Mise à jour du taux d'apprentissage
        self._update_learning_rate(self.policy.optimizer)

        losses = []

        for _ in range(gradient_steps):
            # Échantillonnage avec Prioritized Experience Replay
            replay_data, indices, weights = self.replay_buffer.sample(batch_size)
            weights = th.tensor(weights, device=self.device).unsqueeze(1)

            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Calcul des Q-values cibles
                next_q_values = self.q_net_target(replay_data.next_observations)

                # Meilleure action pour les prochaines observations
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)

                # Cible TD : récompense + discount * max Q(next_state)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Q-values actuelles pour les actions prises
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # TD error
            td_errors = target_q_values - current_q_values

            # Loss pondérée par les poids d'importance pour corriger le biais de l'échantillonnage
            loss = (weights * F.smooth_l1_loss(
                current_q_values,
                target_q_values,
                reduction="none"
            )).mean()

            losses.append(loss.item())

            # Mise à jour des paramètres du réseau
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            # Mise à jour des priorités dans le buffer selon les TD errors
            self.replay_buffer.update_priorities(
                indices,
                td_errors.detach().cpu().numpy().flatten()
            )

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/loss", np.mean(losses))