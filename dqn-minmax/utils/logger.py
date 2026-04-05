import time


class TrainingLogger:
    def __init__(self):
        self.start_time = time.time()

        self.rewards_A = []
        self.rewards_B = []
        self.lengths = []

        self.total_steps = 0
        self.episodes = 0

        self.last_loss = None

    def log_step(self):
        self.total_steps += 1

    def log_loss(self, loss):
        if loss is not None:
            self.last_loss = loss

    def log_episode(self, reward_A, reward_B, length):
        self.rewards_A.append(reward_A)
        self.rewards_B.append(reward_B)
        self.lengths.append(length)
        self.episodes += 1

    def _mean(self, values, last_n=10):
        if len(values) == 0:
            return 0
        return sum(values[-last_n:]) / min(len(values), last_n)

    def print(self, epsilon):
        time_elapsed = int(time.time() - self.start_time)
        fps = int(self.total_steps / (time_elapsed + 1e-8))

        print(f"\n[Episode {self.episodes}]")
        print(f"Steps: {self.total_steps} | FPS: {fps} | Time: {time_elapsed}s")
        print(f"Epsilon: {epsilon:.3f}")

        print("\nRewards:")
        print(f"  agent_A: {self.rewards_A[-1]:.2f}")
        print(f"  agent_B: {self.rewards_B[-1]:.2f}")

        print("\nAverages (last 10):")
        print(f"  reward_A: {self._mean(self.rewards_A):.2f}")
        print(f"  reward_B: {self._mean(self.rewards_B):.2f}")
        print(f"  length:   {self._mean(self.lengths):.0f}")

        if self.last_loss is not None:
            print(f"\nLoss: {self.last_loss:.4f}")

        print("-" * 40)