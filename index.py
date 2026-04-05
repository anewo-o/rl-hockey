import gymnasium as gym
import shimmy
import ale_py
from per.extract_state import get_state, build_state, normalize_state

env = gym.make("ALE/IceHockey-v5", obs_type="ram", render_mode="human")
obs, info = env.reset()

prev_raw_state = None

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Raw state
    raw_state = get_state(obs, prev_raw_state)

    # Build state with velocities and distances
    final_state = build_state(raw_state, prev_raw_state)

    # Normalize state
    normalized_final_state = normalize_state(final_state)

    # Update previous raw state for next step 
    prev_raw_state = raw_state

    # Debug print
    print(f"Step {step}")
    print("Raw state:", raw_state)
    print("Final state:", final_state)
    print("Normalized final state:", normalized_final_state)

    if terminated or truncated:
        obs, info = env.reset()
        prev_raw_state = None

env.close()
