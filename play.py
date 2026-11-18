
import importlib

gym = importlib.import_module("gymnasium")
stable_baselines3 = importlib.import_module("stable_baselines3")

PPO = stable_baselines3.PPO

model = PPO.load("models/lunar_lander_ppo")

env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(50):
    obs, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(
        f"Episode {episode + 1}/50 | "
        f"Reward: {total_reward:.2f}"
    )

env.close()
