
import statistics
from importlib import import_module

gym = import_module("gymnasium")
PPO = import_module("stable_baselines3").PPO


def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                    return_episode_rewards=False):
    """Evaluate a model without depending on SB3's evaluation module."""
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_eval_episodes):
        observation, _ = env.reset()
        total_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            episode_length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return statistics.mean(episode_rewards), statistics.pstdev(episode_rewards)

# Load your trained agent
model = PPO.load("models/lunar_lander_ppo")

# Create an environment for testing
env = gym.make("LunarLander-v3")

try:
    print("Evaluating 200 episodes...")

    rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=200,
        deterministic=True,
        return_episode_rewards=True
    )

    print("\n--- EVALUATION RESULTS ---")
    print(f"Episodes tested: {len(rewards)}")
    print(f"Mean reward: {statistics.mean(rewards):.2f}")
    print(f"Reward standard deviation: {statistics.pstdev(rewards):.2f}")
    print(f"Best reward: {max(rewards):.2f}")
    print(f"Worst reward: {min(rewards):.2f}")
    print(f"Mean episode length: {statistics.mean(lengths):.1f}")

    print(
        "Episodes with positive reward:",
        f"{sum(reward > 0 for reward in rewards)}/200"
    )

    print(
        "Episodes scoring 200 or higher:",
        f"{sum(reward >= 200 for reward in rewards)}/200"
    )

finally:
    env.close()
