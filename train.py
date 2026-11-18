
import os
import importlib
try:
    gym = importlib.import_module("gymnasium")
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise
    raise SystemExit(
        'Missing dependency "gymnasium". Install it in this Python environment with '
        'python -m pip install "gymnasium[box2d]".'
    ) from exc

try:
    sb3 = importlib.import_module("stable_baselines3")
    monitor = importlib.import_module("stable_baselines3.common.monitor")
    callbacks = importlib.import_module("stable_baselines3.common.callbacks")
    PPO = sb3.PPO
    Monitor = monitor.Monitor
    CheckpointCallback = callbacks.CheckpointCallback
except ModuleNotFoundError as exc:
    if not exc.name or not exc.name.startswith("stable_baselines3"):
        raise
    raise SystemExit(
        'Missing dependency "stable_baselines3". Install it in this Python environment with '
        'python -m pip install "stable-baselines3".'
    ) from exc

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = Monitor(
    gym.make("LunarLander-v3"),
    filename="logs/training"
)

model_path = "models/lunar_lander_ppo"

checkpoint = CheckpointCallback(
    save_freq=25_000,
    save_path="./models/",
    name_prefix="lunar_ppo"
)

if os.path.exists(model_path + ".zip"):
    print("Loading existing model to continue training...")
    model = PPO.load(model_path, env=env)
else:
    print("Creating a new PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.0003,
        tensorboard_log="./logs/"
    )

print("Starting or continuing training...")

model.learn(
    total_timesteps=500_000,
    callback=checkpoint,
    reset_num_timesteps=False
)

model.save(model_path)

print("Training complete. Model saved!")

env.close()
