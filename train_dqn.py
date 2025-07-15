from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Register the environment
from gymnasium.envs.registration import register
register(
    id="TrafficEnv-v1",
    entry_point="traffic_env02:TrafficEnv",
)

# Create environment
env = make_vec_env("TrafficEnv-v1", n_envs=1)

# Evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Create model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,  # Slightly higher
    buffer_size=100000,     # Larger buffer
    learning_starts=10000,  # More initial exploration
    batch_size=256,         # Larger batches
    gamma=0.98,            # Slightly longer horizon
    train_freq=4,
    target_update_interval=2000,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./traffic_light_tensorboard/"
)

# Train
model.learn(total_timesteps=500_000, callback=eval_callback)
model.save("dqn_traffic_optimized")

env.close()