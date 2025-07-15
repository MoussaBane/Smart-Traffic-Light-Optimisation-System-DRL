import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from traffic_env02 import TrafficEnv  # Make sure this matches your environment file name

# Register the environment (same as in training)
gym.register(
    id="TrafficEnv-v1",
    entry_point="traffic_env02:TrafficEnv",
    max_episode_steps=200,
)

def test_trained_model(model_path, num_episodes=3, render=True, episode_lengths=None):
    """
    Test trained model with customizable episode lengths
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to run
        render: Whether to show visualization
        episode_lengths: List of episode lengths, or None for default (200 steps each)
                        Can be: [100, 150, 300] for different lengths
                               or "random" for random lengths between 50-400
                               or "progressive" for increasing lengths
    """
    # Create environment
    env = gym.make("TrafficEnv-v1", render_mode="human" if render else None)
    
    # Load the trained model
    model = DQN.load(model_path)
    
    # Set up episode lengths
    if episode_lengths is None:
        episode_lengths = [200] * num_episodes  # Default 200 steps each
    elif episode_lengths == "random":
        episode_lengths = [np.random.randint(50, 401) for _ in range(num_episodes)]
        print(f"Random episode lengths: {episode_lengths}")
    elif episode_lengths == "progressive":
        episode_lengths = [100 + i * 50 for i in range(num_episodes)]
        print(f"Progressive episode lengths: {episode_lengths}")
    elif isinstance(episode_lengths, list):
        if len(episode_lengths) != num_episodes:
            print(f"Warning: {len(episode_lengths)} lengths provided for {num_episodes} episodes")
            # Extend or truncate as needed
            episode_lengths = (episode_lengths * num_episodes)[:num_episodes]
    
    print(f"Testing {num_episodes} episodes with lengths: {episode_lengths}")
    
    for episode in range(num_episodes):
        max_steps = episode_lengths[episode]
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        metrics = {
            'vehicles_passed': 0,
            'total_queue': [],
            'max_wait': [],
            'phase_changes': 0
        }
        
        print(f"\n--- Episode {episode + 1} (Max Steps: {max_steps}) ---")
        
        while not done and step_count < max_steps:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update metrics
            total_reward += reward
            metrics['vehicles_passed'] += info.get('vehicles_passed', 0)
            metrics['total_queue'].append(info.get('total_queues', 0))
            metrics['max_wait'].append(info.get('max_wait_time', 0))
            metrics['phase_changes'] += info.get('phase_changes', 0)
            
            step_count += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        # Calculate performance metrics
        efficiency = (metrics['vehicles_passed'] / step_count) * 100 if step_count > 0 else 0
        avg_queue = np.mean(metrics['total_queue']) if metrics['total_queue'] else 0
        max_wait = np.max(metrics['max_wait']) if metrics['max_wait'] else 0
        
        # Print episode summary
        print(f"Episode {episode + 1} Summary (Length: {max_steps} steps):")
        print(f"  Actual Steps: {step_count}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Vehicles Passed: {metrics['vehicles_passed']}")
        print(f"  Efficiency: {efficiency:.2f} vehicles/step")
        print(f"  Avg Queue Length: {avg_queue:.2f}")
        print(f"  Max Wait Time: {max_wait} steps")
        print(f"  Phase Changes: {metrics['phase_changes']}")
        print(f"  Episode Status: {'Completed naturally' if done else 'Reached max steps'}")
        
        # Plot metrics for each episode if requested
        if render and episode == num_episodes - 1:  # Plot only last episode
            plt.figure(figsize=(15, 10))
            
            plt.subplot(4, 1, 1)
            plt.plot(metrics['total_queue'], 'b-', linewidth=2)
            plt.title(f'Queue Length Over Time (Episode {episode + 1}, {step_count} steps)')
            plt.ylabel('Total Vehicles')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 2)
            plt.plot(metrics['max_wait'], 'r-', linewidth=2)
            plt.title('Maximum Wait Time')
            plt.ylabel('Steps')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 3)
            # Calculate vehicles passed per step
            vehicles_per_step = []
            cumulative = 0
            for i in range(len(metrics['total_queue'])):
                # This is approximate - actual implementation would need step-by-step tracking
                if i > 0 and metrics['total_queue'][i] < metrics['total_queue'][i-1]:
                    vehicles_per_step.append(metrics['total_queue'][i-1] - metrics['total_queue'][i])
                else:
                    vehicles_per_step.append(0)
            
            plt.bar(range(len(vehicles_per_step)), vehicles_per_step, alpha=0.7, color='green')
            plt.title('Vehicles Passed Per Step (Approximate)')
            plt.ylabel('Vehicles')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 4)
            # Show efficiency over time (rolling average)
            if len(metrics['total_queue']) > 10:
                window_size = min(20, len(metrics['total_queue']) // 4)
                rolling_efficiency = []
                for i in range(window_size, len(metrics['total_queue'])):
                    window_vehicles = sum(vehicles_per_step[i-window_size:i])
                    efficiency_val = (window_vehicles / window_size) * 100
                    rolling_efficiency.append(efficiency_val)
                
                plt.plot(range(window_size, len(metrics['total_queue'])), rolling_efficiency, 'purple', linewidth=2)
                plt.title(f'Rolling Efficiency (Window: {window_size} steps)')
                plt.ylabel('Efficiency %')
                plt.xlabel('Time Step')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    env.close()

# Additional convenience functions
def test_short_episodes(model_path="dqn_traffic_optimized.zip"):
    """Test with short episodes (50-100 steps)"""
    test_trained_model(model_path, num_episodes=5, episode_lengths=[50, 75, 100, 60, 90])

def test_long_episodes(model_path="dqn_traffic_optimized.zip"):
    """Test with long episodes (300-500 steps)"""
    test_trained_model(model_path, num_episodes=3, episode_lengths=[300, 400, 500])

def test_varied_episodes(model_path="dqn_traffic_optimized.zip"):
    """Test with varied episode lengths"""
    test_trained_model(model_path, num_episodes=6, episode_lengths=[100, 200, 300, 150, 250, 400])

def test_random_episodes(model_path="dqn_traffic_optimized.zip"):
    """Test with random episode lengths"""
    test_trained_model(model_path, num_episodes=5, episode_lengths="random")

def test_progressive_episodes(model_path="dqn_traffic_optimized.zip"):
    """Test with progressively longer episodes"""
    test_trained_model(model_path, num_episodes=5, episode_lengths="progressive")

if __name__ == "__main__":
    # Automatically run varied episodes test without user input
    print("Running varied episodes test (6 episodes with different lengths)...")
    print("Episode lengths: [100, 200, 300, 150, 250, 400] steps")
    print("="*60)
    
    test_varied_episodes()
    
    print("\n" + "="*60)
    print("Test completed! To run different scenarios, use these commands:")
    print("1. python -c \"from test_agent import *; test_trained_model('dqn_traffic_optimized.zip', 3, True)\"")
    print("2. python -c \"from test_agent import *; test_short_episodes()\"")
    print("3. python -c \"from test_agent import *; test_long_episodes()\"") 
    print("4. python -c \"from test_agent import *; test_varied_episodes()\"")
    print("5. python -c \"from test_agent import *; test_random_episodes()\"")
    print("6. python -c \"from test_agent import *; test_progressive_episodes()\"")
    print("7. python -c \"from test_agent import *; test_trained_model('dqn_traffic_optimized.zip', 3, True, [120, 180, 250])\"")
    print("="*60)