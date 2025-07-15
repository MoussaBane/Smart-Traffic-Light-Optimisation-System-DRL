import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import gymnasium as gym
from traffic_env02 import TrafficEnv

def detailed_model_analysis(model_path="dqn_traffic_optimized.zip", num_episodes=10):
    """Perform detailed analysis of the trained model"""
    
    # Register environment
    try:
        gym.register(
            id="TrafficEnv-v1",
            entry_point="traffic_env02:TrafficEnv",
            max_episode_steps=200,
        )
    except:
        pass  # Already registered
    
    # Load model and create environment
    model = DQN.load(model_path)
    env = gym.make("TrafficEnv-v1", render_mode=None)
    
    # Detailed metrics storage
    episode_data = []
    step_by_step_data = {
        'vehicles_passed_per_step': [],
        'rewards_per_step': [],
        'queue_lengths': [],
        'wait_times': [],
        'phase_changes': [],
        'reward_components': {
            'throughput': [],
            'queue_penalty': [],
            'wait_penalty': [],
            'phase_change': [],
            'efficiency_bonus': [],
            'balance_bonus': []
        }
    }
    
    print("Running detailed analysis...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_metrics = {
            'total_reward': 0,
            'total_vehicles_passed': 0,
            'total_phase_changes': 0,
            'max_queue': 0,
            'max_wait': 0,
            'steps': 0,
            'efficiency_score': 0
        }
        
        episode_step_data = {
            'vehicles_per_step': [],
            'rewards': [],
            'queues': [],
            'waits': [],
            'actions': []
        }
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update episode metrics
            episode_metrics['total_reward'] += reward
            episode_metrics['total_vehicles_passed'] += info.get('vehicles_passed', 0)
            episode_metrics['total_phase_changes'] += info.get('phase_changes', 0)
            episode_metrics['max_queue'] = max(episode_metrics['max_queue'], info.get('total_queues', 0))
            episode_metrics['max_wait'] = max(episode_metrics['max_wait'], info.get('max_wait_time', 0))
            episode_metrics['steps'] += 1
            
            # Store step-by-step data
            episode_step_data['vehicles_per_step'].append(info.get('vehicles_passed', 0))
            episode_step_data['rewards'].append(reward)
            episode_step_data['queues'].append(info.get('total_queues', 0))
            episode_step_data['waits'].append(info.get('max_wait_time', 0))
            episode_step_data['actions'].append(action)
            
            # Store reward components if available
            if 'reward_components' in info:
                for component, value in info['reward_components'].items():
                    if component in step_by_step_data['reward_components']:
                        step_by_step_data['reward_components'][component].append(value)
            
            done = terminated or truncated
        
        # Calculate efficiency score
        episode_metrics['efficiency_score'] = (
            episode_metrics['total_vehicles_passed'] / episode_metrics['steps'] * 100
        )
        
        episode_data.append(episode_metrics)
        
        # Add episode data to step-by-step collection
        step_by_step_data['vehicles_passed_per_step'].extend(episode_step_data['vehicles_per_step'])
        step_by_step_data['rewards_per_step'].extend(episode_step_data['rewards'])
        step_by_step_data['queue_lengths'].extend(episode_step_data['queues'])
        step_by_step_data['wait_times'].extend(episode_step_data['waits'])
        step_by_step_data['phase_changes'].extend([1 if a != episode_step_data['actions'][i-1] else 0 
                                                  for i, a in enumerate(episode_step_data['actions']) if i > 0])
    
    env.close()
    
    # Create comprehensive analysis plots
    create_detailed_plots(episode_data, step_by_step_data, num_episodes)
    print_detailed_statistics(episode_data, step_by_step_data)

def create_detailed_plots(episode_data, step_data, num_episodes):
    """Create comprehensive visualization plots"""
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(24, 20))  # Increased width from 20 to 24 for more horizontal space
    
    # Extract episode-level data
    total_rewards = [ep['total_reward'] for ep in episode_data]
    vehicles_passed = [ep['total_vehicles_passed'] for ep in episode_data]
    phase_changes = [ep['total_phase_changes'] for ep in episode_data]
    efficiency_scores = [ep['efficiency_score'] for ep in episode_data]
    max_queues = [ep['max_queue'] for ep in episode_data]
    max_waits = [ep['max_wait'] for ep in episode_data]
    
    # Plot 1: Episode Performance Overview
    ax1 = plt.subplot(4, 3, 1)  # Changed to 4x3 grid for better spacing
    episodes = range(1, num_episodes + 1)
    ax1.plot(episodes, total_rewards, 'bo-', label='Total Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward per Episode')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Vehicles Passed per Episode
    ax2 = plt.subplot(4, 3, 2)
    ax2.bar(episodes, vehicles_passed, alpha=0.7, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Vehicles Passed')
    ax2.set_title('Total Vehicles Passed per Episode')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency Score
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(episodes, efficiency_scores, 'go-', label='Efficiency Score')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Vehicles per Step (%)')
    ax3.set_title('Traffic Efficiency Score')
    ax3.grid(True)
    ax3.legend()
    
    # Plot 4: Phase Changes
    ax4 = plt.subplot(4, 3, 4)
    ax4.bar(episodes, phase_changes, alpha=0.7, color='orange')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Phase Changes')
    ax4.set_title('Total Phase Changes per Episode')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Vehicles Passed Distribution
    ax5 = plt.subplot(4, 3, 5)
    if step_data['vehicles_passed_per_step']:
        ax5.hist(step_data['vehicles_passed_per_step'], bins=range(0, max(step_data['vehicles_passed_per_step'])+2), 
                 alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Vehicles Passed per Step')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Vehicles Passed per Step')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Reward Distribution
    ax6 = plt.subplot(4, 3, 6)
    ax6.hist(step_data['rewards_per_step'], bins=30, alpha=0.7, edgecolor='black', color='blue')
    ax6.set_xlabel('Reward per Step')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Rewards per Step')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Queue Length Over Time (sample episode)
    ax7 = plt.subplot(4, 3, 7)
    sample_size = min(200, len(step_data['queue_lengths']))
    ax7.plot(step_data['queue_lengths'][:sample_size], color='red', alpha=0.7)
    ax7.set_xlabel('Time Steps')
    ax7.set_ylabel('Total Queue Length')
    ax7.set_title('Queue Length Over Time (Sample)')
    ax7.grid(True)
    
    # Plot 8: Wait Times Over Time
    ax8 = plt.subplot(4, 3, 8)
    ax8.plot(step_data['wait_times'][:sample_size], color='purple', alpha=0.7)
    ax8.set_xlabel('Time Steps')
    ax8.set_ylabel('Max Wait Time')
    ax8.set_title('Max Wait Time Over Time (Sample)')
    ax8.grid(True)
    
    # Plot 9: Reward Components (if available)
    ax9 = plt.subplot(4, 3, 9)
    if step_data['reward_components']['throughput']:
        components = ['throughput', 'queue_penalty', 'wait_penalty', 'efficiency_bonus']
        component_labels = ['Throughput\nReward', 'Queue\nPenalty', 'Wait\nPenalty', 'Efficiency\nBonus']
        means = [np.mean(step_data['reward_components'][comp]) for comp in components if step_data['reward_components'][comp]]
        if means:
            bars = ax9.bar(range(len(means)), means, color=['green', 'red', 'orange', 'blue'])
            ax9.set_xticks(range(len(means)))
            ax9.set_xticklabels(component_labels[:len(means)], fontsize=8, ha='center')
            ax9.set_ylabel('Average Value')
            ax9.set_title('Average Reward Components')
            
            # Add value labels on top of bars
            for i, (bar, value) in enumerate(zip(bars, means)):
                ax9.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=7)
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Performance Correlation
    ax10 = plt.subplot(4, 3, 10)
    scatter = ax10.scatter(vehicles_passed, total_rewards, alpha=0.7, c=efficiency_scores, cmap='viridis')
    ax10.set_xlabel('Vehicles Passed')
    ax10.set_ylabel('Total Reward')
    ax10.set_title('Reward vs Vehicles Passed')
    plt.colorbar(scatter, ax=ax10, label='Efficiency Score')
    ax10.grid(True)
    
    # Plot 11: Queue vs Wait Time Correlation
    ax11 = plt.subplot(4, 3, 11)
    ax11.scatter(max_queues, max_waits, alpha=0.7, color='red')
    ax11.set_xlabel('Max Queue Length')
    ax11.set_ylabel('Max Wait Time')
    ax11.set_title('Queue vs Wait Time Correlation')
    ax11.grid(True)
    
    # Plot 12: Phase Change Efficiency
    ax12 = plt.subplot(4, 3, 12)
    ax12.scatter(phase_changes, efficiency_scores, alpha=0.7, color='orange')
    ax12.set_xlabel('Phase Changes')
    ax12.set_ylabel('Efficiency Score')
    ax12.set_title('Phase Changes vs Efficiency')
    ax12.grid(True)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Increased both vertical and horizontal spacing
    plt.show()

def print_detailed_statistics(episode_data, step_data):
    """Print comprehensive statistics"""
    
    total_rewards = [ep['total_reward'] for ep in episode_data]
    vehicles_passed = [ep['total_vehicles_passed'] for ep in episode_data]
    phase_changes = [ep['total_phase_changes'] for ep in episode_data]
    efficiency_scores = [ep['efficiency_score'] for ep in episode_data]
    
    print("\n" + "="*60)
    print("DETAILED MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"\n[STATS] EPISODE-LEVEL STATISTICS ({len(episode_data)} episodes)")
    print("-" * 40)
    print(f"Average Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Best Episode Reward: {np.max(total_rewards):.2f}")
    print(f"Worst Episode Reward: {np.min(total_rewards):.2f}")
    
    print(f"\n[TRAFFIC] TRAFFIC THROUGHPUT ANALYSIS")
    print("-" * 40)
    print(f"Average Vehicles per Episode: {np.mean(vehicles_passed):.1f} ± {np.std(vehicles_passed):.1f}")
    print(f"Best Episode Throughput: {np.max(vehicles_passed)} vehicles")
    print(f"Average Vehicles per Step: {np.mean(step_data['vehicles_passed_per_step']):.2f}")
    print(f"Peak Vehicles in Single Step: {np.max(step_data['vehicles_passed_per_step'])}")
    
    print(f"\n[EFFICIENCY] EFFICIENCY METRICS")
    print("-" * 40)
    print(f"Average Efficiency Score: {np.mean(efficiency_scores):.2f}%")
    print(f"Best Efficiency Score: {np.max(efficiency_scores):.2f}%")
    print(f"Efficiency Consistency (std): {np.std(efficiency_scores):.2f}%")
    
    print(f"\n[LIGHTS] PHASE CHANGE ANALYSIS")
    print("-" * 40)
    print(f"Average Phase Changes per Episode: {np.mean(phase_changes):.1f}")
    print(f"Phase Change Range: {np.min(phase_changes)} - {np.max(phase_changes)}")
    phase_change_rate = np.sum(step_data['phase_changes']) / len(step_data['vehicles_passed_per_step']) * 100
    print(f"Phase Change Rate: {phase_change_rate:.1f}% of steps")
    
    print(f"\n[QUEUES] QUEUE MANAGEMENT")
    print("-" * 40)
    print(f"Average Queue Length: {np.mean(step_data['queue_lengths']):.2f}")
    print(f"Peak Queue Length: {np.max(step_data['queue_lengths'])}")
    print(f"Queue Length Std Dev: {np.std(step_data['queue_lengths']):.2f}")
    
    print(f"\n[TIME] WAIT TIME ANALYSIS")
    print("-" * 40)
    print(f"Average Max Wait Time: {np.mean(step_data['wait_times']):.2f} steps")
    print(f"Longest Wait Time: {np.max(step_data['wait_times'])} steps")
    print(f"Wait Time Consistency: {np.std(step_data['wait_times']):.2f}")
    
    print(f"\n[REWARDS] REWARD ANALYSIS")
    print("-" * 40)
    print(f"Average Reward per Step: {np.mean(step_data['rewards_per_step']):.2f}")
    print(f"Reward Range: {np.min(step_data['rewards_per_step']):.2f} to {np.max(step_data['rewards_per_step']):.2f}")
    print(f"Reward Volatility (std): {np.std(step_data['rewards_per_step']):.2f}")
    
    # Reward components analysis (if available)
    if step_data['reward_components']['throughput']:
        print(f"\n[COMPONENTS] REWARD COMPONENTS BREAKDOWN")
        print("-" * 40)
        
        # Define better component names for display
        component_display_names = {
            'throughput': 'Throughput Reward',
            'queue_penalty': 'Queue Length Penalty', 
            'wait_penalty': 'Wait Time Penalty',
            'phase_change': 'Phase Change Cost',
            'efficiency_bonus': 'Efficiency Bonus',
            'balance_bonus': 'Balance Bonus'
        }
        
        for component, values in step_data['reward_components'].items():
            if values:
                display_name = component_display_names.get(component, component.replace('_', ' ').title())
                print(f"{display_name:20}: {np.mean(values):8.3f} ± {np.std(values):6.3f} (range: {np.min(values):6.2f} to {np.max(values):6.2f})")
    
    # Performance correlations
    print(f"\n[CORRELATION] PERFORMANCE CORRELATIONS")
    print("-" * 40)
    vehicles_reward_corr = np.corrcoef(vehicles_passed, total_rewards)[0, 1]
    efficiency_reward_corr = np.corrcoef(efficiency_scores, total_rewards)[0, 1]
    phase_efficiency_corr = np.corrcoef(phase_changes, efficiency_scores)[0, 1]
    
    print(f"Vehicles <-> Reward correlation: {vehicles_reward_corr:.3f}")
    print(f"Efficiency <-> Reward correlation: {efficiency_reward_corr:.3f}")
    print(f"Phase Changes <-> Efficiency correlation: {phase_efficiency_corr:.3f}")
    
    # Performance classification
    print(f"\n[CLASSIFICATION] PERFORMANCE CLASSIFICATION")
    print("-" * 40)
    excellent_episodes = sum(1 for r in total_rewards if r > np.mean(total_rewards) + np.std(total_rewards))
    good_episodes = sum(1 for r in total_rewards if np.mean(total_rewards) <= r <= np.mean(total_rewards) + np.std(total_rewards))
    poor_episodes = sum(1 for r in total_rewards if r < np.mean(total_rewards) - np.std(total_rewards))
    
    print(f"Excellent episodes (>mean+std): {excellent_episodes}/{len(episode_data)} ({excellent_episodes/len(episode_data)*100:.1f}%)")
    print(f"Good episodes (mean±std): {good_episodes}/{len(episode_data)} ({good_episodes/len(episode_data)*100:.1f}%)")
    print(f"Poor episodes (<mean-std): {poor_episodes}/{len(episode_data)} ({poor_episodes/len(episode_data)*100:.1f}%)")

def analyze_evaluation_results():
    """Analyze the evaluation results from training"""
    try:
        # Load evaluation data
        eval_data = np.load('logs/evaluations.npz')
        
        # Extract data
        timesteps = eval_data['timesteps']
        results = eval_data['results']
        ep_lengths = eval_data['ep_lengths']
        
        # Calculate statistics
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)
        mean_ep_lengths = np.mean(ep_lengths, axis=1)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean reward over time
        ax1.plot(timesteps, mean_rewards, 'b-', label='Mean Reward')
        ax1.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Training Progress: Mean Reward')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Reward distribution (last evaluation)
        ax2.hist(results[-1], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Reward Distribution (Final Evaluation)\nMean: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}')
        ax2.grid(True)
        
        # Plot 3: Episode lengths
        ax3.plot(timesteps, mean_ep_lengths, 'g-', label='Mean Episode Length')
        ax3.set_xlabel('Timesteps')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Episode Lengths Over Training')
        ax3.grid(True)
        ax3.legend()
        
        # Plot 4: Learning curve smoothed
        window_size = min(10, len(mean_rewards))
        if window_size > 1:
            smoothed_rewards = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
            smoothed_timesteps = timesteps[window_size-1:]
            ax4.plot(smoothed_timesteps, smoothed_rewards, 'r-', label=f'Smoothed (window={window_size})')
        ax4.plot(timesteps, mean_rewards, 'b-', alpha=0.3, label='Raw')
        ax4.set_xlabel('Timesteps')
        ax4.set_ylabel('Mean Reward')
        ax4.set_title('Learning Curve (Smoothed)')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("=== TRAINING SUMMARY ===")
        print(f"Total timesteps: {timesteps[-1]:,}")
        print(f"Number of evaluations: {len(timesteps)}")
        print(f"Final mean reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
        print(f"Best mean reward: {np.max(mean_rewards):.2f}")
        print(f"Final episode length: {mean_ep_lengths[-1]:.1f}")
        
        # Performance improvement
        if len(mean_rewards) > 1:
            improvement = mean_rewards[-1] - mean_rewards[0]
            print(f"Total improvement: {improvement:.2f}")
            print(f"Improvement rate: {improvement/len(mean_rewards):.3f} per evaluation")
        
    except FileNotFoundError:
        print("No evaluation data found. Make sure training with eval_callback was completed.")
    except Exception as e:
        print(f"Error loading evaluation data: {e}")

if __name__ == "__main__":
    # Automatically run detailed model analysis without user input
    print("Running detailed model performance analysis...")
    detailed_model_analysis()
