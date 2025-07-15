import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)  # 0: NS green, 1: EW green
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        self.max_queue = 10
        self.max_steps = 200
        self.reset()

        if self.render_mode == "human":
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.queues = np.zeros(4, dtype=np.int32)  # [N, S, E, W]
        self.wait_times = np.zeros(4, dtype=np.int32)
        self.current_phase = 0  # 0: NS green, 1: EW green
        self.traffic_light_state = "NS_green"
        self.step_count = 0
        return self._get_obs(), {}

    def get_dynamic_arrival_rate(self):
        # More realistic dynamic traffic pattern
        if 0 <= self.step_count < 50:
            return np.array([1.5, 1.5, 2.5, 2.5])  # More EW traffic
        elif 50 <= self.step_count < 100:
            return np.array([2.5, 2.5, 1.5, 1.5])  # More NS traffic
        elif 100 <= self.step_count < 150:
            return np.array([2.0, 2.0, 2.0, 2.0])  # Balanced heavy traffic
        else:
            return np.array([1.0, 1.0, 1.0, 1.0])  # Balanced light traffic

    def _get_obs(self):
        # Normalized observation space (0-1 range)
        normalized_queues = self.queues / self.max_queue
        return np.concatenate((normalized_queues.astype(np.float32), [np.float32(self.current_phase)]))

    def step(self, action):
        self.step_count += 1

        # Phase change penalty (smaller penalty that decays with justification)
        phase_change_cost = 0.5 if action != self.current_phase else 0
        self.current_phase = action
        self.traffic_light_state = "NS_green" if action == 0 else "EW_green"

        # Dynamic vehicle arrivals - more realistic distribution
        dynamic_rate = self.get_dynamic_arrival_rate()
        arrivals = np.random.poisson(dynamic_rate * 1.2)  # Slightly increased traffic
        self.queues = np.minimum(self.queues + arrivals, self.max_queue)

        # Vehicle passing - more vehicles can pass when queues are longer
        if self.current_phase == 0:  # NS green
            base_passing = 2 + int(self.queues[0] > 5) + int(self.queues[1] > 5)
            passed = np.random.randint(base_passing, base_passing + 3, size=2)
            self.queues[0] = max(0, self.queues[0] - passed[0])
            self.queues[1] = max(0, self.queues[1] - passed[1])
            self.wait_times[0:2] = 0
            self.wait_times[2:4] += 1
        else:  # EW green
            base_passing = 2 + int(self.queues[2] > 5) + int(self.queues[3] > 5)
            passed = np.random.randint(base_passing, base_passing + 3, size=2)
            self.queues[2] = max(0, self.queues[2] - passed[0])
            self.queues[3] = max(0, self.queues[3] - passed[1])
            self.wait_times[2:4] = 0
            self.wait_times[0:2] += 1

        vehicles_passed = passed[0] + passed[1]
        
        # New reward components
        throughput_reward = 5.0 * vehicles_passed  # Strong reward for moving vehicles
        
        # Queue management - encourages keeping queues balanced and below threshold
        queue_penalty = 0.1 * np.sum(np.maximum(0, self.queues - 5)**2)  # Only penalize queues >5
        
        # Wait time penalty - non-linear and only for excessive waits
        wait_penalty = 0.05 * np.sum(self.wait_times**1.5)
        
        # Phase change justification bonus
        if phase_change_cost > 0:
            # Calculate if change was justified by queue difference
            if (action == 0 and np.sum(self.queues[0:2]) > np.sum(self.queues[2:4]) + 3) or \
               (action == 1 and np.sum(self.queues[2:4]) > np.sum(self.queues[0:2]) + 3):
                phase_change_cost = 0  # No penalty if justified
        
        # Efficiency bonus - reward for high throughput phases
        efficiency_bonus = 2.0 if vehicles_passed >= 6 else 0
        
        # Balance bonus - reward for maintaining balanced queues
        queue_imbalance = abs(np.sum(self.queues[0:2]) - np.sum(self.queues[2:4]))
        balance_bonus = 1.0 / (1.0 + queue_imbalance)  # 1 when balanced, approaches 0 when imbalanced
        
        # Calculate total reward
        reward = (
            throughput_reward
            - queue_penalty
            - wait_penalty
            - phase_change_cost
            + efficiency_bonus
            + balance_bonus
        )

        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {
            "vehicles_passed": vehicles_passed,
            "total_queues": np.sum(self.queues),
            "max_wait_time": np.max(self.wait_times),
            "phase_changes": int(phase_change_cost > 0),
            "reward_components": {
                "throughput": throughput_reward,
                "queue_penalty": -queue_penalty,
                "wait_penalty": -wait_penalty,
                "phase_change": -phase_change_cost,
                "efficiency_bonus": efficiency_bonus,
                "balance_bonus": balance_bonus
            }
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # For rgb_array mode, create figure without displaying
            fig, ax = plt.subplots(figsize=(5, 5))
            self._render_traffic_state(ax)
            fig.canvas.draw()
            # Convert to RGB array
            buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            rgb_array = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
            plt.close(fig)
            return rgb_array
        
        elif self.render_mode == "human":
            self._render_traffic_state(self.ax)
            plt.pause(0.1)

    def _render_traffic_state(self, ax):
        ax.clear()
        directions = ['N', 'S', 'E', 'W']
        colors = ['red', 'red', 'green', 'green'] if self.traffic_light_state == "NS_green" else ['green', 'green',
                                                                                                    'red', 'red']
        positions = [(0.5, 0.8), (0.5, 0.2), (0.8, 0.5), (0.2, 0.5)]

        for i, (pos, qlen, color) in enumerate(zip(positions, self.queues, colors)):
            ax.text(pos[0], pos[1], f"{directions[i]}: {qlen}\nWait: {self.wait_times[i]}",
                     color=color, fontsize=12, ha='center', va='center')

        ax.set_title(f"Step {self.step_count} - Light: {self.traffic_light_state}\n"
                      f"Total Queue: {np.sum(self.queues)} | Max Wait: {np.max(self.wait_times)}")
        ax.axis('off')

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close(self.fig)