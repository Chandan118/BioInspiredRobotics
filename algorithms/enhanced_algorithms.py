#!/usr/bin/env python3
"""
Bio-Inspired Robotics - Enhanced Algorithms Module
=================================================
Advanced navigation algorithms including Neural Network, ANFIS, 
Swarm Intelligence, and Deep Reinforcement Learning.

Author: Chandan Sheikder
Affiliation: Beijing Institute of Technology (BIT)
"""

import numpy as np
import math
import random
import pickle
import os
from collections import deque
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

# Optional imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from simulations.comprehensive_navigation import (
    NavigationAlgorithm, Environment, RobotState, SimulationResult
)


# =============================================================================
# NEURAL NETWORK BASED NAVIGATION
# =============================================================================

class NeuralNetworkNavigator(NavigationAlgorithm):
    """Navigation using Neural Network for action prediction"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("NeuralNetwork", env, robot_radius)
        
        self.state_dim = 10  # [lidar_8, goal_angle, goal_dist]
        self.hidden_layers = [64, 32, 16]
        
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            self.network = MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden_layers),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            self.is_trained = False
        else:
            self.network = None
            self.is_trained = False
    
    def get_state_features(self, state: RobotState) -> np.ndarray:
        """Extract features from current state"""
        features = []
        
        # Lidar-like obstacles detection (8 sectors)
        for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            dist = self._raycast(state.position[:2], ray_dir)
            features.append(dist / 10.0)  # Normalize
        
        # Goal angle
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        goal_angle = math.atan2(goal_dir[1], goal_dir[0]) - state.heading_rad
        goal_angle = self._normalize_angle(goal_angle)
        features.append(goal_angle / math.pi)
        
        # Goal distance
        goal_dist = np.linalg.norm(goal_dir)
        features.append(goal_dist / 50.0)  # Normalize
        
        return np.array(features)
    
    def _raycast(self, pos: np.ndarray, direction: np.ndarray, 
                max_dist: float = 10.0) -> float:
        """Raycast to find obstacle distance"""
        for dist in np.linspace(0, max_dist, 100):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1], margin=-self.robot_radius):
                    return dist
        return max_dist
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def train(self, training_data: List[Tuple[np.ndarray, np.ndarray]]):
        """Train neural network on collected data"""
        if not HAS_SKLEARN or not training_data:
            return
        
        X = np.array([d[0] for d in training_data])
        y = np.array([d[1] for d in training_data])
        
        # Scale inputs
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.network.fit(X_scaled, y)
        self.is_trained = True
        print(f"  Neural network trained on {len(training_data)} samples")
    
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute action using neural network"""
        features = self.get_state_features(state).reshape(1, -1)
        
        if self.is_trained and self.network is not None:
            features_scaled = self.scaler.transform(features)
            output = self.network.predict(features_scaled)[0]
            
            # Convert output to velocity
            linear_vel = max(0, min(0.35, output[0]))
            angular_vel = output[1]
            
            heading = state.heading_rad + angular_vel * 0.1
            action = np.array([math.cos(heading), math.sin(heading)]) * linear_vel
        else:
            # Fallback to reactive behavior
            action = self._reactive_action(state)
        
        state.mode = 'NEURAL_NET'
        return action
    
    def _reactive_action(self, state: RobotState) -> np.ndarray:
        """Fallback reactive behavior"""
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        goal_dist = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (goal_dist + 1e-6)
        
        # Simple obstacle avoidance
        front_dist = min(self._raycast(state.position[:2], 
                      np.array([math.cos(state.heading_rad), 
                              math.sin(state.heading_rad)])))
        
        if front_dist < 2.0:
            # Turn away from obstacle
            turn = 0.5 if random.random() > 0.5 else -0.5
            heading = state.heading_rad + turn
        else:
            heading = state.heading_rad
        
        return np.array([math.cos(heading), math.sin(heading)]) * state.current_speed


# =============================================================================
# ANFIS NAVIGATION (Adaptive Neuro-Fuzzy Inference System)
# =============================================================================

class ANFISNavigator(NavigationAlgorithm):
    """ANFIS-based navigation controller"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("ANFIS", env, robot_radius)
        
        # ANFIS structure parameters
        self.n_inputs = 3  # distance, angle, obstacle
        self.n_mfs = 3     # memberships per input
        self.rules = []
        
        # Initialize rule consequent parameters
        self.consequent_params = self._initialize_consequents()
        
        # Fuzzy membership function parameters (to be learned)
        self.mf_params = {
            'distance': {'centers': [0.5, 1.5, 3.0], 'widths': [0.5, 0.5, 0.5]},
            'angle': {'centers': [-0.5, 0.0, 0.5], 'widths': [0.3, 0.3, 0.3]},
            'obstacle': {'centers': [0.5, 1.5, 3.0], 'widths': [0.5, 0.5, 0.5]}
        }
        
    def _initialize_consequents(self) -> np.ndarray:
        """Initialize consequent parameters for each rule"""
        n_rules = self.n_mfs ** self.n_inputs
        return np.zeros((n_rules, 3))  # [linear, angular, bias]
    
    def _gaussian_mf(self, x: float, center: float, width: float) -> float:
        """Gaussian membership function"""
        return math.exp(-((x - center) ** 2) / (2 * width ** 2 + 1e-6))
    
    def _compute_firing_strengths(self, inputs: np.ndarray) -> np.ndarray:
        """Compute firing strength for each rule"""
        distances = np.linspace(0, 5, self.n_mfs)
        angles = np.linspace(-math.pi/2, math.pi/2, self.n_mfs)
        obstacles = np.linspace(0, 5, self.n_mfs)
        
        firing = []
        for i, d in enumerate(distances):
            for j, a in enumerate(angles):
                for k, o in enumerate(obstacles):
                    # Gaussian membership for each input
                    w1 = self._gaussian_mf(inputs[0], d, self.mf_params['distance']['widths'][i])
                    w2 = self._gaussian_mf(inputs[1], a, self.mf_params['angle']['widths'][j])
                    w3 = self._gaussian_mf(inputs[2], o, self.mf_params['obstacle']['widths'][k])
                    firing.append(w1 * w2 * w3)
        
        firing = np.array(firing)
        # Normalize
        if firing.sum() > 0:
            firing = firing / firing.sum()
        return firing
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through ANFIS"""
        firing = self._compute_firing_strengths(inputs)
        
        # Compute consequent for each rule
        outputs = []
        for i, (w, params) in enumerate(zip(firing, self.consequent_params)):
            output = np.dot(inputs, params[:2]) + params[2]
            outputs.append(w * output)
        
        # Weighted sum
        if sum(outputs) > 0:
            return np.array([sum(outputs), 0, 0])
        return np.array([0, 0, 0])
    
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute ANFIS navigation action"""
        # Get inputs
        front_dist = self._raycast(state.position[:2], 
                                  np.array([math.cos(state.heading_rad),
                                          math.sin(state.heading_rad)]))
        
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        goal_angle = math.atan2(goal_dir[1], goal_dir[0]) - state.heading_rad
        goal_angle = self._normalize_angle(goal_angle)
        
        goal_dist = np.linalg.norm(goal_dir)
        
        inputs = np.array([goal_dist / 50.0, goal_angle / math.pi, front_dist / 10.0])
        
        # Get ANFIS output
        output = self.forward(inputs)
        
        # Convert to velocity
        linear_vel = max(0.05, min(0.35, abs(output[0])))
        angular_vel = output[1]
        
        heading = state.heading_rad + angular_vel * 0.1
        
        action = np.array([math.cos(heading), math.sin(heading)]) * linear_vel
        state.mode = 'ANFIS'
        
        return action
    
    def _raycast(self, pos: np.ndarray, direction: np.ndarray, 
                max_dist: float = 10.0) -> float:
        """Raycast to find obstacle distance"""
        for dist in np.linspace(0, max_dist, 100):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1], margin=-self.robot_radius):
                    return dist
        return max_dist
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def train_epoch(self, training_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                   learning_rate: float = 0.01):
        """Train ANFIS on one epoch of data"""
        for inputs, targets in training_pairs:
            firing = self._compute_firing_strengths(inputs)
            
            # Gradient descent on consequent parameters
            output = self.forward(inputs)[0]
            error = targets[0] - output
            
            for i, (w, params) in enumerate(zip(firing, self.consequent_params)):
                # Update consequent: y = w * (x1*p1 + x2*p2 + p3)
                self.consequent_params[i] += learning_rate * error * w * np.array([inputs[0], inputs[1], 1])


# =============================================================================
# PARTICLE SWARM NAVIGATION
# =============================================================================

class ParticleSwarmNavigation(NavigationAlgorithm):
    """Navigation using Particle Swarm Optimization for path planning"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5,
                 n_particles: int = 30):
        super().__init__("ParticleSwarm", env, robot_radius)
        
        self.n_particles = n_particles
        self.n_waypoints = 5
        
        # PSO parameters
        self.w = 0.7      # Inertia
        self.c1 = 1.5     # Cognitive
        self.c2 = 2.0     # Social
        
        # Initialize swarm
        self.swarm = self._initialize_swarm()
        self.velocities = np.random.uniform(-1, 1, (n_particles, self.n_waypoints * 2))
        
        # Personal and global best
        self.personal_best = self.swarm.copy()
        self.personal_best_cost = [self._evaluate_path(p) for p in self.swarm]
        
        best_idx = np.argmin(self.personal_best_cost)
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_cost = self.personal_best_cost[best_idx]
        
        self.update_counter = 0
        
    def _initialize_swarm(self) -> np.ndarray:
        """Initialize particle positions"""
        swarm = []
        for _ in range(self.n_particles):
            # Generate random waypoints between start and goal
            waypoints = []
            for i in range(self.n_waypoints):
                t = (i + 1) / (self.n_waypoints + 1)
                x = self.env.start_pos[0] + t * (self.env.goal_pos[0] - self.env.start_pos[0])
                y = self.env.start_pos[1] + t * (self.env.goal_pos[1] - self.env.start_pos[1])
                # Add randomness
                x += np.random.uniform(-10, 10)
                y += np.random.uniform(-10, 10)
                waypoints.extend([x, y])
            swarm.append(np.array(waypoints))
        return np.array(swarm)
    
    def _evaluate_path(self, waypoints: np.ndarray) -> float:
        """Evaluate path cost (lower is better)"""
        path = [self.env.start_pos[:2].copy()]
        
        for i in range(0, len(waypoints), 2):
            path.append(np.array([waypoints[i], waypoints[i+1]]))
        path.append(self.env.goal_pos[:2].copy())
        
        total_cost = 0
        
        # Check path segments
        for i in range(len(path) - 1):
            dist = np.linalg.norm(path[i+1] - path[i])
            
            # Check for collisions
            for obs in self.env.get_all_obstacles():
                if obs.contains_point((path[i][0] + path[i+1][0])/2,
                                     (path[i][1] + path[i+1][1])/2):
                    total_cost += 1000  # Large penalty
            
            total_cost += dist
        
        # Distance to goal
        final_dist = np.linalg.norm(path[-1] - self.env.goal_pos[:2])
        total_cost += final_dist * 10
        
        return total_cost
    
    def _update_swarm(self):
        """Update particle positions and velocities"""
        r1, r2 = np.random.random((2, self.n_particles, self.n_waypoints * 2))
        
        self.velocities = (self.w * self.velocities +
                          self.c1 * r1 * (self.personal_best - self.swarm) +
                          self.c2 * r2 * (self.global_best - self.swarm))
        
        # Limit velocity
        max_vel = 5.0
        self.velocities = np.clip(self.velocities, -max_vel, max_vel)
        
        self.swarm += self.velocities
        
        # Evaluate new positions
        for i in range(self.n_particles):
            cost = self._evaluate_path(self.swarm[i])
            if cost < self.personal_best_cost[i]:
                self.personal_best[i] = self.swarm[i].copy()
                self.personal_best_cost[i] = cost
                
                if cost < self.global_best_cost:
                    self.global_best = self.swarm[i].copy()
                    self.global_best_cost = cost
    
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute action following optimized waypoints"""
        # Update swarm periodically
        if self.update_counter % 10 == 0:
            self._update_swarm()
        self.update_counter += 1
        
        # Get next waypoint from global best
        waypoints = self.global_best.reshape(-1, 2)
        
        # Find closest waypoint
        min_dist = float('inf')
        target_waypoint = waypoints[0]
        
        for wp in waypoints:
            dist = np.linalg.norm(wp - state.position[:2])
            if dist < min_dist:
                min_dist = dist
                target_waypoint = wp
        
        # Also check goal
        goal_dist = np.linalg.norm(self.env.goal_pos[:2] - state.position[:2])
        if goal_dist < min_dist:
            target_waypoint = self.env.goal_pos[:2]
        
        # Steer towards waypoint
        direction = target_waypoint - state.position[:2]
        dist = np.linalg.norm(direction)
        
        if dist < 1.0:
            action = np.array([0.0, 0.0])  # Reached waypoint
        else:
            direction = direction / dist
            action = direction * state.current_speed
        
        state.mode = 'PARTICLE_SWARM'
        return action


# =============================================================================
# ANT COLONY OPTIMIZATION NAVIGATION
# =============================================================================

class AntColonyNavigation(NavigationAlgorithm):
    """Navigation using Ant Colony Optimization for path finding"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5,
                 n_ants: int = 20, n_iterations: int = 50):
        super().__init__("AntColony", env, robot_radius)
        
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        
        # ACO parameters
        self.alpha = 1.0   # Pheromone importance
        self.beta = 2.0    # Heuristic importance
        self.rho = 0.5     # Pheromone evaporation rate
        self.Q = 100       # Pheromone deposit amount
        
        # Discretize space
        self.grid_size = 5.0  # meters
        self.grid = self._create_grid()
        self.pheromone = np.ones_like(self.grid) * 0.1
        
        # Best path found
        self.best_path = None
        self.best_cost = float('inf')
        
        self.iteration = 0
        
    def _create_grid(self) -> np.ndarray:
        """Create discretized grid of the environment"""
        x_range = np.arange(-10, 60, self.grid_size)
        y_range = np.arange(-10, 60, self.grid_size)
        
        grid = np.zeros((len(x_range), len(y_range)))
        
        # Mark obstacle cells
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                for obs in self.env.obstacles:
                    if obs.contains_point(x, y):
                        grid[i, j] = 1  # Obstacle
        
        return grid
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid.shape[0] and 
                    0 <= ny < self.grid.shape[1] and
                    self.grid[nx, ny] == 0):
                    neighbors.append((nx, ny))
        return neighbors
    
    def _position_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world position to grid index"""
        x = int((pos[0] + 10) / self.grid_size)
        y = int((pos[1] + 10) / self.grid_size)
        x = max(0, min(self.grid.shape[0] - 1, x))
        y = max(0, min(self.grid.shape[1] - 1, y))
        return x, y
    
    def _grid_to_position(self, x: int, y: int) -> np.ndarray:
        """Convert grid index to world position"""
        return np.array([x * self.grid_size - 10 + self.grid_size/2,
                         y * self.grid_size - 10 + self.grid_size/2,
                         0])
    
    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Heuristic distance between cells"""
        return 1.0 / (1 + np.sqrt((x1-x2)**2 + (y1-y2)**2))
    
    def _construct_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List:
        """Construct path for one ant using pheromone-guided search"""
        path = [start]
        current = start
        visited = {start}
        
        max_steps = self.grid.shape[0] * self.grid.shape[1]
        
        while current != goal and len(path) < max_steps:
            neighbors = self._get_neighbors(*current)
            neighbors = [n for n in neighbors if n not in visited]
            
            if not neighbors:
                break
            
            # Compute probabilities
            probabilities = []
            goal_x, goal_y = goal
            
            for nx, ny in neighbors:
                tau = self.pheromone[current[0], current[1]] ** self.alpha
                eta = self._heuristic(nx, ny, goal_x, goal_y) ** self.beta
                prob = tau * eta
                probabilities.append(prob)
            
            probabilities = np.array(probabilities)
            if probabilities.sum() > 0:
                probabilities /= probabilities.sum()
            else:
                probabilities = np.ones(len(neighbors)) / len(neighbors)
            
            # Select next cell
            idx = np.random.choice(len(neighbors), p=probabilities)
            next_cell = neighbors[idx]
            
            path.append(next_cell)
            visited.add(next_cell)
            current = next_cell
        
        return path
    
    def _update_pheromones(self, paths: List[List], costs: List[float]):
        """Update pheromone trails"""
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Deposit new pheromones
        for path, cost in zip(paths, costs):
            deposit = self.Q / (cost + 1e-6)
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                self.pheromone[x1, y1] += deposit
                self.pheromone[x2, y2] += deposit
    
    def _run_iteration(self) -> Tuple[List, float]:
        """Run one iteration of ACO"""
        start = self._position_to_grid(self.env.start_pos)
        goal = self._position_to_grid(self.env.goal_pos)
        
        paths = []
        costs = []
        
        for _ in range(self.n_ants):
            path = self._construct_path(start, goal)
            paths.append(path)
            
            # Compute cost
            cost = 0
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2) * self.grid_size
                cost += dist
            costs.append(cost)
            
            # Bonus for reaching goal
            if path[-1] == goal:
                cost -= 50
        
        # Update best
        best_idx = np.argmin(costs)
        if costs[best_idx] < self.best_cost:
            self.best_cost = costs[best_idx]
            self.best_path = paths[best_idx]
        
        # Update pheromones
        self._update_pheromones(paths, costs)
        
        return paths[best_idx], costs[best_idx]
    
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute ACO-guided navigation action"""
        # Run ACO iterations
        if self.iteration < self.n_iterations:
            self._run_iteration()
            self.iteration += 1
        
        if self.best_path is None or len(self.best_path) < 2:
            # Fallback
            goal_dir = self.env.goal_pos[:2] - state.position[:2]
            return goal_dir / (np.linalg.norm(goal_dir) + 1e-6) * state.current_speed
        
        # Find next waypoint on best path
        current_grid = self._position_to_grid(state.position)
        
        # Find closest point in path
        min_dist = float('inf')
        target_idx = 0
        
        for idx, (gx, gy) in enumerate(self.best_path):
            dist = np.sqrt((gx - current_grid[0])**2 + (gy - current_grid[1])**2)
            if dist < min_dist:
                min_dist = dist
                target_idx = idx
        
        # Get next waypoint
        if target_idx < len(self.best_path) - 1:
            target_grid = self.best_path[target_idx + 1]
        else:
            target_grid = self.best_path[-1]
        
        target_pos = self._grid_to_position(*target_grid)
        
        # Steer towards target
        direction = target_pos[:2] - state.position[:2]
        dist = np.linalg.norm(direction)
        
        if dist < 0.5:
            return np.array([0.0, 0.0])
        
        action = direction / dist * state.current_speed
        state.mode = 'ANT_COLONY'
        
        return action


# =============================================================================
# DEEP REINFORCEMENT LEARNING NAVIGATOR
# =============================================================================

class DRLNavigator(NavigationAlgorithm):
    """Deep Reinforcement Learning Navigation with DQN"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5,
                 state_dim: int = 15, action_dim: int = 7):
        super().__init__("DRL", env, robot_radius)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training = True
        self.total_steps = 0
        self.target_update_freq = 100
        
        if HAS_TF:
            self.dqn = self._build_dqn()
            self.target_dqn = self._build_dqn()
            self.target_dqn.set_weights(self.dqn.get_weights())
        else:
            self.dqn = None
            self.target_dqn = None
        
        # Replay buffer
        self.memory = deque(maxlen=10000)
        
        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        # Training history
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon': [],
            'losses': []
        }
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Action space
        self.actions = [
            np.array([0.35, 0.0]),    # Forward
            np.array([0.35, 0.5]),    # Forward-left
            np.array([0.35, -0.5]),   # Forward-right
            np.array([0.15, 0.8]),    # Turn left
            np.array([0.15, -0.8]),   # Turn right
            np.array([0.1, 0.0]),     # Slow forward
            np.array([0.0, 0.0]),     # Stop
        ]
    
    def _build_dqn(self) -> keras.Model:
        """Build Deep Q-Network"""
        model = keras.Sequential([
            layers.Dense(128, input_dim=self.state_dim, activation='relu',
                        kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.action_dim, activation='linear')
        ])
        
        model.compile(
            loss='huber',  # More robust than MSE
            optimizer=keras.optimizers.Adam(learning_rate=0.001)
        )
        
        return model
    
    def get_state(self, state: RobotState) -> np.ndarray:
        """Convert robot state to DQN input"""
        features = []
        
        # Lidar sectors (8 values)
        for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
            ray_dir = np.array([math.cos(angle + state.heading_rad),
                               math.sin(angle + state.heading_rad)])
            dist = self._raycast(state.position[:2], ray_dir)
            features.append(dist / 10.0)
        
        # Goal info (3 values)
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        goal_dist = np.linalg.norm(goal_dir)
        goal_angle = math.atan2(goal_dir[1], goal_dir[0]) - state.heading_rad
        goal_angle = self._normalize_angle(goal_angle)
        
        features.append(goal_dist / 50.0)  # Normalize distance
        features.append(self._normalize_angle(goal_angle) / math.pi)  # Normalize angle
        features.append(math.sin(state.heading_rad))  # Current heading sin
        features.append(math.cos(state.heading_rad))  # Current heading cos
        
        # Speed info (1 value)
        features.append(state.current_speed / 0.35)
        
        # Normalize velocity (2 values)
        if len(state.velocity) >= 2:
            features.append(state.velocity[0] / 0.35)
            features.append(state.velocity[1] / 0.35)
        
        return np.array(features[:self.state_dim])
    
    def _raycast(self, pos: np.ndarray, direction: np.ndarray,
                max_dist: float = 10.0) -> float:
        """Raycast to find obstacle distance"""
        for dist in np.linspace(0, max_dist, 50):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1], margin=-self.robot_radius):
                    return dist
        return max_dist
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if not self.training or random.random() > self.epsilon:
            if self.dqn is not None:
                q_values = self.dqn.predict(state.reshape(1, -1), verbose=0)[0]
                return np.argmax(q_values)
        
        return random.randint(0, self.action_dim - 1)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """Train DQN on batch from replay buffer"""
        if self.dqn is None or len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        # Compute target Q values
        current_q = self.dqn.predict(states, verbose=0)
        next_q = self.target_dqn.predict(next_states, verbose=0)
        
        for i in range(len(batch)):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + 0.99 * np.max(next_q[i])
            current_q[i, actions[i]] = target
        
        # Train
        history = self.dqn.fit(states, current_q, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        self.history['losses'].append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute DRL navigation action"""
        state_vec = self.get_state(state)
        
        # Select action
        action_idx = self.select_action(state_vec)
        action = self.actions[action_idx]
        
        # Store for training
        self.last_state = state_vec
        self.last_action = action_idx
        
        # Execute action
        new_heading = state.heading_rad + action[1] * 0.1
        velocity = np.array([math.cos(new_heading), math.sin(new_heading)]) * action[0]
        
        state.mode = 'DRL'
        
        return velocity
    
    def update(self, state: RobotState, action: int, reward: float,
              next_state_vec: np.ndarray, done: bool):
        """Update after taking action"""
        if self.training and hasattr(self, 'last_state'):
            self.remember(self.last_state, action, reward, next_state_vec, done)
            self.replay()
            
            self.current_episode_reward += reward
            self.current_episode_length += 1
            self.total_steps += 1
            
            # Update target network
            if self.total_steps % self.target_update_freq == 0:
                self.target_dqn.set_weights(self.dqn.get_weights())
            
            # Record history
            if done:
                self.history['episode_rewards'].append(self.current_episode_reward)
                self.history['episode_lengths'].append(self.current_episode_length)
                self.history['epsilon'].append(self.epsilon)
                
                self.current_episode_reward = 0
                self.current_episode_length = 0
    
    def save(self, filepath: str):
        """Save DQN model"""
        if self.dqn is not None:
            self.dqn.save_weights(filepath)
            with open(filepath + '.meta', 'wb') as f:
                pickle.dump({
                    'epsilon': self.epsilon,
                    'history': self.history
                }, f)
    
    def load(self, filepath: str):
        """Load DQN model"""
        if self.dqn is not None and os.path.exists(filepath):
            self.dqn.load_weights(filepath)
            self.target_dqn.set_weights(self.dqn.get_weights())
            
            if os.path.exists(filepath + '.meta'):
                with open(filepath + '.meta', 'rb') as f:
                    meta = pickle.load(f)
                    self.epsilon = meta.get('epsilon', self.epsilon_min)


# =============================================================================
# HYBRID NAVIGATION SYSTEM
# =============================================================================

class HybridNavigationSystem(NavigationAlgorithm):
    """Hybrid system combining multiple navigation strategies"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("Hybrid", env, robot_radius)
        
        # Component algorithms
        self.components = {
            'bug': BugFollower(env, robot_radius),
            'potential': PotentialFieldFollower(env, robot_radius),
            'dwa': DWAFollower(env, robot_radius),
        }
        
        # Weights for fusion
        self.weights = np.array([0.3, 0.3, 0.4])
        
        # Fusion parameters
        self.confidence_thresholds = {
            'obstacle_density': 0.7,
            'goal_visibility': 0.8,
            'tight_space': 0.3
        }
        
    def _compute_confidence(self, state: RobotState) -> Dict[str, float]:
        """Compute confidence scores for each component"""
        confidence = {}
        
        # Obstacle density
        min_dist = float('inf')
        for angle in np.linspace(0, 2*math.pi, 8):
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            dist = self._raycast(state.position[:2], ray_dir)
            min_dist = min(min_dist, dist)
        confidence['obstacle_density'] = 1.0 - min(min_dist / 5.0, 1.0)
        
        # Goal visibility
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        goal_clear = not self._has_obstacle_in_direction(state.position[:2], goal_dir)
        confidence['goal_visibility'] = 1.0 if goal_clear else 0.0
        
        # Tight space
        confidence['tight_space'] = confidence['obstacle_density']
        
        return confidence
    
    def _raycast(self, pos: np.ndarray, direction: np.ndarray,
                max_dist: float = 10.0) -> float:
        """Raycast to find obstacle distance"""
        for dist in np.linspace(0, max_dist, 50):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1], margin=-self.robot_radius):
                    return dist
        return max_dist
    
    def _has_obstacle_in_direction(self, pos: np.ndarray, direction: np.ndarray) -> bool:
        """Check if obstacle blocks direction"""
        dist = self._raycast(pos, direction)
        return dist < np.linalg.norm(self.env.goal_pos[:2] - pos)
    
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute hybrid navigation action"""
        confidence = self._compute_confidence(state)
        
        # Adjust weights based on confidence
        if confidence['tight_space'] > self.confidence_thresholds['tight_space']:
            # More Bug algorithm in tight spaces
            self.weights = np.array([0.5, 0.2, 0.3])
        elif confidence['goal_visibility'] > self.confidence_thresholds['goal_visibility']:
            # More DWA when goal is visible
            self.weights = np.array([0.2, 0.3, 0.5])
        else:
            # Balanced
            self.weights = np.array([0.3, 0.3, 0.4])
        
        # Get actions from components
        actions = []
        for name, component in self.components.items():
            action = component.compute_action(state)
            actions.append(action)
        
        # Weighted fusion
        fused_action = np.zeros(2)
        for action, weight in zip(actions, self.weights):
            fused_action += action * weight
        
        # Normalize
        if np.linalg.norm(fused_action) > 0:
            fused_action = fused_action / np.linalg.norm(fused_action) * state.current_speed
        
        state.mode = 'HYBRID'
        return fused_action


# Helper classes for HybridNavigationSystem

class BugFollower(NavigationAlgorithm):
    def __init__(self, env, robot_radius):
        super().__init__("Bug", env, robot_radius)
        self.detection_radius = 8.0
        
    def compute_action(self, state: RobotState) -> np.ndarray:
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dir_to_goal = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
        
        min_dist = float('inf')
        for angle in np.linspace(0, 2*math.pi, 8):
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            dist = self._raycast(state.position[:2], ray_dir)
            min_dist = min(min_dist, dist)
        
        if min_dist < self.detection_radius:
            # Wall follow
            tangent = np.array([-dir_to_goal[1], dir_to_goal[0]])
            return tangent * state.current_speed
        
        return dir_to_goal * state.current_speed
    
    def _raycast(self, pos, direction, max_dist=10.0):
        for dist in np.linspace(0, max_dist, 50):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1], margin=-self.robot_radius):
                    return dist
        return max_dist


class PotentialFieldFollower(NavigationAlgorithm):
    def __init__(self, env, robot_radius):
        super().__init__("Potential", env, robot_radius)
        
    def compute_action(self, state: RobotState) -> np.ndarray:
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        attractive = goal_dir / (np.linalg.norm(goal_dir) + 1e-6) * 2.0
        
        repulsive = np.array([0.0, 0.0])
        for obs in self.env.get_all_obstacles():
            center = np.array([obs.x + obs.width/2, obs.y + obs.height/2])
            to_robot = state.position[:2] - center
            dist = np.linalg.norm(to_robot)
            if dist < 5.0:
                repulsive += to_robot / (dist**2 + 1e-6)
        
        force = attractive + repulsive
        if np.linalg.norm(force) > 0:
            return force / np.linalg.norm(force) * state.current_speed
        return attractive / np.linalg.norm(attractive) * state.current_speed


class DWAFollower(NavigationAlgorithm):
    def __init__(self, env, robot_radius):
        super().__init__("DWA", env, robot_radius)
        
    def compute_action(self, state: RobotState) -> np.ndarray:
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        goal_dist = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (goal_dist + 1e-6)
        
        # Simple DWA-like: align with goal but avoid nearby obstacles
        front_dist = float('inf')
        for angle_offset in np.linspace(-0.5, 0.5, 5):
            ray_dir = np.array([math.cos(state.heading_rad + angle_offset),
                               math.sin(state.heading_rad + angle_offset)])
            dist = self._raycast(state.position[:2], ray_dir)
            front_dist = min(front_dist, dist)
        
        if front_dist < 2.0:
            # Steer away
            tangent = np.array([-dir_to_goal[1], dir_to_goal[0]])
            return tangent * state.current_speed * 0.5
        
        return dir_to_goal * state.current_speed
    
    def _raycast(self, pos, direction, max_dist=10.0):
        for dist in np.linspace(0, max_dist, 50):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1], margin=-self.robot_radius):
                    return dist
        return max_dist
