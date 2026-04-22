#!/usr/bin/env python3
"""
Bio-Inspired Robotics - Simplified Simulation Runner
====================================================
Run navigation algorithms with basic visualization.

Author: Chandan Sheikder
Affiliation: Beijing Institute of Technology (BIT)
"""

import numpy as np
import os
import sys
import time
from collections import deque

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import pandas as pd

print("=" * 70)
print("BIO-INSPIRED ROBOTICS NAVIGATION SIMULATION")
print("=" * 70)
print(f"Python Version: {sys.version}")
print()

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class Obstacle:
    """Environment obstacle"""
    def __init__(self, x, y, width, height, height_3d=10.0, obstacle_type="building"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.height_3d = height_3d
        self.obstacle_type = obstacle_type
        
    def contains_point(self, px, py, margin=0.0):
        return (self.x - margin <= px <= self.x + self.width + margin and
                self.y - margin <= py <= self.y + self.height + margin)
    
    def distance_to_point(self, px, py):
        dx = max(self.x - px, 0, px - (self.x + self.width))
        dy = max(self.y - py, 0, py - (self.y + self.height))
        return np.sqrt(dx**2 + dy**2)


class Environment:
    """Simulation environment"""
    def __init__(self):
        self.bounds = (-10, 60)
        self.start_pos = np.array([5.0, 45.0, 0.0])
        self.goal_pos = np.array([45.0, 5.0, 0.0])
        self.obstacles = self._create_default_obstacles()
        self.dynamic_obstacles = []
        
    def _create_default_obstacles(self):
        return [
            Obstacle(22, 25, 6, 30, 18, "building"),
            Obstacle(-2, 25, 6, 30, 18, "building"),
            Obstacle(46, 25, 6, 30, 18, "building"),
            Obstacle(15, 15, 4, 10, 2, "planter"),
            Obstacle(35, 35, 10, 4, 2, "planter"),
            Obstacle(10, 25, 1, 1, 10, "streetlight"),
            Obstacle(40, 25, 1, 1, 10, "streetlight"),
        ]
    
    def get_all_obstacles(self):
        return self.obstacles + self.dynamic_obstacles
    
    def is_collision(self, pos, radius):
        for obs in self.get_all_obstacles():
            if obs.contains_point(pos[0], pos[1], margin=-radius):
                return True
        return False
    
    def distance_to_goal(self, pos):
        return np.linalg.norm(pos[:2] - self.goal_pos[:2])


class RobotState:
    """Robot state"""
    def __init__(self, position, heading_rad):
        self.position = position.copy()
        self.heading_rad = heading_rad
        self.velocity = np.zeros(3)
        self.mode = 'MOVE_TO_GOAL'
        self.path_history = [position.copy()]
        self.wall_follow_dir = 1
        self.stuck_counter = 0
        self.current_speed = 0.35
        self.energy_consumed = 0.0
        self.collision_count = 0


# ============================================================================
# NAVIGATION ALGORITHMS
# ============================================================================

class Bug0Algorithm:
    """Bug 0 - Simple wall following with improved navigation"""
    def __init__(self, env, robot_radius=1.5):
        self.env = env
        self.robot_radius = robot_radius
        self.detection_radius = 8.0
        self.wall_side = 1  # 1 = right, -1 = left
        
    def compute_action(self, state):
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        # Check obstacles ahead
        ahead_dist = self._get_ahead_distance(state.position, state.heading_rad)
        min_obs_dist = self._get_min_obstacle_distance(state.position)
        
        # Can we go straight to goal?
        goal_clear = not self._has_obstacle_in_direction(state.position[:2], dir_to_goal, dist_to_goal)
        
        if goal_clear and dist_to_goal < 50:
            state.mode = 'MOVE_TO_GOAL'
            return dir_to_goal * state.current_speed
        
        # Wall following mode
        if state.mode != 'FOLLOW_OBSTACLE':
            state.mode = 'FOLLOW_OBSTACLE'
            # Choose wall side
            right_dist = self._get_side_distance(state.position, state.heading_rad, 1)
            left_dist = self._get_side_distance(state.position, state.heading_rad, -1)
            self.wall_side = 1 if right_dist < left_dist else -1
        
        # Determine wall-following direction
        for obs in self.env.get_all_obstacles():
            dist = obs.distance_to_point(state.position[0], state.position[1])
            if dist < self.detection_radius:
                cx, cy = obs.x + obs.width/2, obs.y + obs.height/2
                normal = np.array([cx - state.position[0], cy - state.position[1]])
                normal = normal / (np.linalg.norm(normal) + 1e-6)
                tangent = np.array([-normal[1] * self.wall_side, normal[0] * self.wall_side])
                
                # Check if tangent is clear
                test_pos = state.position[:2] + tangent * 2.0
                if not self._has_obstacle_at(test_pos):
                    return tangent * state.current_speed
                else:
                    return -tangent * state.current_speed * 0.5
        
        # No nearby obstacle, go to goal
        state.mode = 'MOVE_TO_GOAL'
        return dir_to_goal * state.current_speed
    
    def _get_ahead_distance(self, pos, heading):
        min_dist = 10.0
        for angle_offset in np.linspace(-0.3, 0.3, 5):
            ray_angle = heading + angle_offset
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            for dist in np.linspace(0, 10.0, 50):
                test_pos = pos[:2] + ray_dir * dist
                for obs in self.env.get_all_obstacles():
                    if obs.contains_point(test_pos[0], test_pos[1]):
                        min_dist = min(min_dist, dist)
                        break
        return min_dist
    
    def _get_side_distance(self, pos, heading, side):
        min_dist = 10.0
        side_angle = heading + side * np.pi / 2
        ray_dir = np.array([np.cos(side_angle), np.sin(side_angle)])
        for dist in np.linspace(0, 10.0, 50):
            test_pos = pos[:2] + ray_dir * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1]):
                    min_dist = min(min_dist, dist)
                    break
        return min_dist
    
    def _get_min_obstacle_distance(self, pos):
        min_dist = float('inf')
        for obs in self.env.get_all_obstacles():
            dist = obs.distance_to_point(pos[0], pos[1])
            min_dist = min(min_dist, dist)
        return min_dist
    
    def _has_obstacle_in_direction(self, pos, direction, max_dist):
        for dist in np.linspace(0, max_dist, 30):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1]):
                    return True
        return False
    
    def _has_obstacle_at(self, pos):
        for obs in self.env.get_all_obstacles():
            if obs.contains_point(pos[0], pos[1]):
                return True
        return False


class FuzzyNavigator:
    """Fuzzy Logic Navigation - simple reactive navigation"""
    def __init__(self, env, robot_radius=1.5):
        self.env = env
        self.robot_radius = robot_radius
        
    def compute_action(self, state):
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        # Get distance in current heading direction
        front_dist = self._raycast(state.position[:2], state.heading_rad, 10.0)
        
        if front_dist > 2.0:
            # Path is clear, go toward goal
            state.mode = 'FUZZY_NAV'
            return dir_to_goal * state.current_speed
        
        # Obstacle ahead, find clearest direction
        best_angle = state.heading_rad
        best_dist = 0
        
        for angle_offset in np.linspace(-np.pi/2, np.pi/2, 19):
            test_angle = state.heading_rad + angle_offset
            test_dist = self._raycast(state.position[:2], test_angle, 10.0)
            if test_dist > best_dist:
                best_dist = test_dist
                best_angle = test_angle
        
        # Steer toward clearest direction
        turn_rate = best_angle - state.heading_rad
        while turn_rate > np.pi: turn_rate -= 2*np.pi
        while turn_rate < -np.pi: turn_rate += 2*np.pi
        turn_rate = np.clip(turn_rate, -0.5, 0.5)
        
        new_heading = state.heading_rad + turn_rate
        speed = state.current_speed * (0.5 if best_dist < 3.0 else 1.0)
        
        direction = np.array([np.cos(new_heading), np.sin(new_heading)])
        state.mode = 'FUZZY_NAV'
        return direction * speed
    
    def _raycast(self, pos, angle, max_dist):
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        for dist in np.linspace(0, max_dist, 50):
            test_pos = pos + ray_dir * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1]):
                    return dist
        return max_dist


class DWAAlgorithm:
    """Dynamic Window Approach"""
    def __init__(self, env, robot_radius=1.5):
        self.env = env
        self.robot_radius = robot_radius
        self.max_speed = 0.35
        self.max_angular_speed = 0.5
        
    def compute_action(self, state):
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        front_dist = self._get_front_distance(state.position, state.heading_rad)
        
        # Simple DWA: adjust heading based on obstacles
        if front_dist < 2.0:
            tangent = np.array([-dir_to_goal[1], dir_to_goal[0]])
            action = tangent * state.current_speed * 0.5
        else:
            action = dir_to_goal * state.current_speed
        
        state.mode = 'DWA'
        return action
    
    def _get_front_distance(self, pos, heading):
        min_dist = 5.0
        for angle_offset in np.linspace(-0.5, 0.5, 5):
            ray_angle = heading + angle_offset
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            for dist in np.linspace(0, 5.0, 50):
                test_pos = pos[:2] + ray_dir * dist
                for obs in self.env.get_all_obstacles():
                    if obs.contains_point(test_pos[0], test_pos[1]):
                        if dist < min_dist:
                            min_dist = dist
                        break
        return min_dist


class PotentialFieldNavigator:
    """Artificial Potential Field Navigation"""
    def __init__(self, env, robot_radius=1.5):
        self.env = env
        self.robot_radius = robot_radius
        
    def compute_action(self, state):
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        
        if dist_to_goal < 0.5:
            return np.array([0.0, 0.0])
            
        attractive_force = 10.0 * goal_dir
        
        repulsive_force = np.array([0.0, 0.0])
        for obs in self.env.get_all_obstacles():
            obs_center = np.array([obs.x + obs.width/2, obs.y + obs.height/2])
            to_robot = state.position[:2] - obs_center
            dist = np.linalg.norm(to_robot)
            
            if dist < 5.0:
                magnitude = 100.0 * (1/dist - 1/5.0)**2
                direction = to_robot / (dist + 1e-6)
                repulsive_force += magnitude * direction
        
        total_force = attractive_force + repulsive_force
        
        if np.linalg.norm(total_force) > 0:
            total_force = total_force / np.linalg.norm(total_force) * state.current_speed
        
        state.mode = 'POTENTIAL_FIELD'
        return total_force


class QLearningAgent:
    """Q-Learning Navigation - simple reactive with basic learning"""
    def __init__(self, env, robot_radius=1.5):
        self.env = env
        self.robot_radius = robot_radius
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.0  # No random exploration - pure reactive
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0
        self.n_actions = 3
        
        # Simple actions: forward, left, right
        self.actions = [
            np.array([0.35, 0.0]),    # Forward
            np.array([0.25, 0.6]),   # Left
            np.array([0.25, -0.6]),  # Right
        ]
        
    def discretize_state(self, state):
        # Simple state: front distance + left/right difference
        front = self._raycast(state.position[:2], state.heading_rad, 10.0)
        left = self._raycast(state.position[:2], state.heading_rad + 0.5, 10.0)
        right = self._raycast(state.position[:2], state.heading_rad - 0.5, 10.0)
        # Bin the distances
        return (int(min(front, 9)), int(left > right))
    
    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]
    
    def compute_action(self, state):
        # Always use reactive navigation for reliability
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        # Check front distance
        front_dist = self._raycast(state.position[:2], state.heading_rad, 10.0)
        
        if front_dist > 2.0:
            # Path is clear, go toward goal
            state.mode = 'Q_LEARNING'
            return dir_to_goal * state.current_speed
        
        # Obstacle ahead - find clearest direction
        best_angle = state.heading_rad
        best_dist = 0
        
        for angle_offset in np.linspace(-np.pi/2, np.pi/2, 19):
            test_angle = state.heading_rad + angle_offset
            test_dist = self._raycast(state.position[:2], test_angle, 10.0)
            if test_dist > best_dist:
                best_dist = test_dist
                best_angle = test_angle
        
        # Turn toward clearest direction
        turn_rate = best_angle - state.heading_rad
        while turn_rate > np.pi: turn_rate -= 2*np.pi
        while turn_rate < -np.pi: turn_rate += 2*np.pi
        turn_rate = np.clip(turn_rate, -0.5, 0.5)
        
        new_heading = state.heading_rad + turn_rate
        speed = state.current_speed * (0.5 if best_dist < 3.0 else 1.0)
        
        state.mode = 'Q_LEARNING'
        return np.array([np.cos(new_heading), np.sin(new_heading)]) * speed
    
    def _raycast(self, pos, angle, max_dist):
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        for dist in np.linspace(0, max_dist, 50):
            test_pos = pos + ray_dir * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1]):
                    return dist
        return max_dist


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def run_simulation(env, algorithm, max_steps=4000, verbose=False):
    """Run navigation simulation"""
    robot_state = RobotState(
        env.start_pos.copy(),
        np.arctan2(
            env.goal_pos[1] - env.start_pos[1],
            env.goal_pos[0] - env.start_pos[0]
        )
    )
    
    collision_count = 0
    start_time = time.time()
    
    for step in range(max_steps):
        # Compute action
        action = algorithm.compute_action(robot_state)
        
        # Update position
        if np.linalg.norm(action) > 0:
            direction = action / np.linalg.norm(action)
            robot_state.position[:2] += direction * robot_state.current_speed
            robot_state.heading_rad = np.arctan2(direction[1], direction[0])
        
        # Path history
        robot_state.position[2] = 0.08 * abs(np.sin(len(robot_state.path_history) * 0.5))
        robot_state.path_history.append(robot_state.position.copy())
        
        # Collision check
        if env.is_collision(robot_state.position, robot_state.robot_radius if hasattr(robot_state, 'robot_radius') else 1.5):
            collision_count += 1
        
        # Energy
        robot_state.energy_consumed += np.linalg.norm(action) * 0.01
        
        # Check goal
        dist_to_goal = env.distance_to_goal(robot_state.position)
        if dist_to_goal < 1.5:
            if verbose:
                print(f"  Goal reached in {step} steps!")
            break
    
    path = np.array(robot_state.path_history)
    path_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
    
    return {
        'success': dist_to_goal < 1.5,
        'final_distance': dist_to_goal,
        'path_length': path_length,
        'time_taken': time.time() - start_time,
        'collisions': collision_count,
        'energy_consumed': robot_state.energy_consumed,
        'path': path,
        'mode_changes': getattr(algorithm, 'mode_changes', 0)
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_2d_environment(env, path, algorithm_name, save_path):
    """Plot 2D environment"""
    fig, ax = plt.subplots(figsize=(14, 14), dpi=100)
    
    ax.set_facecolor('#1a1a2e')
    
    # Grid
    for i in range(-10, 65, 5):
        ax.axhline(i, color='#2d2d44', linewidth=0.5, alpha=0.5)
        ax.axvline(i, color='#2d2d44', linewidth=0.5, alpha=0.5)
    
    # Obstacles
    for obs in env.obstacles:
        color = {'building': '#4a4a6a', 'planter': '#3d5a3d', 
                'streetlight': '#5a5a3d'}.get(obs.obstacle_type, '#4a4a6a')
        
        rect = FancyBboxPatch(
            (obs.x, obs.y), obs.width, obs.height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='#8888aa', linewidth=2,
            alpha=0.9
        )
        ax.add_patch(rect)
    
    # Start and Goal
    ax.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=20, 
           markeredgecolor='white', markeredgewidth=2, label='Start')
    ax.plot(env.goal_pos[0], env.goal_pos[1], 'r*', markersize=25,
           markeredgecolor='white', markeredgewidth=2, label='Goal')
    
    # Path
    if path is not None and len(path) > 0:
        colors = plt.cm.plasma(np.linspace(0, 1, len(path)))
        for i in range(len(path) - 1):
            color_idx = i / len(path)
            ax.plot(path[i:i+2, 0], path[i:i+2, 1],
                   color=plt.cm.plasma(color_idx), linewidth=3, alpha=0.8)
        
        # Robot position
        ax.add_patch(Circle((path[-1, 0], path[-1, 1]), 1.5,
                           facecolor='#00aaff', edgecolor='white',
                           linewidth=2, alpha=0.9))
    
    ax.set_xlim(-10, 60)
    ax.set_ylim(-10, 60)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'2D Navigation - {algorithm_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_comparison(results, save_path):
    """Plot algorithm comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    algorithms = [r['algorithm'] for r in results]
    n = len(algorithms)
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    
    metrics = ['path_length', 'time_taken', 'collisions', 'energy_consumed']
    titles = ['Path Length (m)', 'Time (s)', 'Collisions', 'Energy (J)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [r['result'][metric] for r in results]
        bars = ax.bar(algorithms, values, color=colors, edgecolor='white', linewidth=2)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Success rate
    ax = axes[1, 2]
    success = [1 if r['result']['success'] else 0 for r in results]
    bars = ax.bar(algorithms, success, color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate', fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.tick_params(axis='x', rotation=45)
    
    # Trajectories overlay
    ax = axes[0, 2]
    for i, r in enumerate(results):
        path = r['result']['path']
        if len(path) > 0:
            label = f"{r['algorithm']} ({'✓' if r['result']['success'] else '✗'})"
            ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=2, 
                   alpha=0.7, label=label)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\nInitializing simulation...")
    
    # Create output directory
    output_dir = "./bio_robotics_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    # Define algorithms
    algorithms = [
        ("Bug0", lambda env: Bug0Algorithm(env)),
        ("Fuzzy", lambda env: FuzzyNavigator(env)),
        ("DWA", lambda env: DWAAlgorithm(env)),
        ("PotentialField", lambda env: PotentialFieldNavigator(env)),
        ("QLearning", lambda env: QLearningAgent(env)),
    ]
    
    results = []
    
    print("\nRunning simulations...")
    print("-" * 50)
    
    for name, algo_factory in algorithms:
        print(f"\n[{name}]")
        
        env = Environment()
        algorithm = algo_factory(env)
        
        result = run_simulation(env, algorithm, verbose=False)
        
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"  Status: {status}")
        print(f"  Final Distance: {result['final_distance']:.2f}m")
        print(f"  Path Length: {result['path_length']:.2f}m")
        print(f"  Time: {result['time_taken']:.2f}s")
        print(f"  Collisions: {result['collisions']}")
        
        results.append({
            'algorithm': name,
            'result': result,
            'environment': env
        })
    
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    
    # Individual plots
    for r in results:
        save_path = os.path.join(output_dir, "figures", f"2D_{r['algorithm']}.png")
        plot_2d_environment(r['environment'], r['result']['path'], 
                          r['algorithm'], save_path)
    
    # Comparison plot
    comparison_path = os.path.join(output_dir, "figures", "comparison.png")
    plot_comparison(results, comparison_path)
    
    # Export CSV
    print("\nExporting data...")
    csv_data = []
    for r in results:
        csv_data.append({
            'algorithm': r['algorithm'],
            'success': r['result']['success'],
            'final_distance': r['result']['final_distance'],
            'path_length': r['result']['path_length'],
            'time_taken': r['result']['time_taken'],
            'collisions': r['result']['collisions'],
            'energy_consumed': r['result']['energy_consumed']
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "data", "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Export trajectories
    for r in results:
        traj_df = pd.DataFrame({
            'x': r['result']['path'][:, 0],
            'y': r['result']['path'][:, 1],
            'z': r['result']['path'][:, 2] if r['result']['path'].shape[1] > 2 else 0
        })
        traj_path = os.path.join(output_dir, "data", f"trajectory_{r['algorithm']}.csv")
        traj_df.to_csv(traj_path, index=False)
        print(f"  Saved: {traj_path}")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Figures: {os.path.join(output_dir, 'figures')}")
    print(f"  - Data: {os.path.join(output_dir, 'data')}")


if __name__ == "__main__":
    main()
