#!/usr/bin/env python3
"""
Bio-Inspired Robotics - Extended Simulation
==========================================
Additional algorithms: Neural Network, ANFIS, PSO, ACO, Hybrid
"""

import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import pandas as pd

print("=" * 70)
print("EXTENDED BIO-INSPIRED ROBOTICS SIMULATION")
print("Advanced Algorithms: Neural Network, ANFIS, PSO, ACO, Hybrid")
print("=" * 70)

# ============================================================================
# ENVIRONMENT (same as before)
# ============================================================================

class Obstacle:
    def __init__(self, x, y, width, height, height_3d=10.0, obstacle_type="building"):
        self.x, self.y = x, y
        self.width, self.height = width, height
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
    def __init__(self):
        self.bounds = (-10, 60)
        self.start_pos = np.array([5.0, 45.0, 0.0])
        self.goal_pos = np.array([45.0, 5.0, 0.0])
        self.obstacles = [
            Obstacle(22, 25, 6, 30, 18),
            Obstacle(-2, 25, 6, 30, 18),
            Obstacle(46, 25, 6, 30, 18),
            Obstacle(15, 15, 4, 10, 2, "planter"),
            Obstacle(35, 35, 10, 4, 2, "planter"),
        ]
        
    def is_collision(self, pos, radius):
        for obs in self.obstacles:
            if obs.contains_point(pos[0], pos[1], margin=-radius):
                return True
        return False
    
    def distance_to_goal(self, pos):
        return np.linalg.norm(pos[:2] - self.goal_pos[:2])


# ============================================================================
# ADVANCED ALGORITHMS
# ============================================================================

class NeuralNetworkNavigator:
    """Neural Network based navigation"""
    def __init__(self, env):
        self.env = env
        # Simple perceptron-based approach
        self.weights = np.random.randn(5, 2) * 0.1
        self.training_data = []
        
    def get_features(self, pos, heading):
        # Features: front dist, left dist, right dist, goal angle, goal dist
        front = self._raycast(pos, heading, 5.0)
        left = self._raycast(pos, heading + 0.5, 5.0)
        right = self._raycast(pos, heading - 0.5, 5.0)
        goal_dir = self.env.goal_pos[:2] - pos[:2]
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0]) - heading
        goal_dist = np.linalg.norm(goal_dir)
        return np.array([front/5, left/5, right/5, goal_angle/np.pi, goal_dist/60])
    
    def _raycast(self, pos, angle, max_dist):
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        for dist in np.linspace(0, max_dist, 30):
            test = pos[:2] + ray_dir * dist
            for obs in self.env.obstacles:
                if obs.contains_point(test[0], test[1]):
                    return dist
        return max_dist
    
    def compute_action(self, state):
        # Handle both array and object access for compatibility
        if hasattr(state, 'position'):
            pos = state.position
            heading = state.heading_rad
        else:
            pos = state[:3]
            heading = state[2] if len(state) > 2 else 0
        
        features = self.get_features(pos, heading)
        
        # Simple NN forward pass
        output = np.tanh(np.dot(features, self.weights))
        
        # Convert to velocity
        turn = output[0]
        speed = (output[1] + 1) / 2 * 0.35
        
        new_heading = heading + turn * 0.3
        direction = np.array([np.cos(new_heading), np.sin(new_heading)])
        velocity = direction * speed
        
        # Obstacle avoidance: check if path is clear
        front_dist = self._raycast(pos, heading, 5.0)
        left_dist = self._raycast(pos, heading + 0.5, 5.0)
        right_dist = self._raycast(pos, heading - 0.5, 5.0)
        
        if front_dist < 2.0:
            # Emergency turn - choose clearer direction
            if left_dist > right_dist:
                new_heading = heading + 0.6
            else:
                new_heading = heading - 0.6
            direction = np.array([np.cos(new_heading), np.sin(new_heading)])
            velocity = direction * speed * 0.5
        
        # Also check goal direction for biased steering
        goal_dir = self.env.goal_pos[:2] - pos[:2]
        goal_dist = np.linalg.norm(goal_dir)
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0])
        angle_diff = goal_angle - heading
        while angle_diff > np.pi: angle_diff -= 2*np.pi
        while angle_diff < -np.pi: angle_diff += 2*np.pi
        
        # Blend NN output with goal direction
        velocity = velocity + goal_dir / (goal_dist + 1e-6) * 0.1
        velocity = velocity / (np.linalg.norm(velocity) + 1e-6) * speed
        
        state.mode = 'NEURAL_NET'
        return velocity


class ANFISNavigator:
    """ANFIS-based navigation"""
    def __init__(self, env):
        self.env = env
        # Rule parameters
        self.rules = [
            {'dist_center': 1.0, 'angle_center': 0.0, 'steering': 0.0},
            {'dist_center': 2.0, 'angle_center': 0.5, 'steering': 0.5},
            {'dist_center': 2.0, 'angle_center': -0.5, 'steering': -0.5},
            {'dist_center': 3.0, 'angle_center': 0.0, 'steering': 0.0},
        ]
        
    def gaussian_mf(self, x, center, width):
        return np.exp(-((x - center)**2) / (2 * width**2 + 1e-6))
    
    def compute_action(self, state):
        # Handle both array and object access for compatibility
        if hasattr(state, 'position'):
            pos = state.position
            heading = state.heading_rad
        else:
            pos = state[:3]
            heading = state[2] if len(state) > 2 else 0
        
        goal_dir = self.env.goal_pos[:2] - pos[:2]
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0]) - heading
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        goal_dist = np.linalg.norm(goal_dir)
        
        # Compute firing strengths
        steering = 0.0
        total_weight = 0.0
        
        for rule in self.rules:
            dist_mf = self.gaussian_mf(goal_dist/10, rule['dist_center'], 0.5)
            angle_mf = self.gaussian_mf(goal_angle, rule['angle_center'], 0.3)
            weight = dist_mf * angle_mf
            
            steering += weight * rule['steering']
            total_weight += weight
        
        if total_weight > 0:
            steering /= total_weight
        
        new_heading = heading + steering * 0.2
        direction = np.array([np.cos(new_heading), np.sin(new_heading)])
        
        # Get current speed from state or use default
        speed = state.current_speed if hasattr(state, 'current_speed') else 0.35
        velocity = direction * speed
        
        state.mode = 'ANFIS'
        return velocity


class ParticleSwarmNavigator:
    """Particle Swarm Optimization navigation"""
    def __init__(self, env):
        self.env = env
        self.n_particles = 20
        self.best_path = None
        self.best_cost = float('inf')
        
    def compute_action(self, state):
        # Simplified PSO: generate waypoints towards goal
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        goal_dist = np.linalg.norm(goal_dir)
        
        if goal_dist < 1.0:
            return np.array([0.0, 0.0])
        
        # Normalize goal_dir properly
        direction = goal_dir / (goal_dist + 1e-6)
        
        # Add some randomness based on particles
        if self.best_cost < 100 and self.best_cost > 0:
            random_offset = np.random.randn(2) * 2 * (1 - self.best_cost / 100)
        else:
            random_offset = np.random.randn(2) * 0.5
        
        direction = direction + random_offset * 0.1
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        velocity = direction * state.current_speed
        
        state.mode = 'PSO'
        return velocity


class AntColonyNavigator:
    """Ant Colony Optimization navigation"""
    def __init__(self, env):
        self.env = env
        self.pheromone = np.ones((10, 10)) * 0.1
        self.best_trail = None
        
    def compute_action(self, state):
        # Handle both array and object access for compatibility
        if hasattr(state, 'position'):
            pos = state.position
            speed = state.current_speed
        else:
            pos = state[:3]
            speed = 0.35
        
        # Simplified ACO: follow gradient towards goal with pheromone reinforcement
        goal_dir = self.env.goal_pos[:2] - pos[:2]
        goal_dist = np.linalg.norm(goal_dir)
        
        if goal_dist < 1.0:
            return np.array([0.0, 0.0])
        
        # Get pheromone level at current position
        gx = int((pos[0] + 10) / 70 * 10)
        gy = int((pos[1] + 10) / 70 * 10)
        gx = np.clip(gx, 0, 9)
        gy = np.clip(gy, 0, 9)
        
        # Normalize goal direction
        direction = goal_dir / (goal_dist + 1e-6)
        
        # Add pheromone bias
        pheromone_bias = self.pheromone[gx, gy] - 0.1
        
        # Add some randomness
        angle = np.arctan2(direction[1], direction[0])
        angle += np.random.randn() * 0.2 * max(pheromone_bias, 0.1)
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Update pheromone
        self.pheromone[gx, gy] += 0.01 * (1 if goal_dist < 10 else 0.1)
        self.pheromone *= 0.99  # Evaporation
        
        velocity = direction * speed
        
        state.mode = 'ACO'
        return velocity


class HybridNavigator:
    """Hybrid navigation combining multiple strategies"""
    def __init__(self, env):
        self.env = env
        self.components = ['bug', 'potential', 'fuzzy']
        
    def compute_action(self, state):
        # Handle both array and object access for compatibility
        if hasattr(state, 'position'):
            pos = state.position
            heading = state.heading_rad
            speed = state.current_speed
        else:
            pos = state[:3]
            heading = state[2] if len(state) > 2 else 0
            speed = 0.35
        
        goal_dir = self.env.goal_pos[:2] - pos[:2]
        goal_dist = np.linalg.norm(goal_dir)
        
        if goal_dist < 1.0:
            return np.array([0.0, 0.0])
        
        dir_to_goal = goal_dir / (goal_dist + 1e-6)
        
        # Check obstacle proximity
        min_obs = min(obs.distance_to_point(pos[0], pos[1]) 
                     for obs in self.env.obstacles)
        
        # Check if we can go directly to goal
        goal_clear = not self._has_obstacle_in_direction(pos[:2], dir_to_goal, goal_dist)
        
        if goal_clear:
            # Can go straight to goal
            state.mode = 'GO_TO_GOAL'
            return dir_to_goal * speed
        
        # Need obstacle avoidance
        # Get distances in different directions
        ahead_dist = self._get_raycast_distance(pos[:2], heading)
        left_dist = self._get_raycast_distance(pos[:2], heading + 0.5)
        right_dist = self._get_raycast_distance(pos[:2], heading - 0.5)
        
        # Choose direction based on clear space
        if ahead_dist > 3.0:
            # Go forward but slightly toward goal
            action = dir_to_goal * speed
        elif left_dist > right_dist:
            # Turn left
            new_heading = heading + 0.5
            direction = np.array([np.cos(new_heading), np.sin(new_heading)])
            action = direction * speed * 0.8
        else:
            # Turn right
            new_heading = heading - 0.5
            direction = np.array([np.cos(new_heading), np.sin(new_heading)])
            action = direction * speed * 0.8
        
        state.mode = 'HYBRID'
        return action
    
    def _has_obstacle_in_direction(self, pos, direction, max_dist):
        """Check if there's an obstacle in the given direction"""
        for dist in np.linspace(0, max_dist, 30):
            test_pos = pos + direction * dist
            for obs in self.env.obstacles:
                if obs.contains_point(test_pos[0], test_pos[1]):
                    return True
        return False
    
    def _get_raycast_distance(self, pos, angle):
        """Get distance to nearest obstacle in given direction"""
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        for dist in np.linspace(0, 10.0, 50):
            test_pos = pos + ray_dir * dist
            for obs in self.env.obstacles:
                if obs.contains_point(test_pos[0], test_pos[1]):
                    return dist
        return 10.0


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class RobotState:
    def __init__(self, position, heading):
        self.position = position.copy()
        self.heading_rad = heading
        self.current_speed = 0.35
        self.mode = 'MOVE_TO_GOAL'


def run_simulation(env, algorithm, max_steps=3000):
    state = RobotState(
        env.start_pos.copy(),
        np.arctan2(env.goal_pos[1] - env.start_pos[1], 
                  env.goal_pos[0] - env.start_pos[0])
    )
    
    path = [state.position.copy()]
    collisions = 0
    start = time.time()
    
    for _ in range(max_steps):
        action = algorithm.compute_action(state)
        
        if np.linalg.norm(action) > 0:
            direction = action / np.linalg.norm(action)
            state.position[:2] += direction * state.current_speed
            state.heading_rad = np.arctan2(direction[1], direction[0])
        
        path.append(state.position.copy())
        
        if env.is_collision(state.position, 1.5):
            collisions += 1
        
        if env.distance_to_goal(state.position) < 1.5:
            break
    
    path_arr = np.array(path)
    path_length = sum(np.linalg.norm(path_arr[i+1] - path_arr[i]) 
                     for i in range(len(path_arr)-1))
    
    return {
        'success': env.distance_to_goal(state.position) < 1.5,
        'final_distance': env.distance_to_goal(state.position),
        'path_length': path_length,
        'time': time.time() - start,
        'collisions': collisions,
        'path': path_arr
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    algorithms = [
        ("NeuralNetwork", lambda env: NeuralNetworkNavigator(env)),
        ("ANFIS", lambda env: ANFISNavigator(env)),
        ("ParticleSwarm", lambda env: ParticleSwarmNavigator(env)),
        ("AntColony", lambda env: AntColonyNavigator(env)),
        ("Hybrid", lambda env: HybridNavigator(env)),
    ]
    
    results = []
    
    print("\nRunning extended algorithms...")
    print("-" * 50)
    
    for name, factory in algorithms:
        print(f"\n[{name}]")
        env = Environment()
        algo = factory(env)
        result = run_simulation(env, algo)
        
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"  {status} | Distance: {result['final_distance']:.2f}m | "
              f"Path: {result['path_length']:.2f}m | "
              f"Time: {result['time']:.3f}s | Collisions: {result['collisions']}")
        
        results.append({'algorithm': name, 'result': result, 'env': env})
    
    # Save results
    output_dir = "./bio_robotics_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    for r in results:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('#1a1a2e')
        
        for obs in r['env'].obstacles:
            color = '#4a4a6a' if obs.obstacle_type == 'building' else '#3d5a3d'
            rect = FancyBboxPatch((obs.x, obs.y), obs.width, obs.height,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=color, edgecolor='#8888aa', linewidth=2)
            ax.add_patch(rect)
        
        ax.plot(r['env'].start_pos[0], r['env'].start_pos[1], 'go', 
               markersize=15, markeredgecolor='white')
        ax.plot(r['env'].goal_pos[0], r['env'].goal_pos[1], 'r*', markersize=20)
        
        path = r['result']['path']
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, alpha=0.7)
            ax.add_patch(Circle((path[-1, 0], path[-1, 1]), 1.5,
                              facecolor='#00aaff', edgecolor='white'))
        
        ax.set_xlim(-10, 60)
        ax.set_ylim(-10, 60)
        ax.set_aspect('equal')
        ax.set_title(f'{r["algorithm"]} Navigation', fontweight='bold')
        
        save_path = os.path.join(output_dir, "figures", f"2D_{r['algorithm']}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # Export CSV
    csv_data = [{
        'algorithm': r['algorithm'],
        'success': r['result']['success'],
        'final_distance': r['result']['final_distance'],
        'path_length': r['result']['path_length'],
        'time_taken': r['result']['time'],
        'collisions': r['result']['collisions']
    } for r in results]
    
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(output_dir, "data", "extended_results.csv"), index=False)
    
    print("\n" + "=" * 70)
    print("EXTENDED SIMULATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
