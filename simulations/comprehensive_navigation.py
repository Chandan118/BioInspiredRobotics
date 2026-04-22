#!/usr/bin/env python3
"""
Bio-Inspired Robotics Navigation Simulation
==========================================
A comprehensive Python implementation of bio-inspired navigation algorithms
for tethered robots in dynamic environments.

Author: Chandan Sheikder
Affiliation: Beijing Institute of Technology (BIT)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Arrow
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib import cm
import pandas as pd
import json
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from enum import Enum
import random
import pickle
import os
from abc import ABC, abstractmethod

# Try to import optional dependencies
try:
    import scipy.interpolate as interpolate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Obstacle:
    """Base class for obstacles in the environment"""
    x: float
    y: float
    width: float
    height: float
    height_3d: float = 10.0
    obstacle_type: str = "building"
    
    def contains_point(self, px: float, py: float, margin: float = 0.0) -> bool:
        """Check if point is inside obstacle"""
        return (self.x - margin <= px <= self.x + self.width + margin and
                self.y - margin <= py <= self.y + self.height + margin)
    
    def distance_to_point(self, px: float, py: float) -> float:
        """Calculate minimum distance to obstacle edge"""
        dx = max(self.x - px, 0, px - (self.x + self.width))
        dy = max(self.y - py, 0, py - (self.y + self.height))
        return math.sqrt(dx**2 + dy**2)


@dataclass
class RobotState:
    """Complete state of the robot"""
    position: np.ndarray  # [x, y, z]
    heading_rad: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mode: str = 'MOVE_TO_GOAL'
    path_history: List = field(default_factory=list)
    wall_follow_dir: int = 1
    stuck_counter: int = 0
    current_speed: float = 0.0
    energy_consumed: float = 0.0
    collision_count: int = 0
    time_elapsed: float = 0.0
    
    
@dataclass
class HumanState:
    """State of dynamic obstacle (human)"""
    position: np.ndarray  # [x, y, z]
    heading_rad: float
    path_segment: int = 0
    speed: float = 0.1


@dataclass
class TetherInfo:
    """Tether cable state information"""
    path: np.ndarray = field(default_factory=np.array)
    is_snagged: bool = False
    tension: float = 0.0
    length: float = 0.0


@dataclass
class SimulationResult:
    """Results from a simulation run"""
    success: bool
    final_distance: float
    path_length: float
    time_taken: float
    collisions: int
    energy_consumed: float
    mode_changes: int
    path: np.ndarray
    states_log: List = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


# =============================================================================
# ENVIRONMENT CLASS
# =============================================================================

class Environment:
    """Simulation environment with obstacles and boundaries"""
    
    def __init__(self, 
                 bounds: Tuple[float, float] = (-10, 60),
                 start_pos: Tuple[float, float] = (5, 45),
                 goal_pos: Tuple[float, float] = (45, 5)):
        self.bounds = bounds  # (min, max) for x and y
        self.start_pos = np.array([start_pos[0], start_pos[1], 0])
        self.goal_pos = np.array([goal_pos[0], goal_pos[1], 0])
        
        # Build static obstacles
        self.obstacles: List[Obstacle] = self._create_default_obstacles()
        self.dynamic_obstacles: List[Obstacle] = []
        
    def _create_default_obstacles(self) -> List[Obstacle]:
        """Create default obstacle layout matching the MATLAB simulation"""
        return [
            # Buildings (4 perimeter buildings forming a corridor)
            Obstacle(22, 25, 6, 30, 18, "building"),   # Center vertical
            # Left building
            Obstacle(-2, 25, 6, 30, 18, "building"),
            # Right building  
            Obstacle(46, 25, 6, 30, 18, "building"),
            # Planters
            Obstacle(15, 15, 4, 10, 2, "planter"),
            Obstacle(35, 35, 10, 4, 2, "planter"),
            # Streetlights
            Obstacle(10, 25, 1, 1, 10, "streetlight"),
            Obstacle(40, 25, 1, 1, 10, "streetlight"),
        ]
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the environment"""
        self.obstacles.append(obstacle)
        
    def get_all_obstacles(self) -> List[Obstacle]:
        """Get all current obstacles (static + dynamic)"""
        return self.obstacles + self.dynamic_obstacles
    
    def is_collision(self, pos: np.ndarray, radius: float) -> bool:
        """Check if robot at position collides with any obstacle"""
        px, py = pos[0], pos[1]
        for obs in self.get_all_obstacles():
            if obs.contains_point(px, py, margin=-radius):
                return True
            # Also check circular distance
            if obs.distance_to_point(px, py) < radius:
                return True
        return False
    
    def is_in_bounds(self, pos: np.ndarray) -> bool:
        """Check if position is within environment bounds"""
        return (self.bounds[0] <= pos[0] <= self.bounds[1] and
                self.bounds[0] <= pos[1] <= self.bounds[1])
    
    def distance_to_goal(self, pos: np.ndarray) -> float:
        """Distance from position to goal"""
        return np.linalg.norm(pos[:2] - self.goal_pos[:2])


# =============================================================================
# ABSTRACT NAVIGATION ALGORITHM
# =============================================================================

class NavigationAlgorithm(ABC):
    """Base class for all navigation algorithms"""
    
    def __init__(self, name: str, env: Environment, robot_radius: float = 1.5):
        self.name = name
        self.env = env
        self.robot_radius = robot_radius
        self.reset()
        
    @abstractmethod
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute action (velocity) from current state"""
        pass
    
    def reset(self):
        """Reset algorithm state"""
        self.state_log = []
        self.mode_changes = 0
        self.last_mode = None
        
    def log_state(self, state: RobotState, action: np.ndarray, reward: float = 0):
        """Log current state for analysis"""
        self.state_log.append({
            'position': state.position.copy(),
            'heading': state.heading_rad,
            'mode': state.mode,
            'action': action.copy(),
            'reward': reward,
            'time': state.time_elapsed
        })


# =============================================================================
# BUG ALGORITHMS
# =============================================================================

class Bug0Algorithm(NavigationAlgorithm):
    """Bug 0 Algorithm - Simple wall following"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("Bug0", env, robot_radius)
        self.detection_radius = 8.0
        self.follow_distance = 3.0
        self.wall_side = 1
        
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute action using Bug 0 logic"""
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        min_obs_dist, obs_normal = self._detect_obstacles(state.position)
        ahead_dist = self._get_ahead_distance(state.position, state.heading_rad)
        
        if min_obs_dist < self.detection_radius or ahead_dist < 2.0:
            if state.mode != 'FOLLOW_OBSTACLE':
                state.mode = 'FOLLOW_OBSTACLE'
                right_dist = self._get_side_distance(state.position, state.heading_rad, 1)
                left_dist = self._get_side_distance(state.position, state.heading_rad, -1)
                self.wall_side = 1 if right_dist < left_dist else -1
                self.mode_changes += 1
            
            action = self._wall_follow(state, obs_normal)
        else:
            state.mode = 'MOVE_TO_GOAL'
            action = dir_to_goal * state.current_speed
            
        return action
    
    def _get_ahead_distance(self, pos: np.ndarray, heading: float) -> float:
        """Get distance to obstacles directly ahead"""
        min_dist = 10.0
        for angle_offset in np.linspace(-0.3, 0.3, 5):
            ray_angle = heading + angle_offset
            ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)])
            for dist in np.linspace(0, 10.0, 50):
                test_pos = pos[:2] + ray_dir * dist
                for obs in self.env.get_all_obstacles():
                    if obs.contains_point(test_pos[0], test_pos[1]):
                        min_dist = min(min_dist, dist)
                        break
        return min_dist
    
    def _get_side_distance(self, pos: np.ndarray, heading: float, side: int) -> float:
        """Get distance to obstacles on the side (1=right, -1=left)"""
        min_dist = 10.0
        side_angle = heading + side * math.pi / 2
        ray_dir = np.array([math.cos(side_angle), math.sin(side_angle)])
        for dist in np.linspace(0, 10.0, 50):
            test_pos = pos[:2] + ray_dir * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1]):
                    min_dist = min(min_dist, dist)
                    break
        return min_dist
    
    def _detect_obstacles(self, pos: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect nearest obstacle and get surface normal"""
        min_dist = float('inf')
        normal = np.array([0.0, 0.0])
        
        for obs in self.env.get_all_obstacles():
            dist = obs.distance_to_point(pos[0], pos[1])
            if dist < min_dist:
                min_dist = dist
                # Approximate normal as direction to obstacle center
                cx, cy = obs.x + obs.width/2, obs.y + obs.height/2
                normal = np.array([cx - pos[0], cy - pos[1]])
                normal = normal / (np.linalg.norm(normal) + 1e-6)
                
        return min_dist, normal
    
    def _wall_follow(self, state: RobotState, obs_normal: np.ndarray) -> np.ndarray:
        """Compute wall following action"""
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        # Compute tangent (perpendicular to normal)
        tangent = np.array([-obs_normal[1] * self.wall_side, obs_normal[0] * self.wall_side])
        
        # Check if tangent direction is clear
        test_pos = state.position[:2] + tangent * 2.0
        tangent_clear = not self._has_obstacle_at(test_pos)
        
        # Check if going toward goal is possible
        toward_goal_clear = not self._has_obstacle_in_direction(state.position[:2], dir_to_goal, dist_to_goal)
        
        # Decide action
        if toward_goal_clear and dist_to_goal < 50:
            return dir_to_goal * state.current_speed
        elif tangent_clear:
            return tangent * state.current_speed
        else:
            return -tangent * state.current_speed * 0.5
    
    def _has_obstacle_at(self, pos: np.ndarray) -> bool:
        """Check if there's an obstacle at position"""
        for obs in self.env.get_all_obstacles():
            if obs.contains_point(pos[0], pos[1], margin=-self.robot_radius):
                return True
        return False
    
    def _has_obstacle_in_direction(self, pos: np.ndarray, direction: np.ndarray, max_dist: float) -> bool:
        """Check if there's an obstacle in the given direction"""
        for dist in np.linspace(0, max_dist, 50):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1], margin=-self.robot_radius):
                    return True
        return False


class Bug1Algorithm(NavigationAlgorithm):
    """Bug 1 Algorithm - Complete perimeter following"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("Bug1", env, robot_radius)
        self.detection_radius = 8.0
        self.leave_point = None
        self.leave_distance = float('inf')
        
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute action using Bug 1 logic"""
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        min_obs_dist, obs_normal = self._detect_obstacles(state.position)
        
        if min_obs_dist < self.detection_radius:
            if state.mode == 'MOVE_TO_GOAL':
                state.mode = 'FOLLOW_OBSTACLE'
                self.leave_point = state.position.copy()
                self.leave_distance = dist_to_goal
                self.mode_changes += 1
                state.wall_follow_dir = self._choose_best_direction(state, dir_to_goal)
            
            action = self._perimeter_follow(state, obs_normal)
            
            # Check if we should leave
            if dist_to_goal < self.leave_distance - 0.5:
                # Can see goal, try to go straight
                test_pos = state.position[:2] + dir_to_goal * 2.0
                if not self._has_obstacle_in_direction(state.position[:2], dir_to_goal):
                    state.mode = 'MOVE_TO_GOAL'
                    self.mode_changes += 1
        else:
            state.mode = 'MOVE_TO_GOAL'
            action = dir_to_goal * state.current_speed
            
        return action
    
    def _detect_obstacles(self, pos: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect nearest obstacle"""
        min_dist = float('inf')
        normal = np.array([0.0, 0.0])
        
        for obs in self.env.get_all_obstacles():
            dist = obs.distance_to_point(pos[0], pos[1])
            if dist < min_dist:
                min_dist = dist
                cx, cy = obs.x + obs.width/2, obs.y + obs.height/2
                normal = np.array([cx - pos[0], cy - pos[1]])
                normal = normal / (np.linalg.norm(normal) + 1e-6)
                
        return min_dist, normal
    
    def _choose_best_direction(self, state: RobotState, dir_to_goal: np.ndarray) -> int:
        """Choose best wall follow direction"""
        return 1  # Clockwise by default
    
    def _perimeter_follow(self, state: RobotState, obs_normal: np.ndarray) -> np.ndarray:
        """Follow obstacle perimeter"""
        tangent = np.array([-obs_normal[1], obs_normal[0]]) * state.wall_follow_dir
        return tangent * state.current_speed
    
    def _has_obstacle_in_direction(self, pos: np.ndarray, direction: np.ndarray) -> bool:
        """Check if obstacle blocks direction"""
        for dist in np.linspace(0, 10, 50):
            test_pos = pos + direction * dist
            for obs in self.env.get_all_obstacles():
                if obs.contains_point(test_pos[0], test_pos[1]):
                    return True
        return False


class TangentBugAlgorithm(NavigationAlgorithm):
    """Tangent Bug Algorithm - Uses tangent points for efficient navigation"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("TangentBug", env, robot_radius)
        self.detection_radius = 10.0
        self.visited_points = []
        self.tangent_points = []
        
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute action using Tangent Bug logic"""
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        # Get closest obstacles in sensor range
        obstacles_info = self._get_visible_obstacles(state.position)
        
        if obstacles_info and dist_to_goal > 1.0:
            # Find tangent points
            tangent = self._find_tangent_point(state.position, obstacles_info, dir_to_goal)
            
            if tangent is not None:
                state.mode = 'TANGENT_NAV'
                direction = tangent - state.position[:2]
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                return direction * state.current_speed
            else:
                # Follow obstacle boundary
                state.mode = 'FOLLOW_OBSTACLE'
                return self._follow_boundary(state, obstacles_info)
        else:
            state.mode = 'MOVE_TO_GOAL'
            return dir_to_goal * state.current_speed
    
    def _get_visible_obstacles(self, pos: np.ndarray) -> List[Dict]:
        """Get obstacles visible from current position"""
        visible = []
        for obs in self.env.get_all_obstacles():
            dist = obs.distance_to_point(pos[0], pos[1])
            if dist < self.detection_radius:
                visible.append({
                    'obstacle': obs,
                    'distance': dist,
                    'angle': math.atan2(obs.y + obs.height/2 - pos[1],
                                        obs.x + obs.width/2 - pos[0])
                })
        return visible
    
    def _find_tangent_point(self, pos: np.ndarray, obstacles: List[Dict], 
                           dir_to_goal: np.ndarray) -> Optional[np.ndarray]:
        """Find optimal tangent point to navigate around obstacles"""
        # Simplified: just pick direction that gets closest to goal while avoiding obstacles
        best_dir = dir_to_goal.copy()
        best_score = -float('inf')
        
        for angle in np.linspace(-math.pi/2, math.pi/2, 37):
            test_dir = np.array([math.cos(angle), math.sin(angle)])
            test_dir = test_dir * dir_to_goal[0] + np.array([-test_dir[1], test_dir[0]]) * dir_to_goal[1]
            test_dir = test_dir / (np.linalg.norm(test_dir) + 1e-6)
            
            # Check if direction is clear
            clear = True
            min_clear_dist = float('inf')
            for dist in np.linspace(0, self.detection_radius, 50):
                test_pos = pos + test_dir * dist
                for obs in self.env.get_all_obstacles():
                    if obs.contains_point(test_pos[0], test_pos[1]):
                        clear = False
                        break
                if not clear:
                    break
                min_clear_dist = dist
            
            if clear:
                # Score based on alignment with goal
                score = np.dot(test_dir, dir_to_goal)
                if score > best_score:
                    best_score = score
                    best_dir = test_dir
        
        return pos + best_dir if best_score > 0 else None
    
    def _follow_boundary(self, state: RobotState, obstacles: List[Dict]) -> np.ndarray:
        """Follow obstacle boundary"""
        closest = min(obstacles, key=lambda x: x['distance'])
        obs = closest['obstacle']
        
        # Get direction tangent to obstacle
        dx = state.position[0] - (obs.x + obs.width/2)
        dy = state.position[1] - (obs.y + obs.height/2)
        
        tangent = np.array([-dy, dx])
        tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
        
        return tangent * state.current_speed


# =============================================================================
# FUZZY LOGIC NAVIGATION
# =============================================================================

class FuzzyNavigator(NavigationAlgorithm):
    """Fuzzy Logic based Navigation Controller"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("Fuzzy", env, robot_radius)
        self.fuzzy_rules = self._create_fuzzy_rules()
        self.membership_functions = self._create_membership_functions()
        
    def _create_membership_functions(self) -> Dict:
        """Define fuzzy membership functions"""
        return {
            'distance': {
                'near': (0, 0, 0.5, 1.0),
                'medium': (0.5, 1.5, 2.5),
                'far': (2.0, 3.0, 5.0, 5.0)
            },
            'angle': {
                'left': (-math.pi, -math.pi/2, 0),
                'straight': (-math.pi/4, 0, math.pi/4),
                'right': (0, math.pi/2, math.pi)
            },
            'steering': {
                'sharp_left': (-1.5, -1.2, -0.8),
                'left': (-1.0, -0.5, 0),
                'straight': (-0.2, 0, 0.2),
                'right': (0, 0.5, 1.0),
                'sharp_right': (0.8, 1.2, 1.5)
            }
        }
    
    def _create_fuzzy_rules(self) -> List[Tuple]:
        """Define fuzzy inference rules (antecedent, consequent)"""
        return [
            # (distance_mf, angle_mf, steering_mf)
            ('near', 'left', 'sharp_right'),
            ('near', 'straight', 'sharp_right'),
            ('near', 'right', 'sharp_left'),
            ('medium', 'left', 'right'),
            ('medium', 'straight', 'right'),
            ('medium', 'right', 'left'),
            ('far', 'left', 'left'),
            ('far', 'straight', 'straight'),
            ('far', 'right', 'right'),
        ]
    
    def _fuzzify(self, value: float, mf_params: Tuple, mf_type: str) -> float:
        """Compute membership value"""
        if len(mf_params) == 4:
            # Trapezoidal
            x0, x1, x2, x3 = mf_params
            if x0 <= value <= x3:
                if x0 <= value <= x1:
                    return (value - x0) / (x1 - x0 + 1e-6)
                elif x2 <= value <= x3:
                    return (x3 - value) / (x3 - x2 + 1e-6)
                else:
                    return 1.0
            return 0.0
        else:
            # Triangular
            x0, x1, x2 = mf_params
            if x0 <= value <= x2:
                if value <= x1:
                    return (value - x0) / (x1 - x0 + 1e-6)
                else:
                    return (x2 - value) / (x2 - x1 + 1e-6)
            return 0.0
    
    def _defuzzify(self, steering_outputs: Dict[str, float]) -> float:
        """Center of gravity defuzzification"""
        num = 0.0
        den = 0.0
        
        for mf_name, membership in steering_outputs.items():
            if membership > 0:
                # Use center of membership function
                params = self.membership_functions['steering'][mf_name]
                center = params[1]
                num += membership * center
                den += membership
                
        return num / (den + 1e-6)
    
    def _inference(self, distance: float, angle: float) -> float:
        """Perform fuzzy inference"""
        # Fuzzify inputs
        distance_memberships = {}
        for mf_name, params in self.membership_functions['distance'].items():
            distance_memberships[mf_name] = self._fuzzify(distance, params, 'distance')
            
        angle_memberships = {}
        for mf_name, params in self.membership_functions['angle'].items():
            angle_memberships[mf_name] = self._fuzzify(angle, params, 'angle')
        
        # Apply rules
        steering_outputs = {k: 0.0 for k in self.membership_functions['steering'].keys()}
        
        for dist_mf, ang_mf, steer_mf in self.fuzzy_rules:
            # Rule strength is min of antecedent memberships
            strength = min(distance_memberships[dist_mf], angle_memberships[ang_mf])
            steering_outputs[steer_mf] = max(steering_outputs[steer_mf], strength)
        
        # Defuzzify
        return self._defuzzify(steering_outputs)
    
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute fuzzy navigation action"""
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        dir_to_goal = goal_dir / (dist_to_goal + 1e-6)
        
        # Calculate angle to goal relative to current heading
        angle_to_goal = math.atan2(dir_to_goal[1], dir_to_goal[0])
        relative_angle = self._normalize_angle(angle_to_goal - state.heading_rad)
        
        # Get obstacle distance (minimum in front)
        front_dist = self._get_front_obstacle_distance(state.position, state.heading_rad)
        
        # Fuzzy inference
        steering = self._inference(front_dist, relative_angle)
        
        # Convert steering to velocity command
        linear_speed = state.current_speed
        if abs(steering) > 0.8 or front_dist < 0.5:
            linear_speed *= 0.5
            
        # Compute new heading
        new_heading = state.heading_rad - steering * 0.1
        
        # Compute action
        action = np.array([math.cos(new_heading), math.sin(new_heading)]) * linear_speed
        
        state.mode = 'FUZZY_NAV'
        return action
    
    def _get_front_obstacle_distance(self, pos: np.ndarray, heading: float) -> float:
        """Get minimum obstacle distance in front of robot"""
        min_dist = 5.0  # Max sensor range
        
        for angle_offset in np.linspace(-math.pi/4, math.pi/4, 9):
            ray_angle = heading + angle_offset
            ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)])
            
            for dist in np.linspace(0, 5.0, 100):
                test_pos = pos[:2] + ray_dir * dist
                for obs in self.env.get_all_obstacles():
                    if obs.contains_point(test_pos[0], test_pos[1], margin=0.1):
                        if dist < min_dist:
                            min_dist = dist
                        break
                        
        return min_dist
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# =============================================================================
# GENETIC ALGORITHM OPTIMIZED NAVIGATION
# =============================================================================

class GeneticAlgorithmOptimizer:
    """Genetic Algorithm for optimizing navigation parameters"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.gene_count = 6  # [speed, turn_rate, safety_dist, wall_follow_dist, lookahead, tolerance]
        
    def initialize_population(self) -> np.ndarray:
        """Create initial random population"""
        population = np.zeros((self.population_size, self.gene_count))
        # bounds: [min_speed, max_speed, turn_rate, safety_dist, wall_dist, tolerance]
        bounds = [(0.1, 0.4), (0.1, 1.0), (0.5, 3.0), (0.2, 1.0), (0.5, 3.0), (0.1, 0.5)]
        
        for i in range(self.population_size):
            for j in range(self.gene_count):
                population[i, j] = np.random.uniform(bounds[j][0], bounds[j][1])
                
        return population
    
    def fitness(self, individual: np.ndarray, env: Environment, 
                algorithm: NavigationAlgorithm) -> float:
        """Evaluate fitness of an individual"""
        # Create a modified algorithm with these parameters
        modified_algo = self._create_modified_algorithm(algorithm, individual)
        
        # Run simulation
        result = run_simulation(env, modified_algo, max_steps=2000)
        
        # Fitness: prefer successful, fast, collision-free paths
        if result.success:
            fitness = 1000.0 - result.time_taken - result.collisions * 100
        else:
            fitness = -result.final_distance  # Want to get as close as possible
            
        return fitness
    
    def _create_modified_algorithm(self, base_algo: NavigationAlgorithm, 
                                   params: np.ndarray) -> NavigationAlgorithm:
        """Create algorithm with modified parameters"""
        # For now, just adjust speed
        class ModifiedAlgorithm(base_algo.__class__):
            def __init__(self, env, algo, params):
                super().__init__(env, algo.robot_radius)
                self.params = params
                
            def compute_action(self, state):
                action = super().compute_action(state)
                # Scale speed by genetic parameter
                speed_factor = self.params[0]
                action = action * speed_factor
                return action
                
        return ModifiedAlgorithm(env, base_algo, params)
    
    def select_parents(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection"""
        parents = []
        for _ in range(self.population_size // 2):
            # Select 3 random individuals
            indices = np.random.choice(len(population), 3, replace=False)
            best = indices[np.argmax(fitness[indices])]
            parents.append(population[best])
        return np.array(parents)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Blend crossover"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        alpha = np.random.uniform(-0.5, 1.5)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation"""
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
                mutated[i] = max(0.1, mutated[i])  # Ensure positive
        return mutated
    
    def evolve(self, env: Environment, base_algorithm: NavigationAlgorithm) -> Tuple[np.ndarray, List]:
        """Run genetic algorithm optimization"""
        population = self.initialize_population()
        best_individual = None
        best_fitness_history = []
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness = np.array([self.fitness(ind, env, base_algorithm) 
                              for ind in population])
            
            # Track best
            best_idx = np.argmax(fitness)
            if best_individual is None or fitness[best_idx] > self.fitness(best_individual, env, base_algorithm):
                best_individual = population[best_idx].copy()
            
            best_fitness_history.append(np.max(fitness))
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {np.max(fitness):.2f}")
            
            # Selection
            parents = self.select_parents(population, fitness)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.crossover(parents[i], parents[i+1])
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
            
            # Fill remaining slots
            while len(new_population) < self.population_size:
                idx = np.random.randint(len(population))
                new_population.append(self.mutate(population[idx]))
            
            population = np.array(new_population[:self.population_size])
            
        return best_individual, best_fitness_history


# =============================================================================
# PARTICLE SWARM OPTIMIZATION
# =============================================================================

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for navigation"""
    
    def __init__(self, n_particles: int = 30, iterations: int = 100):
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        
    def optimize(self, env: Environment, algorithm: NavigationAlgorithm,
                objective_fn: Callable) -> Tuple[np.ndarray, List]:
        """Run PSO optimization"""
        # Parameter bounds
        bounds = np.array([
            [0.1, 0.5],   # speed
            [0.1, 2.0],   # detection_radius
            [0.2, 1.0],   # safety_margin
            [0.5, 3.0],   # lookahead
            [0.1, 0.5],   # goal_tolerance
            [0.1, 1.0],   # wall_follow_speed
        ])
        
        # Initialize particles
        positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.n_particles, 6))
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, 6))
        
        # Personal best
        p_best = positions.copy()
        p_best_fitness = np.array([objective_fn(pos, env, algorithm) for pos in positions])
        
        # Global best
        g_best_idx = np.argmax(p_best_fitness)
        g_best = p_best[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]
        
        fitness_history = [g_best_fitness]
        
        for iteration in range(self.iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(6), np.random.rand(6)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (p_best[i] - positions[i]) +
                                 self.c2 * r2 * (g_best - positions[i]))
                
                # Update position
                positions[i] += velocities[i]
                
                # Clamp to bounds
                positions[i] = np.clip(positions[i], bounds[:, 0], bounds[:, 1])
                
                # Evaluate fitness
                fitness = objective_fn(positions[i], env, algorithm)
                
                # Update personal best
                if fitness > p_best_fitness[i]:
                    p_best[i] = positions[i].copy()
                    p_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness > g_best_fitness:
                        g_best = positions[i].copy()
                        g_best_fitness = fitness
            
            fitness_history.append(g_best_fitness)
            
            if iteration % 10 == 0:
                print(f"PSO Iteration {iteration}: Best fitness = {g_best_fitness:.2f}")
                
        return g_best, fitness_history


# =============================================================================
# Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    """Q-Learning agent for navigation"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state discretized bins -> action values
        self.q_table = {}
        
        # Action definitions (velocity commands)
        self.actions = [
            np.array([0.3, 0.0]),    # Forward
            np.array([0.3, 0.5]),    # Forward-left
            np.array([0.3, -0.5]),   # Forward-right
            np.array([0.1, 0.8]),    # Turn left
            np.array([0.1, -0.8]),   # Turn right
            np.array([0.0, 0.0]),    # Stop
            np.array([-0.1, 0.0]),   # Backward
        ]
        
    def discretize_state(self, state: np.ndarray) -> Tuple:
        """Convert continuous state to discrete bins"""
        # Extract key features
        if len(state) >= 10:
            lidar_sectors = tuple(int(x * 10) for x in state[:8])
            gas = tuple(int(x * 10) for x in state[8:10])
        else:
            lidar_sectors = tuple(int(x * 10) for x in state[:5])
            gas = ()
            
        return lidar_sectors + gas + (int(state[-2] * 5),)  # Distance to goal bin
        
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions"""
        key = self.discretize_state(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)
        return self.q_table[key]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, 
               reward: float, next_state: np.ndarray, done: bool):
        """Update Q-value"""
        key = self.discretize_state(state)
        next_key = self.discretize_state(next_state)
        
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_dim)
        
        current_q = self.q_table[key][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_key])
            
        self.q_table[key][action] = current_q + self.lr * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filepath: str):
        """Save Q-table"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, filepath: str):
        """Load Q-table"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)


# =============================================================================
# DEEP Q-NETWORK (DQN)
# =============================================================================

class DQNAgent:
    """Deep Q-Network agent for navigation"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 2000,
                 batch_size: int = 32):
        
        if not HAS_TF:
            raise ImportError("TensorFlow not available for DQNAgent")
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Main network and target network
        self.model = self._build_network()
        self.target_model = self._build_network()
        self.update_target_network()
        
        # Action definitions
        self.actions = [
            np.array([0.3, 0.0]),
            np.array([0.3, 0.5]),
            np.array([0.3, -0.5]),
            np.array([0.1, 0.8]),
            np.array([0.1, -0.8]),
            np.array([0.0, 0.0]),
            np.array([-0.1, 0.0]),
        ]
        
    def _build_network(self) -> keras.Model:
        """Build neural network for Q-value approximation"""
        model = keras.Sequential([
            layers.Dense(64, input_dim=self.state_dim, activation='relu',
                        kernel_initializer='he_normal'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(loss='mse',
                     optimizer=keras.optimizers.Adam(learning_rate=self.gamma))
        return model
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(
                    self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save model weights"""
        self.model.save_weights(filepath)
        
    def load(self, filepath: str):
        """Load model weights"""
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            self.update_target_network()


# =============================================================================
# DYNAMIC WINDOW APPROACH
# =============================================================================

class DynamicWindowApproach(NavigationAlgorithm):
    """Dynamic Window Approach (DWA) for robot navigation"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("DWA", env, robot_radius)
        
        # Robot parameters
        self.max_speed = 0.35
        self.max_angular_speed = 0.5
        self.max_acceleration = 2.5
        self.max_angular_acceleration = 3.5
        
        # DWA parameters
        self.v_resolution = 0.1
        self.w_resolution = 0.1
        self.predict_time = 3.0
        self.to_goal_cost_gain = 1.0
        self.speed_cost_gain = 0.1
        self.obstacle_cost_gain = 1.0
        
        # Velocity sampling windows
        self.min_v = 0
        self.max_v = self.max_speed
        self.min_w = -self.max_angular_speed
        self.max_w = self.max_angular_speed
        
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute action using Dynamic Window Approach"""
        # Get velocity samples within dynamic window
        v_samples, w_samples = self._sample_velocities(state)
        
        best_cost = -float('inf')
        best_v, best_w = 0, 0
        
        for v in v_samples:
            for w in w_samples:
                # Simulate trajectory
                trajectory = self._simulate_trajectory(state, v, w)
                
                # Calculate costs
                to_goal_cost = self._to_goal_cost(trajectory)
                speed_cost = self._speed_cost(v)
                obstacle_cost = self._obstacle_cost(trajectory)
                
                total_cost = (self.to_goal_cost_gain * to_goal_cost +
                             self.speed_cost_gain * speed_cost +
                             self.obstacle_cost_gain * obstacle_cost)
                
                if total_cost > best_cost:
                    best_cost = total_cost
                    best_v, best_w = v, w
        
        # Convert to action
        action = np.array([math.cos(state.heading_rad), 
                          math.sin(state.heading_rad)]) * best_v
        state.mode = 'DWA'
        
        return action
    
    def _sample_velocities(self, state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """Sample velocities within dynamic window"""
        # Dynamic window based on current velocity and acceleration
        v_min = max(self.min_v, state.current_speed - self.max_acceleration * 0.1)
        v_max = min(self.max_v, state.current_speed + self.max_acceleration * 0.1)
        w_min = max(self.min_w, -self.max_angular_speed + self.max_angular_acceleration * 0.1)
        w_max = min(self.max_w, self.max_angular_speed - self.max_angular_acceleration * 0.1)
        
        v_samples = np.arange(v_min, v_max + self.v_resolution, self.v_resolution)
        w_samples = np.arange(w_min, w_max + self.w_resolution, self.w_resolution)
        
        return v_samples, w_samples
    
    def _simulate_trajectory(self, state: RobotState, v: float, w: float) -> np.ndarray:
        """Simulate robot trajectory given velocities"""
        trajectory = [state.position[:2].copy()]
        pos = state.position[:2].copy()
        heading = state.heading_rad
        
        dt = 0.1
        for _ in np.linspace(0, self.predict_time, int(self.predict_time/dt)):
            heading += w * dt
            pos += v * dt * np.array([math.cos(heading), math.sin(heading)])
            trajectory.append(pos.copy())
            
        return np.array(trajectory)
    
    def _to_goal_cost(self, trajectory: np.ndarray) -> float:
        """Cost for heading towards goal"""
        final_pos = trajectory[-1]
        dist_to_goal = np.linalg.norm(final_pos - self.env.goal_pos[:2])
        return 1.0 / (dist_to_goal + 0.1)
    
    def _speed_cost(self, v: float) -> float:
        """Cost for high speed"""
        return v / self.max_speed
    
    def _obstacle_cost(self, trajectory: np.ndarray) -> float:
        """Cost for proximity to obstacles"""
        min_dist = float('inf')
        for pos in trajectory:
            for obs in self.env.get_all_obstacles():
                dist = obs.distance_to_point(pos[0], pos[1])
                if dist < min_dist:
                    min_dist = dist
        return 1.0 / (min_dist + 0.1)


# =============================================================================
# POTENTIAL FIELD NAVIGATION
# =============================================================================

class PotentialFieldNavigation(NavigationAlgorithm):
    """Artificial Potential Field navigation"""
    
    def __init__(self, env: Environment, robot_radius: float = 1.5):
        super().__init__("PotentialField", env, robot_radius)
        
        # Potential field gains
        self.attractive_gain = 10.0
        self.repulsive_gain = 100.0
        self.repulsive_radius = 5.0
        self.goal_tolerance = 0.5
        
    def compute_action(self, state: RobotState) -> np.ndarray:
        """Compute action using potential field method"""
        # Attractive force towards goal
        goal_dir = self.env.goal_pos[:2] - state.position[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        
        if dist_to_goal < self.goal_tolerance:
            return np.array([0.0, 0.0])
            
        attractive_force = self.attractive_gain * goal_dir
        
        # Repulsive force from obstacles
        repulsive_force = np.array([0.0, 0.0])
        for obs in self.env.get_all_obstacles():
            obs_center = np.array([obs.x + obs.width/2, obs.y + obs.height/2])
            to_robot = state.position[:2] - obs_center
            dist = np.linalg.norm(to_robot)
            
            if dist < self.repulsive_radius:
                # Repulsive potential: inversely proportional to distance
                magnitude = self.repulsive_gain * (1/dist - 1/self.repulsive_radius)**2
                direction = to_robot / (dist + 1e-6)
                repulsive_force += magnitude * direction
        
        # Combined force
        total_force = attractive_force + repulsive_force
        
        # Limit force magnitude
        max_force = 5.0
        total_force_mag = np.linalg.norm(total_force)
        if total_force_mag > max_force:
            total_force = total_force * max_force / total_force_mag
        
        # Convert to velocity
        action = total_force * state.current_speed * 0.1
        action = action / (np.linalg.norm(action) + 1e-6) * min(np.linalg.norm(action), state.current_speed)
        
        state.mode = 'POTENTIAL_FIELD'
        return action


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_simulation(env: Environment, algorithm: NavigationAlgorithm,
                 max_steps: int = 4000,
                 robot_params: Dict = None,
                 human_params: Dict = None,
                 verbose: bool = False) -> SimulationResult:
    """Run complete navigation simulation"""
    
    if robot_params is None:
        robot_params = {
            'radius': 1.5,
            'max_speed': 0.35,
            'cautious_speed': 0.12,
            'detection_radius': 8.0
        }
    
    # Initialize robot state
    robot_state = RobotState(
        position=env.start_pos.copy(),
        heading_rad=math.atan2(
            env.goal_pos[1] - env.start_pos[1],
            env.goal_pos[0] - env.start_pos[0]
        ),
        current_speed=robot_params['max_speed']
    )
    robot_state.path_history = [robot_state.position.copy()]
    
    # Initialize human (dynamic obstacle)
    if human_params is None:
        human_params = {
            'path': np.array([[10, 40], [40, 40], [40, 10], [10, 10], [10, 40]]),
            'speed': 0.1,
            'radius': 1.0
        }
    
    human_state = HumanState(
        position=np.array([human_params['path'][0, 0], human_params['path'][0, 1], 0]),
        heading_rad=0,
        path_segment=0
    )
    
    # Tracking variables
    start_time = time.time()
    collision_count = 0
    mode_changes = 0
    last_mode = robot_state.mode
    
    # Main simulation loop
    for step in range(max_steps):
        # Update human position
        human_state = update_human(human_state, human_params)
        
        # Update dynamic obstacles
        env.dynamic_obstacles = [
            Obstacle(
                human_state.position[0] - human_params['radius'],
                human_state.position[1] - human_params['radius'],
                human_params['radius'] * 2,
                human_params['radius'] * 2,
                4, "human"
            )
        ]
        
        # Check for collisions
        if env.is_collision(robot_state.position, robot_params['radius']):
            collision_count += 1
            robot_state.collision_count = collision_count
            if verbose:
                print(f"Collision at step {step}")
        
        # Adaptive speed based on proximity to obstacles
        is_near = any(
            obs.distance_to_point(robot_state.position[0], robot_state.position[1]) 
            < robot_params['detection_radius']
            for obs in env.get_all_obstacles()
        )
        robot_state.current_speed = (robot_params['cautious_speed'] if is_near 
                                     else robot_params['max_speed'])
        
        # Compute action
        action = algorithm.compute_action(robot_state)
        
        # Track mode changes
        if robot_state.mode != last_mode:
            mode_changes += 1
            last_mode = robot_state.mode
        
        # Update robot position
        if np.linalg.norm(action) > 0:
            direction = action / np.linalg.norm(action)
            robot_state.position[:2] += direction * robot_state.current_speed
            robot_state.heading_rad = math.atan2(direction[1], direction[0])
        
        # Update path history
        robot_state.position[2] = 0.08 * abs(math.sin(len(robot_state.path_history) * 0.5))
        robot_state.path_history.append(robot_state.position.copy())
        
        # Update time
        robot_state.time_elapsed = time.time() - start_time
        
        # Energy calculation
        robot_state.energy_consumed += np.linalg.norm(action) * 0.01
        
        # Log state
        algorithm.log_state(robot_state, action)
        
        # Check goal
        dist_to_goal = env.distance_to_goal(robot_state.position)
        if dist_to_goal < robot_params['radius']:
            if verbose:
                print(f"Goal reached in {step} steps!")
            break
            
    # Compute final metrics
    final_distance = env.distance_to_goal(robot_state.position)
    path_length = sum(
        np.linalg.norm(robot_state.path_history[i] - robot_state.path_history[i+1])
        for i in range(len(robot_state.path_history) - 1)
    )
    
    return SimulationResult(
        success=final_distance < robot_params['radius'],
        final_distance=final_distance,
        path_length=path_length,
        time_taken=time.time() - start_time,
        collisions=collision_count,
        energy_consumed=robot_state.energy_consumed,
        mode_changes=mode_changes,
        path=np.array(robot_state.path_history),
        states_log=algorithm.state_log
    )


def update_human(human_state: HumanState, params: Dict) -> HumanState:
    """Update human (dynamic obstacle) position"""
    target = params['path'][human_state.path_segment]
    
    direction = target - human_state.position[:2]
    dist = np.linalg.norm(direction)
    
    if dist < params['speed'] * 1.5:
        human_state.path_segment = (human_state.path_segment + 1) % len(params['path'])
        target = params['path'][human_state.path_segment]
        direction = target - human_state.position[:2]
    
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
        human_state.position[:2] += direction * params['speed']
        human_state.heading_rad = math.atan2(direction[1], direction[0])
    
    return human_state


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BIO-INSPIRED ROBOTICS NAVIGATION SIMULATION")
    print("=" * 70)
    
    # Create environment
    env = Environment()
    
    # Test multiple algorithms
    algorithms = [
        Bug0Algorithm(env),
        Bug1Algorithm(env),
        TangentBugAlgorithm(env),
        FuzzyNavigator(env),
        DynamicWindowApproach(env),
        PotentialFieldNavigation(env),
    ]
    
    results = []
    
    for algo in algorithms:
        print(f"\nRunning {algo.name} algorithm...")
        result = run_simulation(env, algo, verbose=True)
        results.append({
            'algorithm': algo.name,
            'result': result
        })
        
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  Status: {status}")
        print(f"  Final distance: {result.final_distance:.2f}m")
        print(f"  Path length: {result.path_length:.2f}m")
        print(f"  Time: {result.time_taken:.2f}s")
        print(f"  Collisions: {result.collisions}")
        print(f"  Energy: {result.energy_consumed:.2f}")
        print(f"  Mode changes: {result.mode_changes}")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
