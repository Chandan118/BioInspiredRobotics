#!/usr/bin/env python3
"""
Bio-Inspired Robotics - Visualization Module
============================================
Generates figures, plots, and CSV data exports from simulation results.

Author: Chandan Sheikder
Affiliation: Beijing Institute of Technology (BIT)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Wedge, Arrow
from matplotlib.collections import PatchCollection, LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

# Optional 3D animation
try:
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False


@dataclass
class PlotConfig:
    """Configuration for plots"""
    figure_size: Tuple[int, int] = (16, 12)
    dpi: int = 150
    colormap: str = 'viridis'
    style: str = 'seaborn-v0_8-darkgrid'
    save_formats: List[str] = None
    
    def __post_init__(self):
        if self.save_formats is None:
            self.save_formats = ['png', 'pdf', 'svg']


class Visualizer:
    """Comprehensive visualization for bio-inspired robotics simulations"""
    
    def __init__(self, output_dir: str = './results', config: PlotConfig = None):
        self.output_dir = output_dir
        self.config = config or PlotConfig()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
        # Apply style
        try:
            plt.style.use(self.config.style)
        except:
            plt.style.use('default')
            
    def save_figure(self, fig: plt.Figure, name: str):
        """Save figure in configured formats"""
        for fmt in self.config.save_formats:
            filepath = os.path.join(self.output_dir, 'figures', f'{name}.{fmt}')
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            print(f"  Saved: {filepath}")
            
    def plot_2d_environment(self, env, robot_path: np.ndarray = None,
                          human_path: np.ndarray = None,
                          algorithm_name: str = "Navigation",
                          show_tether: bool = True,
                          title: str = None) -> plt.Figure:
        """Plot 2D top-down view of the environment"""
        fig, ax = plt.subplots(figsize=(14, 14), dpi=self.config.dpi)
        
        # Set background
        ax.set_facecolor('#1a1a2e')
        
        # Draw grid
        for i in range(-10, 65, 5):
            ax.axhline(i, color='#2d2d44', linewidth=0.5, alpha=0.5)
            ax.axvline(i, color='#2d2d44', linewidth=0.5, alpha=0.5)
        
        # Draw obstacles with different colors
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
            
            # Add 3D height indicator
            height_ratio = obs.height_3d / 20.0
            ax.text(obs.x + obs.width/2, obs.y + obs.height/2, 
                   f'{obs.height_3d:.0f}m',
                   ha='center', va='center', fontsize=8, color='white', alpha=0.7)
        
        # Draw start and goal
        ax.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=20, 
               markeredgecolor='white', markeredgewidth=2, label='Start')
        ax.plot(env.goal_pos[0], env.goal_pos[1], 'r*', markersize=25,
               markeredgecolor='white', markeredgewidth=2, label='Goal')
        
        # Draw robot path
        if robot_path is not None and len(robot_path) > 0:
            # Color path by segments
            colors = plt.cm.plasma(np.linspace(0, 1, len(robot_path)))
            
            for i in range(len(robot_path) - 1):
                color_idx = i / len(robot_path)
                ax.plot(robot_path[i:i+2, 0], robot_path[i:i+2, 1],
                       color=plt.cm.plasma(color_idx), linewidth=3, alpha=0.8)
            
            # Draw tether
            if show_tether:
                self._draw_tether(ax, robot_path, env)
            
            # Draw robot position
            final_pos = robot_path[-1]
            robot_circle = Circle((final_pos[0], final_pos[1]), 1.5,
                                 facecolor='#00aaff', edgecolor='white',
                                 linewidth=2, alpha=0.9)
            ax.add_patch(robot_circle)
            
            # Draw heading arrow
            if len(robot_path) > 1:
                dx = robot_path[-1, 0] - robot_path[-2, 0]
                dy = robot_path[-1, 1] - robot_path[-2, 1]
                if np.linalg.norm([dx, dy]) > 0.1:
                    dx, dy = dx/3, dy/3
                    ax.arrow(robot_path[-1, 0], robot_path[-1, 1], dx, dy,
                            head_width=0.5, head_length=0.3, fc='white', ec='white')
        
        # Draw human path
        if human_path is not None and len(human_path) > 0:
            ax.plot(human_path[:, 0], human_path[:, 1], 'r--', 
                   linewidth=2, alpha=0.5, label='Human Path')
            ax.plot(human_path[-1, 0], human_path[-1, 1], 'rs',
                   markersize=15, label='Human')
        
        # Draw dynamic obstacles
        for obs in env.dynamic_obstacles:
            circle = Circle((obs.x + obs.width/2, obs.y + obs.height/2),
                           obs.width/2, facecolor='#ff4444', edgecolor='white',
                           linewidth=2, alpha=0.7)
            ax.add_patch(circle)
        
        # Labels and styling
        ax.set_xlim(-10, 60)
        ax.set_ylim(-10, 60)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(title or f'2D Navigation Environment - {algorithm_name}',
                    fontsize=14, fontweight='bold', color='white')
        
        # Legend
        legend = ax.legend(loc='upper right', fontsize=10, 
                         facecolor='#2d2d44', edgecolor='white')
        for text in legend.get_texts():
            text.set_color('white')
        
        # Colorbar for path
        sm = plt.cm.ScalarMappable(cmap='plasma', 
                                   norm=plt.Normalize(0, len(robot_path) if robot_path is not None else 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Time Step', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        fig.tight_layout()
        return fig
    
    def _draw_tether(self, ax, path: np.ndarray, env):
        """Draw tether cable from base to robot"""
        # Simplified tether - direct line with sag
        base_pos = env.start_pos[:2]
        robot_pos = path[-1, :2]
        
        # Calculate sag based on distance
        dist = np.linalg.norm(robot_pos - base_pos)
        sag = 0.5 + dist * 0.02
        
        # Draw curved tether
        mid_x = (base_pos[0] + robot_pos[0]) / 2
        mid_y = (base_pos[1] + robot_pos[1]) / 2 - sag
        
        t = np.linspace(0, 1, 50)
        tether_x = (1-t)**2 * base_pos[0] + 2*(1-t)*t * mid_x + t**2 * robot_pos[0]
        tether_y = (1-t)**2 * base_pos[1] + 2*(1-t)*t * mid_y + t**2 * robot_pos[1]
        
        ax.plot(tether_x, tether_y, 'y-', linewidth=2.5, alpha=0.7, label='Tether')
        
        # Check for snagging (simplified)
        for obs in env.obstacles:
            if obs.contains_point(mid_x, mid_y + sag):
                ax.plot(mid_x, mid_y + sag, 'r*', markersize=15)
                ax.text(mid_x + 0.5, mid_y + sag, 'SNAGGED!', color='red', fontsize=10)
                break
    
    def plot_3d_trajectory(self, robot_path: np.ndarray,
                           env = None,
                           algorithm_name: str = "Navigation") -> plt.Figure:
        """Plot 3D trajectory of robot navigation"""
        fig = plt.figure(figsize=(16, 12), dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_facecolor('#1a1a2e')
        ax.yaxis.pane.set_facecolor('#1a1a2e')
        ax.zaxis.pane.set_facecolor('#1a1a2e')
        
        if robot_path is not None and len(robot_path) > 0:
            # Color by time step
            colors = plt.cm.viridis(np.linspace(0, 1, len(robot_path)))
            
            # Plot trajectory
            for i in range(len(robot_path) - 1):
                ax.plot(robot_path[i:i+2, 0], 
                       robot_path[i:i+2, 1], 
                       robot_path[i:i+2, 2],
                       color=colors[i], linewidth=2.5)
            
            # Plot obstacles as 3D boxes
            if env:
                for obs in env.obstacles:
                    x, y = obs.x, obs.y
                    z = 0
                    w, d, h = obs.width, obs.height, obs.height_3d
                    
                    # Draw box faces
                    vertices = [
                        [x, y, z], [x+w, y, z], [x+w, y+d, z], [x, y+d, z],  # bottom
                        [x, y, z+h], [x+w, y, z+h], [x+w, y+d, z+h], [x, y+d, z+h]  # top
                    ]
                    faces = [
                        [0, 1, 2, 3],  # bottom
                        [4, 5, 6, 7],  # top
                        [0, 1, 5, 4],  # front
                        [2, 3, 7, 6],  # back
                        [0, 3, 7, 4],  # left
                        [1, 2, 6, 5],  # right
                    ]
                    
                    color = {'building': '#4a4a6a', 'planter': '#3d5a3d',
                            'streetlight': '#5a5a3d'}.get(obs.obstacle_type, '#4a4a6a')
                    
                    for face in faces:
                        face_verts = [vertices[v] for v in face]
                        verts = [[v[0] for v in face_verts],
                                [v[1] for v in face_verts],
                                [v[2] for v in face_verts]]
                        ax.plot_surface(verts[0], verts[1], verts[2],
                                       color=color, alpha=0.7)
                
                # Start and goal markers
                ax.scatter([env.start_pos[0]], [env.start_pos[1]], [env.start_pos[2]],
                          c='green', s=200, marker='o', edgecolors='white', linewidths=2)
                ax.scatter([env.goal_pos[0]], [env.goal_pos[1]], [env.goal_pos[2]],
                          c='red', s=300, marker='*', edgecolors='white', linewidths=2)
            
            # Final robot position
            ax.scatter([robot_path[-1, 0]], [robot_path[-1, 1]], [robot_path[-1, 2]],
                     c='cyan', s=300, marker='o', edgecolors='white', linewidths=2)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'3D Trajectory - {algorithm_name}', fontsize=14, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        fig.tight_layout()
        return fig
    
    def plot_algorithm_comparison(self, results: List[Dict]) -> plt.Figure:
        """Create comparison plots for multiple algorithms"""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        algorithms = [r['algorithm'] for r in results]
        n_algos = len(algorithms)
        
        # Prepare data
        success = [1 if r['result'].success else 0 for r in results]
        path_lengths = [r['result'].path_length for r in results]
        times = [r['result'].time_taken for r in results]
        collisions = [r['result'].collisions for r in results]
        distances = [r['result'].final_distance for r in results]
        energies = [r['result'].energy_consumed for r in results]
        
        # Color palette
        colors = plt.cm.Set2(np.linspace(0, 1, n_algos))
        
        # 1. Success Rate (bar)
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(algorithms, success, color=colors, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Algorithm', fontweight='bold')
        ax1.set_ylim(0, 1.2)
        for bar, s in zip(bars, success):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{s*100:.0f}%', ha='center', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Path Length (bar)
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(algorithms, path_lengths, color=colors, edgecolor='white', linewidth=2)
        ax2.set_ylabel('Path Length (m)')
        ax2.set_title('Total Path Length', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Time Taken (bar)
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(algorithms, times, color=colors, edgecolor='white', linewidth=2)
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Execution Time', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Collisions (bar)
        ax4 = fig.add_subplot(gs[1, 0])
        bars = ax4.bar(algorithms, collisions, color=colors, edgecolor='white', linewidth=2)
        ax4.set_ylabel('Collisions')
        ax4.set_title('Number of Collisions', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Final Distance (bar)
        ax5 = fig.add_subplot(gs[1, 1])
        bars = ax5.bar(algorithms, distances, color=colors, edgecolor='white', linewidth=2)
        ax5.set_ylabel('Distance to Goal (m)')
        ax5.set_title('Final Distance to Goal', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Energy Consumption (bar)
        ax6 = fig.add_subplot(gs[1, 2])
        bars = ax6.bar(algorithms, energies, color=colors, edgecolor='white', linewidth=2)
        ax6.set_ylabel('Energy (J)')
        ax6.set_title('Energy Consumption', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Radar Chart
        ax7 = fig.add_subplot(gs[2, 0], projection='polar')
        self._draw_radar_chart(ax7, results, algorithms, colors)
        
        # 8. Trajectories overlay
        ax8 = fig.add_subplot(gs[2, 1:])
        for i, r in enumerate(results):
            if r['result'].path is not None and len(r['result'].path) > 0:
                ax8.plot(r['result'].path[:, 0], r['result'].path[:, 1],
                        color=colors[i], linewidth=2, alpha=0.7,
                        label=f"{r['algorithm']} ({'✓' if r['result'].success else '✗'})")
        ax8.set_xlabel('X (m)')
        ax8.set_ylabel('Y (m)')
        ax8.set_title('Trajectory Comparison', fontweight='bold')
        ax8.legend(loc='upper right')
        ax8.set_aspect('equal')
        ax8.grid(True, alpha=0.3)
        
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def _draw_radar_chart(self, ax, results: List[Dict], algorithms: List[str], colors):
        """Draw radar chart for multi-dimensional comparison"""
        # Metrics for radar
        categories = ['Path Length\n(inv)', 'Speed', 'Safety', 'Energy\n(eff)', 'Goal Accuracy']
        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Normalize metrics
        max_path = max(r['result'].path_length for r in results) + 1e-6
        max_time = max(r['result'].time_taken for r in results) + 1e-6
        max_collisions = max(r['result'].collisions for r in results) + 1
        max_energy = max(r['result'].energy_consumed for r in results) + 1e-6
        max_dist = max(r['result'].final_distance for r in results) + 1e-6
        
        for i, r in enumerate(results):
            # Normalize to 0-1 scale (higher is better)
            values = [
                1 - r['result'].path_length / max_path,
                1 - r['result'].time_taken / max_time,
                1 - r['result'].collisions / max_collisions,
                1 - r['result'].energy_consumed / max_energy,
                1 - r['result'].final_distance / max_dist,
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=algorithms[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-dimensional Comparison', fontweight='bold', size=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    def plot_trajectory_analysis(self, robot_path: np.ndarray,
                                states_log: List[Dict] = None,
                                algorithm_name: str = "Navigation") -> plt.Figure:
        """Detailed trajectory analysis"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        if robot_path is None or len(robot_path) < 2:
            return fig
        
        # Time array
        time_steps = np.arange(len(robot_path))
        
        # 1. Distance to Goal over Time
        ax1 = fig.add_subplot(gs[0, 0])
        goal_pos = np.array([45, 5, 0])
        distances = np.linalg.norm(robot_path[:, :2] - goal_pos[:2], axis=1)
        ax1.plot(time_steps, distances, 'b-', linewidth=2)
        ax1.fill_between(time_steps, 0, distances, alpha=0.3)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Distance to Goal (m)')
        ax1.set_title('Distance to Goal Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Velocity Profile
        ax2 = fig.add_subplot(gs[0, 1])
        velocities = []
        for i in range(len(robot_path) - 1):
            vel = np.linalg.norm(robot_path[i+1] - robot_path[i])
            velocities.append(vel)
        velocities = np.array(velocities)
        ax2.plot(time_steps[1:], velocities, 'g-', linewidth=2)
        ax2.fill_between(time_steps[1:], 0, velocities, alpha=0.3, color='green')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Speed (m/step)')
        ax2.set_title('Velocity Profile', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Path Length
        ax3 = fig.add_subplot(gs[0, 2])
        step_distances = np.array([0] + list(np.linalg.norm(
            robot_path[i+1] - robot_path[i] for i in range(len(robot_path)-1))))
        cumulative = np.cumsum(step_distances)
        ax3.plot(time_steps, cumulative, 'orange', linewidth=2)
        ax3.fill_between(time_steps, 0, cumulative, alpha=0.3, color='orange')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Cumulative Distance (m)')
        ax3.set_title('Cumulative Path Length', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Heading Angle
        ax4 = fig.add_subplot(gs[1, 0])
        headings = []
        for i in range(len(robot_path) - 1):
            dx = robot_path[i+1, 0] - robot_path[i, 0]
            dy = robot_path[i+1, 1] - robot_path[i, 1]
            headings.append(np.arctan2(dy, dx))
        headings = np.array(headings)
        ax4.plot(time_steps[1:], np.degrees(headings), 'purple', linewidth=2)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Heading (degrees)')
        ax4.set_title('Heading Angle Over Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Heading Change Rate
        ax5 = fig.add_subplot(gs[1, 1])
        heading_changes = np.abs(np.diff(headings))
        heading_changes = np.rad2deg(heading_changes)
        ax5.plot(time_steps[2:], heading_changes, 'red', linewidth=2)
        ax5.fill_between(time_steps[2:], 0, heading_changes, alpha=0.3, color='red')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Heading Change (deg/step)')
        ax5.set_title('Heading Change Rate', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 2D Position (X and Y separately)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(time_steps, robot_path[:, 0], 'b-', linewidth=2, label='X')
        ax6.plot(time_steps, robot_path[:, 1], 'r-', linewidth=2, label='Y')
        ax6.plot(time_steps, robot_path[:, 2] * 10, 'g-', linewidth=2, label='Z×10')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Position (m)')
        ax6.set_title('Position Components Over Time', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Heatmap of position density
        ax7 = fig.add_subplot(gs[2, 0])
        hist, xedges, yedges = np.histogram2d(robot_path[:, 0], robot_path[:, 1], bins=20)
        im = ax7.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap='hot', aspect='auto')
        ax7.set_xlabel('X (m)')
        ax7.set_ylabel('Y (m)')
        ax7.set_title('Position Density Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax7, label='Visits')
        
        # 8. Curvature
        ax8 = fig.add_subplot(gs[2, 1])
        curvatures = []
        for i in range(1, len(robot_path) - 1):
            v1 = robot_path[i] - robot_path[i-1]
            v2 = robot_path[i+1] - robot_path[i]
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            angle = np.arctan2(cross, dot)
            curvatures.append(angle)
        curvatures = np.abs(np.rad2deg(np.array(curvatures)))
        ax8.plot(time_steps[1:-1], curvatures, 'brown', linewidth=2)
        ax8.fill_between(time_steps[1:-1], 0, curvatures, alpha=0.3, color='brown')
        ax8.set_xlabel('Time Step')
        ax8.set_ylabel('Curvature (deg)')
        ax8.set_title('Path Curvature', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 9. Phase space
        ax9 = fig.add_subplot(gs[2, 2])
        scatter = ax9.scatter(robot_path[:-1, 0], velocities, c=time_steps[1:], 
                            cmap='viridis', s=10, alpha=0.7)
        ax9.set_xlabel('X Position (m)')
        ax9.set_ylabel('Speed (m/step)')
        ax9.set_title('Phase Space (Position vs Speed)', fontweight='bold')
        plt.colorbar(scatter, ax=ax9, label='Time Step')
        ax9.grid(True, alpha=0.3)
        
        fig.suptitle(f'Trajectory Analysis - {algorithm_name}', fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_learning_curves(self, training_history: Dict,
                            algorithm_name: str = "RL Agent") -> plt.Figure:
        """Plot learning curves for RL algorithms"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Episode rewards
        ax1 = axes[0, 0]
        if 'episode_rewards' in training_history:
            rewards = training_history['episode_rewards']
            ax1.plot(rewards, 'b-', alpha=0.3, linewidth=0.5)
            # Moving average
            window = min(100, len(rewards) // 10)
            if window > 1:
                avg_rewards = pd.Series(rewards).rolling(window).mean()
                ax1.plot(avg_rewards, 'r-', linewidth=2, label=f'{window}-episode moving avg')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Episode Rewards Over Training', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Episode lengths
        ax2 = axes[0, 1]
        if 'episode_lengths' in training_history:
            lengths = training_history['episode_lengths']
            ax2.plot(lengths, 'g-', alpha=0.3, linewidth=0.5)
            window = min(100, len(lengths) // 10)
            if window > 1:
                avg_lengths = pd.Series(lengths).rolling(window).mean()
                ax2.plot(avg_lengths, 'orange', linewidth=2, label=f'{window}-episode moving avg')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode Length (steps)')
            ax2.set_title('Episode Length Over Training', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Epsilon decay
        ax3 = axes[1, 0]
        if 'epsilon' in training_history:
            epsilon = training_history['epsilon']
            ax3.plot(epsilon, 'purple', linewidth=2)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Epsilon')
            ax3.set_title('Exploration Rate (Epsilon) Decay', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Q-value evolution
        ax4 = axes[1, 1]
        if 'q_values' in training_history:
            q_vals = training_history['q_values']
            for i, q_val in enumerate(q_vals[:5]):  # Show first 5 Q-values
                ax4.plot(q_val, alpha=0.5, label=f'Action {i}')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Q-Value')
            ax4.set_title('Q-Value Evolution', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        fig.suptitle(f'Learning Curves - {algorithm_name}', fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout()
        
        return fig
    
    def plot_fuzzy_membership(self, navigator) -> plt.Figure:
        """Visualize fuzzy logic membership functions and rules"""
        fig = plt.figure(figsize=(20, 16))
        
        # Distance membership
        ax1 = fig.add_subplot(3, 3, 1)
        x_dist = np.linspace(0, 5, 200)
        for mf_name, params in navigator.membership_functions['distance'].items():
            if len(params) == 4:
                y = np.piecewise(x_dist,
                                [(x_dist >= params[0]) & (x_dist <= params[1]),
                                 (x_dist >= params[1]) & (x_dist <= params[2]),
                                 (x_dist >= params[2]) & (x_dist <= params[3])],
                                [lambda x, p=params: (x - p[0]) / (p[1] - p[0]),
                                 lambda x, p=params: 1.0,
                                 lambda x, p=params: (p[3] - x) / (p[3] - p[2])])
            else:
                y = np.piecewise(x_dist,
                                [(x_dist >= params[0]) & (x_dist <= params[1]),
                                 (x_dist >= params[1]) & (x_dist <= params[2])],
                                [lambda x, p=params: (x - p[0]) / (p[1] - p[0]),
                                 lambda x, p=params: (p[2] - x) / (p[2] - p[1])])
                y = np.nan_to_num(y)
            ax1.plot(x_dist, y, linewidth=2, label=mf_name)
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Membership')
        ax1.set_title('Distance Membership Functions', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Angle membership
        ax2 = fig.add_subplot(3, 3, 2)
        x_angle = np.linspace(-np.pi, np.pi, 200)
        for mf_name, params in navigator.membership_functions['angle'].items():
            y = np.piecewise(x_angle,
                           [(x_angle >= params[0]) & (x_angle <= params[1]),
                            (x_angle >= params[1]) & (x_angle <= params[2])],
                           [lambda x, p=params: (x - p[0]) / (p[1] - p[0] + 1e-6),
                            lambda x, p=params: (p[2] - x) / (p[2] - p[1] + 1e-6)])
            y = np.nan_to_num(y)
            ax2.plot(x_angle, y, linewidth=2, label=mf_name)
        ax2.set_xlabel('Angle (rad)')
        ax2.set_ylabel('Membership')
        ax2.set_title('Angle Membership Functions', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Steering membership
        ax3 = fig.add_subplot(3, 3, 3)
        x_steer = np.linspace(-1.5, 1.5, 200)
        for mf_name, params in navigator.membership_functions['steering'].items():
            y = np.piecewise(x_steer,
                           [(x_steer >= params[0]) & (x_steer <= params[1]),
                            (x_steer >= params[1]) & (x_steer <= params[2])],
                           [lambda x, p=params: (x - p[0]) / (p[1] - p[0] + 1e-6),
                            lambda x, p=params: (p[2] - x) / (p[2] - p[1] + 1e-6)])
            y = np.nan_to_num(y)
            ax3.plot(x_steer, y, linewidth=2, label=mf_name)
        ax3.set_xlabel('Steering Output')
        ax3.set_ylabel('Membership')
        ax3.set_title('Steering Membership Functions', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Surface plot
        ax4 = fig.add_subplot(3, 3, 4, projection='3d')
        X = np.linspace(0, 5, 50)
        Y = np.linspace(-np.pi, np.pi, 50)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = navigator._inference(X[i, j], Y[i, j])
        
        surf = ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax4.set_xlabel('Distance')
        ax4.set_ylabel('Angle')
        ax4.set_zlabel('Steering')
        ax4.set_title('Fuzzy Inference Surface', fontweight='bold')
        
        # Rule strength visualization
        ax5 = fig.add_subplot(3, 3, 5)
        rule_names = [f"Rule {i+1}" for i in range(len(navigator.fuzzy_rules))]
        rule_strengths = [0.3 + 0.7 * np.random.random() for _ in navigator.fuzzy_rules]  # Simulated
        bars = ax5.barh(rule_names, rule_strengths, color='steelblue', edgecolor='white')
        ax5.set_xlabel('Activation Strength')
        ax5.set_title('Fuzzy Rule Activation (Example)', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Input-output mapping
        ax6 = fig.add_subplot(3, 3, 6)
        distances = np.linspace(0, 5, 100)
        for angle in [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]:
            outputs = [navigator._inference(d, angle) for d in distances]
            label = f'angle={np.degrees(angle):.0f}°'
            ax6.plot(distances, outputs, linewidth=2, label=label)
        ax6.set_xlabel('Distance (m)')
        ax6.set_ylabel('Steering Output')
        ax6.set_title('Fuzzy Input-Output Mapping', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle('Fuzzy Logic Controller Visualization', fontsize=16, fontweight='bold')
        fig.tight_layout()
        
        return fig
    
    def plot_convergence_comparison(self, convergence_data: Dict[str, List[float]],
                                    title: str = "Optimization Convergence") -> plt.Figure:
        """Compare convergence curves for different optimization methods"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(convergence_data)))
        
        for i, (method, values) in enumerate(convergence_data.items()):
            iterations = np.arange(len(values))
            ax.plot(iterations, values, '-', linewidth=2.5, color=colors[i], 
                   label=method, alpha=0.8)
            # Add confidence band
            if len(values) > 20:
                window = min(50, len(values) // 4)
                if window > 1:
                    smoothed = pd.Series(values).rolling(window, center=True).mean()
                    std = pd.Series(values).rolling(window, center=True).std()
                    ax.fill_between(iterations, 
                                   np.maximum(0, smoothed - std),
                                   smoothed + std,
                                   color=colors[i], alpha=0.2)
        
        ax.set_xlabel('Iteration / Generation', fontsize=12)
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def create_animation(self, env, robot_path: np.ndarray,
                        human_path: np.ndarray = None,
                        filename: str = 'navigation_animation.gif',
                        interval: int = 50) -> str:
        """Create animation of robot navigation"""
        if not HAS_ANIMATION:
            print("Animation not available (ffmpeg required)")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Initialize plot elements
        ax.set_facecolor('#1a1a2e')
        
        # Draw static environment
        for obs in env.obstacles:
            color = {'building': '#4a4a6a', 'planter': '#3d5a3d',
                    'streetlight': '#5a5a3d'}.get(obs.obstacle_type, '#4a4a6a')
            rect = Rectangle((obs.x, obs.y), obs.width, obs.height,
                            facecolor=color, edgecolor='#8888aa', linewidth=2)
            ax.add_patch(rect)
        
        ax.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=15)
        ax.plot(env.goal_pos[0], env.goal_pos[1], 'r*', markersize=20)
        
        # Trail and robot
        trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
        robot, = ax.plot([], [], 'co', markersize=15, markeredgecolor='white')
        human, = ax.plot([], [], 'rs', markersize=10)
        
        ax.set_xlim(-10, 60)
        ax.set_ylim(-10, 60)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        def init():
            trail.set_data([], [])
            robot.set_data([], [])
            human.set_data([], [])
            return trail, robot, human
        
        def update(frame):
            # Robot trail
            if frame < len(robot_path):
                trail.set_data(robot_path[:frame+1, 0], robot_path[:frame+1, 1])
                robot.set_data([robot_path[frame, 0]], [robot_path[frame, 1]])
            
            # Human position
            if human_path is not None and frame < len(human_path):
                human.set_data([human_path[frame, 0]], [human_path[frame, 1]])
            
            return trail, robot, human
        
        anim = FuncAnimation(fig, update, frames=len(robot_path),
                           init_func=init, interval=interval, blit=True)
        
        filepath = os.path.join(self.output_dir, 'figures', filename)
        anim.save(filepath, writer='pillow', fps=20)
        print(f"  Animation saved: {filepath}")
        
        plt.close(fig)
        return filepath


class DataExporter:
    """Export simulation data to various formats"""
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    def export_to_csv(self, results: List[Dict], filename: str = 'simulation_results.csv'):
        """Export results to CSV"""
        data = []
        for r in results:
            row = {
                'algorithm': r['algorithm'],
                'success': r['result'].success,
                'final_distance': r['result'].final_distance,
                'path_length': r['result'].path_length,
                'time_taken': r['result'].time_taken,
                'collisions': r['result'].collisions,
                'energy_consumed': r['result'].energy_consumed,
                'mode_changes': r['result'].mode_changes,
                'path_points': len(r['result'].path)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.output_dir, 'data', filename)
        df.to_csv(filepath, index=False)
        print(f"  CSV exported: {filepath}")
        return df
    
    def export_trajectory_csv(self, algorithm_name: str, path: np.ndarray,
                             states_log: List[Dict] = None,
                             filename: str = None) -> str:
        """Export trajectory data to CSV"""
        if filename is None:
            filename = f'trajectory_{algorithm_name.lower().replace(" ", "_")}.csv'
        
        data = []
        for i, pos in enumerate(path):
            row = {
                'step': i,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2] if len(pos) > 2 else 0
            }
            
            if states_log and i < len(states_log):
                row['heading'] = states_log[i].get('heading', 0)
                row['mode'] = states_log[i].get('mode', 'UNKNOWN')
                row['reward'] = states_log[i].get('reward', 0)
            
            # Calculate metrics
            if i > 0:
                row['step_distance'] = np.linalg.norm(pos - path[i-1])
                row['cumulative_distance'] = sum(
                    np.linalg.norm(path[j+1] - path[j])
                    for j in range(i)
                )
            
            data.append(row)
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.output_dir, 'data', filename)
        df.to_csv(filepath, index=False)
        print(f"  Trajectory CSV exported: {filepath}")
        return filepath
    
    def export_training_history(self, history: Dict, 
                               filename: str = 'training_history.json'):
        """Export training history to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                try:
                    serializable[key] = [float(v) if isinstance(v, (np.floating, float)) 
                                        else int(v) if isinstance(v, (np.integer, int))
                                        else v for v in value]
                except:
                    serializable[key] = str(value)
            else:
                serializable[key] = value
        
        filepath = os.path.join(self.output_dir, 'data', filename)
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"  Training history exported: {filepath}")
        return filepath
    
    def export_summary_report(self, results: List[Dict],
                             filename: str = 'summary_report.txt'):
        """Export summary report as text file"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("BIO-INSPIRED ROBOTICS NAVIGATION - SIMULATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 70 + "\n\n")
            
            for r in results:
                f.write(f"Algorithm: {r['algorithm']}\n")
                f.write(f"  Status: {'SUCCESS' if r['result'].success else 'FAILED'}\n")
                f.write(f"  Final Distance: {r['result'].final_distance:.4f} m\n")
                f.write(f"  Path Length: {r['result'].path_length:.4f} m\n")
                f.write(f"  Time: {r['result'].time_taken:.4f} s\n")
                f.write(f"  Collisions: {r['result'].collisions}\n")
                f.write(f"  Energy: {r['result'].energy_consumed:.4f} J\n")
                f.write(f"  Mode Changes: {r['result'].mode_changes}\n")
                f.write(f"  Path Points: {len(r['result'].path)}\n")
                f.write("\n")
            
            # Find best algorithm
            successful = [r for r in results if r['result'].success]
            if successful:
                best = min(successful, key=lambda x: x['result'].path_length)
                f.write("-" * 70 + "\n")
                f.write(f"BEST ALGORITHM: {best['algorithm']}\n")
                f.write(f"  Shortest successful path: {best['result'].path_length:.4f} m\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")
        
        print(f"  Summary report exported: {filepath}")
        return filepath


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Visualization Module Test")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    
    # Sample trajectory
    t = np.linspace(0, 4 * np.pi, 200)
    x = 25 + 15 * np.cos(t/4) + np.random.randn(200) * 2
    y = 25 + 15 * np.sin(t/4) + np.random.randn(200) * 2
    z = np.abs(np.sin(t)) * 2
    sample_path = np.column_stack([x, y, z])
    
    # Create visualizer
    vis = Visualizer(output_dir='./test_results')
    
    # Test 2D environment plot
    from simulations.comprehensive_navigation import Environment
    env = Environment()
    
    fig2d = vis.plot_2d_environment(env, sample_path, algorithm_name="Sample")
    vis.save_figure(fig2d, 'test_2d_environment')
    
    # Test 3D trajectory
    fig3d = vis.plot_3d_trajectory(sample_path, env, "Sample")
    vis.save_figure(fig3d, 'test_3d_trajectory')
    
    # Test trajectory analysis
    fig_analysis = vis.plot_trajectory_analysis(sample_path, algorithm_name="Sample")
    vis.save_figure(fig_analysis, 'test_trajectory_analysis')
    
    # Export data
    exporter = DataExporter(output_dir='./test_results')
    
    # Sample results
    sample_results = [
        {'algorithm': 'Bug0', 'result': type('obj', (object,), {
            'success': True, 'final_distance': 0.5, 'path_length': 65.2,
            'time_taken': 2.3, 'collisions': 0, 'energy_consumed': 15.2,
            'mode_changes': 4, 'path': sample_path
        })()},
        {'algorithm': 'Bug1', 'result': type('obj', (object,), {
            'success': True, 'final_distance': 0.3, 'path_length': 68.1,
            'time_taken': 2.1, 'collisions': 0, 'energy_consumed': 16.5,
            'mode_changes': 3, 'path': sample_path * 0.95
        })()},
        {'algorithm': 'Fuzzy', 'result': type('obj', (object,), {
            'success': False, 'final_distance': 2.5, 'path_length': 70.0,
            'time_taken': 2.5, 'collisions': 1, 'energy_consumed': 18.2,
            'mode_changes': 6, 'path': sample_path * 0.9
        })()},
    ]
    
    exporter.export_to_csv(sample_results)
    exporter.export_trajectory_csv("Sample", sample_path)
    exporter.export_summary_report(sample_results)
    
    print("\nVisualization tests complete!")
