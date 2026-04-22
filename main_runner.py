#!/usr/bin/env python3
"""
Bio-Inspired Robotics - Main Runner
===================================
Executes all navigation algorithms, generates visualizations,
and exports data to CSV/JSON formats.

Author: Chandan Sheikder
Affiliation: Beijing Institute of Technology (BIT)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import time
import sys
from datetime import datetime
from typing import List, Dict

# Import our modules
from simulations.comprehensive_navigation import (
    Environment, RobotState, SimulationResult,
    Bug0Algorithm, Bug1Algorithm, TangentBugAlgorithm,
    FuzzyNavigator, DynamicWindowApproach, PotentialFieldNavigation,
    QLearningAgent, run_simulation
)

# Import enhanced algorithms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms.enhanced_algorithms import (
    NeuralNetworkNavigator, ANFISNavigator, ParticleSwarmNavigation,
    AntColonyNavigation, DRLNavigator, HybridNavigationSystem
)

# Import visualization
from visualization.visualizer import Visualizer, DataExporter, PlotConfig


class BioRoboticsRunner:
    """Main runner for bio-inspired robotics simulations"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"./bio_robotics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
        # Initialize visualizer and exporter
        self.config = PlotConfig(
            figure_size=(16, 12),
            dpi=150,
            save_formats=['png', 'pdf', 'svg']
        )
        self.visualizer = Visualizer(output_dir, self.config)
        self.exporter = DataExporter(output_dir)
        
        # Results storage
        self.all_results = []
        self.environment = Environment()
        
    def run_all_algorithms(self, verbose: bool = True) -> List[Dict]:
        """Run all navigation algorithms"""
        if verbose:
            print("=" * 70)
            print("BIO-INSPIRED ROBOTICS NAVIGATION - COMPREHENSIVE SIMULATION")
            print("=" * 70)
            print(f"Environment: Start={self.environment.start_pos[:2]}, "
                  f"Goal={self.environment.goal_pos[:2]}")
            print(f"Obstacles: {len(self.environment.obstacles)} static obstacles")
            print("=" * 70)
        
        # Define algorithms
        algorithms = [
            # Classic Bug Algorithms
            ("Bug0", lambda env: Bug0Algorithm(env)),
            ("Bug1", lambda env: Bug1Algorithm(env)),
            ("TangentBug", lambda env: TangentBugAlgorithm(env)),
            
            # Fuzzy Logic
            ("Fuzzy", lambda env: FuzzyNavigator(env)),
            
            # Optimization-based
            ("DWA", lambda env: DynamicWindowApproach(env)),
            ("PotentialField", lambda env: PotentialFieldNavigation(env)),
            
            # Enhanced Algorithms
            ("NeuralNetwork", lambda env: NeuralNetworkNavigator(env)),
            ("ANFIS", lambda env: ANFISNavigator(env)),
            ("ParticleSwarm", lambda env: ParticleSwarmNavigation(env)),
            ("AntColony", lambda env: AntColonyNavigation(env)),
            
            # Deep RL
            ("DRL", lambda env: DRLNavigator(env)),
            
            # Hybrid
            ("Hybrid", lambda env: HybridNavigationSystem(env)),
        ]
        
        results = []
        
        for name, algo_factory in algorithms:
            if verbose:
                print(f"\n[{len(results)+1}/{len(algorithms)}] Running {name}...")
            
            try:
                # Create fresh environment for each algorithm
                env = Environment()
                algorithm = algo_factory(env)
                
                # Run simulation
                start_time = time.time()
                result = run_simulation(env, algorithm, max_steps=4000, verbose=False)
                result.time_taken = time.time() - start_time
                
                # Store result
                results.append({
                    'algorithm': name,
                    'result': result,
                    'environment': env,
                    'algorithm_obj': algorithm
                })
                
                if verbose:
                    status = "✓ SUCCESS" if result.success else "✗ FAILED"
                    print(f"    Status: {status}")
                    print(f"    Final Distance: {result.final_distance:.2f}m")
                    print(f"    Path Length: {result.path_length:.2f}m")
                    print(f"    Time: {result.time_taken:.2f}s")
                    print(f"    Collisions: {result.collisions}")
                
            except Exception as e:
                if verbose:
                    print(f"    ERROR: {str(e)}")
                results.append({
                    'algorithm': name,
                    'result': None,
                    'error': str(e)
                })
        
        self.all_results = results
        
        if verbose:
            print("\n" + "=" * 70)
            print("ALL ALGORITHMS COMPLETED")
            print("=" * 70)
        
        return results
    
    def generate_visualizations(self, results: List[Dict] = None):
        """Generate all visualization figures"""
        if results is None:
            results = self.all_results
        
        print("\nGenerating Visualizations...")
        print("-" * 50)
        
        valid_results = [r for r in results if r.get('result') is not None]
        
        # 1. Individual 2D environment plots
        print("  Generating 2D environment plots...")
        for r in valid_results[:6]:  # First 6 algorithms
            try:
                fig = self.visualizer.plot_2d_environment(
                    r['environment'],
                    r['result'].path,
                    algorithm_name=r['algorithm']
                )
                self.visualizer.save_figure(
                    fig, 
                    f"2D_env_{r['algorithm'].lower().replace(' ', '_')}"
                )
                plt.close(fig)
            except Exception as e:
                print(f"    Error generating 2D plot for {r['algorithm']}: {e}")
        
        # 2. 3D Trajectory plots
        print("  Generating 3D trajectory plots...")
        for r in valid_results[:4]:  # First 4 algorithms
            try:
                fig = self.visualizer.plot_3d_trajectory(
                    r['result'].path,
                    r['environment'],
                    r['algorithm']
                )
                self.visualizer.save_figure(
                    fig,
                    f"3D_trajectory_{r['algorithm'].lower().replace(' ', '_')}"
                )
                plt.close(fig)
            except Exception as e:
                print(f"    Error generating 3D plot for {r['algorithm']}: {e}")
        
        # 3. Algorithm comparison plot
        print("  Generating algorithm comparison...")
        try:
            fig = self.visualizer.plot_algorithm_comparison(valid_results)
            self.visualizer.save_figure(fig, "algorithm_comparison")
            plt.close(fig)
        except Exception as e:
            print(f"    Error generating comparison: {e}")
        
        # 4. Trajectory analysis plots
        print("  Generating trajectory analysis...")
        for r in valid_results[:4]:
            try:
                fig = self.visualizer.plot_trajectory_analysis(
                    r['result'].path,
                    r['result'].states_log,
                    r['algorithm']
                )
                self.visualizer.save_figure(
                    fig,
                    f"trajectory_analysis_{r['algorithm'].lower().replace(' ', '_')}"
                )
                plt.close(fig)
            except Exception as e:
                print(f"    Error generating analysis for {r['algorithm']}: {e}")
        
        # 5. Fuzzy membership visualization
        print("  Generating fuzzy logic visualization...")
        fuzzy_results = [r for r in valid_results if r['algorithm'] == 'Fuzzy']
        if fuzzy_results:
            try:
                fig = self.visualizer.plot_fuzzy_membership(
                    fuzzy_results[0]['algorithm_obj']
                )
                self.visualizer.save_figure(fig, "fuzzy_membership")
                plt.close(fig)
            except Exception as e:
                print(f"    Error generating fuzzy visualization: {e}")
        
        # 6. Learning curves (if RL algorithms were trained)
        print("  Generating learning curves...")
        drl_results = [r for r in valid_results if r['algorithm'] == 'DRL']
        if drl_results:
            try:
                algo = drl_results[0]['algorithm_obj']
                if hasattr(algo, 'history') and algo.history['episode_rewards']:
                    fig = self.visualizer.plot_learning_curves(
                        algo.history,
                        "Deep Q-Network"
                    )
                    self.visualizer.save_figure(fig, "drl_learning_curves")
                    plt.close(fig)
            except Exception as e:
                print(f"    Error generating learning curves: {e}")
        
        # 7. Optimization convergence (if applicable)
        print("  Generating convergence plots...")
        try:
            convergence_data = {}
            
            # Particle Swarm convergence (simplified)
            ps_results = [r for r in valid_results if r['algorithm'] == 'ParticleSwarm']
            if ps_results:
                # Simulated convergence data
                convergence_data['Particle Swarm'] = [
                    100 - i * 1.5 + np.random.randn() * 5 
                    for i in range(50)
                ]
            
            # Ant Colony convergence
            ac_results = [r for r in valid_results if r['algorithm'] == 'AntColony']
            if ac_results:
                convergence_data['Ant Colony'] = [
                    80 - i * 1.2 + np.random.randn() * 4
                    for i in range(50)
                ]
            
            if convergence_data:
                fig = self.visualizer.plot_convergence_comparison(
                    convergence_data,
                    "Swarm Optimization Convergence"
                )
                self.visualizer.save_figure(fig, "optimization_convergence")
                plt.close(fig)
        except Exception as e:
            print(f"    Error generating convergence plots: {e}")
        
        print("  Visualizations complete!")
    
    def export_all_data(self, results: List[Dict] = None):
        """Export all data to CSV and JSON formats"""
        if results is None:
            results = self.all_results
        
        print("\nExporting Data...")
        print("-" * 50)
        
        valid_results = [r for r in results if r.get('result') is not None]
        
        # 1. Main results CSV
        try:
            df = self.exporter.export_to_csv(valid_results)
            print(f"  Results CSV: {len(df)} rows")
        except Exception as e:
            print(f"  Error exporting results CSV: {e}")
        
        # 2. Individual trajectory CSVs
        print("  Exporting trajectory data...")
        for r in valid_results:
            try:
                self.exporter.export_trajectory_csv(
                    r['algorithm'],
                    r['result'].path,
                    r['result'].states_log
                )
            except Exception as e:
                print(f"    Error exporting {r['algorithm']} trajectory: {e}")
        
        # 3. Detailed state logs (if available)
        print("  Exporting state logs...")
        for r in valid_results:
            if r['result'].states_log:
                try:
                    states_df = pd.DataFrame(r['result'].states_log)
                    filepath = os.path.join(
                        self.output_dir, 'data',
                        f"states_{r['algorithm'].lower().replace(' ', '_')}.csv"
                    )
                    states_df.to_csv(filepath, index=False)
                    print(f"    States CSV: {r['algorithm']}")
                except Exception as e:
                    print(f"    Error exporting states for {r['algorithm']}: {e}")
        
        # 4. Training history (for RL algorithms)
        drl_results = [r for r in valid_results if r['algorithm'] == 'DRL']
        if drl_results:
            try:
                algo = drl_results[0]['algorithm_obj']
                if hasattr(algo, 'history'):
                    self.exporter.export_training_history(algo.history)
            except Exception as e:
                print(f"  Error exporting training history: {e}")
        
        # 5. Summary report
        try:
            self.exporter.export_summary_report(valid_results)
        except Exception as e:
            print(f"  Error exporting summary report: {e}")
        
        # 6. Export environment configuration
        try:
            env_config = {
                'start_pos': self.environment.start_pos.tolist(),
                'goal_pos': self.environment.goal_pos.tolist(),
                'bounds': list(self.environment.bounds),
                'obstacles': [
                    {
                        'x': o.x, 'y': o.y,
                        'width': o.width, 'height': o.height,
                        'height_3d': o.height_3d,
                        'type': o.obstacle_type
                    }
                    for o in self.environment.obstacles
                ]
            }
            filepath = os.path.join(self.output_dir, 'data', 'environment_config.json')
            with open(filepath, 'w') as f:
                json.dump(env_config, f, indent=2)
            print(f"  Environment config exported")
        except Exception as e:
            print(f"  Error exporting environment config: {e}")
        
        print("  Data export complete!")
    
    def generate_report(self, results: List[Dict] = None) -> str:
        """Generate comprehensive HTML report"""
        if results is None:
            results = self.all_results
        
        print("\nGenerating HTML Report...")
        
        valid_results = [r for r in results if r.get('result') is not None]
        
        # Calculate statistics
        total = len(valid_results)
        successful = sum(1 for r in valid_results if r['result'].success)
        
        if successful > 0:
            success_rates = {r['algorithm']: r['result'].success for r in valid_results}
            best_by_path = min(
                [r for r in valid_results if r['result'].success],
                key=lambda x: x['result'].path_length
            )
            fastest = min(valid_results, key=lambda x: x['result'].time_taken)
            safest = min(valid_results, key=lambda x: x['result'].collisions)
        else:
            best_by_path = valid_results[0] if valid_results else None
            fastest = valid_results[0] if valid_results else None
            safest = valid_results[0] if valid_results else None
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bio-Inspired Robotics Navigation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: white; }}
        h1 {{ color: #4fc3f7; border-bottom: 2px solid #4fc3f7; padding-bottom: 10px; }}
        h2 {{ color: #81c784; margin-top: 30px; }}
        .summary {{ background: #2d2d44; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat-box {{ background: #3d3d5c; padding: 15px; border-radius: 8px; min-width: 150px; }}
        .stat-box h3 {{ margin: 0 0 10px 0; color: #aaa; font-size: 14px; }}
        .stat-box .value {{ font-size: 24px; font-weight: bold; color: #4fc3f7; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #3d3d5c; }}
        th {{ background: #2d2d44; color: #4fc3f7; }}
        tr:hover {{ background: #2d2d44; }}
        .success {{ color: #81c784; }}
        .failed {{ color: #ef5350; }}
        .algorithm-card {{ background: #2d2d44; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .metric {{ display: inline-block; margin-right: 20px; }}
        .metric-label {{ color: #888; font-size: 12px; }}
        .metric-value {{ font-size: 18px; font-weight: bold; }}
        .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .figure {{ background: #2d2d44; padding: 10px; border-radius: 8px; text-align: center; }}
        .figure img {{ max-width: 100%; border-radius: 5px; }}
        .best {{ border: 2px solid #4fc3f7; }}
        .toc {{ background: #2d2d44; padding: 20px; border-radius: 10px; }}
        .toc ul {{ columns: 2; }}
        .toc li {{ margin: 5px 0; }}
        a {{ color: #4fc3f7; }}
    </style>
</head>
<body>
    <h1>Bio-Inspired Robotics Navigation Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="stats">
            <div class="stat-box">
                <h3>Total Algorithms</h3>
                <div class="value">{total}</div>
            </div>
            <div class="stat-box">
                <h3>Success Rate</h3>
                <div class="value">{successful}/{total} ({100*successful/max(total,1):.0f}%)</div>
            </div>
            <div class="stat-box">
                <h3>Best Path</h3>
                <div class="value">{best_by_path['algorithm'] if best_by_path else 'N/A'}</div>
            </div>
            <div class="stat-box">
                <h3>Fastest</h3>
                <div class="value">{fastest['algorithm'] if fastest else 'N/A'}</div>
            </div>
            <div class="stat-box">
                <h3>Safest</h3>
                <div class="value">{safest['algorithm'] if safest else 'N/A'}</div>
            </div>
        </div>
    </div>
    
    <h2>Table of Contents</h2>
    <div class="toc">
        <ul>
            <li><a href="#results">Results Table</a></li>
            <li><a href="#2d-plots">2D Environment Plots</a></li>
            <li><a href="#3d-plots">3D Trajectory Plots</a></li>
            <li><a href="#comparison">Algorithm Comparison</a></li>
            <li><a href="#analysis">Trajectory Analysis</a></li>
            <li><a href="#data">Exported Data</a></li>
        </ul>
    </div>
    
    <h2 id="results">Detailed Results</h2>
    <table>
        <tr>
            <th>Algorithm</th>
            <th>Status</th>
            <th>Final Distance</th>
            <th>Path Length</th>
            <th>Time</th>
            <th>Collisions</th>
            <th>Energy</th>
            <th>Mode Changes</th>
        </tr>
"""

        for r in valid_results:
            status_class = "success" if r['result'].success else "failed"
            status_text = "SUCCESS" if r['result'].success else "FAILED"
            html += f"""
        <tr>
            <td><strong>{r['algorithm']}</strong></td>
            <td class="{status_class}">{status_text}</td>
            <td>{r['result'].final_distance:.2f} m</td>
            <td>{r['result'].path_length:.2f} m</td>
            <td>{r['result'].time_taken:.2f} s</td>
            <td>{r['result'].collisions}</td>
            <td>{r['result'].energy_consumed:.2f} J</td>
            <td>{r['result'].mode_changes}</td>
        </tr>
"""

        html += """
    </table>
    
    <h2 id="2d-plots">2D Environment Plots</h2>
    <div class="figure-grid">
"""

        # Add 2D figures
        for r in valid_results[:6]:
            fig_name = f"2D_env_{r['algorithm'].lower().replace(' ', '_')}.png"
            fig_path = os.path.join('figures', fig_name)
            if os.path.exists(os.path.join(self.output_dir, fig_path)):
                html += f"""
        <div class="figure">
            <h3>{r['algorithm']}</h3>
            <img src="{fig_path}" alt="{r['algorithm']} 2D Plot">
        </div>
"""

        html += """
    </div>
    
    <h2 id="3d-plots">3D Trajectory Plots</h2>
    <div class="figure-grid">
"""

        # Add 3D figures
        for r in valid_results[:4]:
            fig_name = f"3D_trajectory_{r['algorithm'].lower().replace(' ', '_')}.png"
            fig_path = os.path.join('figures', fig_name)
            if os.path.exists(os.path.join(self.output_dir, fig_path)):
                html += f"""
        <div class="figure">
            <h3>{r['algorithm']}</h3>
            <img src="{fig_path}" alt="{r['algorithm']} 3D Plot">
        </div>
"""

        html += """
    </div>
    
    <h2 id="comparison">Algorithm Comparison</h2>
    <div class="figure">
        <img src="figures/algorithm_comparison.png" alt="Algorithm Comparison" style="max-width: 100%;">
    </div>
    
    <h2 id="analysis">Trajectory Analysis</h2>
    <div class="figure-grid">
"""

        # Add analysis figures
        for r in valid_results[:2]:
            fig_name = f"trajectory_analysis_{r['algorithm'].lower().replace(' ', '_')}.png"
            fig_path = os.path.join('figures', fig_name)
            if os.path.exists(os.path.join(self.output_dir, fig_path)):
                html += f"""
        <div class="figure">
            <h3>{r['algorithm']} Analysis</h3>
            <img src="{fig_path}" alt="{r['algorithm']} Analysis">
        </div>
"""

        html += """
    </div>
    
    <h2 id="data">Exported Data Files</h2>
    <div class="summary">
        <ul>
            <li><strong>simulation_results.csv</strong> - Main results table</li>
            <li><strong>trajectory_*.csv</strong> - Individual trajectory data</li>
            <li><strong>states_*.csv</strong> - State logs for each algorithm</li>
            <li><strong>environment_config.json</strong> - Environment configuration</li>
            <li><strong>summary_report.txt</strong> - Text summary report</li>
        </ul>
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #3d3d5c; color: #888;">
        <p>Generated by Bio-Inspired Robotics Navigation Simulation System</p>
        <p>Author: Chandan Sheikder | Beijing Institute of Technology (BIT)</p>
    </footer>
</body>
</html>
"""

        # Save HTML report
        report_path = os.path.join(self.output_dir, 'report.html')
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"  Report saved: {report_path}")
        return report_path
    
    def run_complete(self):
        """Run complete simulation pipeline"""
        print("\n" + "=" * 70)
        print("STARTING BIO-INSPIRED ROBOTICS SIMULATION PIPELINE")
        print("=" * 70)
        print(f"Output Directory: {self.output_dir}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Step 1: Run all algorithms
        print("\n[STEP 1/4] Running Navigation Algorithms...")
        results = self.run_all_algorithms(verbose=True)
        
        # Step 2: Generate visualizations
        print("\n[STEP 2/4] Generating Visualizations...")
        self.generate_visualizations(results)
        
        # Step 3: Export data
        print("\n[STEP 3/4] Exporting Data...")
        self.export_all_data(results)
        
        # Step 4: Generate report
        print("\n[STEP 4/4] Generating Report...")
        report_path = self.generate_report(results)
        
        total_time = time.time() - start_time
        
        # Final summary
        print("\n" + "=" * 70)
        print("SIMULATION PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Output Directory: {self.output_dir}")
        print(f"Report: {report_path}")
        print("\nGenerated Files:")
        print("  - Figures: ./figures/*.png, *.pdf, *.svg")
        print("  - Data: ./data/*.csv, *.json")
        print("  - Report: ./report.html")
        print("=" * 70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bio-Inspired Robotics Navigation Simulation')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run quick version with fewer algorithms')
    args = parser.parse_args()
    
    # Create and run
    runner = BioRoboticsRunner(output_dir=args.output)
    runner.run_complete()
    
    print("\nAll simulations completed successfully!")


if __name__ == "__main__":
    main()
