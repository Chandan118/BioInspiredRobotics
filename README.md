# Bio-Inspired Robotics Navigation Simulation

A comprehensive Python implementation of bio-inspired navigation algorithms for tethered robots in dynamic environments. This project implements multiple navigation strategies including Bug algorithms, Fuzzy Logic, Genetic Algorithms, Particle Swarm Optimization, Ant Colony Optimization, and Deep Reinforcement Learning.

## Author

**Chandan Sheikder**  
Beijing Institute of Technology (BIT)  
Email: chandan@bit.edu.cn  
Phone: +8618222390506

## Project Structure

```
BioInspiredRobotics/
├── simulations/
│   └── comprehensive_navigation.py    # Core simulation engine
├── algorithms/
│   └── enhanced_algorithms.py          # Advanced algorithms
├── visualization/
│   └── visualizer.py                  # Visualization module
├── main_runner.py                     # Main execution script
├── requirements.txt                   # Dependencies
└── README.md                         # This file
```

## Algorithms Implemented

### Classical Navigation Algorithms
1. **Bug0 Algorithm** - Simple wall-following navigation
2. **Bug1 Algorithm** - Complete perimeter following
3. **Tangent Bug Algorithm** - Uses tangent points for efficient navigation
4. **Dynamic Window Approach (DWA)** - Velocity space sampling
5. **Potential Field Navigation** - Attractive/repulsive force fields

### Soft Computing Techniques
6. **Fuzzy Logic Navigator** - Mamdani-style fuzzy inference
7. **Neural Network Navigator** - ML-based action prediction
8. **ANFIS Navigator** - Adaptive Neuro-Fuzzy Inference System
9. **Genetic Algorithm Optimizer** - Parameter optimization
10. **Particle Swarm Navigator** - PSO-based path planning
11. **Ant Colony Optimizer** - ACO-based path finding

### Advanced Learning
12. **Q-Learning Agent** - Tabular reinforcement learning
13. **Deep Q-Network (DQN)** - Deep RL for navigation
14. **Hybrid Navigation System** - Multi-algorithm fusion

## Installation

```bash
# Clone or download the project
cd BioInspiredRobotics

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run all algorithms with full visualization
python main_runner.py

# Run with custom output directory
python main_runner.py --output ./my_results
```

### Running Individual Components

```python
from simulations.comprehensive_navigation import (
    Environment, Bug0Algorithm, FuzzyNavigator, run_simulation
)
from visualization.visualizer import Visualizer

# Create environment
env = Environment()

# Create algorithm
algorithm = FuzzyNavigator(env)

# Run simulation
result = run_simulation(env, algorithm)

# Visualize
vis = Visualizer(output_dir='./results')
fig = vis.plot_2d_environment(env, result.path)
vis.save_figure(fig, 'navigation_result')
```

## Features

### Simulation Engine
- 2D/3D environment with static and dynamic obstacles
- Multiple robot types with different kinematics
- Real-time collision detection
- Tether cable visualization
- Dynamic obstacle (human) simulation

### Algorithms
- Complete implementation of Bug family algorithms
- Fuzzy logic with customizable membership functions
- Neural network training and inference
- Swarm intelligence optimization
- Deep reinforcement learning with experience replay
- Hybrid multi-algorithm fusion

### Visualization
- 2D top-down environment plots
- 3D trajectory visualization
- Algorithm comparison charts
- Trajectory analysis (velocity, distance, curvature)
- Fuzzy membership function visualization
- Learning curves for RL algorithms
- Animation export (GIF)

### Data Export
- CSV export of simulation results
- Trajectory data with timestamps
- State logs for analysis
- JSON configuration export
- HTML summary report generation

## Results

Each algorithm is evaluated on:
- **Success Rate** - Goal reached successfully
- **Path Length** - Total distance traveled
- **Execution Time** - Computation efficiency
- **Collisions** - Number of obstacle contacts
- **Energy Consumption** - Estimated energy usage
- **Mode Changes** - Algorithm complexity indicator

## Environment Configuration

The default environment includes:
- Start Position: (5, 45)
- Goal Position: (45, 5)
- 4 Building obstacles
- 2 Planter obstacles
- 2 Streetlight obstacles
- 1 Dynamic human obstacle

## Citation

If you use this code in your research, please cite:

```
Chandan Sheikder, Weimin Zhang, Xiaopeng Chen, et al. 
Soft Computing Techniques Applied to Adaptive Hybrid Navigation Methods 
for Tethered Robots in Dynamic Environments. 
Authorea. January 24, 2025.
DOI: 10.1002/rob.70222
```

## License

MIT License

## Acknowledgments

Beijing Institute of Technology (BIT)  
Navigation and Robotics Laboratory

<!-- MIT License -->