# sar-evaluator
This project implements a multi-victim SAR simulator comparing three strategies:

1. `ground`: ground robot only
2. `constant`: ground robot with regularly deployed drone assistance
3. `smart`: ground robot with score-based drone deployment

### Usage
1. install dependencies: pip install -r requirements.txt
2. Configure settings in config.py (set EXPERIMENT_MODE variable)
3. python main.py

### Dependencies
- Python 3.13
- numpy
- matplotlib

### Features
- Multiple victims per environment
- Occupancy-grid mapping
- Frontier-based exploration
- A* path planning
- Ground robot victim rescue
- Drone-based aerial sensing
- Constant and score-based drone deployment
- Mission metrics for victim detection, rescue, exploration, distance, energy, and drone deployments

[Demo Link](https://youtu.be/44GyYag86Cg) : https://youtu.be/44GyYag86Cg
