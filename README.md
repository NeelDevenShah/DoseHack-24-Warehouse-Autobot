# Warehouse Autobot

## Team: MediHacker

DoseHack'24

Achievement: First Prize (25,000 INR)

### Team Members

- Neel Shah (neeldevenshah@gmail.com) (Mentor)
- Sandeep Swain (sandeepswain2004@gmail.com)
- Divyarajsinh Vala (divyarajsinhvala06@gmail.com)
- Apurv Chudasama (apurvchudasama.edu@gmail.com)

# Problem Description

Dosepacker Inc. runs a large automated warehouse, where robots (called autobots) move products from one place to another. However, the system recently started facing issues like traffic jams and collisions, slowing down operations. Our goal was to create a smart system to control these autobots, helping them move safely and efficiently within the warehouse without bumping into each other or obstacles.

## The Mission

Guide Autobots across a matrix grid from their starting point (A) to their destination (B), avoiding obstacles and each other. Autobots can only understand simple movement commands: Forward, Reverse, Left, Right, and Wait. Ensure all Autobots reach their destination without any collisions or blocking each other.

## Key Challenges

- Collision Avoidance: Ensure smooth navigation, avoiding deadlocks and collisions as multiple autobots move simultaneously on the grid.
- Dynamic Movement: Autobots execute one command at a time. They may pause using the Wait() command. Multiple Wait() commands are needed for longer pauses.
- Matrix Layout: The grid layout varies in size and contains obstacles (blocked cells). More complex grids are introduced at higher difficulty levels.
- Real-Time Coordination: Autobots must coordinate their movements dynamically in real time to avoid blocking each other.

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Pygame](https://img.shields.io/badge/Pygame-v2.0.1-lightgreen)
![Numpy](https://img.shields.io/badge/Numpy-v1.19.5-orange)

## Features

- **Multi-agent navigation:** Multiple Autobots (agents) navigate the grid avoiding obstacles.
- **Reinforcement learning:** Autobots learn optimal paths through Q-learning.
- **Collision and deadlock avoidance:** Autobots avoid each other and deadlocks.
- **Backtracking capability:** Autobots can reverse their steps when needed.
- **Real-time visualization:** The simulation uses `pygame` to display Autobots' movements.
- **Statistics:** Tracks each Autobot's performance, such as total steps, waiting time, and movement patterns.

## Project Structure

```
Main_dir/
├── main.py             # Core logic for Autobot movement and simulation
├── README.md           # Project documentation
├── app.py              # Test cases and additional examples
└── requirements.txt    # Python dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DoseHACK-24/Medihackers.git
cd autobot-simulation
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the simulation:

```bash
python main.py
```

## How It Works

### Autobot Class

- `get_action()`: Selects an action (move or wait) based on the Q-table or exploration (epsilon-greedy).
- `move()`: Updates the Autobot's position.
- `learn()`: Updates the Q-table after an action.
- `backtrack()`: Moves the Autobot back to avoid deadlocks.

### Simulation Class

- `run_episode()`: Runs a single episode where Autobots navigate the grid and learn from actions.
- `train()`: Trains Autobots over multiple episodes using Q-learning.
- `run_inference()`: Runs the simulation to test Autobots after training.
- `display_grid()`: Uses pygame to render the current grid state.

## Demo Video

<a href="https://ibb.co/3s1YcXP"><img src="https://i.ibb.co/fSMGHRT/example-ezgif-com-video-to-gif-converter.gif" alt="example-ezgif-com-video-to-gif-converter" border="0"></a>

## Example Simulation

**Grid Size:** 5x5  
**Obstacles:** `[(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)]`  
**Autobots:**

- **Autobot A:** Start: `(0, 0)`, Goal: `(4, 4)`
- **Autobot B:** Start: `(4, 0)`, Goal: `(0, 4)`

```python
grid_size = (5, 5)
obstacles = [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)]
autobots = [
    Autobot("A", (0, 0), (4, 4), grid_size, obstacles),
    Autobot("B", (4, 0), (0, 4), grid_size, obstacles),
]

simulation = Simulation(grid_size, autobots, obstacles)
simulation.train(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)
simulation.run_inference()
```

### Terminal Output Example

```
==================================================
Step 1:
--------------------------------------------------
A . . . .
. # # # .
. . . . .
B # # # .
. . . . .
--------------------------------------------------
All Autobots have reached their destinations!
==================================================
Simulation Statistics:
--------------------------------------------------
Autobot A:
  Total Time: 10
  Total Steps: 10
  Waiting Time: 2
  Movements:
    Forward: 4
    Backward: 1
    Right: 2
    Wait: 3
--------------------------------------------------
Autobot B:
  Total Time: 12
  Total Steps: 12
  Waiting Time: 1
  Movements:
    Forward: 6
    Backward: 0
    Right: 4
    Wait: 2
--------------------------------------------------
```

## Algorithm Flow

1. **Initialization:** Autobots are placed in the grid with a start and end position, avoiding obstacles.
2. **Training:** Autobots use Q-learning to learn optimal paths over multiple episodes.
3. **Inference:** After training, Autobots use the learned Q-table to navigate the grid.
4. **Backtracking:** Autobots backtrack if they encounter a deadlock.

## Requirements

- Python 3.6+
- `pygame` for visualization
- `numpy` for matrix operations

## Dependencies

The required dependencies are listed in `requirements.txt`:

```txt
pygame==2.0.1
numpy==1.19.5
```

## Acknowledgment

We would like to express our gratitude to Meditab, Dosepacker Inc., and Charusat University for organizing the 24-hour DoseHack'24 hackathon.

Special thanks to Dosepacker Inc., which operates one of the largest automated warehouses, for inspiring this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to give any suggestions or recommendations, regarding our approach!
