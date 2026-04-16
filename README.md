# 🚗 Autonomous Braking Control System using Fuzzy Logic

This project implements a **fuzzy inference system (FIS)** for an autonomous vehicle braking scenario. The system evaluates relative speed and distance to a lead vehicle and determines an appropriate braking force. The implementation includes both a physics-based simulation and a real-time visualization using Pygame.

---

## 📌 Overview

- **Inputs:**
  - Relative Speed (ft/s)
  - Distance to Lead Vehicle (ft)

- **Output:**
  - Brake Force (lbf)

- **Technologies Used:**
  - Python
  - NumPy
  - scikit-fuzzy
  - Matplotlib
  - Pygame

---

## 🎯 Key Features

- Fuzzy logic-based decision making
- Triangular membership function design
- Rule-based inference system
- Centroid defuzzification
- Physics-based vehicle simulation
- Interactive Pygame visualization with multiple scenarios

---

## 📊 Membership Functions

The system uses triangular membership functions for all variables:

### Relative Speed
- Negative
- Zero
- Positive

### Distance
- Close
- Medium
- Far

### Brake Force
- Light
- Moderate
- Hard

---

## 🧠 Fuzzy Rule Base

The controller uses the following rules:

1. IF Distance is Close AND Relative Speed is Positive → Hard Brake  
2. IF Distance is Medium AND Relative Speed is Positive → Moderate Brake  
3. IF Distance is Far AND Relative Speed is Positive → Light Brake  
4. IF Distance is Close AND Relative Speed is Zero → Moderate Brake  
5. IF Distance is Far AND Relative Speed is Negative → Light Brake  

These rules balance safety and smooth braking behavior.

---

## ⚙️ System Dynamics

The vehicle dynamics are modeled as:

m * p̈(t) = f_b(t)

- Mass (m) = 100 slug  
- Lead vehicle moves at constant velocity  
- Brake force is computed using fuzzy logic  

---

## 📈 Simulation Scenarios

Three scenarios are simulated:

| Scenario | Initial Speed (ft/s) | Lead Speed (ft/s) |
|----------|----------------------|-------------------|
| 1        | 45                   | 50                |
| 2        | 75                   | 50                |
| 3        | 100                  | 50                |

Outputs:
- Position vs Time
- Velocity vs Time

---

## 🎮 Pygame Visualization

The simulation includes an interactive UI:

- Scenario selection menu
- Real-time vehicle movement
- Live speed and distance display
- Graph visualization (velocity & distance)
- Replay and navigation controls

### Demo

![Demo](assets/demo.gif)

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/autonomous-braking-control-system.git
cd autonomous-braking-control-system
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the simulation
```bash
python src/fuzzy_braking_system.py
```

---

### 🌍 Real-World Considerations

While this model uses only relative speed and distance, real autonomous systems consider:

- Road conditions (wet, dry, icy)
- Sensor noise and uncertainty
- Vehicle mass and load
- Tire-road friction
- Driver comfort constraints
- Reaction time and system delays

---

### 🚀 Future Improvements

- Add more input variables (acceleration, road friction)
- Adaptive or learning-based fuzzy rules
- Integration with reinforcement learning
- 2D/3D simulation environment
- Sensor noise modeling

---

### ⭐ If you found this useful

Feel free to star the repository!
