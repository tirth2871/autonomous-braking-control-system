"""
Fuzzy Braking System for Autonomous Vehicle
By: [Your Name]

Parts Included:
(a) Membership Functions and Plots
(b) Fuzzy Rules
(c) Real-World Discussion (Printed)
(d) Physics-Based Simulation
(e) Pygame Visualization

Requirements:
pip install pygame numpy matplotlib scikit-fuzzy
"""

# ------------------------------
# PART A: Membership Definitions
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Variables
rel_speed = np.arange(-150, 151, 1)
distance = np.arange(0, 201, 1)
brake_force = np.arange(0, 2501, 1)

# Membership functions
rel_speed_neg = fuzz.trimf(rel_speed, [-150, -50, 0])
rel_speed_zero = fuzz.trimf(rel_speed, [-50, 0, 50])
rel_speed_pos = fuzz.trimf(rel_speed, [0, 50, 150])

distance_close = fuzz.trimf(distance, [0, 0, 50])
distance_medium = fuzz.trimf(distance, [50, 150, 250])
distance_far = fuzz.trimf(distance, [150, 300, 300])

brake_light = fuzz.trimf(brake_force, [0, 0, 1000])
brake_moderate = fuzz.trimf(brake_force, [500, 1250, 2000])
brake_hard = fuzz.trimf(brake_force, [1500, 2500, 2500])

# Plotting
def plot_mfs(x, mfs, labels, title):
    plt.figure(figsize=(8, 4))
    for mf, label in zip(mfs, labels):
        plt.plot(x, mf, label=label)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot input MFs
plot_mfs(rel_speed, [rel_speed_neg, rel_speed_zero, rel_speed_pos], ['Negative', 'Zero', 'Positive'], 'Relative Speed')
plot_mfs(distance, [distance_close, distance_medium, distance_far], ['Close', 'Medium', 'Far'], 'Distance')
plot_mfs(brake_force, [brake_light, brake_moderate, brake_hard], ['Light', 'Moderate', 'Hard'], 'Brake Force')

# Plot all
def plot_memberships():
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(rel_speed, rel_speed_neg, label='Negative')
    axs[0].plot(rel_speed, rel_speed_zero, label='Zero')
    axs[0].plot(rel_speed, rel_speed_pos, label='Positive')
    axs[0].set_title('Relative Speed (ft/s)')
    axs[0].legend()

    axs[1].plot(distance, distance_close, label='Close')
    axs[1].plot(distance, distance_medium, label='Medium')
    axs[1].plot(distance, distance_far, label='Far')
    axs[1].set_title('Distance (ft)')
    axs[1].legend()

    axs[2].plot(brake_force, brake_light, label='Light')
    axs[2].plot(brake_force, brake_moderate, label='Moderate')
    axs[2].plot(brake_force, brake_hard, label='Hard')
    axs[2].set_title('Brake Force (lbf)')
    axs[2].legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# PART B: Fuzzy Rule Logic
# ------------------------------
def compute_brake_force(rel_speed_val, distance_val):
    # Fuzzify inputs
    rs_neg = fuzz.interp_membership(rel_speed, rel_speed_neg, rel_speed_val)
    rs_zero = fuzz.interp_membership(rel_speed, rel_speed_zero, rel_speed_val)
    rs_pos = fuzz.interp_membership(rel_speed, rel_speed_pos, rel_speed_val)

    d_close = fuzz.interp_membership(distance, distance_close, distance_val)
    d_med = fuzz.interp_membership(distance, distance_medium, distance_val)
    d_far = fuzz.interp_membership(distance, distance_far, distance_val)

    print(f"RS: neg={rs_neg}, zero={rs_zero}, pos={rs_pos}")
    print(f"DIST: close={d_close}, med={d_med}, far={d_far}")

    # IF-THEN rule activations
    rule1 = np.fmin(d_close, rs_pos)       # Close & Approaching → Hard brake
    rule2 = np.fmin(d_med, rs_pos)         # Medium & Approaching → Moderate
    rule3 = np.fmin(d_far, rs_pos)         # Far & Approaching → Light
    rule4 = np.fmin(d_close, rs_zero)      # Close & Same speed → Moderate
    rule5 = np.fmin(d_far, rs_neg)         # Far & Receding → Light

    print(f"Rule activations: {rule1}, {rule2}, {rule3}, {rule4}, {rule5}")

    # Rule output memberships
    brake_activation_hard = np.fmin(rule1, brake_hard)
    brake_activation_moderate = np.fmax(np.fmin(rule2, brake_moderate), np.fmin(rule4, brake_moderate))
    brake_activation_light = np.fmax(np.fmin(rule3, brake_light), np.fmin(rule5, brake_light))

    # Aggregate all output membership functions
    aggregated = np.fmax(brake_activation_hard,
                  np.fmax(brake_activation_moderate,
                          brake_activation_light))

    # Ensure we avoid empty aggregation
    if np.all(aggregated == 0):
        print(f"[WARNING] Empty aggregated MF for rel_speed={rel_speed_val}, distance={distance_val}")
        return 0.0

    # Defuzzify to get crisp output
    return fuzz.defuzz(brake_force, aggregated, 'centroid')


# ------------------------------
# PART D + E: Pygame Visualization
# ------------------------------
def plot_graphs(time, p, pB, v, vB):
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the positions (p(t) and pB(t)) in the first subplot
    axs[0].plot(time, p, label="Ego Car Position", color='blue')
    axs[0].plot(time, pB, label="Lead Vehicle Position", color='red')
    axs[0].set_title("Position vs. Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position (ft)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot the velocities (v(t) and vB(t)) in the second subplot
    axs[1].plot(time, v, label="Ego Car Velocity", color='blue')
    axs[1].plot(time, vB, label="Lead Vehicle Velocity", color='red')
    axs[1].set_title("Velocity vs. Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (ft/s)")
    axs[1].legend()
    axs[1].grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()

from fuzzy_braking_game import main_menu  # This is the full Pygame code from earlier

# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    print("Fuzzy Braking System Initialized")
    print("Running Parts (a) to (e)...")

    print("\nShowing Membership Function Plots (Part A)...")
    plot_memberships()

    print("\nFuzzy Rules Implemented (Part B)")

    print("Launching Pygame Simulation (Parts D & E)...")
    main_menu()