import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics import orbital_derivatives, rk4_step

def simulate_orbit(initial_state, dt, num_steps):
    """
    Runs the simulation loop and stores the trajectory.
    """
    # Create an empty matrix to hold our data: [num_steps, 4 variables]
    trajectory = np.zeros((num_steps, 4))
    trajectory[0] = initial_state
    
    current_state = initial_state
    t = 0.0
    
    # The integration loop
    for i in range(1, num_steps):
        current_state = rk4_step(current_state, t, dt, orbital_derivatives, GM=1.0)
        trajectory[i] = current_state
        t += dt
        
    return trajectory

if __name__ == "__main__":
   
    # Start at x=1.0, y=0.0. Give it an initial upward velocity (vy)
    # vy = 1.0 creates a perfect circle when GM=1 and x=1
    # vy > 1.0 creates an ellipse. vy >= 1.414 (sqrt(2)) escapes the system.
    initial_state = np.array([1.0, 0.0, 0.0, 1.2]) 
    
    dt = 0.01        # Time step resolution
    num_steps = 10000 # Total time steps to simulate
    
    
    print("Simulating trajectory...")
    trajectory = simulate_orbit(initial_state, dt, num_steps)
    
    
    os.makedirs('data/trajectories', exist_ok=True)
    np.save('data/trajectories/orbit_data.npy', trajectory)
    print("Data saved to data/trajectories/orbit_data.npy")
    
   
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    
    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, label="Planet Trajectory", color="blue")
    plt.scatter([0], [0], color="orange", s=200, label="Star (Origin)")
    plt.title("Ground-Truth Orbital Dynamics")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis("equal") 
    
    
    plt.savefig('data/trajectories/orbit_plot.png')
    plt.show()