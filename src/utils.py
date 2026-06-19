import matplotlib.pyplot as plt
import os

def plot_trajectory_comparison(true_traj, pred_traj, save_path, title="Trajectory Comparison"):
    """
    Plots the ground-truth physical trajectory against the AI's prediction.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot true trajectory
    plt.plot(true_traj[:, 0], true_traj[:, 1], label="True Orbit (RK4)", color="blue", alpha=0.6)
    
    # Plot AI predicted trajectory
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], label="AI Prediction", color="red", linestyle='--')
    
    # Plot the central star
    plt.scatter([0], [0], color="orange", s=200, label="Star")
    
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis("equal")
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved successfully to {save_path}")
    plt.close() # Close the figure to free up memory