import matplotlib.pyplot as plt
import os

def plot_trajectory_comparison(true_traj, pred_traj, save_path, title="Trajectory Comparison"):
    """
    Plots the ground-truth physical trajectory against the AI's prediction.
    """
    plt.figure(figsize=(8, 8))
    
    plt.plot(true_traj[:, 0], true_traj[:, 1], label="True Orbit (RK4)", color="blue", alpha=0.6)
    
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], label="AI Prediction", color="red", linestyle='--')
    
    plt.scatter([0], [0], color="orange", s=200, label="Star")
    
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis("equal")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved successfully to {save_path}")
    plt.close() 