import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import our custom modules
from pinns import OrbitalPINN
from utils import plot_trajectory_comparison

def prepare_pinn_data(filepath, dt=0.01):
    """
    Loads trajectory data and creates the corresponding time tensor.
    Unlike the naive MLP, the PINN maps Time -> State.
    """
    true_states = np.load(filepath)
    num_steps = len(true_states)
    
    # Reconstruct the time vector (t=0, t=0.01, t=0.02, ...)
    t_array = np.arange(0, num_steps * dt, dt).reshape(-1, 1)
    
    # Convert to PyTorch tensors and ensure they require gradients if needed
    t_tensor = torch.tensor(t_array, dtype=torch.float32)
    x_tensor = torch.tensor(true_states, dtype=torch.float32)
    
    return t_tensor, x_tensor

if __name__ == "__main__":
    print("Loading orbital data for PINN...")
    # Using the 10,000 step data you just generated
    data_path = 'data/trajectories/orbit_data.npy'
    if not os.path.exists(data_path):
        data_path = '../data/trajectories/orbit_data.npy'
        
    t_data, x_data = prepare_pinn_data(data_path)
    
    
    train_cutoff = int(len(t_data) * 0.2)
    t_train_data = t_data[:train_cutoff]
    x_train_data = x_data[:train_cutoff]
    
    print(f"Total steps: {len(t_data)}. Training on first {train_cutoff} steps.")

    # Initialize the PINN and Optimizer
    model = OrbitalPINN(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_data = nn.MSELoss()
    
    epochs = 3000
    lambda_physics = 0.1 # Weighting factor for the physics loss
    max_time = t_data[-1].item() # The maximum time in our simulation

    print("Beginning Physics-Informed Training...")
    
    # The Multi-Objective Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        
        pred_x_data = model(t_train_data)
        loss_data = criterion_data(pred_x_data, x_train_data)
        
        
        t_physics = torch.rand((2000, 1), requires_grad=True) * max_time
        loss_physics = model.physics_loss(t_physics)
        
        total_loss = loss_data + (lambda_physics * loss_physics)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs} | Total: {total_loss.item():.5f} | "
                  f"Data: {loss_data.item():.5f} | Phys: {loss_physics.item():.5f}")

    print("Training complete. Generating continuous trajectory...")
    
    model.eval()
    with torch.no_grad():
        pred_trajectory = model(t_data).numpy()
    
    true_trajectory = x_data.numpy()
    
    os.makedirs('data/trajectories', exist_ok=True)
    save_path = 'data/trajectories/pinn_prediction.png'
    
    plot_trajectory_comparison(
        true_traj=true_trajectory,
        pred_traj=pred_trajectory,
        save_path=save_path,
        title="PINN Extrapolation (Trained on only 20% of data)"
    )