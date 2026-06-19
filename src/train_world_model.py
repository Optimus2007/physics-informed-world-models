import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from models import PhysicsInformedWorldModel
from utils import plot_trajectory_comparison

def load_trajectory_data(filepath, dt=0.01):
    data = np.load(filepath)
    num_steps = len(data)
    t_array = np.arange(0, num_steps * dt, dt)
    
    t_tensor = torch.tensor(t_array, dtype=torch.float32)
    x_tensor = torch.tensor(data, dtype=torch.float32)
    return t_tensor, x_tensor

if __name__ == "__main__":
    print("Initializing Physics-Informed World Model...")
    
    # Load the 10,000 step data
    t_data, x_data = load_trajectory_data('data/trajectories/orbit_data.npy')
    
   # Reduce the training horizon to stabilize early learning
    # We will train on the first 400 steps (same as the PINN)
    train_cutoff = 400 
    t_train = t_data[:train_cutoff]
    x_train = x_data[:train_cutoff].unsqueeze(1) 
    
    x0 = x_train[0] 
    
    model = PhysicsInformedWorldModel(state_dim=4, latent_dim=4, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_data = nn.MSELoss()
    
    epochs = 1000 
    lambda_energy = 0.1 
    
    print(f"Training on {train_cutoff} steps using Adaptive Neural ODE integration...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass (Integrate ODE)
        x_pred = model(x0, t_train)
        
        # 2. Data Loss
        loss_data = criterion_data(x_pred, x_train)
        
        # 3. Physics Loss
        energies = model.compute_energy(x_pred)
        loss_energy = torch.var(energies)
        
        # 4. Total Loss & Backprop
        total_loss = loss_data + (lambda_energy * loss_energy)
        total_loss.backward()
        
        # --- THE NEW FIX: Gradient Clipping ---
        # This prevents the "vibrations" by stopping the weights from updating too aggressively
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Total: {total_loss.item():.6f} | Data: {loss_data.item():.6f}")

    print("Training complete. Performing infinite-horizon rollout (10,000 steps)...")
    
    # --- The Ultimate Test ---
    # We now ask the model to predict the ENTIRE 10,000 steps using only x0.
    model.eval()
    with torch.no_grad():
        full_prediction = model(x0, t_data)
        
    pred_data = full_prediction.squeeze(1).numpy()
    true_data = x_data.numpy()
    
    # Visualize
    os.makedirs('data/trajectories', exist_ok=True)
    plot_trajectory_comparison(
        true_traj=true_data,
        pred_traj=pred_data,
        save_path='data/trajectories/world_model_success.png',
        title="Physics-Informed World Model (Infinite Horizon Rollout)"
    )