import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import plot_trajectory_comparison
import os

# Define the Standard Neural Network Architecture
class NaiveMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super(NaiveMLP, self).__init__()
        # A simple feedforward network with 3 hidden layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def load_and_prep_data(filepath):
    """Loads trajectory data and prepares (X_t, X_{t+1}) training pairs."""
    data = np.load(filepath)
    
    # Inputs (X) are all steps except the last one
    # Targets (Y) are all steps except the first one (shifted by 1 time step)
    X = data[:-1, :]
    Y = data[1:, :]
    
    # Convert pure numbers into PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    return X_tensor, Y_tensor, data

if __name__ == "__main__":
    print("Loading data...")
    # Adjust path if running from root vs src
    data_path = 'data/trajectories/orbit_data.npy'
    if not os.path.exists(data_path):
        data_path = '../data/trajectories/orbit_data.npy'
        
    X_train, Y_train, true_trajectory = load_and_prep_data(data_path)
    
    # Initialize Model, Loss function, and Optimizer
    model = NaiveMLP()
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 1500
    print("Training standard Neural Network...")
    
    # Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()       
        predictions = model(X_train)
        loss = criterion(predictions, Y_train) 
        loss.backward()             
        optimizer.step()            
        
        if epoch % 300 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f}")

    print("Training complete. Beginning Autoregressive Rollout...")
    
    # The Autoregressive Rollout (Testing Phase)
    # We give the AI ONLY the very first state. It must predict all 2000 future steps itself.
    num_steps = len(true_trajectory)
    predicted_trajectory = torch.zeros((num_steps, 4))
    predicted_trajectory[0] = torch.tensor(true_trajectory[0], dtype=torch.float32)
    
    current_state = predicted_trajectory[0].unsqueeze(0) # Shape: [1, 4]
    
    model.eval() # Put model in evaluation mode
    with torch.no_grad(): 
        for i in range(1, num_steps):
            next_state = model(current_state)
            predicted_trajectory[i] = next_state[0]
            current_state = next_state # The AI's prediction becomes its next input
            
    # Convert back to numpy for plotting
    pred_data = predicted_trajectory.numpy()
    
    os.makedirs('data/trajectories', exist_ok=True)
    np.save('data/trajectories/naive_mlp_prediction.npy', pred_data)
    
    plot_trajectory_comparison(
        true_traj=true_trajectory,
        pred_traj=pred_data,
        save_path='data/trajectories/naive_mlp_failure.png',
        title="Failure of Pure ML: Accumulation of Rollout Error"
    )