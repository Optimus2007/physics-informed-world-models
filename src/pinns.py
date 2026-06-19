import torch
import torch.nn as nn

class OrbitalPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(OrbitalPINN, self).__init__()
        
        # Note the input is 1-dimensional (Time 't')
        # The output is 4-dimensional (x, y, vx, vy)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)
        )
        
        # Gravitational constant (normalized for our simulation)
        self.GM = 1.0

    def forward(self, t):
        """
        Takes time 't' and predicts the state [x, y, vx, vy].
        """
        return self.net(t)

    def physics_loss(self, t):
        """
        Calculates how badly the network's predictions violate Newton's laws.
        """
        t.requires_grad_(True)
        state_pred = self.forward(t)
        
        # 1. Extract individual components FIRST
        x_pred = state_pred[:, 0:1]
        y_pred = state_pred[:, 1:2]
        vx_pred = state_pred[:, 2:3]
        vy_pred = state_pred[:, 3:4]
        
        # 2. Compute gradients INDIVIDUALLY to prevent PyTorch from squashing them
        dx_dt_net = torch.autograd.grad(x_pred, t, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
        dy_dt_net = torch.autograd.grad(y_pred, t, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        dvx_dt_net = torch.autograd.grad(vx_pred, t, grad_outputs=torch.ones_like(vx_pred), create_graph=True)[0]
        dvy_dt_net = torch.autograd.grad(vy_pred, t, grad_outputs=torch.ones_like(vy_pred), create_graph=True)[0]
        
        # 3. THE GROUND TRUTH PHYSICS
        epsilon = 1e-6 # Our singularity protection
        r_cubed = (x_pred**2 + y_pred**2 + epsilon)**1.5
        
        dx_dt_physics = vx_pred
        dy_dt_physics = vy_pred
        dvx_dt_physics = -self.GM * x_pred / r_cubed
        dvy_dt_physics = -self.GM * y_pred / r_cubed
        
        # 4. THE RESIDUALS
        loss_x  = torch.mean((dx_dt_net - dx_dt_physics)**2)
        loss_y  = torch.mean((dy_dt_net - dy_dt_physics)**2)
        loss_vx = torch.mean((dvx_dt_net - dvx_dt_physics)**2)
        loss_vy = torch.mean((dvy_dt_net - dvy_dt_physics)**2)
        
        return loss_x + loss_y + loss_vx + loss_vy