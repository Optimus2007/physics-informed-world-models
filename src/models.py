import torch
import torch.nn as nn
from torchdiffeq import odeint

class LatentODEFunc(nn.Module):
    """
    This is the Neural ODE core. It learns the continuous differential equation 
    dz/dt = f(z) purely inside the latent space.
    """
    def __init__(self, latent_dim=4, hidden_dim=64):
        super(LatentODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim) # Outputs dz/dt
        )

    def forward(self, t, z):
        # The torchdiffeq solver requires the function signature (t, y)
        return self.net(z)

class PhysicsInformedWorldModel(nn.Module):
    """
    The complete World Model pipeline: Encode -> Integrate ODE -> Decode.
    """
    def __init__(self, state_dim=4, latent_dim=4, hidden_dim=64):
        super(PhysicsInformedWorldModel, self).__init__()
        
        # 1. The Encoder (Compresses Reality)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 2. The Dynamics Model (The Neural ODE)
        self.ode_func = LatentODEFunc(latent_dim, hidden_dim)
        
        # 3. The Decoder (Translates back to Reality)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.GM = 1.0 # Gravitational constant for physics loss

    def forward(self, x0, t_eval):
        """
        x0: The initial state at t=0 [batch, 4]
        t_eval: The time steps we want to evaluate [T]
        """
        # Step 1: Encode initial state into latent space
        z0 = self.encoder(x0)
        
        # Step 2: Use the ODE solver to roll the latent state forward in time
        # CHANGED: 'rk4' to 'dopri5' (Adaptive step size for stiff equations)
        z_t = odeint(self.ode_func, z0, t_eval, method='dopri5')
        
        # Step 3: Decode all latent states back to physical coordinates
        # z_t has shape [T, batch, latent_dim]
        x_pred = self.decoder(z_t)
        
        return x_pred

    def compute_energy(self, x_pred):
        """
        Calculates the physical Hamiltonian (Total Energy) of the predicted trajectory.
        """
        x = x_pred[:, :, 0]
        y = x_pred[:, :, 1]
        vx = x_pred[:, :, 2]
        vy = x_pred[:, :, 3]
        
        r = torch.sqrt(x**2 + y**2 + 1e-6)
        
        kinetic = 0.5 * (vx**2 + vy**2)
        potential = -self.GM / r
        
        total_energy = kinetic + potential
        return total_energy