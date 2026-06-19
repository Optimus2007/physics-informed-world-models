import numpy as np

def orbital_derivatives(state, t, GM=1.0):
    """
    Computes the rate of change for a 2-body orbital system.
    """
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    r_cubed = r**3
    
    # Calculate velocities (change in position)
    dxdt = vx
    dydt = vy
    
    # Calculate accelerations (change in velocity via gravity)
    dvxdt = -GM * x / r_cubed
    dvydt = -GM * y / r_cubed
    
    return np.array([dxdt, dydt, dvxdt, dvydt])

def rk4_step(state, t, dt, dynamics_fn, **kwargs):
    """
    Advances the system by one time step (dt) using Runge-Kutta 4.
    """
    k1 = dynamics_fn(state, t, **kwargs)
    k2 = dynamics_fn(state + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = dynamics_fn(state + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = dynamics_fn(state + dt * k3, t + dt, **kwargs)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)