import unittest
import numpy as np
import sys
import os

# Ensure the test can find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from dynamics import orbital_derivatives

class TestPhysicsEngine(unittest.TestCase):
    def test_gravity_direction(self):
        """
        Tests if the physics engine correctly calculates gravitational pull.
        If the planet is at x=1.0, y=0.0, gravity MUST pull it left (negative X).
        """
        # State: x=1.0, y=0.0, vx=0.0, vy=1.0
        test_state = np.array([1.0, 0.0, 0.0, 1.0])
        
        # Calculate derivatives
        derivatives = orbital_derivatives(test_state, t=0)
        
        # Extract acceleration in X (dvx/dt)
        acceleration_x = derivatives[2]
        
        # Assert that the acceleration is pulling back towards the origin (0,0)
        self.assertTrue(acceleration_x < 0, "Gravity is pointing the wrong way!")

if __name__ == '__main__':
    unittest.main()