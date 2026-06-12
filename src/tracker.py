import numpy as np

class UncertaintyKalmanFilter:
    """
    Kalman Filter state tracker with dynamic observation covariance scaling.
    State vector: [x_pos, y_pos, x_vel, y_vel]^T
    """
    def __init__(self, dt=1.0):
        self.dt = dt
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        self.Q = np.eye(4) * 1e-5
        self.P = np.eye(4) * 1.0
        
    def predict(self, state, current_drift):
        drift_vec = np.array([0.0, 0.0, current_drift[0], current_drift[1]])
        state_pred = self.F @ state + drift_vec
        self.P = self.F @ self.P @ self.F.T + self.Q
        return state_pred
        
    def update(self, state_pred, measurement, spatial_cov, epistemic_unc, gamma=2.0):
        R = spatial_cov + gamma * epistemic_unc * np.eye(2)
        S_cov = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S_cov)
        innovation = measurement - self.H @ state_pred
        state_updated = state_pred + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return state_updated
