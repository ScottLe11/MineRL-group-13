"""
Observation Wrapper - Adds scalar features to the observation dictionary.

This wrapper tracks and exposes:
- time: normalized time remaining (1.0 → 0.0)
- yaw: agent's horizontal facing direction (degrees, normalized to [-1, 1])
- pitch: agent's vertical head angle (degrees, normalized to [-1, 1])
"""

import gym
import numpy as np


class ObservationWrapper(gym.Wrapper):
    """
    Wraps environment to add scalar observations needed by the DQN network.
    
    Should be placed AFTER StackAndProcessWrapper and BEFORE action wrappers
    in the wrapper chain.
    """
    
    def __init__(self, env, max_steps: int = 8000):
        """
        Args:
            env: The wrapped environment (should have stacked POV observations).
            max_steps: Maximum episode steps for time normalization.
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        
        # Track agent orientation (updated externally when camera actions execute)
        self.yaw = 0.0    # Horizontal rotation: -180 to 180
        self.pitch = 0.0  # Vertical rotation: -90 (down) to 90 (up)
        
        # Update observation space to include scalars
        # Keep existing spaces and add new scalar spaces
        self.observation_space = gym.spaces.Dict({
            'pov': env.observation_space.spaces.get('pov', env.observation_space),
            'time': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'yaw': gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'pitch': gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })
    
    def reset(self):
        """Reset environment and observation tracking."""
        obs = self.env.reset()
        self.current_step = 0
        self.yaw = 0.0
        self.pitch = 0.0
        return self._add_scalars(obs)
    
    def step(self, action):
        """Take a step and add scalar observations."""
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1
        return self._add_scalars(obs), reward, done, info
    
    def _add_scalars(self, obs: dict) -> dict:
        """
        Add time, yaw, and pitch scalars to observation dict.
        
        Args:
            obs: Original observation dict with 'pov' key.
            
        Returns:
            Extended observation dict with 'pov', 'time', 'yaw', 'pitch'.
        """
        # Normalized time remaining: 1.0 at start, 0.0 at max_steps
        time_remaining = max(0.0, (self.max_steps - self.current_step) / self.max_steps)
        
        # Normalize yaw to [-1, 1] (from -180 to 180)
        normalized_yaw = self.yaw / 180.0
        
        # Normalize pitch to [-1, 1] (from -90 to 90)
        normalized_pitch = self.pitch / 90.0
        
        # Build extended observation
        extended_obs = {
            'pov': obs['pov'] if isinstance(obs, dict) else obs,
            'time': np.array([time_remaining], dtype=np.float32),
            'yaw': np.array([normalized_yaw], dtype=np.float32),
            'pitch': np.array([normalized_pitch], dtype=np.float32),
        }
        
        return extended_obs
    
    def update_orientation(self, delta_yaw: float, delta_pitch: float):
        """
        Update agent orientation (called by action wrapper after camera actions).
        
        Args:
            delta_yaw: Change in horizontal rotation (degrees).
            delta_pitch: Change in vertical rotation (degrees).
        """
        self.yaw = (self.yaw + delta_yaw) % 360
        if self.yaw > 180:
            self.yaw -= 360
        
        self.pitch = np.clip(self.pitch + delta_pitch, -90, 90)
    
    def get_time_fraction(self) -> float:
        """Returns current step / max_steps (useful for logging)."""
        return self.current_step / self.max_steps


if __name__ == "__main__":
    print("✅ ObservationWrapper Test")
    
    # Create a mock environment for testing
    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
            })
            self.action_space = gym.spaces.Discrete(23)
            self.step_count = 0
        
        def reset(self):
            self.step_count = 0
            return {'pov': np.zeros((4, 84, 84), dtype=np.uint8)}
        
        def step(self, action):
            self.step_count += 1
            obs = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)}
            reward = -0.001
            done = self.step_count >= 100
            return obs, reward, done, {}
    
    mock_env = MockEnv()
    wrapped_env = ObservationWrapper(mock_env, max_steps=100)
    
    # Test reset
    obs = wrapped_env.reset()
    print(f"  After reset:")
    print(f"    POV shape: {obs['pov'].shape}")
    print(f"    Time: {obs['time'][0]:.3f} (should be 1.0)")
    print(f"    Yaw: {obs['yaw'][0]:.3f}")
    print(f"    Pitch: {obs['pitch'][0]:.3f}")
    
    assert obs['time'][0] == 1.0, "Time should be 1.0 at reset"
    assert obs['yaw'][0] == 0.0, "Yaw should be 0.0 at reset"
    assert obs['pitch'][0] == 0.0, "Pitch should be 0.0 at reset"
    
    # Test stepping
    for i in range(50):
        obs, _, _, _ = wrapped_env.step(0)
    
    print(f"\n  After 50 steps:")
    print(f"    Time: {obs['time'][0]:.3f} (should be ~0.5)")
    assert 0.49 <= obs['time'][0] <= 0.51, "Time should be ~0.5 after 50/100 steps"
    
    # Test orientation update
    wrapped_env.update_orientation(delta_yaw=90, delta_pitch=30)
    obs, _, _, _ = wrapped_env.step(0)
    
    print(f"\n  After orientation update (+90 yaw, +30 pitch):")
    print(f"    Yaw: {obs['yaw'][0]:.3f} (should be 0.5 = 90/180)")
    print(f"    Pitch: {obs['pitch'][0]:.3f} (should be 0.33 = 30/90)")
    
    assert abs(obs['yaw'][0] - 0.5) < 0.01, "Yaw should be 0.5"
    assert abs(obs['pitch'][0] - 0.333) < 0.01, "Pitch should be ~0.33"
    
    # Test pitch clamping
    wrapped_env.update_orientation(delta_yaw=0, delta_pitch=100)  # Should clamp to 90
    obs, _, _, _ = wrapped_env.step(0)
    print(f"\n  After pitch overflow (+100 more pitch):")
    print(f"    Pitch: {obs['pitch'][0]:.3f} (should be 1.0 = clamped to 90)")
    assert obs['pitch'][0] == 1.0, "Pitch should be clamped to 1.0"
    
    print("\n✅ ObservationWrapper validated!")

