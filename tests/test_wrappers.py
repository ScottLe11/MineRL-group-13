"""
Tests for environment wrappers.

Run with: pytest tests/test_wrappers.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
from wrappers.observation import ObservationWrapper


class MockBaseEnv:
    """Mock environment for testing wrappers."""
    
    def __init__(self, frame_shape=(64, 64)):
        self.frame_shape = frame_shape
        self.observation_space = gym.spaces.Dict({
            'pov': gym.spaces.Box(low=0, high=255, shape=(4, *frame_shape), dtype=np.uint8)
        })
        self.action_space = gym.spaces.Discrete(8)
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        return {'pov': np.zeros((4, *self.frame_shape), dtype=np.uint8)}
    
    def step(self, action):
        self.step_count += 1
        obs = {'pov': np.random.randint(0, 256, (4, *self.frame_shape), dtype=np.uint8)}
        reward = -0.001
        done = self.step_count >= 100
        info = {}
        return obs, reward, done, info


class TestObservationWrapper:
    """Tests for the ObservationWrapper."""
    
    def test_reset_adds_scalars(self):
        """Reset should return observation with time, yaw, pitch."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=100)
        
        obs = wrapped.reset()
        
        assert 'pov' in obs
        assert 'time' in obs
        assert 'yaw' in obs
        assert 'pitch' in obs
    
    def test_initial_time_is_one(self):
        """Time should be 1.0 at reset."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=100)
        
        obs = wrapped.reset()
        
        assert obs['time'][0] == 1.0
    
    def test_time_decreases(self):
        """Time should decrease as steps progress."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=100)
        
        obs = wrapped.reset()
        initial_time = obs['time'][0]
        
        for _ in range(50):
            obs, _, _, _ = wrapped.step(0)
        
        assert obs['time'][0] < initial_time
        assert abs(obs['time'][0] - 0.5) < 0.01  # Should be ~0.5 after 50/100 steps
    
    def test_orientation_update(self):
        """Orientation should update correctly."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=100)
        
        wrapped.reset()
        wrapped.update_orientation(delta_yaw=90, delta_pitch=30)
        
        obs, _, _, _ = wrapped.step(0)
        
        # Normalized: yaw 90/180 = 0.5, pitch 30/90 = 0.33
        assert abs(obs['yaw'][0] - 0.5) < 0.01
        assert abs(obs['pitch'][0] - 0.333) < 0.02
    
    def test_pitch_clamping(self):
        """Pitch should be clamped to [-90, 90]."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=100)
        
        wrapped.reset()
        wrapped.update_orientation(delta_yaw=0, delta_pitch=100)  # Over limit
        
        obs, _, _, _ = wrapped.step(0)
        
        assert obs['pitch'][0] == 1.0  # Clamped to 90, normalized to 1.0
    
    def test_yaw_wrapping(self):
        """Yaw should wrap around at 180/-180."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=100)
        
        wrapped.reset()
        wrapped.update_orientation(delta_yaw=270, delta_pitch=0)  # 270 -> -90
        
        obs, _, _, _ = wrapped.step(0)
        
        # 270 degrees wraps to -90, normalized: -90/180 = -0.5
        assert abs(obs['yaw'][0] - (-0.5)) < 0.01
    
    def test_observation_space_updated(self):
        """Wrapped env should have updated observation space."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=100)
        
        assert 'pov' in wrapped.observation_space.spaces
        assert 'time' in wrapped.observation_space.spaces
        assert 'yaw' in wrapped.observation_space.spaces
        assert 'pitch' in wrapped.observation_space.spaces
    
    def test_pov_passthrough(self):
        """POV observation should pass through unchanged."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=100)
        
        obs = wrapped.reset()
        
        assert obs['pov'].shape == (4, 64, 64)
        assert obs['pov'].dtype == np.uint8


class TestObservationWrapperEdgeCases:
    """Edge case tests for ObservationWrapper."""
    
    def test_max_steps_reached(self):
        """Time should be 0 at max_steps."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=10)
        
        wrapped.reset()
        
        for _ in range(10):
            obs, _, _, _ = wrapped.step(0)
        
        assert obs['time'][0] == 0.0
    
    def test_beyond_max_steps(self):
        """Time should stay at 0 beyond max_steps."""
        env = MockBaseEnv()
        wrapped = ObservationWrapper(env, max_steps=5)
        
        wrapped.reset()
        
        for _ in range(10):
            obs, _, _, _ = wrapped.step(0)
        
        assert obs['time'][0] == 0.0  # Should not go negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


