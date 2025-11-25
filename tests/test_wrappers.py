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
from wrappers.actions import (
    ExtendedActionWrapper, 
    ACTION_NAMES, 
    NUM_ACTIONS,
    FRAMES_PER_ACTION,
    get_action_name,
    get_action_space_info,
)


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


class TestExtendedActionWrapper:
    """Tests for the ExtendedActionWrapper."""
    
    def test_action_space_size(self):
        """Action space should have 23 actions."""
        assert NUM_ACTIONS == 23
        assert len(ACTION_NAMES) == 23
    
    def test_action_names(self):
        """All actions should have names."""
        for i in range(NUM_ACTIONS):
            name = get_action_name(i)
            assert name is not None
            assert 'unknown' not in name
    
    def test_action_space_info(self):
        """Action space info should be complete."""
        info = get_action_space_info()
        
        assert info['num_actions'] == 23
        assert info['frames_per_action'] == 4
        assert len(info['primitives']) == 7  # noop through attack
        assert len(info['camera']) == 12     # 4 left + 4 right + 2 up + 2 down
        assert len(info['macros']) == 4      # planks, make_table, sticks, axe
    
    def test_primitive_action_indices(self):
        """Primitive actions should be indices 0-6."""
        info = get_action_space_info()
        assert info['primitives'] == [0, 1, 2, 3, 4, 5, 6]
    
    def test_camera_action_indices(self):
        """Camera actions should be indices 7-18."""
        info = get_action_space_info()
        assert info['camera'] == list(range(7, 19))
    
    def test_macro_action_indices(self):
        """Macro actions should be indices 19-22."""
        info = get_action_space_info()
        assert info['macros'] == [19, 20, 21, 22]
    
    def test_frames_per_action(self):
        """Each action should execute for 4 frames."""
        assert FRAMES_PER_ACTION == 4


class TestActionNames:
    """Tests for action naming."""
    
    def test_primitive_names(self):
        """Primitive action names should be correct."""
        assert ACTION_NAMES[0] == 'noop'
        assert ACTION_NAMES[1] == 'forward'
        assert ACTION_NAMES[6] == 'attack'
    
    def test_camera_names(self):
        """Camera action names should be correct."""
        assert ACTION_NAMES[7] == 'turn_left_30'
        assert ACTION_NAMES[10] == 'turn_left_90'
        assert ACTION_NAMES[11] == 'turn_right_30'
        assert ACTION_NAMES[14] == 'turn_right_90'
        assert ACTION_NAMES[15] == 'look_up_12'
        assert ACTION_NAMES[18] == 'look_down_20'
    
    def test_macro_names(self):
        """Macro action names should be correct."""
        assert ACTION_NAMES[19] == 'craft_planks'
        assert ACTION_NAMES[20] == 'make_table'
        assert ACTION_NAMES[21] == 'craft_sticks'
        assert ACTION_NAMES[22] == 'craft_axe'
    
    def test_get_action_name_invalid(self):
        """Invalid action index should return unknown."""
        assert 'unknown' in get_action_name(100)
        assert 'unknown' in get_action_name(-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


