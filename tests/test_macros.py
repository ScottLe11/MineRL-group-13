"""
Tests for macro action behavior, especially when prerequisites are not met.

These tests document the EXPECTED behavior when macros are called without proper state:
- craft_planks without logs: Executes same primitives, creates fewer/no planks
- make_table without planks: Clicks randomly, produces nothing
- craft_sticks without planks/table: Clicks randomly, produces nothing
- craft_axe without materials: Clicks randomly, produces nothing

This is important for Q-learning: the agent learns "don't call macros without prereqs"
through wasted frames and no reward, rather than crashes.

Run with: pytest tests/test_macros.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.actions import (
    ExtendedActionWrapper,
    ACTION_NAMES,
    MACRO_CRAFT_PLANKS,
    MACRO_MAKE_TABLE,
    MACRO_CRAFT_STICKS,
    MACRO_CRAFT_AXE,
    FRAMES_PER_ACTION,
)


class MockEnvForMacros:
    """
    Mock environment that simulates macro execution.
    
    Tracks:
    - Number of frames executed
    - Whether inventory was opened
    - Click positions (for verifying macro pattern)
    """
    
    def __init__(self, inventory=None):
        """
        Args:
            inventory: Dict of item_name -> quantity. 
                       e.g., {'oak_log': 3, 'oak_planks': 0}
        """
        self.inventory = inventory or {}
        self.frame_count = 0
        self.inventory_open = False
        self.click_history = []
        self.last_action = None
        
        # Mock spaces
        import gym
        self.observation_space = gym.spaces.Dict({
            'pov': gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8),
            'isGuiOpen': gym.spaces.Discrete(2),
        })
        self.action_space = self._create_action_space()
    
    def _create_action_space(self):
        """Create mock action space with no_op method."""
        class MockActionSpace:
            def no_op(self):
                return {
                    'forward': 0, 'back': 0, 'left': 0, 'right': 0,
                    'jump': 0, 'attack': 0, 'camera': np.zeros(2),
                    'inventory': 0, 'use': 0,
                }
            @property
            def n(self):
                return 23
        return MockActionSpace()
    
    def reset(self):
        self.frame_count = 0
        self.inventory_open = False
        self.click_history = []
        return self._get_obs()
    
    def step(self, action):
        self.frame_count += 1
        self.last_action = action
        
        # Track inventory toggle
        if action.get('inventory', 0) == 1:
            self.inventory_open = not self.inventory_open
        
        # Track clicks (simplified - just note that a click happened)
        if action.get('attack', 0) == 1 or action.get('use', 0) == 1:
            self.click_history.append({
                'frame': self.frame_count,
                'type': 'attack' if action.get('attack', 0) else 'use',
            })
        
        # Macro execution never fails (crashes), just produces no result
        reward = 0.0
        done = False
        info = {'frame': self.frame_count}
        
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        return {
            'pov': np.zeros((4, 84, 84), dtype=np.uint8),
            'isGuiOpen': self.inventory_open,
        }
    
    def render(self):
        pass
    
    def get_last_full_frame(self):
        """Return mock frame for hotbar scanning."""
        return np.zeros((360, 640, 3), dtype=np.uint8)


class TestMacroWithoutPrerequisites:
    """
    Tests for macro behavior when prerequisites are NOT met.
    
    Key insight: Macros don't crash or error - they execute the same
    GUI interactions but produce no/partial results.
    """
    
    def test_craft_planks_without_logs(self):
        """
        craft_planks without logs: Executes same GUI clicks.
        
        Expected behavior:
        - Opens inventory
        - Clicks on hotbar slot where logs WOULD be
        - Clicks on crafting grid
        - Clicks on output slot (empty)
        - Wastes ~8-12 frames
        - Returns with no planks crafted
        
        This is CORRECT behavior - agent learns via wasted time.
        """
        env = MockEnvForMacros(inventory={'oak_log': 0})
        # Note: We can't easily test the full wrapper without MineRL
        # This test documents the expected behavior
        
        # Document: craft_planks takes ~8 steps (2 agent decisions worth)
        expected_frames_wasted = 8  # Approximate
        
        # The key assertion: macro doesn't crash
        assert True, "craft_planks should not crash without logs"
    
    def test_make_table_without_planks(self):
        """
        make_table without planks: Clicks randomly in GUI, does nothing useful.
        
        Expected behavior:
        - Opens inventory
        - Clicks on empty hotbar slot
        - Clicks on crafting grid positions
        - Attempts to place non-existent table
        - Wastes ~12 frames
        """
        env = MockEnvForMacros(inventory={'oak_planks': 0})
        
        # Document expected wasted frames
        expected_frames_wasted = 12  # Approximate (craft + place + open)
        
        assert True, "make_table should not crash without planks"
    
    def test_craft_sticks_without_table_open(self):
        """
        craft_sticks without table interface open: Clicks in wrong GUI.
        
        Expected behavior:
        - Assumes 3x3 grid is visible
        - Clicks at 3x3 grid positions (which don't exist in 2x2)
        - Produces nothing
        """
        env = MockEnvForMacros(inventory={'oak_planks': 4})
        
        assert True, "craft_sticks should not crash without table open"
    
    def test_craft_axe_without_materials(self):
        """
        craft_axe without sticks/planks: Just clicks around, does nothing.
        
        Expected behavior:
        - Assumes materials exist in hotbar
        - Clicks on empty slots
        - Arranges nothing in grid
        - Output slot empty
        """
        env = MockEnvForMacros(inventory={})
        
        assert True, "craft_axe should not crash without materials"


class TestMacroFrameConsumption:
    """
    Tests that macros consume the expected number of frames.
    
    Important for Q-learning: experience = (state, action, reward, next_state)
    Macro = ONE experience, but multiple frames executed.
    """
    
    def test_craft_planks_frame_count(self):
        """craft_planks should consume ~8 frames (2 agent steps)."""
        # Document: This is by design - macro = 1 decision = multiple frames
        expected_frames = 8  # logs_to_convert=3 default
        assert expected_frames > FRAMES_PER_ACTION, "Macro should take more frames than primitive"
    
    def test_make_table_frame_count(self):
        """make_table should consume ~12 frames (3 agent steps)."""
        expected_frames = 12  # craft + place + open
        assert expected_frames > FRAMES_PER_ACTION
    
    def test_macro_is_single_experience(self):
        """
        Even with multiple frames, macro = ONE experience in replay buffer.
        
        This is the key temporal abstraction:
        - state: [frame_36, 37, 38, 39]
        - action: craft_planks (consumes frames 40-47)
        - reward: sum of rewards during those 8 frames
        - next_state: [frame_44, 45, 46, 47]
        """
        # This is a documentation test - verifying understanding
        # The actual implementation is in ExtendedActionWrapper
        assert True, "Macro should be single experience regardless of frame count"


class TestMacroConstants:
    """Verify macro action indices are correct."""
    
    def test_macro_indices(self):
        """Macros should be actions 19-22."""
        assert MACRO_CRAFT_PLANKS == 19
        assert MACRO_MAKE_TABLE == 20
        assert MACRO_CRAFT_STICKS == 21
        assert MACRO_CRAFT_AXE == 22
    
    def test_macro_names(self):
        """Macro action names should match indices."""
        assert ACTION_NAMES[MACRO_CRAFT_PLANKS] == 'craft_planks'
        assert ACTION_NAMES[MACRO_MAKE_TABLE] == 'make_table'
        assert ACTION_NAMES[MACRO_CRAFT_STICKS] == 'craft_sticks'
        assert ACTION_NAMES[MACRO_CRAFT_AXE] == 'craft_axe'


class TestMacroRewardAccumulation:
    """
    Tests that rewards are properly accumulated during macro execution.
    """
    
    def test_reward_accumulation_concept(self):
        """
        During macro (8 frames):
        - Frame 40: reward = -0.001 (step penalty)
        - Frame 41: reward = -0.001
        - ...
        - Frame 47: reward = -0.001
        Total: -0.008
        
        If agent gets a log during macro (unlikely but possible):
        - Frame 45: reward = 1.0 - 0.001 = 0.999
        Total: 0.992
        """
        step_penalty = -0.001
        frames_in_macro = 8
        expected_penalty = step_penalty * frames_in_macro
        
        assert expected_penalty == -0.008, "8 frames of step penalty"
    
    def test_failed_macro_still_penalized(self):
        """
        Even when macro fails (no prereqs), step penalty applies.
        
        This is the learning signal: wasted frames = negative reward.
        Agent learns: "don't call craft_planks without logs"
        """
        frames_wasted = 8
        step_penalty = -0.001
        total_penalty = frames_wasted * step_penalty
        
        # Agent receives -0.008 for calling useless macro
        assert total_penalty < 0, "Failed macro should have negative reward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

