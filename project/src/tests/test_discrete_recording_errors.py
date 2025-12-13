"""
Error Handling and Edge Case Tests for Discrete Action Recording

Tests edge cases, error conditions, and robustness:
- Invalid action indices
- Empty trajectories
- Corrupted data
- Mismatched configurations
- ActionQueue edge cases
- Config validation
"""

import sys
import os
import tempfile
import pickle
import numpy as np
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly to avoid cv2 dependency
import importlib.util

# Load discrete_actions module
discrete_actions_path = Path(__file__).parent.parent / "wrappers" / "discrete_actions.py"
spec = importlib.util.spec_from_file_location("discrete_actions", discrete_actions_path)
discrete_actions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discrete_actions)
DISCRETE_ACTION_POOL = discrete_actions.DISCRETE_ACTION_POOL
get_enabled_actions = discrete_actions.get_enabled_actions

# Load action_queue module
action_queue_path = Path(__file__).parent.parent / "recording" / "action_queue.py"
spec = importlib.util.spec_from_file_location("action_queue", action_queue_path)
action_queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(action_queue)
ActionQueue = action_queue.ActionQueue

from pkl_parser import extract_bc_data


def test_action_queue_invalid_action():
    """Test ActionQueue rejects invalid action indices."""
    print("\n" + "="*70)
    print("ERROR TEST 1: ActionQueue Invalid Action Index")
    print("="*70)

    enabled_actions = get_enabled_actions([1, 6, 25])
    queue = ActionQueue(enabled_actions)

    # Try to queue an action not in enabled_actions
    try:
        queue.queue_action(99)  # Invalid action
        assert False, "Should have raised ValueError for invalid action"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    # Try to queue an action not in the enabled set
    try:
        queue.queue_action(8)  # Valid action but not enabled
        assert False, "Should have raised ValueError for non-enabled action"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    print("\n✅ ERROR TEST 1 PASSED: Invalid actions properly rejected\n")


def test_action_queue_full_buffer():
    """Test ActionQueue properly handles queue full condition."""
    print("="*70)
    print("ERROR TEST 2: ActionQueue Full Buffer Handling")
    print("="*70)

    enabled_actions = get_enabled_actions([1, 6, 25])
    queue = ActionQueue(enabled_actions)

    # Queue first action (starts immediately)
    success1 = queue.queue_action(25)  # attack_10 (duration=10)
    assert success1, "First action should queue successfully"
    print(f"  ✓ Action 1 queued (attack_10, duration=10)")

    # Queue second action (goes to buffer)
    success2 = queue.queue_action(1)  # forward
    assert success2, "Second action should queue successfully"
    print(f"  ✓ Action 2 queued (forward)")

    # Try to queue third action (buffer full)
    success3 = queue.queue_action(6)  # attack
    assert not success3, "Third action should be rejected (buffer full)"
    print(f"  ✓ Action 3 rejected (buffer full)")

    # Verify statistics
    stats = queue.get_statistics()
    assert stats['total_queue_rejections'] == 1, "Should have 1 rejection"
    print(f"  ✓ Queue rejection count correct: {stats['total_queue_rejections']}")

    print("\n✅ ERROR TEST 2 PASSED: Queue full condition handled correctly\n")


def test_parser_empty_trajectory():
    """Test parser handles empty trajectories gracefully."""
    print("="*70)
    print("ERROR TEST 3: Parser Empty Trajectory Handling")
    print("="*70)

    empty_trajectory = []
    config = {
        'action_space': {
            'preset': 'custom',
            'enabled_actions': [1, 6, 8, 12, 25]
        }
    }

    try:
        data = extract_bc_data(empty_trajectory, config)
        assert False, "Should have raised ValueError for empty trajectory"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    print("\n✅ ERROR TEST 3 PASSED: Empty trajectory handled correctly\n")


def test_parser_corrupted_observations():
    """Test parser skips corrupted observations gracefully."""
    print("="*70)
    print("ERROR TEST 4: Parser Corrupted Observation Handling")
    print("="*70)

    # Mix of valid and corrupted transitions
    mixed_trajectory = [
        # Valid transition
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.8], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': 1,
            'reward': 1.0,
            'terminated': False,
        },
        # Corrupted: missing scalar keys
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
            },
            'action': 6,
            'reward': 1.0,
            'terminated': False,
        },
        # Corrupted: state is not a dict
        {
            'state': None,
            'action': 8,
            'reward': 1.0,
            'terminated': False,
        },
        # Valid transition
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.6], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': 12,
            'reward': 1.0,
            'terminated': False,
        },
    ]

    config = {
        'action_space': {
            'preset': 'custom',
            'enabled_actions': [1, 6, 8, 12, 25]
        }
    }

    # Should extract only valid transitions (2 out of 4)
    data = extract_bc_data(mixed_trajectory, config)

    assert len(data['actions']) == 2, f"Expected 2 valid transitions, got {len(data['actions'])}"
    print(f"  ✓ Extracted {len(data['actions'])} valid transitions from 4 total")
    print(f"  ✓ Skipped {4 - len(data['actions'])} corrupted transitions")
    print(f"  ✓ Valid actions: {data['actions'].tolist()}")

    print("\n✅ ERROR TEST 4 PASSED: Corrupted observations skipped gracefully\n")


def test_parser_mismatched_action_indices():
    """Test parser handles actions not in enabled_actions."""
    print("="*70)
    print("ERROR TEST 5: Parser Mismatched Action Indices")
    print("="*70)

    # Recording used actions [1, 6, 25]
    # But config only enables [1, 8, 12]
    trajectory = [
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.8], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': 1,  # Valid
            'reward': 1.0,
            'terminated': False,
        },
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.7], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': 6,  # NOT in enabled_actions
            'reward': 1.0,
            'terminated': False,
        },
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.6], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': 25,  # NOT in enabled_actions
            'reward': 1.0,
            'terminated': False,
        },
    ]

    config = {
        'action_space': {
            'preset': 'custom',
            'enabled_actions': [1, 8, 12]  # Different from recording!
        }
    }

    data = extract_bc_data(trajectory, config)

    # Should only extract action 1 (the one that's in enabled_actions)
    assert len(data['actions']) == 1, f"Expected 1 valid action, got {len(data['actions'])}"
    assert data['actions'][0] == 0, "Action 1 should map to position 0"
    print(f"  ✓ Extracted {len(data['actions'])} valid transitions from 3 total")
    print(f"  ✓ Skipped 2 actions not in enabled_actions")
    print(f"  ✓ Valid action indices: {data['actions'].tolist()}")

    print("\n✅ ERROR TEST 5 PASSED: Mismatched actions handled correctly\n")


def test_action_queue_state_transitions():
    """Test ActionQueue state transitions are correct."""
    print("="*70)
    print("ERROR TEST 6: ActionQueue State Transitions")
    print("="*70)

    enabled_actions = get_enabled_actions([1, 6, 25])
    queue = ActionQueue(enabled_actions)

    # Initial state: idle
    assert not queue.is_busy(), "Should start idle"
    assert queue.can_queue(), "Should be able to queue"
    state = queue.get_state()
    assert state['current_action'] is None
    assert state['queued_action'] is None
    print(f"  ✓ Initial state: idle, can queue")

    # Queue action: should start immediately
    queue.queue_action(6)  # attack (duration=1)
    assert queue.is_busy(), "Should be busy after queuing"
    assert queue.can_queue(), "Should still be able to queue (buffer empty)"
    state = queue.get_state()
    assert state['current_action'] == 6
    assert state['remaining_steps'] == 1
    assert state['queued_action'] is None
    print(f"  ✓ After queue: busy, current=6, remaining=1")

    # Execute step: action should complete
    action_idx = queue.step()
    assert action_idx == 6, "Should return current action"
    assert not queue.is_busy(), "Should be idle after completing 1-step action"
    assert queue.can_queue(), "Should be able to queue"
    print(f"  ✓ After step: idle again (1-step action completed)")

    # Queue multi-step action
    queue.queue_action(25)  # attack_10 (duration=10)
    queue.queue_action(1)   # forward (queued)
    assert queue.is_busy(), "Should be busy"
    assert not queue.can_queue(), "Should NOT be able to queue (buffer full)"
    state = queue.get_state()
    assert state['current_action'] == 25
    assert state['remaining_steps'] == 10
    assert state['queued_action'] == 1
    print(f"  ✓ Multi-step action: current=25, remaining=10, queued=1")

    # Execute all 10 steps
    for i in range(10):
        action_idx = queue.step()
        assert action_idx == 25, f"Step {i+1} should return 25"

    # After 10 steps, queued action should auto-start
    assert queue.is_busy(), "Should still be busy (queued action started)"
    state = queue.get_state()
    assert state['current_action'] == 1, "Queued action should have started"
    assert state['remaining_steps'] == 1
    assert state['queued_action'] is None
    print(f"  ✓ After 10 steps: queued action auto-started (current=1)")

    print("\n✅ ERROR TEST 6 PASSED: State transitions correct\n")


def test_discrete_actions_coverage():
    """Test all 26 discrete actions are defined correctly."""
    print("="*70)
    print("ERROR TEST 7: Discrete Actions Completeness")
    print("="*70)

    # Verify all indices 0-25 exist
    for i in range(26):
        assert i in DISCRETE_ACTION_POOL, f"Action index {i} missing from DISCRETE_ACTION_POOL"

    print(f"  ✓ All 26 action indices (0-25) present")

    # Verify each action has required fields
    required_fields = ['index', 'name', 'duration', 'display_name', 'default_key']
    for idx, action in DISCRETE_ACTION_POOL.items():
        for field in required_fields:
            assert hasattr(action, field), f"Action {idx} missing field '{field}'"

        # Verify to_minerl_dict is callable
        assert callable(action.to_minerl_dict), f"Action {idx} to_minerl_dict not callable"

        # Verify it returns a dict
        minerl_dict = action.to_minerl_dict()
        assert isinstance(minerl_dict, dict), f"Action {idx} to_minerl_dict didn't return dict"

        # Verify duration is positive
        assert action.duration > 0, f"Action {idx} has invalid duration: {action.duration}"

    print(f"  ✓ All actions have required fields")
    print(f"  ✓ All to_minerl_dict() methods work")
    print(f"  ✓ All durations are valid")

    # Test get_enabled_actions with various subsets
    subset1 = get_enabled_actions([1, 6, 25])
    assert len(subset1) == 3
    assert all(idx in subset1 for idx in [1, 6, 25])
    print(f"  ✓ get_enabled_actions works for subset [1, 6, 25]")

    subset2 = get_enabled_actions(list(range(23)))
    assert len(subset2) == 23
    print(f"  ✓ get_enabled_actions works for base 23 actions")

    print("\n✅ ERROR TEST 7 PASSED: All discrete actions valid\n")


def test_recorder_multiple_episodes():
    """Test TrajectoryRecorder handles multiple episodes correctly."""
    print("="*70)
    print("ERROR TEST 8: Multiple Episodes Recording")
    print("="*70)

    # Import TrajectoryRecorder
    recorder_path = Path(__file__).parent.parent / "wrappers" / "recorder.py"
    spec = importlib.util.spec_from_file_location("recorder", recorder_path)
    recorder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder_module)
    TrajectoryRecorder = recorder_module.TrajectoryRecorder

    import gym

    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
            })
            self.step_count = 0
            self.episode_count = 0

        def reset(self):
            self.step_count = 0
            self.episode_count += 1
            return {'pov': np.zeros((4, 84, 84), dtype=np.uint8)}, {}

        def step(self, action):
            self.step_count += 1
            obs = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)}
            reward = 1.0
            done = self.step_count >= 3  # Short episodes
            info = {'episode': self.episode_count}
            return obs, reward, done, info

        def close(self):
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_env = MockEnv()
        env = TrajectoryRecorder(mock_env, log_dir=tmpdir)

        # Record 3 episodes
        # Note: Recorder uses millisecond precision to prevent file overwrites
        for episode in range(3):
            obs, info = env.reset()

            for step in range(3):
                obs, reward, done, info = env.step((1, DISCRETE_ACTION_POOL[1].to_minerl_dict()))
                if done:
                    break

        env.close()

        # Verify 3 PKL files created
        pkl_files = list(Path(tmpdir).glob("*.pkl"))
        assert len(pkl_files) == 3, f"Expected 3 PKL files, found {len(pkl_files)}"
        print(f"  ✓ Created {len(pkl_files)} PKL files for 3 episodes")

        # Verify each file has correct number of steps
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                trajectory = pickle.load(f)
            assert len(trajectory) == 3, f"Expected 3 steps, got {len(trajectory)}"
            print(f"  ✓ {pkl_file.name}: {len(trajectory)} steps")

    print("\n✅ ERROR TEST 8 PASSED: Multiple episodes handled correctly\n")


def test_action_queue_statistics():
    """Test ActionQueue statistics tracking."""
    print("="*70)
    print("ERROR TEST 9: ActionQueue Statistics Tracking")
    print("="*70)

    enabled_actions = get_enabled_actions([1, 6, 25])
    queue = ActionQueue(enabled_actions)

    # Initial statistics
    stats = queue.get_statistics()
    assert stats['total_actions_executed'] == 0
    assert stats['total_queue_rejections'] == 0
    print(f"  ✓ Initial stats: {stats}")

    # Execute some actions
    queue.queue_action(1)
    queue.step()

    queue.queue_action(6)
    queue.step()

    queue.queue_action(25)
    for _ in range(10):
        queue.step()

    stats = queue.get_statistics()
    assert stats['total_actions_executed'] == 3, f"Expected 3 actions executed, got {stats['total_actions_executed']}"
    print(f"  ✓ After 3 actions: {stats}")

    # Test rejection counting
    queue.queue_action(1)  # Starts immediately
    queue.queue_action(6)  # Queued
    success = queue.queue_action(25)  # Rejected
    assert not success

    stats = queue.get_statistics()
    assert stats['total_queue_rejections'] == 1, f"Expected 1 rejection, got {stats['total_queue_rejections']}"
    print(f"  ✓ After rejection: {stats}")

    print("\n✅ ERROR TEST 9 PASSED: Statistics tracked correctly\n")


def test_parser_action_type_validation():
    """Test parser validates action types correctly."""
    print("="*70)
    print("ERROR TEST 10: Parser Action Type Validation")
    print("="*70)

    # Test with various invalid action types
    invalid_types_trajectory = [
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.8], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': "invalid_string",  # String (invalid)
            'reward': 1.0,
            'terminated': False,
        },
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.7], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': [1, 2, 3],  # List (invalid)
            'reward': 1.0,
            'terminated': False,
        },
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.6], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': 1,  # Valid int
            'reward': 1.0,
            'terminated': False,
        },
    ]

    config = {
        'action_space': {
            'preset': 'custom',
            'enabled_actions': [1, 6, 8]
        }
    }

    data = extract_bc_data(invalid_types_trajectory, config)

    # Should only extract the valid int action
    assert len(data['actions']) == 1, f"Expected 1 valid action, got {len(data['actions'])}"
    print(f"  ✓ Extracted {len(data['actions'])} valid transitions from 3 total")
    print(f"  ✓ Skipped 2 invalid action types (string, list)")

    print("\n✅ ERROR TEST 10 PASSED: Action type validation works\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DISCRETE ACTION RECORDING - ERROR & EDGE CASE TEST SUITE")
    print("="*70)

    try:
        test_action_queue_invalid_action()
        test_action_queue_full_buffer()
        test_parser_empty_trajectory()
        test_parser_corrupted_observations()
        test_parser_mismatched_action_indices()
        test_action_queue_state_transitions()
        test_discrete_actions_coverage()
        test_recorder_multiple_episodes()
        test_action_queue_statistics()
        test_parser_action_type_validation()

        print("="*70)
        print("🎉 ALL ERROR & EDGE CASE TESTS PASSED! 🎉")
        print("="*70)
        print("\nRobustness validated:")
        print("  ✓ Invalid action indices rejected")
        print("  ✓ Queue full condition handled")
        print("  ✓ Empty trajectories rejected")
        print("  ✓ Corrupted observations skipped")
        print("  ✓ Mismatched action configs handled")
        print("  ✓ State transitions correct")
        print("  ✓ All 26 discrete actions valid")
        print("  ✓ Multiple episodes recorded correctly")
        print("  ✓ Statistics tracked accurately")
        print("  ✓ Action type validation works")
        print("\nThe system is robust and production-ready!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
