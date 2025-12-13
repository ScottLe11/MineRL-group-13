"""
Test suite for Discrete Action Recording Pipeline

Tests the complete flow:
1. TrajectoryRecorder accepts discrete action indices
2. PKL files contain discrete indices (not dicts)
3. Parser extracts discrete actions correctly
4. End-to-end pipeline produces valid BC training data
"""

import sys
import os
import tempfile
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly to avoid cv2 dependency in wrappers/__init__.py
import importlib.util

# Load discrete_actions module directly
discrete_actions_path = Path(__file__).parent.parent / "wrappers" / "discrete_actions.py"
spec = importlib.util.spec_from_file_location("discrete_actions", discrete_actions_path)
discrete_actions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discrete_actions)
DISCRETE_ACTION_POOL = discrete_actions.DISCRETE_ACTION_POOL
get_enabled_actions = discrete_actions.get_enabled_actions

# Load action_queue module directly
action_queue_path = Path(__file__).parent.parent / "recording" / "action_queue.py"
spec = importlib.util.spec_from_file_location("action_queue", action_queue_path)
action_queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(action_queue)
ActionQueue = action_queue.ActionQueue

from pkl_parser import extract_bc_data


def test_trajectory_recorder_discrete_format():
    """Test that TrajectoryRecorder accepts and saves discrete action indices."""
    print("\n" + "="*70)
    print("TEST 1: TrajectoryRecorder Discrete Format")
    print("="*70)

    # Import TrajectoryRecorder directly to avoid cv2 dependency
    recorder_path = Path(__file__).parent.parent / "wrappers" / "recorder.py"
    spec = importlib.util.spec_from_file_location("recorder", recorder_path)
    recorder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder_module)
    TrajectoryRecorder = recorder_module.TrajectoryRecorder

    import gym

    # Create a mock environment
    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
            })
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return {'pov': np.zeros((4, 84, 84), dtype=np.uint8)}, {}

        def step(self, action):
            self.step_count += 1
            obs = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)}
            reward = 1.0
            done = self.step_count >= 5
            info = {}
            return obs, reward, done, info

        def close(self):
            pass

    # Create temporary directory for test recordings
    with tempfile.TemporaryDirectory() as tmpdir:
        # Wrap with TrajectoryRecorder
        mock_env = MockEnv()
        env = TrajectoryRecorder(mock_env, log_dir=tmpdir)

        # Reset and run episode with discrete actions
        obs, info = env.reset()

        # Execute discrete actions (as tuples)
        discrete_actions = [1, 6, 8, 12, 25]  # forward, attack, turn_left_45, turn_right_45, attack_10
        for discrete_idx in discrete_actions:
            # Get MineRL dict from action definition
            minerl_dict = DISCRETE_ACTION_POOL[discrete_idx].to_minerl_dict()

            # Pass tuple (discrete_idx, minerl_dict)
            obs, reward, done, info = env.step((discrete_idx, minerl_dict))

            if done:
                break

        env.close()  # This should trigger _save_trajectory()

        # Verify PKL file was created
        pkl_files = list(Path(tmpdir).glob("*.pkl"))
        assert len(pkl_files) == 1, f"Expected 1 PKL file, found {len(pkl_files)}"
        print(f"  ✓ PKL file created: {pkl_files[0].name}")

        # Load and verify contents
        with open(pkl_files[0], 'rb') as f:
            trajectory = pickle.load(f)

        print(f"  ✓ Trajectory length: {len(trajectory)} transitions")

        # Verify actions are discrete indices (not dicts!)
        for i, transition in enumerate(trajectory):
            action = transition['action']
            assert isinstance(action, (int, np.integer)), \
                f"Transition {i}: Expected int action, got {type(action)}"
            print(f"    Step {i}: action={action} (type={type(action).__name__})")

        print("  ✓ All actions are discrete indices!")

    print("\n✅ TEST 1 PASSED: TrajectoryRecorder correctly saves discrete actions\n")


def test_pkl_parser_discrete_extraction():
    """Test that pkl_parser extracts discrete actions correctly."""
    print("="*70)
    print("TEST 2: PKL Parser Discrete Action Extraction")
    print("="*70)

    # Create mock trajectory data with discrete actions
    mock_trajectory = []
    enabled_actions = [1, 6, 8, 12, 15, 17, 25]  # Subset of actions

    for i in range(10):
        # Cycle through enabled actions
        discrete_idx = enabled_actions[i % len(enabled_actions)]

        mock_trajectory.append({
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.8 - i*0.08], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': discrete_idx,  # Discrete index!
            'reward': 1.0,
            'terminated': False,
            'truncated': False,
        })

    # Extract data using parser
    mock_config = {
        'action_space': {
            'preset': 'custom',
            'enabled_actions': enabled_actions
        }
    }

    data = extract_bc_data(mock_trajectory, mock_config)

    print(f"  ✓ Extracted {len(data['actions'])} transitions")
    print(f"  ✓ Action shape: {data['actions'].shape}")
    print(f"  ✓ Action dtype: {data['actions'].dtype}")

    # Verify actions are mapped correctly
    # Original index → Mapped index (position in enabled_actions list)
    expected_mapped = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]  # Cycling pattern
    assert np.array_equal(data['actions'], expected_mapped), \
        f"Expected {expected_mapped}, got {data['actions'].tolist()}"

    print(f"  ✓ Original actions: {[enabled_actions[i % len(enabled_actions)] for i in range(10)]}")
    print(f"  ✓ Mapped actions: {data['actions'].tolist()}")
    print(f"  ✓ Mapping correct: original index → position in enabled_actions")

    # Verify observations
    assert data['obs_pov'].shape == (10, 4, 84, 84), "POV shape mismatch"
    assert data['obs_time'].shape == (10,), "Time shape mismatch"
    assert data['obs_yaw'].shape == (10,), "Yaw shape mismatch"
    assert data['obs_pitch'].shape == (10,), "Pitch shape mismatch"

    print("  ✓ All observation arrays have correct shapes")

    print("\n✅ TEST 2 PASSED: PKL parser correctly extracts discrete actions\n")


def test_legacy_dict_actions():
    """Test that pkl_parser still handles legacy dict actions."""
    print("="*70)
    print("TEST 3: Legacy Dict Action Compatibility")
    print("="*70)

    # Create mock trajectory with dict actions (legacy format)
    mock_trajectory = [
        {
            'state': {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([0.8], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            },
            'action': {'forward': 1, 'attack': 0, 'camera': [0, 0]},  # Dict!
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
            'action': {'forward': 0, 'attack': 1, 'camera': [0, 0]},  # Dict!
            'reward': 1.0,
            'terminated': False,
        },
    ]

    mock_config = {
        'action_space': {
            'preset': 'custom',
            'enabled_actions': [1, 6, 8, 12, 15, 17, 25]
        }
    }

    data = extract_bc_data(mock_trajectory, mock_config)

    print(f"  ✓ Extracted {len(data['actions'])} transitions from legacy format")
    print(f"  ✓ Action 1 (forward): mapped to {data['actions'][0]}")
    print(f"  ✓ Action 2 (attack): mapped to {data['actions'][1]}")

    # forward (index 1) should map to position 0
    # attack (index 6) prioritized to attack_10 (index 25) → position 6
    assert data['actions'][0] == 0, "Forward should map to position 0"
    assert data['actions'][1] == 6, "Attack should prioritize attack_10 (position 6)"

    print("  ✓ Legacy discretization logic still works!")

    print("\n✅ TEST 3 PASSED: Legacy dict actions still supported\n")


def test_action_queue_integration():
    """Test ActionQueue produces correct discrete indices."""
    print("="*70)
    print("TEST 4: ActionQueue Integration")
    print("="*70)

    enabled_actions = get_enabled_actions([1, 6, 25])  # forward, attack, attack_10

    queue = ActionQueue(enabled_actions)

    # Queue attack_10 (duration=10)
    success = queue.queue_action(25)
    assert success, "Should successfully queue attack_10"

    print(f"  ✓ Queued action 25 (attack_10, duration=10)")

    # Execute 10 steps
    executed_actions = []
    for step in range(10):
        action_idx = queue.step()
        executed_actions.append(action_idx)

    # All 10 steps should be attack_10
    assert all(a == 25 for a in executed_actions), \
        f"All steps should be 25, got {executed_actions}"

    print(f"  ✓ Executed 10 steps of action 25: {executed_actions}")

    # Queue should now be idle
    assert not queue.is_busy(), "Queue should be idle after action completes"
    print(f"  ✓ Queue is idle after action completes")

    # Test queuing during execution
    queue.queue_action(25)  # Start attack_10
    queue.queue_action(1)   # Queue forward

    # Execute 10 steps (attack_10)
    for _ in range(10):
        assert queue.step() == 25

    # Next step should auto-start forward
    assert queue.step() == 1, "Queued action should auto-start"
    print(f"  ✓ Queued action auto-starts after current action completes")

    print("\n✅ TEST 4 PASSED: ActionQueue produces correct discrete indices\n")


def test_end_to_end_pipeline():
    """Test complete pipeline: Record → Save → Parse → Extract."""
    print("="*70)
    print("TEST 5: End-to-End Pipeline")
    print("="*70)

    # Import TrajectoryRecorder directly to avoid cv2 dependency
    recorder_path = Path(__file__).parent.parent / "wrappers" / "recorder.py"
    spec = importlib.util.spec_from_file_location("recorder", recorder_path)
    recorder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder_module)
    TrajectoryRecorder = recorder_module.TrajectoryRecorder

    import gym

    # Mock environment
    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
            })
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return {
                'pov': np.zeros((4, 84, 84), dtype=np.uint8),
                'time_left': np.array([1.0], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            }, {}

        def step(self, action):
            self.step_count += 1
            obs = {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'time_left': np.array([1.0 - self.step_count * 0.1], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
                'place_table_safe': np.array([1.0], dtype=np.float32),
            }
            reward = 1.0
            done = self.step_count >= 8
            info = {}
            return obs, reward, done, info

        def close(self):
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        # STEP 1: Record with discrete actions
        print("  Step 1: Recording discrete actions...")
        mock_env = MockEnv()
        env = TrajectoryRecorder(mock_env, log_dir=tmpdir)

        # Include noop (0) in enabled actions to handle idle state
        enabled_actions = get_enabled_actions([0, 1, 6, 8, 12, 25])
        queue = ActionQueue(enabled_actions)

        obs, info = env.reset()

        # Human presses keys → ActionQueue → Discrete indices
        queue.queue_action(1)   # forward (starts immediately)
        queue.queue_action(6)   # attack (queued)

        step_count = 0
        while step_count < 8:
            discrete_idx = queue.step()
            minerl_dict = enabled_actions[discrete_idx].to_minerl_dict()

            obs, reward, done, info = env.step((discrete_idx, minerl_dict))
            step_count += 1

            # Queue more actions during execution
            if step_count == 2:
                queue.queue_action(25)  # attack_10 (duration=10, but episode ends before it finishes)

            if done:
                break

        env.close()
        print(f"    ✓ Recorded {step_count} steps")

        # STEP 2: Load PKL file
        print("  Step 2: Loading PKL file...")
        pkl_files = list(Path(tmpdir).glob("*.pkl"))
        assert len(pkl_files) == 1

        with open(pkl_files[0], 'rb') as f:
            trajectory = pickle.load(f)

        print(f"    ✓ Loaded {len(trajectory)} transitions")

        # STEP 3: Parse with extract_bc_data
        print("  Step 3: Parsing with extract_bc_data...")
        mock_config = {
            'action_space': {
                'preset': 'custom',
                'enabled_actions': [0, 1, 6, 8, 12, 25]  # Updated to include noop
            }
        }

        data = extract_bc_data(trajectory, mock_config)
        print(f"    ✓ Extracted {len(data['actions'])} transitions")

        # STEP 4: Verify data format for BC training
        print("  Step 4: Verifying BC training data format...")
        assert 'obs_pov' in data
        assert 'obs_time' in data
        assert 'obs_yaw' in data
        assert 'obs_pitch' in data
        assert 'obs_place_table_safe' in data
        assert 'actions' in data
        assert 'rewards' in data
        assert 'dones' in data

        assert data['obs_pov'].shape[0] == len(data['actions'])
        assert data['actions'].dtype == np.int64
        assert data['rewards'].dtype == np.float32

        print(f"    ✓ All required keys present")
        print(f"    ✓ Shapes: obs_pov={data['obs_pov'].shape}, actions={data['actions'].shape}")
        print(f"    ✓ Action sequence: {data['actions'].tolist()}")

        # Verify action mapping is correct
        # enabled_actions = [0, 1, 6, 8, 12, 25]
        # Action sequence: [1, 6, 25, 25, 25, 25, 25, 25]
        # Mapped to positions: [1, 2, 5, 5, 5, 5, 5, 5]
        expected_first_three = [1, 2, 5]  # forward, attack, attack_10
        assert data['actions'][:3].tolist() == expected_first_three, \
            f"Expected {expected_first_three}, got {data['actions'][:3].tolist()}"

        print(f"    ✓ Action mapping correct!")

    print("\n✅ TEST 5 PASSED: End-to-end pipeline works!\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DISCRETE ACTION RECORDING PIPELINE - TEST SUITE")
    print("="*70)

    try:
        test_trajectory_recorder_discrete_format()
        test_pkl_parser_discrete_extraction()
        test_legacy_dict_actions()
        test_action_queue_integration()
        test_end_to_end_pipeline()

        print("="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nDiscrete action recording pipeline is ready for use!")
        print("  ✓ TrajectoryRecorder accepts discrete indices")
        print("  ✓ PKL files contain discrete actions")
        print("  ✓ Parser extracts actions correctly")
        print("  ✓ Legacy dict format still supported")
        print("  ✓ ActionQueue integration works")
        print("  ✓ End-to-end pipeline validated")
        print("\nNext steps:")
        print("  1. Run recorder_gameplay_discrete.py to record expert demos")
        print("  2. Use pkl_parser.py to extract BC training data")
        print("  3. Train with scripts/train.py --method bc")

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
