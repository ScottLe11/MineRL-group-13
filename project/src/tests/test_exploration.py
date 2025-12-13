#!/usr/bin/env python3
"""
Quick test to verify epsilon-greedy exploration is working correctly.
"""

import random
import numpy as np
from agent.dqn import DQNAgent

def test_exploration():
    """Test that epsilon-greedy exploration works correctly."""
    print("=" * 70)
    print("Testing Epsilon-Greedy Exploration")
    print("=" * 70)

    # Create agent with high epsilon
    agent = DQNAgent(
        num_actions=16,  # "assisted" preset
        input_channels=4,
        num_scalars=3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
        device='cpu'
    )

    # Mock state
    mock_state = {
        'pov': np.zeros((4, 84, 84), dtype=np.float32),
        'time': 0.5,
        'yaw': 0.0,
        'pitch': 0.0
    }

    print(f"\nAgent settings:")
    print(f"  num_actions: {agent.num_actions}")
    print(f"  epsilon_start: {agent.epsilon_start}")
    print(f"  epsilon_end: {agent.epsilon_end}")
    print(f"  step_count: {agent.step_count}")
    print(f"  current epsilon: {agent.get_epsilon():.3f}")

    # Test 1: With epsilon=1.0, should see diverse actions
    print(f"\n{'='*70}")
    print("Test 1: Pure Exploration (epsilon=1.0)")
    print(f"{'='*70}")

    actions = []
    for i in range(100):
        action = agent.select_action(mock_state, explore=True)
        actions.append(action)

    unique_actions = len(set(actions))
    print(f"Selected 100 actions:")
    print(f"  Unique actions: {unique_actions}/{agent.num_actions}")
    print(f"  Expected: ~{agent.num_actions} (all actions)")

    # Count frequency
    from collections import Counter
    action_counts = Counter(actions)
    print(f"\n  Action distribution (top 5):")
    for action, count in action_counts.most_common(5):
        print(f"    Action {action}: {count} times ({count/len(actions)*100:.1f}%)")

    if unique_actions < agent.num_actions * 0.7:
        print(f"\n  ⚠️  WARNING: Only {unique_actions}/{agent.num_actions} unique actions!")
        print(f"      Expected nearly all actions to be tried with epsilon=1.0")
        print(f"      This suggests exploration is NOT working correctly!")
    else:
        print(f"\n  ✅ PASS: Good action diversity with epsilon=1.0")

    # Test 2: Increment steps and verify epsilon decreases
    print(f"\n{'='*70}")
    print("Test 2: Epsilon Decay")
    print(f"{'='*70}")

    test_steps = [0, 10000, 25000, 50000, 100000]
    for step in test_steps:
        agent.step_count = step
        epsilon = agent.get_epsilon()
        print(f"  Step {step:6d}: epsilon = {epsilon:.3f}")

    expected_at_50k = 0.05  # Should be at epsilon_end
    if abs(agent.get_epsilon() - expected_at_50k) > 0.01:
        print(f"\n  ⚠️  WARNING: Epsilon at 50k steps should be {expected_at_50k}, got {agent.get_epsilon():.3f}")
    else:
        print(f"\n  ✅ PASS: Epsilon decay working correctly")

    # Test 3: With epsilon=0.0, should be deterministic
    print(f"\n{'='*70}")
    print("Test 3: Greedy Selection (epsilon=0.0)")
    print(f"{'='*70}")

    agent.step_count = 100000  # Force epsilon to minimum
    actions_greedy = []
    for i in range(20):
        action = agent.select_action(mock_state, explore=True)
        actions_greedy.append(action)

    unique_greedy = len(set(actions_greedy))
    print(f"Selected 20 actions with epsilon={agent.get_epsilon():.3f}:")
    print(f"  Unique actions: {unique_greedy}")
    print(f"  Expected: 1 (deterministic greedy)")

    if unique_greedy > 1:
        print(f"\n  ⚠️  WARNING: Greedy selection should be deterministic!")
        print(f"      Got {unique_greedy} different actions with epsilon~0")
    else:
        print(f"\n  ✅ PASS: Greedy selection is deterministic")

    # Test 4: Check action space bounds
    print(f"\n{'='*70}")
    print("Test 4: Action Space Bounds")
    print(f"{'='*70}")

    agent.step_count = 0  # Reset for exploration
    actions = [agent.select_action(mock_state, explore=True) for _ in range(1000)]

    min_action = min(actions)
    max_action = max(actions)
    print(f"1000 random actions:")
    print(f"  Min action: {min_action} (expected: 0)")
    print(f"  Max action: {max_action} (expected: {agent.num_actions - 1})")

    invalid = [a for a in actions if a < 0 or a >= agent.num_actions]
    if invalid:
        print(f"\n  ❌ ERROR: Found {len(invalid)} invalid actions!")
        print(f"      Invalid actions: {set(invalid)}")
    else:
        print(f"\n  ✅ PASS: All actions within valid range [0, {agent.num_actions-1}]")

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print("If exploration is working correctly, you should see:")
    print("  ✅ Test 1: ~16 unique actions (good diversity)")
    print("  ✅ Test 2: Epsilon decays from 1.0 to 0.05")
    print("  ✅ Test 3: Only 1 action when greedy (epsilon=0)")
    print("  ✅ Test 4: All actions in range [0, 15]")
    print()
    print("If agent is spamming one action in training:")
    print("  - Check epsilon value in logs (should start at 1.0)")
    print("  - Check action_stats for diversity")
    print("  - May be a Q-network initialization issue")
    print("=" * 70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    test_exploration()
