#!/usr/bin/env python3
"""
Debug script to check model's action selection behavior.
"""

import sys
import numpy as np
import torch
from utils.config import load_config
from utils.agent_factory import create_agent

# Load config
config = load_config('config/bestsimple.yaml')

# Create agent
num_actions = len(config['action_space']['enabled_actions'])
print(f"Number of actions in config: {num_actions}")
print(f"Enabled actions: {config['action_space']['enabled_actions']}")

agent = create_agent(config, num_actions=num_actions)

# Load checkpoint
checkpoint_path = 'best_model/best_model_ppo_ep2050.pt'
print(f"\nLoading checkpoint: {checkpoint_path}")
agent.load(checkpoint_path)

# Create a dummy observation
dummy_obs = {
    'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
    'time_left': 0.8,
    'yaw': 0.0,
    'pitch': 0.0,
    'place_table_safe': 0.0
}

# Get policy info for 10 random observations
print("\n" + "="*60)
print("Testing action selection on random observations")
print("="*60)

action_counts = [0] * num_actions
for i in range(100):
    dummy_obs['pov'] = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
    dummy_obs['time_left'] = np.random.rand()

    action, log_prob, value = agent.select_action(dummy_obs)
    action_counts[action] += 1

    if i < 10:
        policy_info = agent.get_policy_info(dummy_obs)
        print(f"\nObservation {i+1}:")
        print(f"  Selected action: {action}")
        print(f"  Action probabilities: {policy_info['action_probs']}")
        print(f"  Entropy: {policy_info['entropy']:.4f}")
        print(f"  Value: {policy_info['value']:.4f}")

print("\n" + "="*60)
print("Action selection statistics over 100 random observations:")
print("="*60)
for i, count in enumerate(action_counts):
    pct = (count / 100) * 100
    print(f"  Action {i}: {count}/100 ({pct:.1f}%)")

# Check if the model is stuck on action 0
if action_counts[0] > 90:
    print("\n⚠️  WARNING: Model is selecting action 0 more than 90% of the time!")
    print("  This suggests the actor head weights may be corrupted or incorrectly loaded.")

    # Check the actor head weights
    print("\n  Inspecting actor head weights...")
    actor_final_weight = agent.policy.actor[2].weight
    actor_final_bias = agent.policy.actor[2].bias

    print(f"  Actor final layer weight shape: {actor_final_weight.shape}")
    print(f"  Actor final layer bias shape: {actor_final_bias.shape}")
    print(f"  Actor final layer weight stats:")
    print(f"    Mean: {actor_final_weight.mean().item():.6f}")
    print(f"    Std: {actor_final_weight.std().item():.6f}")
    print(f"    Min: {actor_final_weight.min().item():.6f}")
    print(f"    Max: {actor_final_weight.max().item():.6f}")
    print(f"  Actor final layer bias: {actor_final_bias.detach().cpu().numpy()}")
