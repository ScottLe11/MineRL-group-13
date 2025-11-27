#!/usr/bin/env python3
"""
Evaluation script for the MineRL Tree-Chopping DQN Agent.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/final_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/final_model.pt --episodes 10 --render
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from utils.config import load_config
from networks import DQNNetwork
from scripts.train import create_env


def load_agent(checkpoint_path: str, config: dict, device: str):
    """
    Load a trained agent from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.
        
    Returns:
        Loaded DQNNetwork in eval mode.
    """
    # Create network
    network = DQNNetwork(
        input_channels=config['network']['input_channels'],
        num_actions=config['dqn']['num_actions']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['q_network_state_dict'])
    network.eval()
    
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    
    return network


def evaluate_episode(env, network, device: str, render: bool = False):
    """
    Run a single evaluation episode.
    
    Args:
        env: The environment.
        network: Trained Q-network.
        device: Device for inference.
        render: Whether to render the episode.
        
    Returns:
        Tuple of (total_reward, episode_length, wood_collected).
    """
    obs = env.reset()
    total_reward = 0
    episode_length = 0
    wood_collected = 0
    done = False
    
    while not done:
        # Convert observation to tensor
        obs_tensor = {
            'pov': torch.tensor(obs['pov'], dtype=torch.uint8).unsqueeze(0).to(device),
            'time': torch.tensor([obs['time_left']], dtype=torch.float32).to(device),
            'yaw': torch.tensor([obs['yaw']], dtype=torch.float32).to(device),
            'pitch': torch.tensor([obs['pitch']], dtype=torch.float32).to(device),
        }
        
        # Select greedy action (no exploration)
        with torch.no_grad():
            q_values = network(obs_tensor)
            action = q_values.argmax(dim=1).item()
        
        # Take step
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        episode_length += 1
        wood_collected = info.get('wood_count', 0)  # Current wood inventory (net: mining - using)
        
        if render:
            env.render()
    
    return total_reward, episode_length, wood_collected


def evaluate(config: dict, checkpoint_path: str, num_episodes: int, render: bool):
    """
    Evaluate a trained agent.
    
    Args:
        config: Configuration dictionary.
        checkpoint_path: Path to model checkpoint.
        num_episodes: Number of episodes to evaluate.
        render: Whether to render episodes.
    """
    device = config['device']
    
    # Create environment
    env = create_env(config)
    config['dqn']['num_actions'] = env.action_space.n
    
    # Load agent
    network = load_agent(checkpoint_path, config, device)
    
    # Evaluate
    rewards = []
    lengths = []
    wood_counts = []
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("-" * 50)
    
    for ep in range(num_episodes):
        reward, length, wood = evaluate_episode(env, network, device, render)
        rewards.append(reward)
        lengths.append(length)
        wood_counts.append(wood)
        
        print(f"Episode {ep + 1}: Reward={reward:.2f}, Length={length}, Wood={wood}")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes:       {num_episodes}")
    print(f"Mean Reward:    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Length:    {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Mean Wood:      {np.mean(wood_counts):.2f} ± {np.std(wood_counts):.2f}")
    print(f"Total Wood:     {sum(wood_counts)}")
    print(f"Best Episode:   Reward={max(rewards):.2f}, Wood={max(wood_counts)}")
    print("=" * 50)
    
    env.close()
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'mean_wood': np.mean(wood_counts),
        'total_wood': sum(wood_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MineRL Tree-Chopping DQN Agent")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print("=" * 50)
    print("MineRL Tree-Chopping DQN Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes:   {args.episodes}")
    print(f"Render:     {args.render}")
    print(f"Device:     {config['device']}")
    
    # Run evaluation
    evaluate(config, args.checkpoint, args.episodes, args.render)


if __name__ == "__main__":
    main()


