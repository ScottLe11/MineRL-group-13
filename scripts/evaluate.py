#!/usr/bin/env python3
"""
Unified evaluation script for MineRL Tree-Chopping RL Agents.

Supports both DQN and PPO algorithms.
Episodes run for a fixed number of steps (not based on done signal).

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model_dqn.pt --algorithm dqn
    python scripts/evaluate.py --checkpoint checkpoints/final_model_ppo.pt --algorithm ppo --episodes 10 --render
    python scripts/evaluate.py --checkpoint checkpoints/best_model_ppo.pt --algorithm ppo --max-steps 200
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from utils.config import load_config
from utils.env_factory import create_env
from utils.agent_factory import create_agent


def load_agent_from_checkpoint(checkpoint_path: str, config: dict, num_actions: int):
    """
    Load a trained agent from checkpoint (works for both DQN and PPO).

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        num_actions: Number of actions in the environment

    Returns:
        Loaded agent in evaluation mode
    """
    # Create agent
    agent = create_agent(config, num_actions=num_actions)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    agent.load(checkpoint_path)

    # Set to eval mode if applicable
    if hasattr(agent, 'q_network'):
        agent.q_network.eval()
    if hasattr(agent, 'policy'):
        agent.policy.eval()

    episode = checkpoint.get('episode', 'unknown')
    step_count = checkpoint.get('step_count', 'unknown')
    print(f"✅ Loaded checkpoint from episode {episode}, step {step_count}")

    return agent


def evaluate_episode(env, agent, max_steps: int, render: bool = False):
    """
    Run a single evaluation episode for a fixed number of steps.

    Args:
        env: The environment
        agent: Trained agent (DQN or PPO)
        max_steps: Maximum steps per episode (episode ends after this many steps)
        render: Whether to render the episode

    Returns:
        Tuple of (total_reward, episode_length, wood_collected)
    """
    obs = env.reset()
    total_reward = 0
    episode_length = 0
    wood_collected = 0

    for step in range(max_steps):
        # Select action (greedy for DQN, sample from policy for PPO)
        if hasattr(agent, 'select_action'):
            # PPO agent - returns (action, log_prob, value)
            if hasattr(agent, 'policy'):
                with torch.no_grad():
                    action, _, _ = agent.select_action(obs)
            # DQN agent - pass explore=False for greedy action
            else:
                action = agent.select_action(obs, explore=False)
        else:
            raise ValueError(f"Agent {type(agent)} doesn't have select_action method")

        # Take step
        obs, reward, done, info = env.step(action)

        total_reward += reward
        episode_length += 1
        wood_collected = info.get('wood_count', 0)

        if render:
            env.render()

        # Reset environment if done (but continue for max_steps)
        if done:
            obs = env.reset()

    return total_reward, episode_length, wood_collected


def evaluate(config: dict, checkpoint_path: str, num_episodes: int, render: bool, max_steps: int = None):
    """
    Evaluate a trained agent.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        max_steps: Max steps per episode (if None, computed from config)
    """
    device = config['device']
    algorithm = config.get('algorithm', 'dqn')

    # Compute max_steps from config if not provided
    if max_steps is None:
        episode_seconds = config.get('environment', {}).get('episode_seconds', 25)
        max_steps = episode_seconds * 5  # 5 agent steps per second (4 frames per step at 20fps)
    
    # Create environment
    env = create_env(config)
    num_actions = env.action_space.n

    # Load agent
    agent = load_agent_from_checkpoint(checkpoint_path, config, num_actions)

    # Evaluate
    rewards = []
    lengths = []
    wood_counts = []

    print(f"\n{'='*60}")
    print(f"Evaluating {algorithm.upper()} for {num_episodes} episodes...")
    print(f"Max steps per episode: {max_steps}")
    print(f"{'='*60}\n")

    for ep in range(num_episodes):
        reward, length, wood = evaluate_episode(env, agent, max_steps, render)
        rewards.append(reward)
        lengths.append(length)
        wood_counts.append(wood)

        print(f"Episode {ep + 1}/{num_episodes}: Reward={reward:.2f}, Length={length}, Wood={wood}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Algorithm:      {algorithm.upper()}")
    print(f"Episodes:       {num_episodes}")
    print(f"Mean Reward:    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Length:    {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Mean Wood:      {np.mean(wood_counts):.2f} ± {np.std(wood_counts):.2f}")
    print(f"Max Wood:       {max(wood_counts)}")
    print(f"Success Rate:   {sum(1 for w in wood_counts if w > 0) / num_episodes * 100:.1f}%")
    print(f"{'='*60}")

    env.close()

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'mean_wood': np.mean(wood_counts),
        'max_wood': max(wood_counts),
        'success_rate': sum(1 for w in wood_counts if w > 0) / num_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MineRL Tree-Chopping RL Agent")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['dqn', 'ppo'],
                        help='Algorithm type (dqn or ppo)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Max steps per episode (default: computed from config episode_seconds * 5)')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes')
    args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Override algorithm in config with command-line argument
    config['algorithm'] = args.algorithm

    # Compute max_steps for display
    max_steps = args.max_steps
    if max_steps is None:
        episode_seconds = config.get('environment', {}).get('episode_seconds', 25)
        max_steps = episode_seconds * 5

    print("=" * 60)
    print("MineRL Tree-Chopping RL Evaluation")
    print("=" * 60)
    print(f"Algorithm:  {args.algorithm.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes:   {args.episodes}")
    print(f"Max Steps:  {max_steps}")
    print(f"Render:     {args.render}")
    print(f"Device:     {config['device']}")
    print("=" * 60)

    # Run evaluation
    evaluate(config, args.checkpoint, args.episodes, args.render, args.max_steps)


if __name__ == "__main__":
    main()
