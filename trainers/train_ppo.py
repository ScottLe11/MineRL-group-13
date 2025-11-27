#!/usr/bin/env python3
"""
PPO Training script for the MineRL Tree-Chopping Agent.

Usage:
    python train_ppo.py                    # Use default config
    python train_ppo.py --config path.yaml # Use custom config
    python train_ppo.py --render           # Show Minecraft window during training

PPO (Proximal Policy Optimization) uses:
- Rollout collection (not experience replay)
- Advantage estimation with GAE
- Policy gradient updates with clipping
- Shared actor-critic network
"""

import argparse
import os
import sys
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Project imports
from utils.config import load_config
from utils.logger import Logger
from utils.env_factory import create_env
from utils.agent_factory import create_agent
from networks.cnn import get_architecture_info


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def train(config: dict, render: bool = False):
    """
    Main PPO training loop.

    Args:
        config: Configuration dictionary.
        render: If True, render the Minecraft window during training.
    """
    # Setup
    set_seed(config.get('seed'))
    device = config['device']
    print(f"Training on device: {device}")

    # Create environment
    env = create_env(config)
    print(f"Environment created: {config['environment']['name']}")
    print(f"Action space: {env.action_space}")

    # Episode settings
    env_config = config['environment']
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps_per_episode = episode_seconds * 5

    # Create PPO agent
    agent = create_agent(config, num_actions=env.action_space.n)

    # Create logger
    logger = Logger(
        log_dir=config['training']['log_dir'],
        experiment_name=f"treechop_ppo_{config.get('seed', 'noseed')}"
    )

    # Training parameters
    num_episodes = config['training']['num_episodes']
    log_freq = config['training']['log_freq']
    save_freq = config['training']['save_freq']

    # PPO specific
    ppo_config = config['ppo']
    n_steps = ppo_config['n_steps']  # Steps to collect before update

    # Get config for logging
    network_config = config['network']
    arch_name = network_config.get('architecture', 'small')
    arch_info = get_architecture_info().get(arch_name, {})
    curriculum = env_config.get('curriculum', {})

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Algorithm: PPO")
    print(f"Episodes: {num_episodes}")
    print(f"Episode length: {episode_seconds} seconds ({max_steps_per_episode} agent steps)")
    print(f"Network: {arch_name} ({arch_info.get('params', '?'):,} params)")
    print(f"Rollout size: {n_steps} steps")
    print(f"Clip epsilon: {ppo_config['clip_epsilon']}")
    print(f"Entropy coef: {ppo_config['entropy_coef']}")
    print(f"Curriculum: with_logs={curriculum.get('with_logs', 0)}, with_axe={curriculum.get('with_axe', False)}")
    print(f"{'='*60}\n")

    # Training state
    global_step = 0
    best_avg_wood = 0
    recent_wood = []  # Track last 50 episodes
    episode = 0

    # =========================================================================
    # MAIN PPO TRAINING LOOP (rollout-based)
    # =========================================================================
    while episode < num_episodes:
        obs = env.reset()
        episode_reward = 0
        episode_wood = 0
        step_in_episode = 0
        episode += 1

        # Run episode and collect rollout
        while step_in_episode < max_steps_per_episode:
            # Select action (PPO samples from policy)
            action, log_prob, value = agent.select_action(obs)

            # Take step
            next_obs, reward, done, info = env.step(action)

            # Render if requested
            if render:
                env.render()

            # Store transition in rollout buffer
            agent.store_transition(obs, action, log_prob, reward, value, done)

            # Update counters
            if 'macro_steps' in info:
                steps_used = max(1, info['macro_steps'] // 4)
            else:
                steps_used = 1

            obs = next_obs
            episode_reward += reward
            episode_wood = info.get('wood_count', 0)
            step_in_episode += steps_used
            global_step += steps_used

            # PPO update when rollout buffer is full
            if len(agent.buffer.observations) >= n_steps:
                # Compute last value for GAE
                _, _, last_value = agent.select_action(obs)

                # Update policy
                update_metrics = agent.update(last_value)

                # Log update metrics
                if update_metrics:
                    logger.log_training_step(
                        step=global_step,
                        loss=update_metrics.get('total_loss', 0),
                        policy_loss=update_metrics.get('policy_loss', 0),
                        value_loss=update_metrics.get('value_loss', 0),
                        entropy=update_metrics.get('entropy', 0)
                    )

            if done:
                break

        # Track wood for success rate
        recent_wood.append(episode_wood)
        if len(recent_wood) > 50:
            recent_wood.pop(0)

        # Log episode to TensorBoard
        logger.log_episode(
            episode_reward=episode_reward,
            episode_length=step_in_episode,
            wood_collected=episode_wood,
            epsilon=0.0  # PPO doesn't use epsilon
        )

        # Console logging every log_freq episodes
        if episode % log_freq == 0:
            avg_wood = np.mean(recent_wood) if recent_wood else 0
            success_rate = sum(1 for w in recent_wood if w > 0) / len(recent_wood) * 100 if recent_wood else 0
            print(f"Episode {episode}/{num_episodes} | "
                  f"Steps: {global_step} | "
                  f"Wood: {episode_wood} | "
                  f"Avg(50): {avg_wood:.2f} | "
                  f"Success: {success_rate:.0f}% | "
                  f"Buffer: {len(agent.buffer.observations)}")

        # Save checkpoint every save_freq episodes
        if episode % save_freq == 0:
            save_checkpoint(agent, config, episode)

            # Save best model
            avg_wood = np.mean(recent_wood) if recent_wood else 0
            if avg_wood > best_avg_wood:
                best_avg_wood = avg_wood
                save_checkpoint(agent, config, episode, best=True)

    # Final save
    save_checkpoint(agent, config, num_episodes, final=True)
    logger.close()
    env.close()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {global_step}")
    print(f"Best avg wood (50 ep): {best_avg_wood:.2f}")
    print(f"{'='*60}")


def save_checkpoint(agent, config: dict, episode: int, final: bool = False, best: bool = False):
    """Save a training checkpoint."""
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    if final:
        path = os.path.join(checkpoint_dir, "final_model_ppo.pt")
    elif best:
        path = os.path.join(checkpoint_dir, "best_model_ppo.pt")
    else:
        path = os.path.join(checkpoint_dir, f"checkpoint_ppo_ep{episode}.pt")

    torch.save({
        'episode': episode,
        'step_count': agent.step_count,
        'update_count': agent.update_count,
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, path)

    print(f"💾 Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train MineRL Tree-Chopping PPO Agent")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--render', action='store_true',
                        help='Render the Minecraft window during training')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Ensure PPO algorithm is selected
    if config.get('algorithm', '').lower() != 'ppo':
        print("Warning: Config algorithm is not 'ppo'. Setting to 'ppo'.")
        config['algorithm'] = 'ppo'

    print("=" * 60)
    print("MineRL Tree-Chopping PPO Training")
    print("=" * 60)
    print(f"Config: {args.config or 'config/config.yaml'}")
    print(f"Device: {config['device']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Episode length: {config['environment']['episode_seconds']}s")
    print(f"Render: {args.render}")
    print("=" * 60)

    # Train
    train(config, render=args.render)


if __name__ == "__main__":
    main()
