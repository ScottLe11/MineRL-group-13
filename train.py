#!/usr/bin/env python3
"""
Unified training script for MineRL Tree-Chopping RL Agents.

This script provides COMMON INFRASTRUCTURE for all algorithms:
- Environment setup and recreation
- Safe environment reset with retry logic
- Checkpoint saving (algorithm-agnostic)
- Episode statistics logging
- TensorBoard integration

Algorithm-specific training loops are in:
- trainers/train_dqn.py (DQN-specific logic)
- trainers/train_ppo.py (PPO-specific logic)

Usage:
    python train.py                    # Use default config
    python train.py --config path.yaml # Use custom config
    python train.py --render           # Show Minecraft window during training
"""

import argparse
import os
import sys
import random
import numpy as np
import time
import socket

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Project imports
from utils.config import load_config
from utils.logger import Logger
from utils.env_factory import create_env
from utils.agent_factory import create_agent
from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from networks.cnn import get_architecture_info


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def safe_env_reset(env, max_retries: int = 3, retry_delay: float = 2.0):
    """
    Safely reset environment with retry logic for MineRL socket timeouts.

    Args:
        env: The environment to reset
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries (doubles each retry)

    Returns:
        Initial observation from env.reset()

    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            obs = env.reset()
            return obs
        except (socket.timeout, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"⚠️  Environment reset failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                print(f"   Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Environment reset failed after {max_retries} attempts")
                raise

    # Should never reach here, but just in case
    raise RuntimeError("Unexpected error in safe_env_reset")


def print_config_summary(config: dict, agent, env_config: dict):
    """Print training configuration summary."""
    algorithm = config.get('algorithm', 'dqn')
    network_config = config['network']
    arch_name = network_config.get('architecture', 'small')
    arch_info = get_architecture_info().get(arch_name, {})
    curriculum = env_config.get('curriculum', {})
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps = episode_seconds * 5

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Episode length: {episode_seconds} seconds ({max_steps} agent steps)")
    print(f"Network: {arch_name} ({arch_info.get('params', '?'):,} params)")

    # Algorithm-specific config
    if algorithm == 'dqn':
        dqn_config = config['dqn']
        per_config = dqn_config.get('prioritized_replay', {})
        target_config = dqn_config.get('target_update', {})
        use_per = per_config.get('enabled', False)
        target_method = target_config.get('method', 'soft')
        print(f"Replay buffer: {dqn_config['replay_buffer']['capacity']:,} capacity, {dqn_config['replay_buffer']['min_size']:,} min")
        print(f"PER: {'enabled (α={}, β={}->{})'.format(per_config.get('alpha', 0.6), per_config.get('beta_start', 0.4), per_config.get('beta_end', 1.0)) if use_per else 'disabled'}")
        print(f"Target update: {target_method}" + (f" (τ={target_config.get('tau', 0.005)})" if target_method == 'soft' else f" (every {target_config.get('hard_update_freq', 1000)} steps)"))
    elif algorithm == 'ppo':
        ppo_config = config['ppo']
        print(f"Rollout size: {ppo_config['n_steps']} steps")
        print(f"Clip epsilon: {ppo_config['clip_epsilon']}")
        print(f"Entropy coef: {ppo_config['entropy_coef']}")

    print(f"Curriculum: with_logs={curriculum.get('with_logs', 0)}, with_axe={curriculum.get('with_axe', False)}")
    print(f"{'='*60}\n")


def log_episode_stats(episode: int, num_episodes: int, global_step: int,
                     episode_wood: int, recent_wood: list, agent,
                     env, obs: dict, log_freq: int):
    """Print episode statistics and Q-values/action stats."""
    if episode % log_freq != 0:
        return

    avg_wood = np.mean(recent_wood) if recent_wood else 0
    success_rate = sum(1 for w in recent_wood if w > 0) / len(recent_wood) * 100 if recent_wood else 0

    # Build base stats string
    stats_str = (f"Episode {episode}/{num_episodes} | "
                f"Steps: {global_step} | "
                f"Wood: {episode_wood} | "
                f"Avg(50): {avg_wood:.2f} | "
                f"Success: {success_rate:.0f}%")

    # Add epsilon or buffer info depending on algorithm
    if hasattr(agent, 'get_epsilon'):
        stats_str += f" | ε: {agent.get_epsilon():.3f}"

    if hasattr(agent, 'replay_buffer'):
        stats_str += f" | Buffer: {len(agent.replay_buffer)}"
    elif hasattr(agent, 'buffer'):
        stats_str += f" | Buffer: {len(agent.buffer.observations)}"

    print(stats_str)

    # Print Q-values for DQN agents
    if hasattr(agent, 'get_q_values'):
        try:
            q_values = agent.get_q_values(obs)

            # Get action names from environment
            action_names = []
            if hasattr(env, 'action_names'):
                action_names = env.action_names
            else:
                from wrappers.actions import ACTION_NAMES_POOL
                action_names = ACTION_NAMES_POOL[:agent.num_actions]

            # Get top 5 actions by Q-value
            top_indices = np.argsort(q_values)[-5:][::-1]  # Descending order
            top_q_str = ", ".join([
                f"{action_names[idx] if idx < len(action_names) else f'a{idx}'}:{q_values[idx]:.2f}"
                for idx in top_indices
            ])
            print(f"  [Top Q-values] {top_q_str}")
        except Exception as e:
            print(f"  [Top Q-values] Error: {type(e).__name__}: {e}")

    # Print action statistics
    if hasattr(agent, 'get_action_stats'):
        stats = agent.get_action_stats()
        if stats:
            print(f"  [Action Stats] Last 100: {stats['last_100_unique']}/{len(stats['last_100_actions'])} unique")

            # Print top 3 most frequent actions WITH NAMES
            top_actions = sorted(enumerate(stats['action_frequencies']), key=lambda x: x[1], reverse=True)[:3]

            # Get action names from environment
            action_names = []
            if hasattr(env, 'action_names'):
                action_names = env.action_names
            else:
                from wrappers.actions import ACTION_NAMES_POOL
                action_names = ACTION_NAMES_POOL[:agent.num_actions]

            top_str = ", ".join([
                f"{action_names[idx] if idx < len(action_names) else f'a{idx}'}:{freq*100:.1f}%"
                for idx, freq in top_actions
            ])
            print(f"  [Top Actions] {top_str}")


def save_checkpoint(agent, config: dict, episode: int, final: bool = False,
                   best: bool = False, save_buffer: bool = True):
    """
    Save a training checkpoint (algorithm-agnostic).

    Args:
        agent: Agent to save (DQN or PPO)
        config: Configuration dict
        episode: Current episode number
        final: Whether this is the final checkpoint
        best: Whether this is the best model so far
        save_buffer: Whether to save replay buffer (DQN only, large file)
    """
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    algorithm = config.get('algorithm', 'dqn')

    if final:
        path = os.path.join(checkpoint_dir, f"final_model_{algorithm}.pt")
    elif best:
        path = os.path.join(checkpoint_dir, f"best_model_{algorithm}.pt")
    else:
        path = os.path.join(checkpoint_dir, f"checkpoint_{algorithm}_ep{episode}.pt")

    # Build checkpoint dict based on algorithm
    if isinstance(agent, DQNAgent):
        checkpoint = {
            'episode': episode,
            'step_count': agent.step_count,
            'train_count': agent.train_count,
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'action_counts': agent.action_counts,
            'last_actions': agent.last_actions,
        }

        # Optionally save replay buffer (can be large!)
        if save_buffer and len(agent.replay_buffer) > 0:
            if hasattr(agent.replay_buffer, 'get_all_experiences'):
                # PrioritizedReplayBuffer
                checkpoint['replay_buffer'] = agent.replay_buffer.get_all_experiences()
            elif hasattr(agent.replay_buffer, 'buffer'):
                # Regular ReplayBuffer
                checkpoint['replay_buffer'] = list(agent.replay_buffer.buffer)
            else:
                print("⚠️  Warning: Unknown replay buffer type, skipping buffer save")
                save_buffer = False

            if save_buffer:
                buffer_size_mb = len(agent.replay_buffer) * 84 * 84 * 4 * 4 / (1024 * 1024)
                print(f"💾 Saving with replay buffer (~{buffer_size_mb:.1f}MB)...")

    elif isinstance(agent, PPOAgent):
        checkpoint = {
            'episode': episode,
            'step_count': agent.step_count,
            'update_count': agent.update_count,
            'policy_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
        }

        # Add action tracking if available
        if hasattr(agent, 'action_counts'):
            checkpoint['action_counts'] = agent.action_counts
        if hasattr(agent, 'last_actions'):
            checkpoint['last_actions'] = agent.last_actions
    else:
        raise ValueError(f"Unknown agent type: {type(agent)}")

    torch.save(checkpoint, path)
    print(f"💾 Saved: {path}")


def train(config: dict, render: bool = False):
    """
    Main training entry point - provides common infrastructure and routes to algorithm.

    Args:
        config: Configuration dictionary
        render: If True, render the Minecraft window during training
    """
    # Setup
    set_seed(config.get('seed'))
    device = config['device']
    print(f"Training on device: {device}")

    # Create environment
    env = create_env(config)
    print(f"Environment created: {config['environment']['name']}")
    print(f"Action space: {env.action_space}")

    # Create agent
    agent = create_agent(config, num_actions=env.action_space.n)

    # Create logger
    algorithm = config.get('algorithm', 'dqn')
    logger = Logger(
        log_dir=config['training']['log_dir'],
        experiment_name=f"treechop_{algorithm}_{config.get('seed', 'noseed')}"
    )

    # Print configuration summary
    print_config_summary(config, agent, config['environment'])

    # Route to algorithm-specific training loop
    if algorithm == 'dqn':
        from trainers.train_dqn import train_dqn
        env = train_dqn(config, env, agent, logger, render)
    elif algorithm == 'ppo':
        from trainers.train_ppo import train_ppo
        env = train_ppo(config, env, agent, logger, render)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'dqn' or 'ppo'.")

    # Cleanup
    logger.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train MineRL Tree-Chopping RL Agent")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--render', action='store_true',
                        help='Render the Minecraft window during training')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print("MineRL Tree-Chopping RL Training")
    print("=" * 60)
    print(f"Config: {args.config or 'config/config.yaml'}")
    print(f"Algorithm: {config.get('algorithm', 'dqn').upper()}")
    print(f"Device: {config['device']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Episode length: {config['environment']['episode_seconds']}s")
    print(f"Render: {args.render}")
    print("=" * 60)

    # Train
    train(config, render=args.render)


if __name__ == "__main__":
    main()
