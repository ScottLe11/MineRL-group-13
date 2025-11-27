#!/usr/bin/env python3
"""
Training script for the MineRL Tree-Chopping DQN Agent.

Usage:
    python scripts/train.py                    # Use default config
    python scripts/train.py --config path.yaml # Use custom config
    python scripts/train.py --render           # Show Minecraft window during training
    
Features:
    - Supports multiple CNN architectures (tiny, small, medium, wide, deep)
    - Prioritized Experience Replay (PER) support
    - Soft or hard target network updates
    - Curriculum learning with configurable starting conditions
"""

import argparse
import os
import sys
import random
import numpy as np
import time
import socket

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Project imports
from utils.config import load_config
from utils.logger import Logger
from utils.env_factory import create_env
from utils.agent_factory import create_agent
from agent.dqn import DQNAgent
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


def train(config: dict, render: bool = False):
    """
    Main training loop - trains for a fixed number of EPISODES.

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

    # Episode settings - compute max_steps from episode_seconds
    env_config = config['environment']
    episode_seconds = env_config.get('episode_seconds', 20)  # Default 20s for recon
    max_steps_per_episode = episode_seconds * 5  # 1 agent step = 4 frames = 200ms = 0.2s

    # Create agent using factory
    agent = create_agent(config, num_actions=env.action_space.n)
    
    # Create logger
    algorithm = config.get('algorithm', 'dqn')
    logger = Logger(
        log_dir=config['training']['log_dir'],
        experiment_name=f"treechop_{algorithm}_{config.get('seed', 'noseed')}"
    )

    # Training parameters (episode-based)
    num_episodes = config['training']['num_episodes']
    train_freq = config['training']['train_freq']
    log_freq = config['training']['log_freq']      # Episodes between logs
    save_freq = config['training']['save_freq']    # Episodes between saves

    # Get config for logging
    network_config = config['network']
    arch_name = network_config.get('architecture', 'small')
    arch_info = get_architecture_info().get(arch_name, {})
    curriculum = env_config.get('curriculum', {})

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Episodes: {num_episodes}")
    print(f"Episode length: {episode_seconds} seconds ({max_steps_per_episode} agent steps)")
    print(f"Network: {arch_name} ({arch_info.get('params', '?'):,} params)")

    if algorithm == 'dqn':
        dqn_config = config['dqn']
        per_config = dqn_config.get('prioritized_replay', {})
        target_config = dqn_config.get('target_update', {})
        use_per = per_config.get('enabled', False)
        target_method = target_config.get('method', 'soft')
        print(f"Replay buffer: {dqn_config['replay_buffer']['capacity']:,} capacity, {dqn_config['replay_buffer']['min_size']:,} min")
        print(f"PER: {'enabled (α={}, β={}->{})'.format(per_config.get('alpha', 0.6), per_config.get('beta_start', 0.4), per_config.get('beta_end', 1.0)) if use_per else 'disabled'}")
        print(f"Target update: {target_method}" + (f" (τ={target_config.get('tau', 0.005)})" if target_method == 'soft' else f" (every {target_config.get('hard_update_freq', 1000)} steps)"))

    print(f"Curriculum: with_logs={curriculum.get('with_logs', 0)}, with_axe={curriculum.get('with_axe', False)}")
    print(f"{'='*60}\n")
    
    # Training state
    global_step = 0
    best_avg_wood = 0
    recent_wood = []  # Track last 50 episodes

    # Environment recreation interval (helps prevent MineRL memory leaks)
    env_recreation_interval = config['training'].get('env_recreation_interval', 50)

    # =========================================================================
    # MAIN TRAINING LOOP (episode-based)
    # =========================================================================
    for episode in range(1, num_episodes + 1):
        # Periodically recreate environment to prevent memory leaks
        if episode > 1 and episode % env_recreation_interval == 0:
            print(f"\n🔄 Recreating environment at episode {episode} (prevent memory leaks)...")
            env.close()
            env = create_env(config)
            print("✓ Environment recreated\n")

        # Safe reset with retry logic for MineRL socket timeouts
        obs = safe_env_reset(env, max_retries=3, retry_delay=2.0)
        episode_reward = 0
        episode_wood = 0
        step_in_episode = 0

        # Run episode until done or max steps
        while step_in_episode < max_steps_per_episode:
            # Select action
            action = agent.select_action(obs, explore=True)

            # Take step (this executes 4 MineRL frames = 200ms)
            next_obs, reward, done, info = env.step(action)

            # Render if requested
            if render:
                env.render()

            # Store experience
            state = {
                'pov': obs['pov'],
                'time': float(obs['time_left'][0]) if hasattr(obs['time_left'], '__getitem__') else float(obs['time_left']),
                'yaw': float(obs['yaw'][0]) if hasattr(obs['yaw'], '__getitem__') else float(obs['yaw']),
                'pitch': float(obs['pitch'][0]) if hasattr(obs['pitch'], '__getitem__') else float(obs['pitch']),
            }
            next_state = {
                'pov': next_obs['pov'],
                'time': float(next_obs['time_left'][0]) if hasattr(next_obs['time_left'], '__getitem__') else float(next_obs['time_left']),
                'yaw': float(next_obs['yaw'][0]) if hasattr(next_obs['yaw'], '__getitem__') else float(next_obs['yaw']),
                'pitch': float(next_obs['pitch'][0]) if hasattr(next_obs['pitch'], '__getitem__') else float(next_obs['pitch']),
            }
            agent.store_experience(state, action, reward, next_state, done)

            # Train every train_freq steps
            if global_step % train_freq == 0 and agent.replay_buffer.is_ready():
                train_metrics = agent.train_step()
                
                # Log training metrics periodically
                if train_metrics and global_step % (train_freq * 100) == 0:
                    logger.log_training_step(
                        step=global_step,
                        loss=train_metrics.get('loss', 0),
                        q_mean=train_metrics.get('q_mean', 0),
                        td_error=train_metrics.get('td_error_mean', 0),
                        per_beta=train_metrics.get('per_beta', None)
                    )

            # Update counters
            # Macros take multiple frames - count them properly (frames / 4 = agent steps)
            if 'macro_steps' in info:
                # Macro action: macro_steps is number of MineRL frames
                steps_used = max(1, info['macro_steps'] // 4)  # Convert frames to agent steps
            else:
                # Primitive action: 1 agent step = 4 frames
                steps_used = 1

            obs = next_obs
            episode_reward += reward
            episode_wood = info.get('wood_count', 0)  # Current wood inventory (net: mining - using)
            step_in_episode += steps_used
            global_step += steps_used

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
            epsilon=agent.get_epsilon()
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
                  f"ε: {agent.get_epsilon():.3f} | "
                  f"Buffer: {len(agent.replay_buffer)}")

            # Print Q-values for top actions
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
                    # Show error if Q-value logging fails
                    print(f"  [Top Q-values] Error: {type(e).__name__}: {e}")

            # Print action statistics to diagnose exploration issues
            if hasattr(agent, 'get_action_stats'):
                stats = agent.get_action_stats()
                if stats:
                    print(f"  [Action Stats] Last 100: {stats['last_100_unique']}/{len(stats['last_100_actions'])} unique")

                    # Print top 3 most frequent actions WITH NAMES
                    top_actions = sorted(enumerate(stats['action_frequencies']), key=lambda x: x[1], reverse=True)[:3]

                    # Get action names from environment (handles ConfigurableActionWrapper mapping)
                    action_names = []
                    if hasattr(env, 'action_names'):
                        # ConfigurableActionWrapper has action_names
                        action_names = env.action_names
                    else:
                        # Fallback to default action names
                        from wrappers.actions import ACTION_NAMES_POOL
                        action_names = ACTION_NAMES_POOL[:agent.num_actions]

                    top_str = ", ".join([
                        f"{action_names[idx] if idx < len(action_names) else f'a{idx}'}:{freq*100:.1f}%"
                        for idx, freq in top_actions
                    ])
                    print(f"  [Top Actions] {top_str}")
        
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


def save_checkpoint(agent: DQNAgent, config: dict, episode: int, final: bool = False, best: bool = False, save_buffer: bool = True):
    """
    Save a training checkpoint.

    Args:
        agent: DQN agent to save
        config: Configuration dict
        episode: Current episode number
        final: Whether this is the final checkpoint
        best: Whether this is the best model so far
        save_buffer: Whether to save replay buffer (large file, ~100-500MB)
    """
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    if final:
        path = os.path.join(checkpoint_dir, "final_model.pt")
    elif best:
        path = os.path.join(checkpoint_dir, "best_model.pt")
    else:
        path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")

    # Build checkpoint dict
    checkpoint = {
        'episode': episode,
        'step_count': agent.step_count,
        'train_count': agent.train_count,
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        # Action tracking
        'action_counts': agent.action_counts,
        'last_actions': agent.last_actions,
    }

    # Optionally save replay buffer (can be large!)
    if save_buffer and len(agent.replay_buffer) > 0:
        checkpoint['replay_buffer'] = list(agent.replay_buffer.buffer)
        buffer_size_mb = len(agent.replay_buffer) * 84 * 84 * 4 * 4 / (1024 * 1024)  # Rough estimate
        print(f"💾 Saving with replay buffer (~{buffer_size_mb:.1f}MB)...")

    torch.save(checkpoint, path)
    print(f"💾 Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train MineRL Tree-Chopping DQN Agent")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--render', action='store_true',
                        help='Render the Minecraft window during training')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print("MineRL Tree-Chopping DQN Training")
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

