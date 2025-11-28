"""
Shared training helper functions for DQN and PPO trainers.

This module contains common utilities used by both training algorithms:
- Environment reset with retry logic
- Configuration summary printing
- Episode statistics logging
- Checkpoint saving
"""

import os
import time
import socket
import numpy as np
import torch

from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from networks.cnn import get_architecture_info

# Constants
AGENT_STEPS_PER_SECOND = 5  # Each agent step = 4 frames at 20 ticks/sec = 0.2s


def safe_env_reset(env, max_retries: int = 3, retry_delay: float = 3.0, recreate_fn=None):
    """
    Safely reset environment with retry logic for MineRL socket timeouts.

    Args:
        env: The environment to reset
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries (doubles each retry)
        recreate_fn: Optional function to recreate environment on persistent failures

    Returns:
        obs: Initial observation from env.reset()

    Raises:
        Exception: If all retries fail and no recreate_fn provided

    Note:
        When recreate_fn is provided and used, the old env is closed and a new
        env is created. The caller should use the env from their scope, which
        will need to be reassigned if recreation occurs. Consider using the
        trainers' env_recreation_interval mechanism instead.
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            obs = env.reset()
            if attempt > 0:
                print(f"✅ Environment reset succeeded on attempt {attempt + 1}")
            return obs
        except (socket.timeout, TimeoutError, OSError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"⚠️  Environment reset failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                print(f"   Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Environment reset failed after {max_retries} attempts")
                if recreate_fn:
                    print("🔄 Attempting to recreate environment...")
                    print("⚠️  WARNING: Using recreate_fn from safe_env_reset is not recommended.")
                    print("   The caller's env reference will not be updated.")
                    print("   Consider lowering env_recreation_interval in config instead.")
                    raise last_exception
                else:
                    print("💡 HINT: Lower env_recreation_interval in config to prevent timeouts")
                    print("   Current interval triggers recreation every N episodes")
                    raise

    # Should never reach here, but just in case
    raise RuntimeError(f"Unexpected error in safe_env_reset: {last_exception}")


def print_config_summary(config: dict, agent, env_config: dict):
    """Print training configuration summary."""
    algorithm = config.get('algorithm', 'dqn')
    network_config = config['network']
    arch_name = network_config.get('architecture', 'small')
    arch_info = get_architecture_info().get(arch_name, {})
    curriculum = env_config.get('curriculum', {})
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps = episode_seconds * AGENT_STEPS_PER_SECOND

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
