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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gym
from gym.envs.registration import register

# Project imports
from utils.config import load_config
from utils.logger import Logger
from agent.dqn import DQNAgent
from networks.cnn import create_cnn, get_architecture_info
from wrappers.vision import StackAndProcessWrapper
from wrappers.observation import ObservationWrapper
from wrappers.hold_attack import HoldAttackWrapper
from wrappers.actions import ExtendedActionWrapper, ConfigurableActionWrapper
from wrappers.reward import RewardWrapper

# Import the custom treechop spec
from treechop_spec import Treechop, ConfigurableTreechop, handlers


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ============================================================================
# ENVIRONMENT SETUP (matches main.py)
# ============================================================================

# Global curriculum config - set before env registration
_CURRICULUM_CONFIG = {
    'with_logs': 0,
    'with_axe': False,
}


class CustomTreechop(Treechop):
    """Custom treechop environment with configurable starting conditions.
    
    Uses global _CURRICULUM_CONFIG since gym.register doesn't support arguments.
    """
    
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLcustom_treechop-v0'
        super().__init__(*args, **kwargs)

    def create_agent_start(self) -> list:
        """Override to use curriculum config instead of hardcoded values."""
        # Get base handlers from HumanControlEnvSpec (skip Treechop's hardcoded inventory)
        from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
        base_handlers = HumanControlEnvSpec.create_agent_start(self)
        
        # Build inventory from curriculum config
        inventory = []
        
        with_logs = _CURRICULUM_CONFIG.get('with_logs', 0)
        with_axe = _CURRICULUM_CONFIG.get('with_axe', False)
        
        if with_logs > 0:
            inventory.append(dict(type="oak_log", quantity=with_logs))
        
        if with_axe:
            inventory.append(dict(type="wooden_axe", quantity=1))
        
        if inventory:
            base_handlers.append(handlers.SimpleInventoryAgentStart(inventory))
        
        return base_handlers


def _make_base_treechop_env():
    """Factory function for gym.register."""
    spec = CustomTreechop(resolution=(640, 360))
    return spec.make()


# Register the custom environment (only once)
_ENV_REGISTERED = False


def _parse_action_space_config(action_config: dict) -> list:
    """
    Parse action space configuration and return list of enabled action indices.

    Args:
        action_config: Action space configuration dict from config.yaml

    Returns:
        List of action indices to enable (0-25)
    """
    preset = action_config.get('preset', 'base')

    if preset == 'base':
        # Base 23 actions (0-22)
        return list(range(23))
    elif preset == 'assisted':
        # Assisted learning preset: curated action set
        # Movement (0-6) + key camera angles + craft_entire_axe + extended attacks
        return [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 15, 17, 23, 24, 25]
    elif preset == 'custom':
        # Use custom enabled_actions list
        enabled = action_config.get('enabled_actions', [])
        if not enabled:
            print("Warning: preset='custom' but enabled_actions is empty. Defaulting to base 23 actions.")
            return list(range(23))
        return enabled
    else:
        print(f"Warning: Unknown action preset '{preset}'. Defaulting to base 23 actions.")
        return list(range(23))


def _ensure_env_registered():
    """Register custom env if not already registered."""
    global _ENV_REGISTERED
    if not _ENV_REGISTERED:
        try:
            register(
                id='MineRLcustom_treechop-v0',
                entry_point=_make_base_treechop_env,
                max_episode_steps=8000
            )
            _ENV_REGISTERED = True
        except Exception as e:
            # Already registered or other issue
            pass


def create_env(config: dict):
    """
    Create and wrap the MineRL environment.

    Wrapper order:
    1. Base MineRL env (CustomTreechop)
    2. StackAndProcessWrapper (frame processing)
    3. HoldAttackWrapper (attack duration)
    4. RewardWrapper (reward shaping)
    5. ObservationWrapper (add scalars)
    6. ConfigurableActionWrapper (discrete actions with configurable action space)

    Args:
        config: Configuration dictionary
    """
    global _CURRICULUM_CONFIG

    env_config = config['environment']
    reward_config = config.get('rewards', {})
    action_config = config.get('action_space', {})

    # Set curriculum config BEFORE creating env (used by CustomTreechop)
    curriculum = env_config.get('curriculum', {})
    _CURRICULUM_CONFIG = {
        'with_logs': curriculum.get('with_logs', 0),
        'with_axe': curriculum.get('with_axe', False),
    }
    print(f"Curriculum config: with_logs={_CURRICULUM_CONFIG['with_logs']}, with_axe={_CURRICULUM_CONFIG['with_axe']}")

    # Calculate max steps per episode (episode_seconds * 5 because 1 step = 4 frames = 200ms)
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps_per_episode = episode_seconds * 5

    # Parse action space configuration
    enabled_actions = _parse_action_space_config(action_config)
    print(f"Action space: {len(enabled_actions)} actions (preset: {action_config.get('preset', 'base')})")

    # Create real MineRL environment
    _ensure_env_registered()
    env = gym.make(env_config['name'])
    print(f"✓ Created MineRL environment: {env_config['name']}")

    # Apply wrappers in order (same as main.py but with additional wrappers)
    env = StackAndProcessWrapper(env, shape=tuple(env_config['frame_shape']))
    env = HoldAttackWrapper(
        env,
        hold_steps=35,
        lock_aim=True,
        pass_through_move=False,
        yaw_per_tick=0.0,
        fwd_jump_ticks=0
    )
    env = RewardWrapper(
        env,
        wood_value=reward_config.get('wood_value', 1.0),
        step_penalty=reward_config.get('step_penalty', -0.001)
    )
    env = ObservationWrapper(env, max_episode_steps=max_steps_per_episode)
    env = ConfigurableActionWrapper(env, enabled_actions=enabled_actions)

    return env


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
    
    # Update config with actual action space size
    config['dqn']['num_actions'] = env.action_space.n
    
    # Create agent with full configuration
    dqn_config = config['dqn']
    network_config = config['network']
    target_config = dqn_config.get('target_update', {})
    per_config = dqn_config.get('prioritized_replay', {})
    
    # Log network architecture choice
    arch_name = network_config.get('architecture', 'small')
    attention_type = network_config.get('attention', 'none')
    arch_info = get_architecture_info().get(arch_name, {})
    print(f"Network architecture: {arch_name} ({arch_info.get('params', 'unknown'):,} params)")
    print(f"Attention mechanism: {attention_type}")

    # Log PER and target update settings
    use_per = per_config.get('enabled', False)
    target_method = target_config.get('method', 'soft')
    print(f"Prioritized replay: {'enabled' if use_per else 'disabled'}")
    print(f"Target updates: {target_method}")

    agent = DQNAgent(
        num_actions=dqn_config['num_actions'],
        input_channels=network_config['input_channels'],
        num_scalars=3,  # time, yaw, pitch
        learning_rate=dqn_config['learning_rate'],
        gamma=dqn_config['gamma'],
        # Target update settings
        tau=target_config.get('tau', 0.005),
        target_update_method=target_method,
        hard_update_freq=target_config.get('hard_update_freq', 1000),
        # Exploration settings
        epsilon_start=dqn_config['exploration']['epsilon_start'],
        epsilon_end=dqn_config['exploration']['epsilon_end'],
        epsilon_decay_steps=dqn_config['exploration']['epsilon_decay_steps'],
        # Replay buffer settings
        buffer_capacity=dqn_config['replay_buffer']['capacity'],
        buffer_min_size=dqn_config['replay_buffer']['min_size'],
        batch_size=dqn_config['batch_size'],
        # Prioritized Experience Replay
        use_per=use_per,
        per_alpha=per_config.get('alpha', 0.6),
        per_beta_start=per_config.get('beta_start', 0.4),
        per_beta_end=per_config.get('beta_end', 1.0),
        # Network architecture
        cnn_architecture=arch_name,
        attention_type=attention_type,
        device=config['device']
    )
    
    # Create logger
    logger = Logger(
        log_dir=config['training']['log_dir'],
        experiment_name=f"treechop_dqn_{config.get('seed', 'noseed')}"
    )
    
    # Training parameters (episode-based)
    num_episodes = config['training']['num_episodes']
    train_freq = config['training']['train_freq']
    log_freq = config['training']['log_freq']      # Episodes between logs
    save_freq = config['training']['save_freq']    # Episodes between saves
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Episode length: {episode_seconds} seconds ({max_steps_per_episode} agent steps)")
    print(f"Network: {arch_name} ({arch_info.get('params', '?'):,} params)")
    print(f"Replay buffer: {dqn_config['replay_buffer']['capacity']:,} capacity, {dqn_config['replay_buffer']['min_size']:,} min")
    print(f"PER: {'enabled (α={}, β={}->{})'.format(per_config.get('alpha', 0.6), per_config.get('beta_start', 0.4), per_config.get('beta_end', 1.0)) if use_per else 'disabled'}")
    print(f"Target update: {target_method}" + (f" (τ={target_config.get('tau', 0.005)})" if target_method == 'soft' else f" (every {target_config.get('hard_update_freq', 1000)} steps)"))
    print(f"Curriculum: with_logs={_CURRICULUM_CONFIG['with_logs']}, with_axe={_CURRICULUM_CONFIG['with_axe']}")
    print(f"{'='*60}\n")
    
    # Training state
    global_step = 0
    best_avg_wood = 0
    recent_wood = []  # Track last 50 episodes
    
    # =========================================================================
    # MAIN TRAINING LOOP (episode-based)
    # =========================================================================
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
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
            episode_wood += info.get('wood_this_frame', 0)
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


def save_checkpoint(agent: DQNAgent, config: dict, episode: int, final: bool = False, best: bool = False):
    """Save a training checkpoint."""
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if final:
        path = os.path.join(checkpoint_dir, "final_model.pt")
    elif best:
        path = os.path.join(checkpoint_dir, "best_model.pt")
    else:
        path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
    
    torch.save({
        'episode': episode,
        'step_count': agent.step_count,
        'train_count': agent.train_count,
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, path)
    
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

