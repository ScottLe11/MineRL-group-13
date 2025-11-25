#!/usr/bin/env python3
"""
Training script for the MineRL Tree-Chopping DQN Agent.

Usage:
    python scripts/train.py                    # Use default config
    python scripts/train.py --config path.yaml # Use custom config
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
from wrappers.vision import StackAndProcessWrapper
from wrappers.observation import ObservationWrapper
from wrappers.hold_attack import HoldAttackWrapper
from wrappers.actions import ExtendedActionWrapper
from wrappers.reward import RewardWrapper

# Import the custom treechop spec from main.py
from treechop_spec import Treechop, handlers


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


def create_env(config: dict, use_mock: bool = False):
    """
    Create and wrap the MineRL environment.
    
    Wrapper order:
    1. Base MineRL env (CustomTreechop)
    2. StackAndProcessWrapper (frame processing)
    3. HoldAttackWrapper (attack duration)
    4. RewardWrapper (reward shaping)
    5. ObservationWrapper (add scalars)
    6. ExtendedActionWrapper (discrete actions)
    
    Args:
        config: Configuration dictionary
        use_mock: If True, force use of mock environment for testing
    """
    global _CURRICULUM_CONFIG
    
    env_config = config['environment']
    reward_config = config.get('rewards', {})
    
    # Set curriculum config BEFORE creating env (used by CustomTreechop)
    curriculum = env_config.get('curriculum', {})
    _CURRICULUM_CONFIG = {
        'with_logs': curriculum.get('with_logs', 0),
        'with_axe': curriculum.get('with_axe', False),
    }
    print(f"Curriculum config: with_logs={_CURRICULUM_CONFIG['with_logs']}, with_axe={_CURRICULUM_CONFIG['with_axe']}")
    
    if use_mock:
        print("Using mock environment (use_mock=True)")
        return create_mock_env(env_config, reward_config)
    
    # Try to create real MineRL environment
    try:
        _ensure_env_registered()
        env = gym.make(env_config['name'])
        print(f"Created MineRL environment: {env_config['name']}")
    except Exception as e:
        print(f"Warning: Could not create MineRL env '{env_config['name']}': {e}")
        print("Falling back to mock environment...")
        return create_mock_env(env_config, reward_config)
    
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
    # Compute max_steps from episode_seconds (1 step = 4 frames = 200ms)
    max_steps = env_config.get('episode_seconds', 20) * 5
    env = ObservationWrapper(env, max_steps=max_steps)
    env = ExtendedActionWrapper(env)
    
    return env


def create_mock_env(env_config: dict, reward_config: dict = None):
    """Create a mock environment for testing without MineRL installed."""
    
    if reward_config is None:
        reward_config = {'wood_value': 1.0, 'step_penalty': -0.001}
    
    class MockMineRLEnv:
        """Mock environment that mimics the wrapped MineRL interface."""
        
        def __init__(self, config, rewards):
            # Compute max_steps from episode_seconds (1 step = 4 frames = 200ms)
            self.max_steps = config.get('episode_seconds', 20) * 5
            self.frame_shape = tuple(config['frame_shape'])
            self.wood_value = rewards.get('wood_value', 1.0)
            self.step_penalty = rewards.get('step_penalty', -0.001)
            self.current_step = 0
            
            # Observation space matches ObservationWrapper output
            self.observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(0, 255, (4, *self.frame_shape), dtype=np.uint8),
                'time': gym.spaces.Box(0, 1, (1,), dtype=np.float32),
                'yaw': gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
                'pitch': gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
            })
            
            # Action space: 23 actions from ExtendedActionWrapper
            self.action_space = gym.spaces.Discrete(23)
        
        def reset(self):
            self.current_step = 0
            return {
                'pov': np.zeros((4, *self.frame_shape), dtype=np.uint8),
                'time': np.array([1.0], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
            }
        
        def step(self, action):
            self.current_step += 1
            
            # Mock observation
            obs = {
                'pov': np.random.randint(0, 256, (4, *self.frame_shape), dtype=np.uint8),
                'time': np.array([max(0, (self.max_steps - self.current_step) / self.max_steps)], dtype=np.float32),
                'yaw': np.array([0.0], dtype=np.float32),
                'pitch': np.array([0.0], dtype=np.float32),
            }
            
            # Mock reward: log × wood_value + step penalty
            logs = 1 if random.random() < 0.01 else 0  # 1% chance
            reward = (logs * self.wood_value) + self.step_penalty
            
            done = self.current_step >= self.max_steps
            info = {'wood_this_frame': logs}
            
            return obs, reward, done, info
        
        def close(self):
            pass
    
    return MockMineRLEnv(env_config, reward_config)


def train(config: dict, use_mock: bool = False):
    """
    Main training loop - trains for a fixed number of EPISODES.
    
    Args:
        config: Configuration dictionary.
        use_mock: If True, use mock environment instead of real MineRL.
    """
    # Setup
    set_seed(config.get('seed'))
    device = config['device']
    print(f"Training on device: {device}")
    
    # Create environment
    env = create_env(config, use_mock=use_mock)
    print(f"Environment created: {config['environment']['name']}")
    print(f"Action space: {env.action_space}")
    
    # Episode settings - compute max_steps from episode_seconds
    env_config = config['environment']
    episode_seconds = env_config.get('episode_seconds', 20)  # Default 20s for recon
    max_steps_per_episode = episode_seconds * 5  # 1 agent step = 4 frames = 200ms = 0.2s
    
    # Update config with actual action space size
    config['dqn']['num_actions'] = env.action_space.n
    
    # Create agent
    dqn_config = config['dqn']
    agent = DQNAgent(
        num_actions=dqn_config['num_actions'],
        input_channels=config['network']['input_channels'],
        num_scalars=3,  # time, yaw, pitch
        learning_rate=dqn_config['learning_rate'],
        gamma=dqn_config['gamma'],
        tau=dqn_config['target_update']['tau'],
        epsilon_start=dqn_config['exploration']['epsilon_start'],
        epsilon_end=dqn_config['exploration']['epsilon_end'],
        epsilon_decay_steps=dqn_config['exploration']['epsilon_decay_steps'],
        buffer_capacity=dqn_config['replay_buffer']['capacity'],
        buffer_min_size=dqn_config['replay_buffer']['min_size'],
        batch_size=dqn_config['batch_size'],
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
    print(f"Replay buffer min: {dqn_config['replay_buffer']['min_size']}")
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
            
            # Store experience
            state = {
                'pov': obs['pov'],
                'time': float(obs['time'][0]) if hasattr(obs['time'], '__getitem__') else float(obs['time']),
                'yaw': float(obs['yaw'][0]) if hasattr(obs['yaw'], '__getitem__') else float(obs['yaw']),
                'pitch': float(obs['pitch'][0]) if hasattr(obs['pitch'], '__getitem__') else float(obs['pitch']),
            }
            next_state = {
                'pov': next_obs['pov'],
                'time': float(next_obs['time'][0]) if hasattr(next_obs['time'], '__getitem__') else float(next_obs['time']),
                'yaw': float(next_obs['yaw'][0]) if hasattr(next_obs['yaw'], '__getitem__') else float(next_obs['yaw']),
                'pitch': float(next_obs['pitch'][0]) if hasattr(next_obs['pitch'], '__getitem__') else float(next_obs['pitch']),
            }
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train every train_freq steps
            if global_step % train_freq == 0 and agent.replay_buffer.is_ready():
                agent.train_step()
            
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
    parser.add_argument('--mock', action='store_true',
                        help='Use mock environment instead of real MineRL (for testing)')
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
    print(f"Environment: {'MOCK' if args.mock else 'MineRL'}")
    print("=" * 60)
    
    # Train
    train(config, use_mock=args.mock)


if __name__ == "__main__":
    main()

