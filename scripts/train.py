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

# Project imports
from utils.config import load_config
from utils.logger import Logger
from agent.dqn import DQNAgent
from wrappers.vision import StackAndProcessWrapper
from wrappers.observation import ObservationWrapper
from wrappers.hold_attack import HoldAttackWrapper
from wrappers.actions import ExtendedActionWrapper
from wrappers.reward import RewardWrapper


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def create_env(config: dict):
    """
    Create and wrap the MineRL environment.
    
    Wrapper order:
    1. Base MineRL env
    2. StackAndProcessWrapper (frame processing)
    3. HoldAttackWrapper (attack duration)
    4. RewardWrapper (reward shaping)
    5. ObservationWrapper (add scalars)
    6. ExtendedActionWrapper (discrete actions)
    """
    env_config = config['environment']
    reward_config = config.get('rewards', {})
    
    # Create base environment
    # Note: MineRL env name may need adjustment based on your setup
    try:
        import minerl
        env = gym.make(env_config['name'])
    except Exception as e:
        print(f"Warning: Could not create MineRL env '{env_config['name']}': {e}")
        print("Creating a mock environment for testing...")
        env = create_mock_env(env_config, reward_config)
        return env
    
    # Apply wrappers in order
    env = StackAndProcessWrapper(env, shape=tuple(env_config['frame_shape']))
    env = HoldAttackWrapper(env)
    env = RewardWrapper(
        env,
        step_penalty=reward_config.get('step_penalty', -0.001),
        wood_reward_scale=reward_config.get('wood_collected', 1.0)
    )
    env = ObservationWrapper(env, max_steps=env_config['max_steps'])
    env = ExtendedActionWrapper(env)
    
    return env


def create_mock_env(env_config: dict, reward_config: dict = None):
    """Create a mock environment for testing without MineRL installed."""
    
    if reward_config is None:
        reward_config = {'step_penalty': -0.001, 'wood_collected': 1.0}
    
    class MockMineRLEnv:
        """Mock environment that mimics the wrapped MineRL interface."""
        
        def __init__(self, config, rewards):
            self.max_steps = config['max_steps']
            self.frame_shape = tuple(config['frame_shape'])
            self.step_penalty = rewards.get('step_penalty', -0.001)
            self.wood_scale = rewards.get('wood_collected', 1.0)
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
            
            # Mock reward: step penalty + occasional wood collection
            wood_collected = 1 if random.random() < 0.01 else 0  # 1% chance
            reward = self.step_penalty + (wood_collected * self.wood_scale)
            
            done = self.current_step >= self.max_steps
            info = {
                'wood_collected': wood_collected,
                'raw_reward': wood_collected * self.wood_scale,
                'shaped_reward': reward,
            }
            
            return obs, reward, done, info
        
        def close(self):
            pass
    
    return MockMineRLEnv(env_config, reward_config)


def train(config: dict):
    """
    Main training loop.
    
    Args:
        config: Configuration dictionary.
    """
    # Setup
    set_seed(config.get('seed'))
    device = config['device']
    print(f"Training on device: {device}")
    
    # Create environment
    env = create_env(config)
    print(f"Environment created: {config['environment']['name']}")
    print(f"Action space: {env.action_space}")
    
    # Update config with actual action space size
    config['dqn']['num_actions'] = env.action_space.n
    
    # Create agent
    agent = DQNAgent(config)
    
    # Create logger
    logger = Logger(
        log_dir=config['training']['log_dir'],
        experiment_name=f"treechop_dqn_{config.get('seed', 'noseed')}"
    )
    
    # Training parameters
    total_steps = config['training']['total_steps']
    train_freq = config['training']['train_freq']
    log_freq = config['training']['log_freq']
    save_freq = config['training']['save_freq']
    eval_freq = config['training']['eval_freq']
    
    # Training state
    global_step = 0
    episode = 0
    best_eval_reward = float('-inf')
    
    print(f"\nStarting training for {total_steps} steps...")
    print(f"Replay buffer min size: {config['dqn']['replay_buffer']['min_size']}")
    
    while global_step < total_steps:
        episode += 1
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_wood = 0
        done = False
        
        while not done and global_step < total_steps:
            # Select action
            action = agent.select_action(obs, global_step)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            # Convert observation format for replay buffer
            state = {
                'pov': obs['pov'],
                'time': float(obs['time'][0]),
                'yaw': float(obs['yaw'][0]),
                'pitch': float(obs['pitch'][0]),
            }
            next_state = {
                'pov': next_obs['pov'],
                'time': float(next_obs['time'][0]),
                'yaw': float(next_obs['yaw'][0]),
                'pitch': float(next_obs['pitch'][0]),
            }
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train
            if global_step % train_freq == 0 and agent.replay_buffer.is_ready():
                metrics = agent.train_step()
                
                if global_step % log_freq == 0 and metrics:
                    logger.log_training(
                        loss=metrics['loss'],
                        q_mean=metrics['q_mean'],
                        q_std=metrics.get('q_std'),
                        step=global_step
                    )
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            episode_wood += info.get('wood_collected', 0)
            global_step += 1
            logger.set_step(global_step)
            
            # Progress logging
            if global_step % log_freq == 0:
                epsilon = agent.epsilon_schedule.get_epsilon(global_step)
                buffer_size = len(agent.replay_buffer)
                print(f"Step {global_step}/{total_steps} | "
                      f"Buffer: {buffer_size} | "
                      f"Epsilon: {epsilon:.3f}")
            
            # Save checkpoint
            if global_step % save_freq == 0:
                save_checkpoint(agent, config, global_step)
        
        # Log episode
        epsilon = agent.epsilon_schedule.get_epsilon(global_step)
        logger.log_episode(
            episode_reward=episode_reward,
            episode_length=episode_length,
            wood_collected=episode_wood,
            epsilon=epsilon
        )
    
    # Final save
    save_checkpoint(agent, config, global_step, final=True)
    logger.close()
    env.close()
    
    print(f"\nTraining complete! Total steps: {global_step}")


def save_checkpoint(agent: DQNAgent, config: dict, step: int, final: bool = False):
    """Save a training checkpoint."""
    checkpoint_dir = config['training']['checkpoint_dir']
    
    if final:
        path = os.path.join(checkpoint_dir, "final_model.pt")
    else:
        path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    
    torch.save({
        'step': step,
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, path)
    
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train MineRL Tree-Chopping DQN Agent")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/config.yaml)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=" * 60)
    print("MineRL Tree-Chopping DQN Training")
    print("=" * 60)
    print(f"Config: {args.config or 'config/config.yaml'}")
    print(f"Device: {config['device']}")
    print(f"Total steps: {config['training']['total_steps']}")
    print("=" * 60)
    
    # Train
    train(config)


if __name__ == "__main__":
    main()

