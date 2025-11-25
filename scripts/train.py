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
from wrappers.actions import SimpleActionWrapper


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
    4. ObservationWrapper (add scalars)
    5. SimpleActionWrapper (discrete actions)
    """
    env_config = config['environment']
    
    # Create base environment
    # Note: MineRL env name may need adjustment based on your setup
    try:
        import minerl
        import main
        env = gym.make(env_config['name'])
    except Exception as e:
        print(f"Warning: Could not create MineRL env '{env_config['name']}': {e}")
        print("Creating a mock environment for testing...")
        env = create_mock_env(env_config)
        return env
    
    # Apply wrappers in order
    env = StackAndProcessWrapper(env, shape=tuple(env_config['frame_shape']))
    env = HoldAttackWrapper(env)
    env = ObservationWrapper(env, max_steps=env_config['max_steps'])
    env = SimpleActionWrapper(env)
    
    return env


def create_mock_env(env_config: dict):
    """Create a mock environment for testing without MineRL installed."""
    
    class MockMineRLEnv:
        """Mock environment that mimics the wrapped MineRL interface."""
        
        def __init__(self, config):
            self.max_steps = config['max_steps']
            self.frame_shape = tuple(config['frame_shape'])
            self.current_step = 0
            
            # Observation space matches ObservationWrapper output
            self.observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(0, 255, (4, *self.frame_shape), dtype=np.uint8),
                'time': gym.spaces.Box(0, 1, (1,), dtype=np.float32),
                'yaw': gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
                'pitch': gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
            })
            
            # Action space: 8 actions from SimpleActionWrapper (until extended)
            self.action_space = gym.spaces.Discrete(8)
        
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
            
            # Mock reward: small step penalty, occasional positive for "wood"
            reward = -0.001
            if random.random() < 0.01:  # 1% chance of "collecting wood"
                reward += 1.0
            
            done = self.current_step >= self.max_steps
            info = {'wood_collected': 1 if reward > 0 else 0}
            
            return obs, reward, done, info
        
        def close(self):
            pass
    
    return MockMineRLEnv(env_config)

def prefill_replay_buffer(agent: DQNAgent, npz_path: str = "bc_expert_data.npz"):
    try:
        # 1. Load the single, compressed NPZ file with all four arrays
        data = np.load(npz_path, allow_pickle=True)
        observations = data['obs']
        actions = data['actions']
        rewards = data['rewards']
        dones = data['dones']
        
    except FileNotFoundError:
        print(f"[PRE-FILL] WARNING: NPZ file not found at {npz_path}. Skipping pre-fill.")
        return
    except KeyError:
        print(f"[PRE-FILL] ERROR: NPZ file is missing 'rewards' or 'dones' key. Re-run pkl_parser.py.")
        return
    
    num_transitions = observations.shape[0]
    for i in range(num_transitions - 1): # Loop up to second-to-last item for transitions
        
        # 2. Complete the state dictionary with required scalar features (zeroed for recorded data)
        state_i = {
            'pov': observations[i], 
            'time': 0.0, 'yaw': 0.0, 'pitch': 0.0,
        }
        # The next state is the observation from the next index
        next_state_i = {
            'pov': observations[i+1],
            'time': 0.0, 'yaw': 0.0, 'pitch': 0.0,
        }
        
        # 3. Store the complete experience
        agent.store_experience(
            state=state_i, 
            action=actions[i].item(),  # .item() to ensure integer type
            reward=rewards[i].item(), 
            next_state=next_state_i, 
            done=dones[i].item()
        )
    
    print(f"[PRE-FILL] Successfully stored {num_transitions - 1} expert transitions.")
    if agent.replay_buffer.is_ready():
        print(f"[PRE-FILL] Replay buffer is ready for training. Size: {len(agent.replay_buffer)}")
    else:
        print(f"[PRE-FILL] WARNING: Buffer size ({len(agent.replay_buffer)}) is less than min_size ({agent.replay_buffer.min_size}).")
    pass

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
    agent = DQNAgent(
        num_actions=config['dqn']['num_actions'], # (Currently 8, but will be 23 later)
        input_channels=config['network']['input_channels'], # 4
        
        # Learning parameters
        learning_rate=config['dqn']['learning_rate'],
        gamma=config['dqn']['gamma'],
        
        # Target Network parameters
        tau=config['dqn']['target_update']['tau'],
        
        # Exploration parameters
        epsilon_start=config['dqn']['exploration']['epsilon_start'],
        epsilon_end=config['dqn']['exploration']['epsilon_end'],
        epsilon_decay_steps=config['dqn']['exploration']['epsilon_decay_steps'],
        
        # Replay Buffer parameters (nested dicts)
        buffer_capacity=config['dqn']['replay_buffer']['capacity'],
        buffer_min_size=config['dqn']['replay_buffer']['min_size'],
        batch_size=config['dqn']['batch_size'],
        
        # Device setup
        device=device
    )
    
    prefill_replay_buffer(agent, npz_path="bc_expert_data.npz")
    buffer_len = len(agent.replay_buffer)
    min_size = config['dqn']['replay_buffer']['min_size']
    print("\n--- TRAIN START VALIDATION ---")
    print(f" Buffer Length: {buffer_len} / {min_size}")
    print(f" Buffer is Ready: {agent.replay_buffer.is_ready()}")
    
    if buffer_len >= min_size:
        print("Pipeline is verified! Starting main training loop.")
    else:
        print("ERROR: Buffer size is insufficient. Check data.")
        # CRITICAL: We exit here to skip the actual long training run
        # This closes the environment processes created by SubprocVecEnv
        env.close() 
        sys.exit(0) # Exit the script here
    
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

