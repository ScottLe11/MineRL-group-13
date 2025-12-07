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
import glob
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Project imports
from utils.config import load_config
from utils.logger import Logger
from utils.env_factory import create_env
from utils.agent_factory import create_agent
from trainers.helpers import print_config_summary

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def get_sorted_checkpoints(checkpoint_dir, algorithm):
    """Returns a list of checkpoint paths sorted by modification time."""
    pattern = os.path.join(checkpoint_dir, f"checkpoint_{algorithm}_ep*.pt")
    files = glob.glob(pattern)
    if not files: return []
    files.sort(key=os.path.getctime, reverse=True)
    return files

def train(config: dict, render: bool = False, resume_checkpoint: str = None):
    """
    Main training entry point - provides common infrastructure and routes to algorithm.

    Args:
        config: Configuration dictionary
        render: If True, render the Minecraft window during training
        resume_checkpoint: Path to checkpoint file to resume from
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

    algorithm = config.get('algorithm', 'dqn')

    # Only load BC/DQfD checkpoint if running DQN (and not resuming from a checkpoint)
    if algorithm == 'dqn' and not resume_checkpoint:
        checkpoint_dir = config['training']['checkpoint_dir']

        # Try to load from bc_dqn or dqfd checkpoints (in order of preference)
        checkpoint_paths = [
            os.path.join(checkpoint_dir, "final_model_dqfd.pt"),   # Prefer DQfD (multi-objective)
            os.path.join(checkpoint_dir, "final_model_bc_dqn.pt"), # Then BC-DQN (simple)
            os.path.join(checkpoint_dir, "final_model_bc.pt"),     # Legacy name
        ]

        loaded = False
        for bc_checkpoint_path in checkpoint_paths:
            if os.path.exists(bc_checkpoint_path):
                print(f"\n🧠 Loading pre-trained weights: {bc_checkpoint_path}")

                # Load checkpoint data
                checkpoint = torch.load(bc_checkpoint_path, map_location=device, weights_only=False)

                # Load Q-Network weights from checkpoint
                if 'q_network_state_dict' in checkpoint:
                     # Load weights into the online Q-Network
                     agent.q_network.load_state_dict(checkpoint['q_network_state_dict'], strict=False)
                     
                     # Initialize target network with pre-trained weights too
                     agent.target_network.load_state_dict(checkpoint['q_network_state_dict'], strict=False)
                     
                     print("✅ Successfully loaded weights into Q-Network.")

                     # Load replay buffer if it exists (from bc_dqn with buffer pre-filling)
                     if 'replay_buffer' in checkpoint:
                         print("🔄 Restoring replay buffer with expert demonstrations...")
                         buffer_data = checkpoint['replay_buffer']
                        
                         # Restore buffer experiences
                         if hasattr(agent.replay_buffer, 'buffer'):
                             # Regular ReplayBuffer
                             agent.replay_buffer.buffer.extend(buffer_data)
                             agent.replay_buffer.position = len(agent.replay_buffer.buffer) % agent.replay_buffer.capacity
                         elif hasattr(agent.replay_buffer, 'add_experience'):
                             # PrioritizedReplayBuffer
                             for exp in buffer_data:
                                 agent.replay_buffer.add_experience(*exp)
                         print(f"✅ Restored {len(buffer_data)} expert transitions.")
                     else:
                         print("ℹ️  No replay buffer in checkpoint (normal for older bc_dqn)")

                     loaded = True
                     break
                else:
                     print(f"⚠️  Warning: Checkpoint found but 'q_network_state_dict' key is missing.")

        if not loaded:
             print("⚠️  Info: No pre-trained checkpoint found. Starting DQN from scratch.")
    
    # Load checkpoint if resuming (overrides BC weights)
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")
        print(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
        agent.load(resume_checkpoint)
        print("✅ Checkpoint loaded successfully\n")


    # Create logger
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
    elif algorithm == 'bc_dqn':
        from trainers.helpers import train_bc
        # BC with DQN - simple supervised learning (cross-entropy loss only)
        env = train_bc(config, env, agent, logger)
    elif algorithm == 'bc_ppo':
        from trainers.helpers import train_bc_ppo
        # BC with PPO - simple supervised learning (cross-entropy loss only)
        env = train_bc_ppo(config, env, agent, logger)
    elif algorithm == 'dqfd':
        from trainers.helpers import train_dqfd
        # DQfD - advanced imitation learning (multi-objective loss)
        env = train_dqfd(config, env, agent, logger)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'dqn', 'ppo', 'bc_dqn', 'bc_ppo', or 'dqfd'.")

    # Cleanup
    logger.close()
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Train MineRL Tree-Chopping RL Agent")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--render', action='store_true',
                        help='Render the Minecraft window during training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    algorithm = config.get('algorithm', 'dqn')
    checkpoint_dir = config['training']['checkpoint_dir']

    print("=" * 60)
    print("MineRL Tree-Chopping RL Training")
    print("=" * 60)
    print(f"Config: {args.config or 'config/config.yaml'}")
    print(f"Algorithm: {config.get('algorithm', 'dqn').upper()}")
    print(f"Device: {config['device']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Episode length: {config['environment']['episode_seconds']}s")
    print(f"Render: {args.render}")
    if args.resume:
        print(f"Resume: {args.resume}")
    print("=" * 60)
    
    failure_counts = {}
    
    candidate_checkpoint = args.resume 

    while True:
        try:
            train(config, render=args.render, resume_checkpoint=candidate_checkpoint)
            print("\nTraining finished successfully.")
            break 

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Exiting.")
            sys.exit(0)
            
        except Exception as e:
            # 2. CRASH HANDLING
            print("\nCRASH DETECTED. Printing traceback:")
            traceback.print_exc()

            if "comms.py" in str(e) and "recvall" in str(e):
                print("Fatal MineRL Comms Error. Exiting.")
                sys.exit(1)

            if candidate_checkpoint:
                current_fails = failure_counts.get(candidate_checkpoint, 0)
                failure_counts[candidate_checkpoint] = current_fails + 1
            
            print("\nWaiting 5 seconds before Auto-Resume Strategy...")
            time.sleep(5)
            
            checkpoints = get_sorted_checkpoints(checkpoint_dir, algorithm)
            
            if not checkpoints:
                print("No checkpoints found to resume from. Retrying fresh start...")
                candidate_checkpoint = None
            else:
                latest = checkpoints[0]
                
                if failure_counts.get(latest, 0) < 5:
                    candidate_checkpoint = latest
                    attempt_num = failure_counts.get(latest, 0) + 1
                    print(f"Auto-Resuming LATEST checkpoint ({attempt_num}/5): {os.path.basename(latest)}")
                
                elif len(checkpoints) > 1:
                    second_latest = checkpoints[1]
                    if failure_counts.get(second_latest, 0) < 10:
                        candidate_checkpoint = second_latest
                        attempt_num = failure_counts.get(second_latest, 0) + 1
                        print(f"Latest failed 5x. Fallback to 2nd LATEST ({attempt_num}/10): {os.path.basename(second_latest)}")
                    else:
                        print("\nCRITICAL FAILURE: Backups exhausted. Exiting.")
                        sys.exit(1)
                else:
                    print("\nCRITICAL FAILURE: Latest checkpoint failed 5 times and no older checkpoint exists.")
                    sys.exit(1)

if __name__ == "__main__":
    main()