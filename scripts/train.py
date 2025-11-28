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

    # Only load BC checkpoint if running DQN (and not resuming from a checkpoint)
    if algorithm == 'dqn' and not resume_checkpoint:
        checkpoint_dir = config['training']['checkpoint_dir']
        bc_checkpoint_path = os.path.join(checkpoint_dir, "final_model_bc.pt")

        if os.path.exists(bc_checkpoint_path):
            print(f"\n🧠 Loading BC pre-trained weights: {bc_checkpoint_path}")

            # Load checkpoint data
            checkpoint = torch.load(bc_checkpoint_path, map_location=device)

            # Load Q-Network weights from BC checkpoint
            if 'q_network_state_dict' in checkpoint:
                 # Load weights into the online Q-Network
                 agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])

                 # Initialize target network with pre-trained weights too
                 agent.target_network.load_state_dict(checkpoint['q_network_state_dict'])

                 print("✅ Successfully loaded weights into Q-Network and Target Network.")
            else:
                 print("⚠️  Warning: BC checkpoint found but 'q_network_state_dict' key is missing.")
        else:
             print("⚠️  Warning: BC checkpoint not found. Starting DQN from scratch.")

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
    elif algorithm == 'bc': 
        from trainers.helpers import train_bc
        # BC does not require 'render'
        env = train_bc(config, env, agent, logger)
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
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
    if args.resume:
        print(f"Resume: {args.resume}")
    print("=" * 60)

    # Train
    train(config, render=args.render, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
