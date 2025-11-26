#!/usr/bin/env python3
"""
Training dispatcher for MineRL Tree-Chopping Agent.

This script automatically selects and runs the appropriate training script
based on the algorithm specified in the config file.

Usage:
    python scripts/train.py                    # Use default config
    python scripts/train.py --config path.yaml # Use custom config
    python scripts/train.py --render           # Show Minecraft window during training
    python scripts/train.py --algorithm dqn    # Override config algorithm

Supported algorithms:
    - dqn: Deep Q-Network (uses train_dqn.py)
    - ppo: Proximal Policy Optimization (uses train_ppo.py)
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Train MineRL Tree-Chopping Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py                         # Use config file's algorithm
  python scripts/train.py --algorithm dqn         # Force DQN
  python scripts/train.py --algorithm ppo --render # PPO with rendering
  python scripts/train.py --config custom.yaml    # Custom config
        """
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--render', action='store_true',
                        help='Render the Minecraft window during training')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ppo'], default=None,
                        help='Override algorithm from config (dqn or ppo)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine algorithm
    if args.algorithm:
        algorithm = args.algorithm.lower()
        print(f"Algorithm override: using {algorithm.upper()} (from command line)")
    else:
        algorithm = config.get('algorithm', 'dqn').lower()
        print(f"Algorithm: {algorithm.upper()} (from config)")

    # Validate algorithm
    if algorithm not in ['dqn', 'ppo']:
        print(f"Error: Unknown algorithm '{algorithm}'. Must be 'dqn' or 'ppo'.")
        sys.exit(1)

    # Import and run the appropriate trainer
    print(f"\n{'='*60}")
    print(f"Dispatching to {algorithm.upper()} trainer...")
    print(f"{'='*60}\n")

    if algorithm == 'dqn':
        # Import DQN trainer
        from trainers import train_dqn
        # Update config to ensure DQN
        config['algorithm'] = 'dqn'
        # Run DQN training
        train_dqn.train(config, render=args.render)

    elif algorithm == 'ppo':
        # Import PPO trainer
        from trainers import train_ppo
        # Update config to ensure PPO
        config['algorithm'] = 'ppo'
        # Run PPO training
        train_ppo.train(config, render=args.render)


if __name__ == "__main__":
    main()
