#!/usr/bin/env python3
"""
Visualize attention maps from a trained checkpoint.

Usage:
    python scripts/visualize_attention.py --checkpoint checkpoints/best_model_ppo.pt
    python scripts/visualize_attention.py --checkpoint checkpoints/checkpoint_ppo_ep100.pt --num-steps 50
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.env_factory import create_env
from agent.factory import create_agent
from utils.visualization import AttentionVisualizer


def visualize_checkpoint_attention(
    checkpoint_path: str,
    config_path: str = None,
    num_episodes: int = 3,
    num_steps: int = 20,
    save_dir: str = "attention_visualization"
):
    """
    Load checkpoint and visualize attention maps.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config file (optional, will try to infer)
        num_episodes: Number of episodes to visualize
        num_steps: Number of steps per episode to visualize
        save_dir: Directory to save visualizations
    """
    print("="*60)
    print("ATTENTION MAP VISUALIZATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {num_steps}")
    print(f"Save directory: {save_dir}")
    print("="*60 + "\n")

    # Load config
    if config_path is None:
        config_path = "config/config.yaml"

    config = load_config(config_path)
    print(f"✓ Loaded config from: {config_path}\n")

    # Check if attention is enabled
    attention_type = config['network'].get('attention', 'none')
    if attention_type == 'none':
        print("❌ ERROR: Attention is not enabled in config!")
        print("   Set 'attention: spatial' (or 'cbam', 'treechop_bias') in config.yaml")
        return

    print(f"✓ Attention type: {attention_type}\n")

    # Create environment
    env = create_env(config)
    print(f"✓ Environment created: {config['environment']['name']}\n")

    # Create agent
    agent = create_agent(config, num_actions=env.action_space.n)
    print(f"✓ Agent created\n")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=config['device'])

    # Load weights based on agent type
    if hasattr(agent, 'q_network'):  # DQN
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        network = agent.q_network
        print(f"✓ Loaded DQN checkpoint (episode {checkpoint.get('episode', '?')})\n")
    elif hasattr(agent, 'policy'):  # PPO
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        network = agent.policy
        print(f"✓ Loaded PPO checkpoint (episode {checkpoint.get('episode', '?')})\n")
    else:
        print("❌ ERROR: Unknown agent type")
        return

    network.eval()  # Set to evaluation mode

    # Check if network has attention
    if not hasattr(network, 'use_attention') or not network.use_attention:
        print("❌ ERROR: Network does not have attention enabled!")
        print("   The checkpoint was trained without attention.")
        return

    print(f"✓ Attention enabled in network\n")

    # Create visualizer
    visualizer = AttentionVisualizer(save_dir=save_dir)
    print(f"✓ Visualizer ready (saving to {save_dir}/)\n")

    print("="*60)
    print("STARTING VISUALIZATION")
    print("="*60 + "\n")

    total_visualizations = 0

    # Run episodes
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        episode_reward = 0
        step = 0

        while step < num_steps:
            # Prepare observation
            if not isinstance(obs['pov'], torch.Tensor):
                obs_tensor = {
                    k: torch.tensor(v).unsqueeze(0).to(config['device'])
                    for k, v in obs.items()
                }
            else:
                obs_tensor = {
                    k: v.unsqueeze(0).to(config['device']) if v.dim() <= 1 else v.to(config['device'])
                    for k, v in obs.items()
                }

            # Forward pass with attention
            with torch.no_grad():
                if hasattr(agent, 'q_network'):  # DQN
                    result = network(obs_tensor, return_attention=True)
                    if isinstance(result, tuple) and len(result) == 2:
                        q_values, attention_map = result
                        action = q_values.argmax(dim=1).item()
                    else:
                        print(f"  ⚠️  Warning: Network did not return attention map")
                        break
                else:  # PPO
                    result = network(obs_tensor, return_attention=True)
                    if isinstance(result, tuple) and len(result) == 3:
                        logits, value, attention_map = result
                        action_probs = torch.softmax(logits, dim=-1)
                        action = action_probs.argmax(dim=1).item()
                    else:
                        print(f"  ⚠️  Warning: Network did not return attention map")
                        break

            # Extract frame and attention
            pov = obs_tensor['pov'][0, -1].cpu().numpy()  # Last frame
            attention = attention_map[0, 0].cpu().numpy()  # (H, W)

            # Normalize frame to [0, 255]
            if pov.max() <= 1.0:
                pov = (pov * 255).astype(np.uint8)

            # Visualize
            save_name = f'ep{episode+1:02d}_step{step:03d}.png'
            path = visualizer.visualize_attention(
                frame=pov,
                attention_map=attention,
                episode=episode + 1,
                step=step,
                save_name=save_name
            )

            if step == 0 or step == num_steps - 1:  # Print first and last
                print(f"  Step {step:3d}: {save_name}")

            total_visualizations += 1

            # Take action
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward

            obs = next_obs
            step += 1

            if done:
                break

        print(f"  Episode reward: {episode_reward:.2f}")
        print(f"  Saved {step} attention maps\n")

    env.close()

    print("="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Total visualizations: {total_visualizations}")
    print(f"Saved to: {save_dir}/")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize attention maps from checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., checkpoints/best_model_ppo.pt)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--num-episodes', type=int, default=3,
                        help='Number of episodes to visualize (default: 3)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='Number of steps per episode (default: 20)')
    parser.add_argument('--save-dir', type=str, default='attention_visualization',
                        help='Directory to save visualizations (default: attention_visualization)')

    args = parser.parse_args()

    visualize_checkpoint_attention(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
