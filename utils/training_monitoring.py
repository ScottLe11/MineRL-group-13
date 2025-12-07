"""
Training monitoring integration for DQN and PPO.

Handles:
- Real-time metric plotting (rewards, actions)
- Attention map visualization during evaluation
- Periodic monitoring updates
"""

import os
import torch
import numpy as np
from typing import Optional, Dict

try:
    from .visualization import AttentionVisualizer, TrainingMonitor
except ImportError:
    from visualization import AttentionVisualizer, TrainingMonitor


class TrainingMonitoringManager:
    """
    Manages all training monitoring including plots and attention visualization.
    """

    def __init__(
        self,
        enable_plots: bool = True,
        enable_attention_viz: bool = True,
        plot_dir: str = "training_plots",
        attention_dir: str = "attention_maps",
        plot_freq: int = 5,
        attention_freq: int = 50,
        window_size: int = 50
    ):
        """
        Args:
            enable_plots: Enable real-time plotting
            enable_attention_viz: Enable attention visualization
            plot_dir: Directory for training plots
            attention_dir: Directory for attention visualizations
            plot_freq: Plot every N episodes
            attention_freq: Visualize attention every N episodes
            window_size: Sliding window size for plots
        """
        self.enable_plots = enable_plots
        self.enable_attention_viz = enable_attention_viz
        self.plot_freq = plot_freq
        self.attention_freq = attention_freq

        # Create monitors
        if enable_plots:
            self.training_monitor = TrainingMonitor(save_dir=plot_dir, window_size=window_size)
        else:
            self.training_monitor = None

        if enable_attention_viz:
            self.attention_visualizer = AttentionVisualizer(save_dir=attention_dir)
        else:
            self.attention_visualizer = None

    def update_episode(
        self,
        episode: int,
        episode_reward: float,
        wood_collected: int,
        action_frequencies: Optional[Dict[int, float]] = None
    ):
        """
        Update monitoring after each episode.

        Args:
            episode: Episode number
            episode_reward: Total episode reward
            wood_collected: Wood collected this episode
            action_frequencies: Dict mapping action_idx -> frequency
        """
        if self.training_monitor is not None:
            self.training_monitor.add_episode(
                episode=episode,
                reward=episode_reward,
                wood_collected=wood_collected,
                action_frequencies=action_frequencies
            )

    def plot_metrics(
        self,
        episode: int,
        action_names: list,
        force: bool = False
    ):
        """
        Generate training metric plots.

        Args:
            episode: Current episode
            action_names: List of action names
            force: Force plot even if not at plot_freq
        """
        if self.training_monitor is None:
            return

        # Check if we should plot
        if not force and episode % self.plot_freq != 0:
            return

        try:
            # Generate combined metrics plot
            path = self.training_monitor.plot_combined_metrics(
                current_episode=episode,
                action_names=action_names,
                top_k=3
            )
            print(f"  📊 Metrics plot saved: {path}")
        except Exception as e:
            print(f"  ⚠️  Failed to generate plot: {e}")

    def visualize_attention(
        self,
        agent,
        env,
        obs: dict,
        episode: int,
        force: bool = False
    ):
        """
        Visualize attention maps for current observation.

        Args:
            agent: Agent with network that supports attention
            env: Environment (for action names)
            obs: Current observation dict
            episode: Current episode
            force: Force visualization even if not at attention_freq
        """
        if self.attention_visualizer is None:
            return

        # Check if we should visualize
        if not force and episode % self.attention_freq != 0:
            return

        # Check if agent uses attention
        network = None
        if hasattr(agent, 'q_network'):  # DQN
            network = agent.q_network
        elif hasattr(agent, 'policy'):  # PPO
            network = agent.policy
        else:
            return

        if not hasattr(network, 'use_attention') or not network.use_attention:
            return

        try:
            with torch.no_grad():
                # Get observation as tensor
                if not isinstance(obs['pov'], torch.Tensor):
                    obs_tensor = {
                        k: torch.tensor(v).unsqueeze(0) if not isinstance(v, torch.Tensor) else v.unsqueeze(0)
                        for k, v in obs.items()
                    }
                else:
                    obs_tensor = {
                        k: v.unsqueeze(0) if v.dim() == 3 or v.dim() == 0 else v
                        for k, v in obs.items()
                    }

                # Forward pass with attention
                if hasattr(agent, 'q_network'):  # DQN
                    result = network(obs_tensor, return_attention=True)
                    if isinstance(result, tuple) and len(result) == 2:
                        _, attention_map = result
                    else:
                        return  # No attention available
                else:  # PPO
                    result = network(obs_tensor, return_attention=True)
                    if isinstance(result, tuple) and len(result) == 3:
                        _, _, attention_map = result
                    else:
                        return  # No attention available

                # Get frame (last frame from stack)
                pov = obs_tensor['pov'][0, -1].cpu().numpy()  # (H, W)
                attention = attention_map[0, 0].cpu().numpy()  # (H, W)

                # Normalize frame to [0, 255]
                if pov.max() <= 1.0:
                    pov = (pov * 255).astype(np.uint8)

                # Visualize
                path = self.attention_visualizer.visualize_attention(
                    frame=pov,
                    attention_map=attention,
                    episode=episode,
                    step=0,
                    save_name=f'attention_ep{episode:04d}.png'
                )
                print(f"  👁️  Attention visualization saved: {path}")

        except Exception as e:
            print(f"  ⚠️  Failed to visualize attention: {e}")

    def final_summary(self, episode: int, action_names: list):
        """
        Generate final summary plots at end of training.

        Args:
            episode: Final episode number
            action_names: List of action names
        """
        if self.training_monitor is not None:
            try:
                print("\n📊 Generating final training summary...")

                # Reward curve
                reward_path = self.training_monitor.plot_reward_curve(
                    current_episode=episode,
                    save_name=f'final_reward_curve.png'
                )
                print(f"  Reward curve: {reward_path}")

                # Top actions
                actions_path = self.training_monitor.plot_top_actions(
                    current_episode=episode,
                    action_names=action_names,
                    top_k=3,
                    save_name=f'final_top_actions.png'
                )
                print(f"  Top actions: {actions_path}")

                # Combined
                combined_path = self.training_monitor.plot_combined_metrics(
                    current_episode=episode,
                    action_names=action_names,
                    top_k=3,
                    save_name=f'final_metrics.png'
                )
                print(f"  Combined metrics: {combined_path}")

                print("✅ Final summary complete!\n")

            except Exception as e:
                print(f"  ⚠️  Failed to generate final summary: {e}")


def create_monitoring_manager(config: dict) -> Optional[TrainingMonitoringManager]:
    """
    Create monitoring manager from config.

    Args:
        config: Training configuration dict

    Returns:
        TrainingMonitoringManager or None if monitoring disabled
    """
    monitoring_config = config.get('monitoring', {})

    # Check if monitoring is enabled
    enable_plots = monitoring_config.get('enable_plots', True)
    enable_attention = monitoring_config.get('enable_attention_viz', True)

    if not enable_plots and not enable_attention:
        return None

    # Create manager
    return TrainingMonitoringManager(
        enable_plots=enable_plots,
        enable_attention_viz=enable_attention,
        plot_dir=monitoring_config.get('plot_dir', 'training_plots'),
        attention_dir=monitoring_config.get('attention_dir', 'attention_maps'),
        plot_freq=monitoring_config.get('plot_freq', 5),
        attention_freq=monitoring_config.get('attention_freq', 50),
        window_size=monitoring_config.get('window_size', 50)
    )


if __name__ == "__main__":
    print("Testing TrainingMonitoringManager...")

    # Create config
    config = {
        'monitoring': {
            'enable_plots': True,
            'enable_attention_viz': False,
            'plot_freq': 5,
            'window_size': 20
        }
    }

    manager = create_monitoring_manager(config)

    # Simulate training
    action_names = ['forward', 'back', 'left', 'right', 'jump', 'attack', 'turn_left', 'turn_right']

    for ep in range(30):
        reward = np.random.randn() * 2 + ep * 0.1
        wood = np.random.randint(0, 3)
        action_freqs = {i: np.random.rand() for i in range(8)}
        total = sum(action_freqs.values())
        action_freqs = {k: v/total for k, v in action_freqs.items()}

        manager.update_episode(ep, reward, wood, action_freqs)

        # Plot every 5 episodes
        manager.plot_metrics(ep, action_names)

    # Final summary
    manager.final_summary(30, action_names)

    print("\n✅ Monitoring manager test passed!")
