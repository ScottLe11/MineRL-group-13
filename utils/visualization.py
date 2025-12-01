"""
Visualization utilities for attention maps and training monitoring.

Provides:
- Attention map visualization
- Real-time training plots
- Action distribution tracking
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import torch


class AttentionVisualizer:
    """Visualize attention maps overlaid on input frames."""

    def __init__(self, save_dir: str = "attention_maps"):
        """
        Args:
            save_dir: Directory to save attention visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def visualize_attention(
        self,
        frame: np.ndarray,
        attention_map: np.ndarray,
        episode: int,
        step: int,
        save_name: Optional[str] = None
    ):
        """
        Visualize attention map overlaid on input frame.

        Args:
            frame: RGB frame of shape (H, W, 3) or (H, W) grayscale
            attention_map: Attention weights of shape (H, W), values in [0, 1]
            episode: Current episode number
            step: Current step number
            save_name: Optional custom save name
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Handle grayscale frames
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # 1. Original frame
        axes[0].imshow(frame)
        axes[0].set_title('Original Frame')
        axes[0].axis('off')

        # 2. Attention map (heatmap)
        im = axes[1].imshow(attention_map, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # 3. Overlay (attention as alpha channel)
        axes[2].imshow(frame)
        axes[2].imshow(attention_map, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        # 4. Attended region (multiply frame by attention)
        attended = frame * attention_map[..., np.newaxis]
        axes[3].imshow(attended.astype(np.uint8))
        axes[3].set_title('Attended Region')
        axes[3].axis('off')

        plt.suptitle(f'Episode {episode}, Step {step}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = f'attention_ep{episode:04d}_step{step:04d}.png'

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return save_path

    def visualize_attention_batch(
        self,
        frames: torch.Tensor,
        attention_maps: torch.Tensor,
        episode: int,
        save_name: Optional[str] = None,
        max_samples: int = 4
    ):
        """
        Visualize attention for a batch of frames.

        Args:
            frames: Batch of frames (B, C, H, W) or (B, H, W)
            attention_maps: Batch of attention maps (B, 1, H, W)
            episode: Current episode
            save_name: Custom save name
            max_samples: Maximum number of samples to visualize
        """
        batch_size = min(frames.shape[0], max_samples)

        fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
        if batch_size == 1:
            axes = axes[np.newaxis, :]

        for i in range(batch_size):
            # Extract frame and attention
            if frames.dim() == 4:  # (B, C, H, W)
                frame = frames[i, -1].cpu().numpy()  # Last channel (most recent frame)
            else:  # (B, H, W)
                frame = frames[i].cpu().numpy()

            attention = attention_maps[i, 0].cpu().numpy()

            # Normalize frame to [0, 255]
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)

            # Original frame
            axes[i, 0].imshow(frame, cmap='gray')
            axes[i, 0].set_title(f'Sample {i+1}: Frame')
            axes[i, 0].axis('off')

            # Attention heatmap
            im = axes[i, 1].imshow(attention, cmap='jet', vmin=0, vmax=1)
            axes[i, 1].set_title('Attention')
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

            # Overlay
            axes[i, 2].imshow(frame, cmap='gray')
            axes[i, 2].imshow(attention, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')

        plt.suptitle(f'Attention Maps - Episode {episode}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = f'attention_batch_ep{episode:04d}.png'

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return save_path


class TrainingMonitor:
    """Monitor and visualize training metrics in real-time."""

    def __init__(self, save_dir: str = "training_plots", window_size: int = 50):
        """
        Args:
            save_dir: Directory to save plots
            window_size: Number of episodes to display in sliding window
        """
        self.save_dir = save_dir
        self.window_size = window_size
        os.makedirs(save_dir, exist_ok=True)

        # Tracking
        self.episode_rewards = []
        self.episode_wood = []
        self.action_history = []  # List of dicts with episode: action_frequencies

    def add_episode(
        self,
        episode: int,
        reward: float,
        wood_collected: int,
        action_frequencies: Optional[Dict[int, float]] = None
    ):
        """
        Add episode data for tracking.

        Args:
            episode: Episode number
            reward: Episode reward
            wood_collected: Wood collected this episode
            action_frequencies: Dict mapping action_idx -> frequency
        """
        self.episode_rewards.append((episode, reward))
        self.episode_wood.append((episode, wood_collected))

        if action_frequencies is not None:
            self.action_history.append({
                'episode': episode,
                'frequencies': action_frequencies
            })

    def plot_reward_curve(self, current_episode: int, save_name: Optional[str] = None):
        """
        Plot episode rewards over the last window_size episodes.

        Args:
            current_episode: Current episode number
            save_name: Custom save name
        """
        if len(self.episode_rewards) == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Get last window_size episodes
        episodes, rewards = zip(*self.episode_rewards[-self.window_size:])

        # Plot rewards
        ax.plot(episodes, rewards, 'b-', alpha=0.3, label='Episode Reward')

        # Running average (last 10)
        if len(rewards) >= 10:
            running_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
            running_episodes = episodes[9:]
            ax.plot(running_episodes, running_avg, 'r-', linewidth=2, label='Running Avg (10 ep)')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f'Episode Rewards (Last {self.window_size} Episodes)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name is None:
            save_name = f'reward_curve_ep{current_episode:04d}.png'

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return save_path

    def plot_top_actions(
        self,
        current_episode: int,
        action_names: List[str],
        top_k: int = 3,
        save_name: Optional[str] = None
    ):
        """
        Plot top K most frequent actions over the last window_size episodes.

        Args:
            current_episode: Current episode number
            action_names: List of action names
            top_k: Number of top actions to track
            save_name: Custom save name
        """
        if len(self.action_history) == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Get last window_size episodes
        recent_history = self.action_history[-self.window_size:]

        # Extract episodes
        episodes = [entry['episode'] for entry in recent_history]

        # For each episode, find top K actions
        top_actions_per_episode = []
        for entry in recent_history:
            freqs = entry['frequencies']
            # Get top K actions
            top_k_actions = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            top_actions_per_episode.append(top_k_actions)

        # Track each unique action that appears in top K
        all_top_actions = set()
        for top_actions in top_actions_per_episode:
            for action_idx, _ in top_actions:
                all_top_actions.add(action_idx)

        # Plot each action's frequency over time
        for action_idx in sorted(all_top_actions):
            action_name = action_names[action_idx] if action_idx < len(action_names) else f'a{action_idx}'

            # Extract frequencies for this action
            freqs = []
            for entry in recent_history:
                freq = entry['frequencies'].get(action_idx, 0.0)
                freqs.append(freq * 100)  # Convert to percentage

            ax.plot(episodes, freqs, marker='o', label=action_name, linewidth=2, markersize=4)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Frequency (%)', fontsize=12)
        ax.set_title(f'Top {top_k} Action Frequencies (Last {self.window_size} Episodes)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name is None:
            save_name = f'top_actions_ep{current_episode:04d}.png'

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return save_path

    def plot_combined_metrics(
        self,
        current_episode: int,
        action_names: List[str],
        top_k: int = 3,
        save_name: Optional[str] = None
    ):
        """
        Create a combined plot with rewards and top actions.

        Args:
            current_episode: Current episode
            action_names: List of action names
            top_k: Number of top actions to track
            save_name: Custom save name
        """
        if len(self.episode_rewards) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # === Plot 1: Rewards ===
        episodes, rewards = zip(*self.episode_rewards[-self.window_size:])

        ax1.plot(episodes, rewards, 'b-', alpha=0.3, label='Episode Reward')

        if len(rewards) >= 10:
            running_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
            running_episodes = episodes[9:]
            ax1.plot(running_episodes, running_avg, 'r-', linewidth=2, label='Running Avg (10 ep)')

        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # === Plot 2: Top Actions ===
        if len(self.action_history) > 0:
            recent_history = self.action_history[-self.window_size:]
            episodes_actions = [entry['episode'] for entry in recent_history]

            # Find top K actions overall
            all_freqs = {}
            for entry in recent_history:
                for action_idx, freq in entry['frequencies'].items():
                    all_freqs[action_idx] = all_freqs.get(action_idx, 0) + freq

            top_k_overall = sorted(all_freqs.items(), key=lambda x: x[1], reverse=True)[:top_k]

            for action_idx, _ in top_k_overall:
                action_name = action_names[action_idx] if action_idx < len(action_names) else f'a{action_idx}'

                freqs = []
                for entry in recent_history:
                    freq = entry['frequencies'].get(action_idx, 0.0)
                    freqs.append(freq * 100)

                ax2.plot(episodes_actions, freqs, marker='o', label=action_name, linewidth=2, markersize=4)

            ax2.set_xlabel('Episode', fontsize=12)
            ax2.set_ylabel('Frequency (%)', fontsize=12)
            ax2.set_title(f'Top {top_k} Action Frequencies', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Training Metrics - Episode {current_episode}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_name is None:
            save_name = f'combined_metrics_ep{current_episode:04d}.png'

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return save_path


if __name__ == "__main__":
    # Test attention visualizer
    print("Testing AttentionVisualizer...")
    visualizer = AttentionVisualizer(save_dir="test_attention")

    # Create dummy data
    frame = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
    attention = np.random.rand(84, 84)
    attention = (attention - attention.min()) / (attention.max() - attention.min())  # Normalize

    path = visualizer.visualize_attention(frame, attention, episode=1, step=10)
    print(f"  Saved: {path}")

    # Test training monitor
    print("\nTesting TrainingMonitor...")
    monitor = TrainingMonitor(save_dir="test_plots", window_size=20)

    # Simulate training
    for ep in range(30):
        reward = np.random.randn() * 2 + ep * 0.1  # Trending upward
        wood = np.random.randint(0, 3)
        action_freqs = {i: np.random.rand() for i in range(8)}
        # Normalize frequencies
        total = sum(action_freqs.values())
        action_freqs = {k: v/total for k, v in action_freqs.items()}

        monitor.add_episode(ep, reward, wood, action_freqs)

    action_names = ['forward', 'back', 'left', 'right', 'jump', 'attack', 'turn_left', 'turn_right']

    reward_path = monitor.plot_reward_curve(30)
    print(f"  Reward curve saved: {reward_path}")

    actions_path = monitor.plot_top_actions(30, action_names, top_k=3)
    print(f"  Top actions saved: {actions_path}")

    combined_path = monitor.plot_combined_metrics(30, action_names, top_k=3)
    print(f"  Combined metrics saved: {combined_path}")

    print("\n✅ Visualization tests passed!")
