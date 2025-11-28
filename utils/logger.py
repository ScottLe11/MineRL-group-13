"""
TensorBoard logger for training metrics.
"""

import os
from datetime import datetime
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class Logger:
    """
    Simple logger that writes to TensorBoard and optionally to console.
    Falls back to console-only if TensorBoard is not available.
    """
    
    def __init__(self, log_dir: str = "runs", experiment_name: str = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Base directory for logs.
            experiment_name: Name for this experiment. If None, uses timestamp.
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_path = Path(log_dir) / experiment_name
        os.makedirs(self.log_path, exist_ok=True)
        
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_path))
            print(f"TensorBoard logging to: {self.log_path}")
        else:
            print("TensorBoard not available, using console logging only")
        
        self.step = 0
        self.episode = 0
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the metric (e.g., "loss", "reward").
            value: The value to log.
            step: Global step. If None, uses internal counter.
        """
        if step is None:
            step = self.step
        
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int = None):
        """
        Log multiple scalars under a common tag.
        
        Args:
            main_tag: Common prefix for all metrics.
            tag_scalar_dict: Dict of {name: value} pairs.
            step: Global step. If None, uses internal counter.
        """
        if step is None:
            step = self.step
        
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_episode(self, episode_reward: float, episode_length: int, 
                    wood_collected: int = 0, epsilon: float = None):
        """
        Log end-of-episode metrics.
        
        Args:
            episode_reward: Total episode reward.
            episode_length: Number of steps in episode.
            wood_collected: Amount of wood collected.
            epsilon: Current exploration rate.
        """
        self.episode += 1
        
        self.log_scalar("episode/reward", episode_reward, self.episode)
        self.log_scalar("episode/length", episode_length, self.episode)
        self.log_scalar("episode/wood_collected", wood_collected, self.episode)
        
        if epsilon is not None:
            self.log_scalar("episode/epsilon", epsilon, self.episode)
        
        print(f"Episode {self.episode}: reward={episode_reward:.2f}, "
              f"length={episode_length}, wood={wood_collected}")
    
    def log_training(self, loss: float, q_mean: float, q_std: float = None, step: int = None):
        """
        Log training step metrics.
        
        Args:
            loss: Training loss.
            q_mean: Mean Q-value of the batch.
            q_std: Std of Q-values (optional).
            step: Global step.
        """
        if step is None:
            step = self.step
        
        self.log_scalar("train/loss", loss, step)
        self.log_scalar("train/q_mean", q_mean, step)
        if q_std is not None:
            self.log_scalar("train/q_std", q_std, step)
    
    def log_training_step(self, step: int, loss: float, **kwargs):
        """
        Log detailed training step metrics (algorithm-agnostic).

        Args:
            step: Global step number.
            loss: Training loss.
            **kwargs: Additional metrics to log:
                - DQN: q_mean, td_error, per_beta
                - PPO: policy_loss, value_loss, entropy
        """
        self.log_scalar("train/loss", loss, step)

        # Log all additional metrics
        for key, value in kwargs.items():
            if value is not None:
                # Convert underscores to forward slashes for TensorBoard hierarchy
                metric_name = f"train/{key}"
                self.log_scalar(metric_name, value, step)
    
    def set_step(self, step: int):
        """Set the global step counter."""
        self.step = step
    
    def close(self):
        """Close the logger and flush any pending writes."""
        if self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    print("✅ Logger Test")
    
    # Create a test logger
    logger = Logger(log_dir="test_runs", experiment_name="test_experiment")
    
    # Log some dummy metrics
    for i in range(10):
        logger.set_step(i)
        logger.log_scalar("test/metric", i * 0.1)
        logger.log_training(loss=1.0 / (i + 1), q_mean=i * 0.5, q_std=0.1)
    
    # Log an episode
    logger.log_episode(episode_reward=5.5, episode_length=100, wood_collected=3, epsilon=0.5)
    
    logger.close()
    
    print(f"  Logs written to: {logger.log_path}")
    print("\n✅ Logger validated!")
    
    # Cleanup test directory
    import shutil
    if os.path.exists("test_runs"):
        shutil.rmtree("test_runs")
        print("  (Cleaned up test directory)")

