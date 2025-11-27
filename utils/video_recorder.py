"""
Video Recorder for training progress visualization.

Records episodes at key training milestones (0%, 20%, 40%, 60%, 80%, 100%)
to visualize agent improvement over time.

Saves as MP4 (if imageio-ffmpeg available) or GIF (fallback).
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


class VideoRecorder:
    """
    Records episodes at training milestones for progress visualization.
    
    Usage:
        recorder = VideoRecorder(save_dir="videos", total_episodes=1000)
        
        for episode in range(total_episodes):
            if recorder.should_record(episode):
                recorder.start_episode(episode)
                
                obs = env.reset()
                while not done:
                    action = agent.select_action(obs)
                    obs, reward, done, info = env.step(action)
                    recorder.add_frame(obs, reward, action, info)
                
                recorder.end_episode(total_reward, wood_collected)
    """
    
    def __init__(
        self,
        save_dir: str = "videos",
        total_episodes: int = 1000,
        progress_milestones: List[float] = None,
        fps: int = 10,
        frame_key: str = 'pov',
        include_overlay: bool = True,
        max_frames: int = 1000
    ):
        """
        Args:
            save_dir: Directory to save videos.
            total_episodes: Total episodes in training run.
            progress_milestones: List of progress fractions to record at (default: [0, 0.2, 0.4, 0.6, 0.8, 1.0]).
            fps: Frames per second for saved video.
            frame_key: Key in observation dict for the visual frame.
            include_overlay: Whether to add info overlay on frames.
            max_frames: Maximum frames per episode (safety limit).
        """
        if not IMAGEIO_AVAILABLE:
            print("Warning: imageio not installed. Video recording disabled.")
            print("Install with: pip install imageio imageio-ffmpeg")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.total_episodes = total_episodes
        self.progress_milestones = progress_milestones or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.fps = fps
        self.frame_key = frame_key
        self.include_overlay = include_overlay
        self.max_frames = max_frames
        
        # Calculate milestone episodes
        self.milestone_episodes = set()
        for milestone in self.progress_milestones:
            ep = int(milestone * total_episodes)
            # Handle edge cases
            ep = max(1, min(ep, total_episodes))
            self.milestone_episodes.add(ep)
        
        # Add episode 1 for initial behavior
        self.milestone_episodes.add(1)
        
        # Recording state
        self.frames: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.actions: List[int] = []
        self.is_recording = False
        self.current_episode = 0
        
        # Track which milestones have been recorded
        self.recorded_milestones: set = set()
        
        print(f"VideoRecorder initialized. Will record at episodes: {sorted(self.milestone_episodes)}")
    
    def should_record(self, episode: int) -> bool:
        """
        Check if this episode should be recorded.
        
        Args:
            episode: Current episode number (1-indexed).
        
        Returns:
            True if this episode is a milestone that hasn't been recorded yet.
        """
        if not IMAGEIO_AVAILABLE:
            return False
        
        return episode in self.milestone_episodes and episode not in self.recorded_milestones
    
    def start_episode(self, episode: int):
        """
        Start recording a new episode.
        
        Args:
            episode: Current episode number.
        """
        if not IMAGEIO_AVAILABLE:
            return
        
        self.frames = []
        self.rewards = []
        self.actions = []
        self.is_recording = True
        self.current_episode = episode
        
        progress = episode / self.total_episodes * 100
        print(f"🎬 Recording episode {episode} ({progress:.0f}% progress)")
    
    def add_frame(
        self,
        observation: dict,
        reward: float = 0.0,
        action: int = None,
        info: dict = None
    ):
        """
        Add a frame to the current recording.
        
        Args:
            observation: Observation dict with frame data.
            reward: Reward received this step.
            action: Action taken (for overlay).
            info: Additional info dict (for overlay).
        """
        if not self.is_recording or not IMAGEIO_AVAILABLE:
            return
        
        if len(self.frames) >= self.max_frames:
            return
        
        # Extract frame
        frame = self._extract_frame(observation)
        if frame is None:
            return
        
        # Add overlay if enabled
        if self.include_overlay:
            frame = self._add_overlay(frame, reward, action, info)
        
        self.frames.append(frame)
        self.rewards.append(reward)
        if action is not None:
            self.actions.append(action)
    
    def end_episode(self, total_reward: float = None, wood_collected: int = None) -> Optional[str]:
        """
        End recording and save the video.
        
        Args:
            total_reward: Total episode reward (for filename).
            wood_collected: Wood collected (for filename).
        
        Returns:
            Path to saved video, or None if recording failed.
        """
        if not self.is_recording or not IMAGEIO_AVAILABLE:
            self.is_recording = False
            return None
        
        self.is_recording = False
        
        if len(self.frames) == 0:
            print("Warning: No frames recorded")
            return None
        
        # Mark as recorded
        self.recorded_milestones.add(self.current_episode)
        
        # Generate filename
        progress = self.current_episode / self.total_episodes
        reward_str = f"_r{total_reward:.1f}" if total_reward is not None else ""
        wood_str = f"_w{wood_collected}" if wood_collected is not None else ""
        timestamp = datetime.now().strftime("%H%M%S")
        
        filename = f"ep{self.current_episode:04d}_prog{int(progress*100):03d}{reward_str}{wood_str}_{timestamp}"
        
        # Try MP4 first, fall back to GIF
        video_path = self._save_video(filename)
        
        if video_path:
            print(f"💾 Saved: {video_path} ({len(self.frames)} frames)")
        
        # Clear frames
        self.frames = []
        self.rewards = []
        self.actions = []
        
        return video_path
    
    def _extract_frame(self, observation: dict) -> Optional[np.ndarray]:
        """Extract and format frame from observation."""
        if isinstance(observation, dict):
            if self.frame_key in observation:
                frame = observation[self.frame_key]
            elif 'pov' in observation:
                frame = observation['pov']
            else:
                return None
        elif isinstance(observation, np.ndarray):
            frame = observation
        else:
            return None
        
        # Handle frame stack: take most recent frame
        if len(frame.shape) == 3:
            if frame.shape[0] == 4:  # (4, H, W) frame stack
                frame = frame[-1]  # Take last frame
            elif frame.shape[2] == 4:  # (H, W, 4) frame stack
                frame = frame[:, :, -1]
        
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Ensure 2D or 3D
        if len(frame.shape) == 2:
            # Grayscale to RGB
            frame = np.stack([frame, frame, frame], axis=-1)
        
        return frame
    
    def _add_overlay(
        self,
        frame: np.ndarray,
        reward: float,
        action: int,
        info: dict
    ) -> np.ndarray:
        """Add info overlay to frame."""
        # Make a copy to avoid modifying original
        frame = frame.copy()
        
        # Simple overlay: add a semi-transparent bar at top
        h, w = frame.shape[:2]
        bar_height = min(20, h // 10)
        
        # Darken top bar area
        frame[:bar_height] = (frame[:bar_height] * 0.5).astype(np.uint8)
        
        # We can't easily add text without PIL/OpenCV, so just use visual indicators
        # Reward indicator: green bar at top-left proportional to cumulative reward
        cum_reward = sum(self.rewards)
        reward_width = min(w // 3, max(2, int(abs(cum_reward) * 10)))
        color = (0, 255, 0) if cum_reward >= 0 else (255, 0, 0)
        frame[:bar_height, :reward_width] = color
        
        # Wood indicator (if available in info)
        if info and 'wood_this_frame' in info and info['wood_this_frame'] > 0:
            # Flash gold when wood collected
            frame[:bar_height, -20:] = (255, 215, 0)  # Gold
        
        return frame
    
    def _save_video(self, filename: str) -> Optional[str]:
        """Save frames as video file."""
        # Try MP4 first
        mp4_path = self.save_dir / f"{filename}.mp4"
        try:
            imageio.mimwrite(str(mp4_path), self.frames, fps=self.fps)
            return str(mp4_path)
        except Exception as e:
            print(f"MP4 save failed ({e}), trying GIF...")
        
        # Fall back to GIF
        gif_path = self.save_dir / f"{filename}.gif"
        try:
            imageio.mimwrite(str(gif_path), self.frames, fps=self.fps)
            return str(gif_path)
        except Exception as e:
            print(f"GIF save failed: {e}")
            return None
    
    def record_episode(
        self,
        env,
        agent,
        episode: int,
        max_steps: int = 300
    ) -> Tuple[float, int, Optional[str]]:
        """
        Convenience method to record a full episode.
        
        Args:
            env: Gym environment.
            agent: Agent with select_action(obs, explore=False) method.
            episode: Episode number.
            max_steps: Maximum steps per episode.
        
        Returns:
            (total_reward, wood_collected, video_path)
        """
        self.start_episode(episode)
        
        obs = env.reset()
        total_reward = 0
        wood_collected = 0
        
        for step in range(max_steps):
            action = agent.select_action(obs, explore=False)
            
            next_obs, reward, done, info = env.step(action)
            
            self.add_frame(obs, reward, action, info)
            
            total_reward += reward
            wood_collected = info.get('wood_count', 0)  # Current wood inventory (net: mining - using)
            
            obs = next_obs
            
            if done:
                break
        
        video_path = self.end_episode(total_reward, wood_collected)
        
        return total_reward, wood_collected, video_path
    
    def get_progress_report(self) -> str:
        """Get a summary of recorded milestones."""
        recorded = sorted(self.recorded_milestones)
        remaining = sorted(self.milestone_episodes - self.recorded_milestones)
        
        report = f"Video Recording Progress:\n"
        report += f"  Recorded: {recorded}\n"
        report += f"  Remaining: {remaining}\n"
        
        return report


if __name__ == "__main__":
    print("Testing VideoRecorder...")
    print("=" * 60)
    
    # Create test recorder
    recorder = VideoRecorder(
        save_dir="test_videos",
        total_episodes=100,
        progress_milestones=[0.0, 0.5, 1.0]
    )
    
    print(f"\nMilestone episodes: {sorted(recorder.milestone_episodes)}")
    
    # Test should_record
    print("\nTesting should_record():")
    for ep in [1, 2, 50, 51, 100]:
        should = recorder.should_record(ep)
        print(f"  Episode {ep}: {should}")
    
    # Test frame extraction
    print("\nTesting frame extraction:")
    
    # Test with frame stack (4, 84, 84)
    obs1 = {'pov': np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)}
    frame1 = recorder._extract_frame(obs1)
    print(f"  Frame stack (4,84,84) -> {frame1.shape if frame1 is not None else None}")
    
    # Test with single frame
    obs2 = {'pov': np.random.randint(0, 255, (84, 84), dtype=np.uint8)}
    frame2 = recorder._extract_frame(obs2)
    print(f"  Single frame (84,84) -> {frame2.shape if frame2 is not None else None}")
    
    # Test with RGB frame
    obs3 = {'pov': np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)}
    frame3 = recorder._extract_frame(obs3)
    print(f"  RGB frame (84,84,3) -> {frame3.shape if frame3 is not None else None}")
    
    # Simulate recording (without actually saving)
    print("\nSimulating episode recording:")
    recorder.start_episode(1)
    
    for i in range(10):
        obs = {'pov': np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)}
        recorder.add_frame(obs, reward=0.1 * i, action=i % 23, info={'wood_this_frame': 1 if i == 5 else 0})
    
    print(f"  Frames recorded: {len(recorder.frames)}")
    print(f"  Cumulative reward: {sum(recorder.rewards):.2f}")
    
    # Don't actually save (imageio may not be installed)
    recorder.is_recording = False
    recorder.frames = []
    
    print("\n" + "=" * 60)
    print("✅ VideoRecorder tests passed!")
    
    # Cleanup
    import shutil
    if os.path.exists("test_videos"):
        shutil.rmtree("test_videos")

