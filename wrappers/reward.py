"""
Reward Wrapper - Shapes rewards for wood collection.

Simple reward:
- 1 log mined = wood_value points (default 1.0)
- Step penalty = -0.001 per frame
"""

import gym
import numpy as np


class RewardWrapper(gym.Wrapper):
    """
    Shapes rewards: wood_value per log + step penalty per frame.
    
    reward = (logs_mined × wood_value) + step_penalty
    """
    
    def __init__(self, env, wood_value: float = 1.0, step_penalty: float = -0.001):
        """
        Args:
            env: The wrapped environment.
            wood_value: Points per log mined (default: 1.0).
            step_penalty: Penalty per MineRL frame (default: -0.001).
        """
        super().__init__(env)
        self.wood_value = wood_value
        self.step_penalty = step_penalty
        
        # Track episode statistics
        self.episode_wood = 0
        self.episode_frames = 0
    
    def reset(self):
        """Reset environment and statistics."""
        self.episode_wood = 0
        self.episode_frames = 0
        return self.env.reset()
    
    def step(self, action):
        """Apply action and shape reward."""
        obs, raw_reward, done, info = self.env.step(action)
        
        # MineRL gives +1 per log, we convert to wood_value
        logs_mined = int(raw_reward) if raw_reward > 0 else 0
        shaped_reward = (logs_mined * self.wood_value) + self.step_penalty
        
        # Track statistics
        self.episode_frames += 1
        self.episode_wood += logs_mined
        
        # Add info
        info['wood_this_frame'] = logs_mined
        
        if done:
            info['episode_wood'] = self.episode_wood
            info['episode_frames'] = self.episode_frames
        
        return obs, shaped_reward, done, info


if __name__ == "__main__":
    print("✅ RewardWrapper Test")
    
    # Create a mock environment
    class MockEnv:
        def __init__(self):
            self.frame = 0
        
        def reset(self):
            self.frame = 0
            return {'pov': np.zeros((4, 84, 84))}
        
        def step(self, action):
            self.frame += 1
            # MineRL gives +1 per log on frames 10 and 20
            reward = 1.0 if self.frame % 10 == 0 else 0.0
            done = self.frame >= 20
            return {'pov': np.zeros((4, 84, 84))}, reward, done, {}
    
    # Test with default wood_value=1.0
    env = MockEnv()
    wrapped = RewardWrapper(env, wood_value=1.0, step_penalty=-0.001)
    
    wrapped.reset()
    total = 0
    
    print("\n  wood_value=1.0:")
    for i in range(20):
        obs, reward, done, info = wrapped.step(0)
        total += reward
    
    print(f"    2 logs × 1.0 + 20 frames × -0.001 = {total:.3f}")
    assert abs(total - 1.98) < 0.0001
    
    # Test with wood_value=2.0
    env = MockEnv()
    wrapped = RewardWrapper(env, wood_value=2.0, step_penalty=-0.001)
    
    wrapped.reset()
    total = 0
    
    print("\n  wood_value=2.0:")
    for i in range(20):
        obs, reward, done, info = wrapped.step(0)
        total += reward
    
    print(f"    2 logs × 2.0 + 20 frames × -0.001 = {total:.3f}")
    assert abs(total - 3.98) < 0.0001
    
    print("\n✅ RewardWrapper validated!")

