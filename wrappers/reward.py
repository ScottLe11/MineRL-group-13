"""
Reward Wrapper - Applies reward shaping to environment rewards.

Adds:
- Step penalty (small negative reward each step to encourage efficiency)
- Reward scaling (multiply raw rewards by a factor)
"""

import gym
import numpy as np


class RewardWrapper(gym.Wrapper):
    """
    Wraps environment to apply reward shaping.
    
    Reward = (raw_reward * reward_scale) + step_penalty
    
    This goes BEFORE the action wrapper in the wrapper chain so that
    the accumulated rewards in multi-frame actions include the shaped rewards.
    """
    
    def __init__(
        self, 
        env, 
        step_penalty: float = -0.001,
        wood_reward_scale: float = 1.0
    ):
        """
        Args:
            env: The wrapped environment.
            step_penalty: Penalty applied each step (default: -0.001).
            wood_reward_scale: Multiplier for wood collection rewards (default: 1.0).
        """
        super().__init__(env)
        self.step_penalty = step_penalty
        self.wood_reward_scale = wood_reward_scale
        
        # Track episode statistics
        self.episode_raw_reward = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_steps = 0
    
    def reset(self):
        """Reset environment and reward tracking."""
        self.episode_raw_reward = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_steps = 0
        return self.env.reset()
    
    def step(self, action):
        """Apply action and shape the reward."""
        obs, raw_reward, done, info = self.env.step(action)
        
        # Shape the reward
        shaped_reward = self._shape_reward(raw_reward)
        
        # Track statistics
        self.episode_raw_reward += raw_reward
        self.episode_shaped_reward += shaped_reward
        self.episode_steps += 1
        
        # Add reward info to info dict
        info['raw_reward'] = raw_reward
        info['shaped_reward'] = shaped_reward
        info['step_penalty'] = self.step_penalty
        
        # Add episode stats at end of episode
        if done:
            info['episode_raw_reward'] = self.episode_raw_reward
            info['episode_shaped_reward'] = self.episode_shaped_reward
            info['episode_steps'] = self.episode_steps
        
        return obs, shaped_reward, done, info
    
    def _shape_reward(self, raw_reward: float) -> float:
        """
        Apply reward shaping.
        
        Args:
            raw_reward: The raw reward from the environment.
            
        Returns:
            Shaped reward.
        """
        # Scale the raw reward (for wood collection)
        scaled_reward = raw_reward * self.wood_reward_scale
        
        # Add step penalty
        shaped_reward = scaled_reward + self.step_penalty
        
        return shaped_reward


if __name__ == "__main__":
    print("✅ RewardWrapper Test")
    
    # Create a mock environment
    class MockEnv:
        def __init__(self):
            self.step_count = 0
        
        def reset(self):
            self.step_count = 0
            return {'pov': np.zeros((4, 84, 84))}
        
        def step(self, action):
            self.step_count += 1
            # Simulate: most steps give 0, occasionally get +1 for wood
            raw_reward = 1.0 if self.step_count % 10 == 0 else 0.0
            done = self.step_count >= 20
            return {'pov': np.zeros((4, 84, 84))}, raw_reward, done, {}
    
    env = MockEnv()
    wrapped = RewardWrapper(env, step_penalty=-0.001, wood_reward_scale=1.0)
    
    obs = wrapped.reset()
    total_shaped = 0
    total_raw = 0
    
    print("\n  Step-by-step rewards:")
    for i in range(20):
        obs, reward, done, info = wrapped.step(0)
        total_shaped += reward
        total_raw += info['raw_reward']
        
        if info['raw_reward'] > 0:
            print(f"    Step {i+1}: raw={info['raw_reward']:.3f}, shaped={reward:.3f} (wood collected!)")
    
    print(f"\n  Episode totals:")
    print(f"    Raw reward: {info['episode_raw_reward']:.3f}")
    print(f"    Shaped reward: {info['episode_shaped_reward']:.3f}")
    print(f"    Steps: {info['episode_steps']}")
    print(f"    Step penalty contribution: {-0.001 * 20:.3f}")
    
    # Verify math
    expected = 2.0 * 1.0 + (-0.001 * 20)  # 2 wood collections + 20 step penalties
    assert abs(info['episode_shaped_reward'] - expected) < 0.0001, f"Expected {expected}, got {info['episode_shaped_reward']}"
    
    print("\n✅ RewardWrapper validated!")

