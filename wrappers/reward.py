"""
Reward Wrapper - Wood inventory-based rewards.

New reward system:
- Reward = change in wood inventory
- +1 for each log gained (mining)
- -1 for each log used (crafting/placing)
- Step penalty per frame
- Total episode reward = final wood count + step penalties
"""

import gym
import numpy as np


class RewardWrapper(gym.Wrapper):
    """
    Wood inventory-based rewards.

    Tracks wood inventory and rewards based on changes:
    - Mining wood: +wood_value per log
    - Using wood (crafting/placing): -wood_value per log
    - Step penalty per frame

    reward = (wood_delta × wood_value) + step_penalty
    """

    def __init__(self, env, wood_value: float = 1.0, step_penalty: float = -0.001):
        """
        Args:
            env: The wrapped environment.
            wood_value: Points per log change (default: 1.0).
            step_penalty: Penalty per MineRL frame (default: -0.001).
        """
        super().__init__(env)
        self.wood_value = wood_value
        self.step_penalty = step_penalty

        # Track wood inventory
        self.previous_wood_count = 0

        # Track episode statistics
        self.episode_wood_mined = 0
        self.episode_wood_used = 0
        self.episode_frames = 0

    def _get_wood_count(self, obs) -> int:
        """
        Extract wood/log count from observation.

        MineRL observations contain inventory info. This handles:
        - obs['inventory']['log'] (typical MineRL format)
        - Fallback to 0 if not available
        """
        try:
            if isinstance(obs, dict) and 'inventory' in obs:
                inv = obs['inventory']
                if isinstance(inv, dict):
                    # Try different log types
                    return int(inv.get('log', inv.get('oak_log', 0)))
                elif hasattr(inv, 'get'):
                    return int(inv.get('log', inv.get('oak_log', 0)))
            return 0
        except (KeyError, TypeError, AttributeError):
            return 0

    def reset(self):
        """Reset environment and statistics."""
        obs = self.env.reset()

        # Initialize wood count from starting inventory
        self.previous_wood_count = self._get_wood_count(obs)

        # Reset statistics
        self.episode_wood_mined = 0
        self.episode_wood_used = 0
        self.episode_frames = 0

        return obs

    def step(self, action):
        """Apply action and shape reward based on wood inventory change."""
        obs, raw_reward, done, info = self.env.step(action)

        # Get current wood count from inventory
        current_wood_count = self._get_wood_count(obs)

        # Calculate change in wood inventory
        wood_delta = current_wood_count - self.previous_wood_count

        # Reward based on inventory change
        # Positive delta = mined wood (+reward)
        # Negative delta = used wood (-reward)
        wood_reward = wood_delta * self.wood_value
        shaped_reward = wood_reward + self.step_penalty

        # Track statistics
        self.episode_frames += 1
        if wood_delta > 0:
            self.episode_wood_mined += wood_delta
        elif wood_delta < 0:
            self.episode_wood_used += abs(wood_delta)

        # Update previous count
        self.previous_wood_count = current_wood_count

        # Add info
        info['wood_delta'] = wood_delta
        info['wood_count'] = current_wood_count
        info['wood_reward'] = wood_reward

        if done:
            info['episode_wood_mined'] = self.episode_wood_mined
            info['episode_wood_used'] = self.episode_wood_used
            info['episode_wood_final'] = current_wood_count
            info['episode_wood_net'] = self.episode_wood_mined - self.episode_wood_used
            info['episode_frames'] = self.episode_frames

        return obs, shaped_reward, done, info


if __name__ == "__main__":
    print("✅ RewardWrapper Test - Wood Inventory Based Rewards")

    # Create a mock environment that simulates wood inventory changes
    class MockEnv:
        def __init__(self):
            self.frame = 0
            self.wood_count = 0

        def reset(self):
            self.frame = 0
            self.wood_count = 0  # Start with 0 wood
            return {'pov': np.zeros((4, 84, 84)), 'inventory': {'log': self.wood_count}}

        def step(self, action):
            self.frame += 1

            # Simulate wood changes:
            # Frame 5: Mine 2 logs (+2)
            # Frame 10: Mine 3 logs (+3)
            # Frame 15: Use 2 logs for crafting (-2)
            # Frame 20: Mine 1 log (+1)
            if self.frame == 5:
                self.wood_count += 2
            elif self.frame == 10:
                self.wood_count += 3
            elif self.frame == 15:
                self.wood_count -= 2
            elif self.frame == 20:
                self.wood_count += 1

            done = self.frame >= 25
            obs = {'pov': np.zeros((4, 84, 84)), 'inventory': {'log': self.wood_count}}
            return obs, 0.0, done, {}  # raw_reward unused now

    # Test with default wood_value=1.0, step_penalty=-0.001
    env = MockEnv()
    wrapped = RewardWrapper(env, wood_value=1.0, step_penalty=-0.001)

    obs = wrapped.reset()
    total_reward = 0

    print("\n  Simulating wood inventory changes:")
    print(f"    Frame  | Wood Δ | Wood Total | Reward")
    print(f"    -------|--------|------------|-------")

    for i in range(25):
        obs, reward, done, info = wrapped.step(0)
        total_reward += reward

        if info.get('wood_delta', 0) != 0 or i == 24:
            wood_delta = info.get('wood_delta', 0)
            wood_count = info.get('wood_count', 0)
            print(f"    {i+1:6d} | {wood_delta:+6d} | {wood_count:10d} | {reward:+.3f}")

    # Expected: +2 +3 -2 +1 = +4 net wood
    # Reward: 4.0 (wood) + 25 × -0.001 (step penalty) = 3.975
    expected = 4.0 - 0.025

    print(f"\n  Total wood changes: +2 +3 -2 +1 = +4 net")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Expected: 4.0 (wood) - 0.025 (25 frames penalty) = {expected:.3f}")

    assert abs(total_reward - expected) < 0.0001, f"Expected {expected}, got {total_reward}"

    # Verify final stats
    if done:
        print(f"\n  Episode stats:")
        print(f"    Wood mined: {info.get('episode_wood_mined', 0)}")
        print(f"    Wood used: {info.get('episode_wood_used', 0)}")
        print(f"    Final wood: {info.get('episode_wood_final', 0)}")
        print(f"    Net wood: {info.get('episode_wood_net', 0)}")
        assert info['episode_wood_mined'] == 6  # 2 + 3 + 1
        assert info['episode_wood_used'] == 2   # 2 used in crafting
        assert info['episode_wood_final'] == 4  # 6 - 2 = 4
        assert info['episode_wood_net'] == 4    # 6 - 2 = 4

    print("\n✅ Wood inventory-based rewards validated!")
    print("   ✓ Mining gives positive rewards")
    print("   ✓ Crafting/using gives negative rewards")
    print("   ✓ Total episode reward = net wood change + step penalties")

