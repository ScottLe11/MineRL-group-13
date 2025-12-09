"""
Observation Wrapper - Adds scalar features to the observation dictionary.

This wrapper tracks and exposes:
- time_left: normalized episode time remaining (1.0 → 0.0 within each episode)
- yaw: agent's horizontal facing direction (degrees, normalized to [-1, 1])
- pitch: agent's vertical head angle (degrees, normalized to [-1, 1])
- place_table_safe: heuristic flag in [0, 1] for "safe to place crafting table"
"""

import gym
import numpy as np

from crafting.crafting_utils import get_basic_inventory_counts


class ObservationWrapper(gym.Wrapper):
    """
    Wraps environment to add scalar observations needed by the DQN network.

    Should be placed AFTER StackAndProcessWrapper and BEFORE action wrappers
    in the wrapper chain.
    """

    def __init__(self, env, max_episode_steps: int):
        """
        Args:
            env: The wrapped environment (should have stacked POV observations).
            max_episode_steps: Maximum steps per episode for time normalization.
        """
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.current_episode_step = 0

        # Track agent orientation (updated externally when camera actions execute)
        self.yaw = 0.0    # Horizontal rotation: -180 to 180
        self.pitch = 0.0  # Vertical rotation: -90 (down) to 90 (up)

        # build observation_space by preserving existing keys (like inventory') and adding time_left / yaw / pitch.
        if isinstance(env.observation_space, gym.spaces.Dict):
            base_spaces = dict(env.observation_space.spaces)
        else:
            # Fallback: treat full obs space as 'pov' if it's not a Dict
            base_spaces = {'pov': env.observation_space}

        if 'pov' not in base_spaces:
            raise KeyError("ObservationWrapper expects 'pov' key in observation_space.")

        # Add scalar spaces
        base_spaces['time_left'] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        base_spaces['yaw'] = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        base_spaces['pitch'] = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        base_spaces['place_table_safe'] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Inventory Scalars 
        base_spaces['inv_logs'] = gym.spaces.Box(low=0, high=640, shape=(1,), dtype=np.float32)
        base_spaces['inv_planks'] = gym.spaces.Box(low=0, high=640, shape=(1,), dtype=np.float32)
        base_spaces['inv_sticks'] = gym.spaces.Box(low=0, high=640, shape=(1,), dtype=np.float32)
        base_spaces['inv_table'] = gym.spaces.Box(low=0, high=640, shape=(1,), dtype=np.float32)
        base_spaces['inv_axe'] = gym.spaces.Box(low=0, high=640, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(base_spaces)
    
    def reset(self):
        """Reset environment and observation tracking."""
        obs = self.env.reset()
        self.current_episode_step = 0
        self.yaw = 0.0
        self.pitch = 0.0
        return self._add_scalars(obs)

    def step(self, action):
        """Take a step and add scalar observations."""
        obs, reward, done, info = self.env.step(action)
        self.current_episode_step += 1
        return self._add_scalars(obs), reward, done, info

    def _add_scalars(self, obs) -> dict:
        """
        Add time_left, yaw, pitch, place_table_safe scalars to observation dict,
        while preserving other keys like 'inventory'.

        Args:
            obs: Original observation (dict with 'pov' and possibly other keys).

        Returns:
            Extended observation dict.
        """
        # Normalized episode time remaining: 1.0 at episode start, 0.0 at max_episode_steps
        time_normalized = max(
            0.0,
            (self.max_episode_steps - self.current_episode_step) / self.max_episode_steps
        )

        # Normalize yaw to [-1, 1] (from -180 to 180)
        normalized_yaw = self.yaw / 180.0

        # Normalize pitch to [-1, 1] (from -90 to 90)
        normalized_pitch = self.pitch / 90.0

        if isinstance(obs, dict):
            # Start from the original obs (so we keep 'inventory', etc.)
            extended_obs = dict(obs)
            pov = obs.get('pov', obs)
        else:
            # Fallback: obs wasn't a dict, treat it as the pov
            extended_obs = {}
            pov = obs

        # Compute place_table_safe using current pitch and pov 
        place_table_safe = self._estimate_place_table_safe(pov, normalized_pitch)

        # Pass the full 'obs' which contains the 'inventory' dict
        counts = get_basic_inventory_counts(obs)

        # pov and add scalar features
        extended_obs['pov'] = pov
        extended_obs['time_left'] = np.array([time_normalized], dtype=np.float32)
        extended_obs['yaw'] = np.array([normalized_yaw], dtype=np.float32)
        extended_obs['pitch'] = np.array([normalized_pitch], dtype=np.float32)
        extended_obs['place_table_safe'] = np.array([place_table_safe], dtype=np.float32)

        # Inventory Scalars
        extended_obs['inv_logs'] = np.array([float(counts['logs'])], dtype=np.float32)
        extended_obs['inv_planks'] = np.array([float(counts['planks'])], dtype=np.float32)
        extended_obs['inv_sticks'] = np.array([float(counts['sticks'])], dtype=np.float32)
        extended_obs['inv_table'] = np.array([float(counts['crafting_table'])], dtype=np.float32)
        extended_obs['inv_axe'] = np.array([float(counts['wooden_axe'])], dtype=np.float32)

        return extended_obs

    def _estimate_place_table_safe(self, raw_pov, normalized_pitch: float) -> float:
        """ Heuristic flag for whether it's safe to place a crafting table in front of the agent. """
        pitch_deg = normalized_pitch * 90.0

        # Looking straight ahead
        if abs(pitch_deg) < 1e-6:
            return 1.0

        # Require a downward pitch 
        if pitch_deg < 4.0 or pitch_deg > 14.5:
            return 0.0

        # If orientation is in the safe band, return 1.0
        return 1.0
    
    def update_orientation(self, delta_yaw: float, delta_pitch: float):
        """
        Update agent orientation (called by action wrapper after camera actions).
        
        Args:
            delta_yaw: Change in horizontal rotation (degrees).
            delta_pitch: Change in vertical rotation (degrees).
        """
        self.yaw = (self.yaw + delta_yaw) % 360
        if self.yaw > 180:
            self.yaw -= 360
        
        self.pitch = np.clip(self.pitch + delta_pitch, -90, 90)
    
    def get_time_fraction(self) -> float:
        """Returns current episode step / max steps (useful for logging)."""
        return self.current_episode_step / self.max_episode_steps


if __name__ == "__main__":
    print("✅ ObservationWrapper Test")
    
    # Create a mock environment for testing
    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),
                'inventory': gym.spaces.Dict({
                    'oak_log': gym.spaces.Box(low=0, high=64, shape=(1,), dtype=np.int32),
                    'stick': gym.spaces.Box(low=0, high=64, shape=(1,), dtype=np.int32),
                    'wooden_axe': gym.spaces.Box(low=0, high=64, shape=(1,), dtype=np.int32),
                })
            })
            self.action_space = gym.spaces.Discrete(21)
            self.step_count = 0
        
        def reset(self):
            self.step_count = 0
            return {
                'pov': np.zeros((4, 84, 84), dtype=np.uint8),
                'inventory': {
                    'oak_log': 0, 
                    'wooden_axe': 0,
                    'stick': 0, 
                    'planks': 0
                }
            }
        
        def step(self, action):
            self.step_count += 1
            obs = {
                'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                'inventory': {'oak_log': 5, 'wooden_axe': 2, 'stick': 2}
            }
            reward = -0.001
            done = self.step_count >= 100
            return obs, reward, done, {}
    
    mock_env = MockEnv()
    wrapped_env = ObservationWrapper(mock_env, max_episode_steps=100)

    # Test reset
    obs = wrapped_env.reset()
    print(f"  After reset:")
    print(f"    Keys: {list(obs.keys())}")
    print(f"    POV shape: {obs['pov'].shape}")
    print(f"    Time Left: {obs['time_left'][0]:.3f} (should be 1.0)")
    print(f"    Yaw: {obs['yaw'][0]:.3f}")
    print(f"    Pitch: {obs['pitch'][0]:.3f}")
    print(f"    place_table_safe: {obs['place_table_safe'][0]:.3f}")

    assert obs['inv_logs'][0] == 0.0
    assert obs['inv_sticks'][0] == 0.0
    assert obs['inv_axe'][0] == 0.0

    assert obs['time_left'][0] == 1.0, "Time left should be 1.0 at episode start"
    assert obs['yaw'][0] == 0.0, "Yaw should be 0.0 at reset"
    assert obs['pitch'][0] == 0.0, "Pitch should be 0.0 at reset"

    # Test episode step updates
    for _ in range(50):
        obs, _, _, _ = wrapped_env.step(0)

    assert obs['inv_logs'][0] == 5.0
    assert obs['inv_sticks'][0] == 2.0
    assert obs['inv_axe'][0] == 2.0

    print(f"\n  After 50 episode steps:")
    print(f"    Time Left: {obs['time_left'][0]:.3f} (should be ~0.5)")
    assert 0.49 <= obs['time_left'][0] <= 0.51, "Time left should be ~0.5 after 50/100 steps"
    
    # Test orientation update
    wrapped_env.update_orientation(delta_yaw=90, delta_pitch=30)
    obs, _, _, _ = wrapped_env.step(0)
    
    print(f"\n  After orientation update (+90 yaw, +30 pitch):")
    print(f"    Yaw: {obs['yaw'][0]:.3f} (should be 0.5 = 90/180)")
    print(f"    Pitch: {obs['pitch'][0]:.3f} (should be 0.33 = 30/90)")
    
    assert abs(obs['yaw'][0] - 0.5) < 0.01, "Yaw should be 0.5"
    assert abs(obs['pitch'][0] - 0.333) < 0.01, "Pitch should be ~0.33"
    
    # Test pitch clamping
    wrapped_env.update_orientation(delta_yaw=0, delta_pitch=100)  # Should clamp to 90
    obs, _, _, _ = wrapped_env.step(0)
    print(f"\n  After pitch overflow (+100 more pitch):")
    print(f"    Pitch: {obs['pitch'][0]:.3f} (should be 1.0 = clamped to 90)")
    assert obs['pitch'][0] == 1.0, "Pitch should be clamped to 1.0"

    print(f"\n  place_table_safe at extreme pitch: {obs['place_table_safe'][0]:.3f} (should be 0.0)")
    assert obs['place_table_safe'][0] == 0.0, "place_table_safe should be 0.0 when pitch is out of band"
    
    print("\n✅ ObservationWrapper validated!")
