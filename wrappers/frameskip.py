"""
FrameSkip Wrapper - Repeats actions for multiple frames and accumulates rewards.

This wrapper:
- Takes one action and repeats it for N frames (default: 4)
- Accumulates rewards over those frames
- Tracks camera movement and updates ObservationWrapper
- Ensures proper frame stacking without overlap between agent steps
"""

import gym
import numpy as np


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeats each action for multiple frames and accumulates rewards.

    This wrapper is essential for:
    1. Reducing transition frequency (e.g., 30 FPS -> 7.5 agent steps/sec)
    2. Accumulating rewards over multiple frames
    3. Matching agent step granularity to MineRL's temporal dynamics

    Should be placed AFTER StackAndProcessWrapper and BEFORE ObservationWrapper.
    """

    def __init__(self, env, skip: int = 4):
        """
        Args:
            env: The wrapped environment (should have stacked POV).
            skip: Number of frames to repeat each action (default: 4).
        """
        super().__init__(env)
        self.skip = skip

        # Track camera deltas for ObservationWrapper
        self.yaw_delta_accumulated = 0.0
        self.pitch_delta_accumulated = 0.0

    def reset(self, **kwargs):
        """Reset environment and clear camera tracking."""
        self.yaw_delta_accumulated = 0.0
        self.pitch_delta_accumulated = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Execute action for self.skip frames and accumulate rewards.

        Args:
            action: MineRL action dictionary (with 'camera' key if camera movement)

        Returns:
            Tuple of (observation, accumulated_reward, done, info)
        """
        # Reset camera tracking for this step
        self.yaw_delta_accumulated = 0.0
        self.pitch_delta_accumulated = 0.0

        # Extract camera deltas from action (if present)
        camera_delta = action.get('camera', [0.0, 0.0])
        pitch_per_frame = float(camera_delta[0])
        yaw_per_frame = float(camera_delta[1])

        # Repeat action for self.skip frames
        total_reward = 0.0
        obs = None
        done = False
        info = {}

        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            # Track camera movement
            self.yaw_delta_accumulated += yaw_per_frame
            self.pitch_delta_accumulated += pitch_per_frame

            if done:
                break

        # Update ObservationWrapper with accumulated camera movement
        self._update_observation_wrapper()

        # Add frame skip info
        info['frames_skipped'] = self.skip
        info['yaw_delta'] = self.yaw_delta_accumulated
        info['pitch_delta'] = self.pitch_delta_accumulated

        return obs, total_reward, done, info

    def _update_observation_wrapper(self):
        """
        Find and update ObservationWrapper with camera rotation if present.

        This ensures that pitch/yaw scalars in observations reflect the
        accumulated camera movement over the skipped frames.
        """
        cur = self.env
        while hasattr(cur, "env"):
            if hasattr(cur, "update_orientation"):
                cur.update_orientation(
                    delta_yaw=self.yaw_delta_accumulated,
                    delta_pitch=self.pitch_delta_accumulated
                )
                return
            cur = cur.env


if __name__ == "__main__":
    print("✅ FrameSkipWrapper Test")

    # Create a mock environment for testing
    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                'pov': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
            })
            self.action_space = gym.spaces.Dict({
                'camera': gym.spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),
                'forward': gym.spaces.Discrete(2),
                'attack': gym.spaces.Discrete(2),
            })
            self.step_count = 0
            self.reward_per_frame = 0.1

        def reset(self):
            self.step_count = 0
            return {'pov': np.zeros((4, 84, 84), dtype=np.uint8)}

        def step(self, action):
            self.step_count += 1
            obs = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)}
            reward = self.reward_per_frame
            done = self.step_count >= 20
            info = {}
            return obs, reward, done, info

        def no_op(self):
            return {
                'camera': np.array([0.0, 0.0], dtype=np.float32),
                'forward': 0,
                'attack': 0,
            }

    mock_env = MockEnv()
    wrapped_env = FrameSkipWrapper(mock_env, skip=4)

    # Test reset
    obs = wrapped_env.reset()
    print(f"  After reset:")
    print(f"    POV shape: {obs['pov'].shape}")

    # Test single step with frame skip
    action = mock_env.no_op()
    action['forward'] = 1
    action['camera'] = np.array([5.0, 10.0], dtype=np.float32)  # Pitch, Yaw

    obs, reward, done, info = wrapped_env.step(action)

    print(f"\n  After 1 agent step (4 frames):")
    print(f"    Accumulated reward: {reward:.3f} (should be {0.1 * 4:.3f})")
    print(f"    Frames skipped: {info['frames_skipped']}")
    print(f"    Yaw delta: {info['yaw_delta']:.1f}° (should be {10.0 * 4:.1f}°)")
    print(f"    Pitch delta: {info['pitch_delta']:.1f}° (should be {5.0 * 4:.1f}°)")
    print(f"    Env step count: {mock_env.step_count} (should be 4)")

    assert reward == 0.4, f"Expected reward 0.4, got {reward}"
    assert info['frames_skipped'] == 4, "Should skip 4 frames"
    assert info['yaw_delta'] == 40.0, "Yaw should accumulate to 40°"
    assert info['pitch_delta'] == 20.0, "Pitch should accumulate to 20°"
    assert mock_env.step_count == 4, "Should have taken 4 env steps"

    # Test multiple agent steps
    for _ in range(4):
        obs, reward, done, info = wrapped_env.step(action)

    print(f"\n  After 5 total agent steps (20 frames):")
    print(f"    Env step count: {mock_env.step_count} (should be 20)")
    print(f"    Done: {done} (should be True)")

    assert mock_env.step_count == 20, "Should have taken 20 env steps total"
    assert done, "Episode should be done after 20 frames"

    print("\n✅ FrameSkipWrapper validated!")
    print("   ✓ Actions repeated for N frames")
    print("   ✓ Rewards accumulated correctly")
    print("   ✓ Camera deltas tracked")
