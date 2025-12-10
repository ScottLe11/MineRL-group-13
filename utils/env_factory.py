"""
Environment factory for creating and configuring MineRL environments.

This module handles:
- Environment registration
- Wrapper application
- Configuration parsing
"""

import gym
from gym.envs.registration import register

from treechop_spec import ConfigurableTreechop
from wrappers.vision import StackAndProcessWrapper
from wrappers.observation import ObservationWrapper
from wrappers.hold_attack import HoldAttackWrapper
from wrappers.actions import ConfigurableActionWrapper
from wrappers.reward import RewardWrapper

# Constants
AGENT_STEPS_PER_SECOND = 5  # Each agent step = 4 frames at 20 ticks/sec = 0.2s


def parse_action_space_config(action_config: dict) -> list:
    """
    Parse action space configuration and return list of enabled action indices.

    Args:
        action_config: Action space configuration dict from config.yaml

    Returns:
        List of action indices to enable (0-24)
    """
    preset = action_config.get('preset', 'base')

    if preset == 'base':
        # Base 22 actions (0-21)
        return list(range(22))
    elif preset == 'assisted':
        # Assisted learning preset: curated action set
        # Movement (0-6) + key camera angles + craft_entire_axe + extended attacks
        return [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 16, 18, 22, 23, 24]
    elif preset == 'custom':
        # Use custom enabled_actions list
        enabled = action_config.get('enabled_actions', [])
        if not enabled:
            print("Warning: preset='custom' but enabled_actions is empty. Defaulting to base 22 actions.")
            return list(range(22))
        return enabled
    else:
        print(f"Warning: Unknown action preset '{preset}'. Defaulting to base 22 actions.")
        return list(range(22))


def register_custom_env(env_id: str, curriculum_config: dict, max_episode_steps: int = 8000):
    """
    Register a custom MineRL environment with gym.

    Args:
        env_id: Environment ID for gym registration
        curriculum_config: Dict with 'spawn_type', 'with_logs', 'with_axe'
        max_episode_steps: Maximum steps per episode

    Note:
        This function is idempotent - calling it multiple times with the same
        env_id will not cause errors.
    """
    try:
        # Try to register the environment
        def _make_env():
            spec = ConfigurableTreechop(
                spawn_type=curriculum_config.get('spawn_type', 'random'),
                with_logs=curriculum_config.get('with_logs', 0),
                with_axe=curriculum_config.get('with_axe', False),
                resolution=(640, 360),
                max_episode_steps=max_episode_steps
            )
            return spec.make()

        register(
            id=env_id,
            entry_point=_make_env
        )
    except Exception as e:
        # Environment already registered, which is fine
        if "Cannot re-register" not in str(e):
            # If it's a different error, raise it
            raise


def create_env(config: dict, wrap: bool = True):
    """
    Create and wrap the MineRL environment.

    Wrapper order (if wrap=True):
    1. Base MineRL env (ConfigurableTreechop)
    2. StackAndProcessWrapper (frame processing)
    3. HoldAttackWrapper (attack duration)
    4. RewardWrapper (reward shaping)
    5. ObservationWrapper (add scalars)
    6. ConfigurableActionWrapper (discrete actions with configurable action space)

    Args:
        config: Configuration dictionary
        wrap: If False, returns the raw, unwrapped environment.

    Returns:
        Fully wrapped or raw environment.
    """
    env_config = config['environment']
    
    # Get curriculum configuration
    curriculum = env_config.get('curriculum', {})
    print(f"Curriculum config: with_logs={curriculum.get('with_logs', 0)}, "
          f"with_axe={curriculum.get('with_axe', False)}, "
          f"spawn_type={curriculum.get('spawn_type', 'random')}")

    # Register and create environment
    env_id = env_config['name']
    register_custom_env(env_id, curriculum, max_episode_steps=8000)
    env = gym.make(env_id)
    print(f"✓ Created MineRL environment: {env_id}")

    if not wrap:
        print("✓ Returning raw, unwrapped environment.")
        return env

    # --- Apply wrappers if wrap=True ---
    print("Applying environment wrappers...")
    reward_config = config.get('rewards', {})
    action_config = config.get('action_space', {})
    
    # Calculate max steps per episode
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps_per_episode = episode_seconds * AGENT_STEPS_PER_SECOND

    # Parse action space configuration
    enabled_actions = parse_action_space_config(action_config)
    print(f"Action space: {len(enabled_actions)} actions (preset: {action_config.get('preset', 'base')})")

    # Apply wrappers in order
    env = StackAndProcessWrapper(env, shape=tuple(env_config['frame_shape']))
    env = HoldAttackWrapper(
        env,
        hold_steps=35,
        lock_aim=True,
        pass_through_move=False,
        yaw_per_tick=0.0,
        fwd_jump_ticks=0
    )
    env = RewardWrapper(
        env,
        wood_value=reward_config.get('wood_value', 1.0),
        step_penalty=reward_config.get('step_penalty', -0.001)
    )
    env = ObservationWrapper(env, max_episode_steps=max_steps_per_episode)
    env = ConfigurableActionWrapper(env, enabled_actions=enabled_actions)
    print("✓ All wrappers applied.")

    return env
