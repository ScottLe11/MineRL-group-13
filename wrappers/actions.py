"""
Extended Action Wrapper for the MineRL Tree-Chopping RL Agent.

Provides 23 discrete actions:
- 0-6: Movement primitives (noop, forward, back, right, left, jump, attack)
- 7-18: Camera primitives (turn left/right at various angles, look up/down)
- 19-22: Crafting macros (planks, make_table, sticks, axe)

All primitives (0-18) work identically: execute for 4 frames, accumulate rewards.
The only difference is the action dict content (movement vs camera deltas).

Each action corresponds to one agent "step" which equals 4 MineRL frames (200ms).

Designed to work with both DQN and PPO policies.
"""

import gym
import numpy as np
from gym.spaces import Discrete
from gym import ActionWrapper
from crafting import (
    craft_pipeline_make_and_equip_axe,
    craft_planks_from_logs,
    craft_table_in_inventory,
    place_and_open_table,
    craft_sticks_in_table,
    craft_wooden_axe,
    GuiClicker,
)


# =============================================================================
# ACTION DEFINITIONS
# =============================================================================

# Movement primitives (0-6): executed for 4 frames each
MOVEMENT_PRIMITIVES = [
    {},               # 0: noop
    {'forward': 1},   # 1: forward
    {'back': 1},      # 2: back
    {'right': 1},     # 3: strafe right
    {'left': 1},      # 4: strafe left
    {'jump': 1},      # 5: jump
    {'attack': 1},    # 6: attack
]

# Camera primitives (7-18): (yaw_delta, pitch_delta) per frame × 4 frames
# These are ALSO primitives - they just modify camera instead of movement
CAMERA_PRIMITIVES = [
    # Turn left (negative yaw)
    (-7.5, 0),    # 7: turn_left_30  (7.5 * 4 = 30°)
    (-11.25, 0),  # 8: turn_left_45  (11.25 * 4 = 45°)
    (-15.0, 0),   # 9: turn_left_60  (15 * 4 = 60°)
    (-22.5, 0),   # 10: turn_left_90 (22.5 * 4 = 90°)
    # Turn right (positive yaw)
    (7.5, 0),     # 11: turn_right_30
    (11.25, 0),   # 12: turn_right_45
    (15.0, 0),    # 13: turn_right_60
    (22.5, 0),    # 14: turn_right_90
    # Look up (negative pitch in MineRL)
    (0, -3.0),    # 15: look_up_12  (3 * 4 = 12°)
    (0, -5.0),    # 16: look_up_20  (5 * 4 = 20°)
    # Look down (positive pitch in MineRL)
    (0, 3.0),     # 17: look_down_12
    (0, 5.0),     # 18: look_down_20
]

# Combined primitives for easy indexing
NUM_MOVEMENT_PRIMITIVES = len(MOVEMENT_PRIMITIVES)  # 7
NUM_CAMERA_PRIMITIVES = len(CAMERA_PRIMITIVES)      # 12
NUM_PRIMITIVES = NUM_MOVEMENT_PRIMITIVES + NUM_CAMERA_PRIMITIVES  # 19

# Legacy alias for backward compatibility
PRIMITIVE_ACTIONS = MOVEMENT_PRIMITIVES
CAMERA_ACTIONS = CAMERA_PRIMITIVES

# Macro action indices (base set)
MACRO_CRAFT_PLANKS = 19
MACRO_MAKE_TABLE = 20
MACRO_CRAFT_STICKS = 21
MACRO_CRAFT_AXE = 22

# Extended action indices (can be added to action pool)
MACRO_CRAFT_ENTIRE_AXE = 23  # Full pipeline: planks -> table -> sticks -> axe
ACTION_ATTACK_5 = 24         # Attack for 5 steps (20 frames)
ACTION_ATTACK_10 = 25        # Attack for 10 steps (40 frames)

# Action names for debugging (full action pool)
ACTION_NAMES_POOL = [
    'noop', 'forward', 'back', 'right', 'left', 'jump', 'attack',
    'turn_left_30', 'turn_left_45', 'turn_left_60', 'turn_left_90',
    'turn_right_30', 'turn_right_45', 'turn_right_60', 'turn_right_90',
    'look_up_12', 'look_up_20', 'look_down_12', 'look_down_20',
    'craft_planks', 'make_table', 'craft_sticks', 'craft_axe',
    'craft_entire_axe', 'attack_5', 'attack_10',
]

# Default action names (23 actions - backward compatible)
ACTION_NAMES = ACTION_NAMES_POOL[:23]

NUM_ACTIONS = 23  # Default (backward compatible)
NUM_ACTIONS_POOL = 26  # Total actions in the pool
FRAMES_PER_ACTION = 4  # Each agent decision = 4 MineRL frames


class ExtendedActionWrapper(ActionWrapper):
    """
    Extended action wrapper with 23 discrete actions.
    
    Each action executes for 4 MineRL frames (200ms at 20Hz).
    Rewards are accumulated across frames (step penalty already applied per frame by RewardWrapper).
    Camera movements are tracked for the ObservationWrapper.
    """
    
    def __init__(self, env, logs_to_convert=3, gui_size=(640, 360)):
        """
        Args:
            env: The wrapped environment (should have ObservationWrapper above it).
            logs_to_convert: Number of logs to convert to planks in craft_planks macro.
            gui_size: GUI dimensions for crafting macros.
        """
        super().__init__(env)
        self.logs_to_convert = logs_to_convert
        self.gui_size = gui_size
        
        # Persistent hotbar mapping for crafting macros
        self.hotbar_map = {}
        
        # Track camera rotation for ObservationWrapper
        self.yaw_delta_accumulated = 0.0
        self.pitch_delta_accumulated = 0.0
        
        self.action_space = Discrete(NUM_ACTIONS)
    
    def action(self, action_index: int) -> dict:
        """Convert discrete action to MineRL action dict (for single frame)."""
        if action_index < len(PRIMITIVE_ACTIONS):
            full_action = self.env.action_space.no_op()
            for k, v in PRIMITIVE_ACTIONS[action_index].items():
                full_action[k] = v
            return full_action
        else:
            # Camera and macro actions are handled in step()
            return self.env.action_space.no_op()
    
    def step(self, action_index: int):
        """
        Execute action for 4 frames and accumulate rewards.

        Args:
            action_index: The discrete action index (0-22 for base, up to 25 for extended).

        Returns:
            Tuple of (observation, accumulated_reward, done, info).
        """
        # Reset camera tracking
        self.yaw_delta_accumulated = 0.0
        self.pitch_delta_accumulated = 0.0

        # Primitive actions (0-6)
        if action_index < len(PRIMITIVE_ACTIONS):
            return self._execute_primitive(action_index)

        # Camera actions (7-18)
        camera_idx = action_index - len(PRIMITIVE_ACTIONS)
        if camera_idx < len(CAMERA_ACTIONS):
            return self._execute_camera(camera_idx)

        # Extended attack actions (24-25)
        if action_index == ACTION_ATTACK_5:
            return self._execute_extended_attack(5)  # 5 steps
        elif action_index == ACTION_ATTACK_10:
            return self._execute_extended_attack(10)  # 10 steps

        # Macro actions (19-23)
        return self._execute_macro(action_index)
    
    def _execute_primitive(self, action_index: int):
        """Execute a primitive action for FRAMES_PER_ACTION frames."""
        total_reward = 0.0
        obs, done, info = None, False, {}
        
        action_dict = self.action(action_index)
        
        for _ in range(FRAMES_PER_ACTION):
            obs, reward, done, info = self.env.step(action_dict)
            total_reward += reward  # Includes step penalty from RewardWrapper
            if done:
                break
        
        info['action_name'] = ACTION_NAMES[action_index]
        return obs, total_reward, done, info
    
    def _execute_camera(self, camera_idx: int):
        """Execute a camera action for FRAMES_PER_ACTION frames."""
        yaw_per_frame, pitch_per_frame = CAMERA_ACTIONS[camera_idx]
        
        total_reward = 0.0
        obs, done, info = None, False, {}
        
        for _ in range(FRAMES_PER_ACTION):
            action_dict = self.env.action_space.no_op()
            action_dict['camera'] = np.array([pitch_per_frame, yaw_per_frame], dtype=np.float32)
            
            obs, reward, done, info = self.env.step(action_dict)
            total_reward += reward
            
            # Track rotation
            self.yaw_delta_accumulated += yaw_per_frame
            self.pitch_delta_accumulated += pitch_per_frame
            
            if done:
                break
        
        # Update ObservationWrapper if present
        self._update_observation_wrapper()
        
        action_index = len(PRIMITIVE_ACTIONS) + camera_idx
        info['action_name'] = ACTION_NAMES[action_index]
        info['yaw_delta'] = self.yaw_delta_accumulated
        info['pitch_delta'] = self.pitch_delta_accumulated
        
        return obs, total_reward, done, info
    
    def _execute_extended_attack(self, num_steps: int):
        """
        Execute an extended attack action for a specified number of steps.

        Args:
            num_steps: Number of agent steps (each step = 4 frames).
                      E.g., num_steps=5 means 20 frames total.
        """
        total_reward = 0.0
        obs, done, info = None, False, {}

        action_dict = self.env.action_space.no_op()
        action_dict['attack'] = 1

        # Each step = FRAMES_PER_ACTION frames
        total_frames = num_steps * FRAMES_PER_ACTION

        for _ in range(total_frames):
            obs, reward, done, info = self.env.step(action_dict)
            total_reward += reward
            if done:
                break

        action_name = f'attack_{num_steps}'
        info['action_name'] = action_name
        info['attack_steps'] = num_steps
        info['attack_frames'] = total_frames
        return obs, total_reward, done, info

    def _execute_macro(self, action_index: int):
        """Execute a crafting macro action."""
        # Suppress HoldAttackWrapper during macro
        suppressor = self._find_hold_attack_wrapper()
        if suppressor is not None:
            suppressor.set_hold_suppressed(True)

        try:
            if action_index == MACRO_CRAFT_PLANKS:
                return self._macro_craft_planks()
            elif action_index == MACRO_MAKE_TABLE:
                return self._macro_make_table()
            elif action_index == MACRO_CRAFT_STICKS:
                return self._macro_craft_sticks()
            elif action_index == MACRO_CRAFT_AXE:
                return self._macro_craft_axe()
            elif action_index == MACRO_CRAFT_ENTIRE_AXE:
                return self._macro_craft_entire_axe()
            else:
                # Fallback: no-op
                return self.env.step(self.env.action_space.no_op())
        finally:
            if suppressor is not None:
                suppressor.set_hold_suppressed(False)
    
    def _macro_craft_planks(self):
        """Craft planks from logs using 2x2 inventory grid."""
        tracer = _RewardTracer(self.env)
        helper = GuiClicker(tracer)
        
        craft_planks_from_logs(
            tracer, helper,
            logs_to_convert=self.logs_to_convert,
            width=self.gui_size[0], height=self.gui_size[1],
            hotbar_map=self.hotbar_map
        )
        
        return self._finalize_macro(tracer, 'craft_planks')
    
    def _macro_make_table(self):
        """
        Make and place crafting table.
        Combines craft_table_in_inventory() + place_and_open_table().
        """
        tracer = _RewardTracer(self.env)
        helper = GuiClicker(tracer)
        
        # Craft the table in inventory
        craft_table_in_inventory(
            tracer, helper,
            width=self.gui_size[0], height=self.gui_size[1],
            hotbar_map=self.hotbar_map
        )
        
        # Place and open the table
        place_and_open_table(
            tracer, helper,
            hotbar_map=self.hotbar_map
        )
        
        return self._finalize_macro(tracer, 'make_table')
    
    def _macro_craft_sticks(self):
        """Craft sticks in the 3x3 crafting table."""
        tracer = _RewardTracer(self.env)
        helper = GuiClicker(tracer)
        
        craft_sticks_in_table(
            tracer, helper,
            width=self.gui_size[0], height=self.gui_size[1],
            hotbar_map=self.hotbar_map
        )
        
        return self._finalize_macro(tracer, 'craft_sticks')
    
    def _macro_craft_axe(self):
        """Craft wooden axe in the 3x3 crafting table."""
        tracer = _RewardTracer(self.env)
        helper = GuiClicker(tracer)

        craft_wooden_axe(
            tracer, helper,
            width=self.gui_size[0], height=self.gui_size[1],
            hotbar_map=self.hotbar_map
        )

        return self._finalize_macro(tracer, 'craft_axe')

    def _macro_craft_entire_axe(self):
        """
        Craft entire axe from scratch.
        Full pipeline: planks -> table -> sticks -> axe.
        """
        tracer = _RewardTracer(self.env)
        helper = GuiClicker(tracer)

        craft_pipeline_make_and_equip_axe(
            tracer, helper,
            logs_to_convert=self.logs_to_convert,
            width=self.gui_size[0], height=self.gui_size[1],
            hotbar_map=self.hotbar_map
        )

        return self._finalize_macro(tracer, 'craft_entire_axe')
    
    def _finalize_macro(self, tracer, macro_name: str):
        """Finalize macro execution and return results."""
        if tracer.last is None:
            tracer.step(self.env.action_space.no_op())
        
        obs, _, done, info = tracer.last
        info = dict(info or {})
        info['action_name'] = macro_name
        info['macro'] = macro_name
        info['macro_steps'] = tracer.steps
        info['macro_reward'] = tracer.total_reward
        
        return obs, tracer.total_reward, done, info
    
    def _find_hold_attack_wrapper(self):
        """Find HoldAttackWrapper in the wrapper chain."""
        cur = self.env
        while hasattr(cur, "env"):
            if hasattr(cur, "set_hold_suppressed"):
                return cur
            cur = cur.env
        return None
    
    def _update_observation_wrapper(self):
        """Update ObservationWrapper with camera rotation if present."""
        cur = self.env
        while hasattr(cur, "env"):
            if hasattr(cur, "update_orientation"):
                cur.update_orientation(
                    delta_yaw=self.yaw_delta_accumulated,
                    delta_pitch=self.pitch_delta_accumulated
                )
                return
            cur = cur.env
    
    def reset(self):
        """Reset environment and clear hotbar mapping."""
        self.hotbar_map = {}
        self.yaw_delta_accumulated = 0.0
        self.pitch_delta_accumulated = 0.0
        return self.env.reset()


class _RewardTracer(gym.Wrapper):
    """
    Helper wrapper that tracks rewards during macro execution.
    Used internally by macro actions.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.last = None
        self.total_reward = 0.0
        self.steps = 0
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last = (obs, reward, done, info)
        self.total_reward += float(reward)
        self.steps += 1
        return obs, reward, done, info


# =============================================================================
# CONFIGURABLE ACTION SPACE
# =============================================================================

class ConfigurableActionWrapper(ExtendedActionWrapper):
    """
    Configurable action wrapper that allows selecting which actions to include
    from the full action pool.

    The action pool consists of 26 actions (indices 0-25):
    - 0-6: Movement primitives (noop, forward, back, right, left, jump, attack)
    - 7-18: Camera primitives (turn left/right, look up/down at various angles)
    - 19-22: Basic crafting macros (planks, make_table, sticks, axe)
    - 23: craft_entire_axe (full pipeline: planks -> table -> sticks -> axe)
    - 24: attack_5 (attack for 5 steps = 20 frames)
    - 25: attack_10 (attack for 10 steps = 40 frames)

    You can enable/disable ANY action(s) like switches - any combination works!

    Examples:
        # Use only base 23 actions
        env = ConfigurableActionWrapper(env, enabled_actions=list(range(23)))

        # Use base actions + craft_entire_axe
        env = ConfigurableActionWrapper(env, enabled_actions=list(range(23)) + [23])

        # Use base actions + extended attacks
        env = ConfigurableActionWrapper(env, enabled_actions=list(range(23)) + [24, 25])

        # Use everything
        env = ConfigurableActionWrapper(env, enabled_actions=list(range(26)))

        # Custom: Only movement + attack + one camera turn + craft_entire_axe
        env = ConfigurableActionWrapper(env, enabled_actions=[0, 1, 2, 6, 11, 23])

        # Custom: No movement, only camera + attacks
        env = ConfigurableActionWrapper(env, enabled_actions=[0, 7, 8, 9, 10, 11, 12, 13, 14, 24, 25])

        # Minimal: Only forward, attack, and one camera angle
        env = ConfigurableActionWrapper(env, enabled_actions=[1, 6, 11])

        # Any arbitrary combination works - the wrapper handles all the mapping!
    """

    def __init__(self, env, enabled_actions=None, logs_to_convert=3, gui_size=(640, 360)):
        """
        Args:
            env: The wrapped environment.
            enabled_actions: List of action indices to include (0-25).
                           If None, uses all 26 actions.
            logs_to_convert: Number of logs to convert in crafting macros.
            gui_size: GUI dimensions for crafting macros.
        """
        super().__init__(env, logs_to_convert=logs_to_convert, gui_size=gui_size)

        # Default: use all actions in the pool
        if enabled_actions is None:
            enabled_actions = list(range(NUM_ACTIONS_POOL))

        # Validate and store enabled actions
        self.enabled_actions = sorted(list(set(enabled_actions)))

        # Validate that all indices are within pool range
        invalid = [a for a in self.enabled_actions if a < 0 or a >= NUM_ACTIONS_POOL]
        if invalid:
            raise ValueError(f"Invalid action indices: {invalid}. Must be in range [0, {NUM_ACTIONS_POOL-1}]")

        # Create mapping from new action space to original action space
        # new_action_index -> original_action_index
        self.action_mapping = {i: orig for i, orig in enumerate(self.enabled_actions)}

        # Create reverse mapping for debugging
        # original_action_index -> new_action_index
        self.reverse_mapping = {orig: i for i, orig in enumerate(self.enabled_actions)}

        # Update action space to reflect selected actions
        self.action_space = Discrete(len(self.enabled_actions))

        # Create action names for selected actions
        self.action_names = [ACTION_NAMES_POOL[i] for i in self.enabled_actions]

    def step(self, action_index: int):
        """
        Execute the selected action.

        Args:
            action_index: Index in the NEW action space (0 to len(enabled_actions)-1).

        Returns:
            Tuple of (observation, accumulated_reward, done, info).
        """
        if action_index < 0 or action_index >= len(self.enabled_actions):
            raise ValueError(f"Action {action_index} out of range [0, {len(self.enabled_actions)-1}]")

        # Map to original action space and execute
        original_action = self.action_mapping[action_index]
        obs, reward, done, info = super().step(original_action)

        # Update action name in info to reflect the mapped action
        info['action_name'] = self.action_names[action_index]
        info['original_action_index'] = original_action
        info['mapped_action_index'] = action_index

        return obs, reward, done, info

    def get_action_info(self):
        """Get information about the configured action space."""
        return {
            'num_actions': len(self.enabled_actions),
            'enabled_actions': self.enabled_actions,
            'action_names': self.action_names,
            'action_mapping': self.action_mapping,
        }


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

class SimpleActionWrapper(ExtendedActionWrapper):
    """
    Alias for backward compatibility.
    Use ExtendedActionWrapper for new code.
    """
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_action_name(action_index: int) -> str:
    """Get the human-readable name for an action index."""
    if 0 <= action_index < len(ACTION_NAMES):
        return ACTION_NAMES[action_index]
    return f"unknown_{action_index}"


def get_action_space_info() -> dict:
    """Get information about the action space."""
    return {
        'num_actions': NUM_ACTIONS,
        'frames_per_action': FRAMES_PER_ACTION,
        # All primitives (both movement and camera)
        'primitives': list(range(NUM_PRIMITIVES)),  # 0-18
        # Breakdown of primitive types
        'movement_primitives': list(range(NUM_MOVEMENT_PRIMITIVES)),  # 0-6
        'camera_primitives': list(range(NUM_MOVEMENT_PRIMITIVES, NUM_PRIMITIVES)),  # 7-18
        # Macros (multi-step sequences)
        'macros': [MACRO_CRAFT_PLANKS, MACRO_MAKE_TABLE, MACRO_CRAFT_STICKS, MACRO_CRAFT_AXE],
        'action_names': ACTION_NAMES,
    }


if __name__ == "__main__":
    print("✅ ExtendedActionWrapper Info")
    print(f"  Total actions: {NUM_ACTIONS}")
    print(f"  Frames per action: {FRAMES_PER_ACTION}")
    print("\n  Action mapping:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"    {i:2d}: {name}")
    
    info = get_action_space_info()
    print(f"\n  Primitive actions: {info['primitives']}")
    print(f"  Camera actions: {info['camera']}")
    print(f"  Macro actions: {info['macros']}")
