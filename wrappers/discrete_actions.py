"""
Discrete Action Definitions for MineRL Tree-Chopping.

Defines all discrete actions with:
- Index (matches wrappers/actions.py)
- Duration (how many agent steps to execute)
- MineRL dictionary conversion
- Display names and key bindings
"""

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class DiscreteAction:
    """Definition of a single discrete action."""
    index: int                      # Action index (0-25)
    name: str                       # Internal name
    duration: int                   # Number of agent steps to execute
    to_minerl_dict: Callable        # Function: () -> MineRL action dict
    display_name: str               # Short name for UI (max 8 chars)
    default_key: str               # Default keyboard binding


def _create_minerl_noop():
    """Create a no-op action."""
    return {
        'forward': 0, 'back': 0, 'left': 0, 'right': 0, 'jump': 0,
        'sneak': 0, 'sprint': 0, 'attack': 0, 'camera': [0.0, 0.0],
        'place': 0, 'equip': 0, 'craft': 0, 'nearbyCraft': 0, 'nearbySmelt': 0
    }


def _create_forward():
    action = _create_minerl_noop()
    action['forward'] = 1
    return action


def _create_back():
    action = _create_minerl_noop()
    action['back'] = 1
    return action


def _create_left():
    action = _create_minerl_noop()
    action['left'] = 1
    return action


def _create_right():
    action = _create_minerl_noop()
    action['right'] = 1
    return action


def _create_jump():
    action = _create_minerl_noop()
    action['jump'] = 1
    return action


def _create_attack():
    action = _create_minerl_noop()
    action['attack'] = 1
    return action


def _create_camera(pitch_delta: float, yaw_delta: float):
    """Create camera movement action."""
    action = _create_minerl_noop()
    action['camera'] = [pitch_delta, yaw_delta]
    return action


# Full action pool (26 actions)
DISCRETE_ACTION_POOL = {
    0: DiscreteAction(
        index=0,
        name='noop',
        duration=1,
        to_minerl_dict=_create_minerl_noop,
        display_name='NOOP',
        default_key='n'
    ),
    1: DiscreteAction(
        index=1,
        name='forward',
        duration=1,
        to_minerl_dict=_create_forward,
        display_name='FWD',
        default_key='w'
    ),
    2: DiscreteAction(
        index=2,
        name='back',
        duration=1,
        to_minerl_dict=_create_back,
        display_name='BACK',
        default_key='s'
    ),
    3: DiscreteAction(
        index=3,
        name='right',
        duration=1,
        to_minerl_dict=_create_right,
        display_name='RIGHT',
        default_key='d'
    ),
    4: DiscreteAction(
        index=4,
        name='left',
        duration=1,
        to_minerl_dict=_create_left,
        display_name='LEFT',
        default_key='a'
    ),
    5: DiscreteAction(
        index=5,
        name='jump',
        duration=1,
        to_minerl_dict=_create_jump,
        display_name='JUMP',
        default_key='space'
    ),
    6: DiscreteAction(
        index=6,
        name='attack',
        duration=1,
        to_minerl_dict=_create_attack,
        display_name='ATK',
        default_key='k'
    ),
    7: DiscreteAction(
        index=7,
        name='turn_left_30',
        duration=1,
        to_minerl_dict=lambda: _create_camera(0.0, -30.0),
        display_name='TL30',
        default_key='q'
    ),
    8: DiscreteAction(
        index=8,
        name='turn_left_45',
        duration=1,
        to_minerl_dict=lambda: _create_camera(0.0, -45.0),
        display_name='TL45',
        default_key='1'
    ),
    9: DiscreteAction(
        index=9,
        name='turn_left_60',
        duration=1,
        to_minerl_dict=lambda: _create_camera(0.0, -60.0),
        display_name='TL60',
        default_key='2'
    ),
    10: DiscreteAction(
        index=10,
        name='turn_left_90',
        duration=1,
        to_minerl_dict=lambda: _create_camera(0.0, -90.0),
        display_name='TL90',
        default_key='3'
    ),
    11: DiscreteAction(
        index=11,
        name='turn_right_30',
        duration=1,
        to_minerl_dict=lambda: _create_camera(0.0, 30.0),
        display_name='TR30',
        default_key='e'
    ),
    12: DiscreteAction(
        index=12,
        name='turn_right_45',
        duration=1,
        to_minerl_dict=lambda: _create_camera(0.0, 45.0),
        display_name='TR45',
        default_key='4'
    ),
    13: DiscreteAction(
        index=13,
        name='turn_right_60',
        duration=1,
        to_minerl_dict=lambda: _create_camera(0.0, 60.0),
        display_name='TR60',
        default_key='5'
    ),
    14: DiscreteAction(
        index=14,
        name='turn_right_90',
        duration=1,
        to_minerl_dict=lambda: _create_camera(0.0, 90.0),
        display_name='TR90',
        default_key='6'
    ),
    15: DiscreteAction(
        index=15,
        name='look_up_12',
        duration=1,
        to_minerl_dict=lambda: _create_camera(-12.0, 0.0),
        display_name='LU12',
        default_key='r'
    ),
    16: DiscreteAction(
        index=16,
        name='look_up_20',
        duration=1,
        to_minerl_dict=lambda: _create_camera(-20.0, 0.0),
        display_name='LU20',
        default_key='t'
    ),
    17: DiscreteAction(
        index=17,
        name='look_down_12',
        duration=1,
        to_minerl_dict=lambda: _create_camera(12.0, 0.0),
        display_name='LD12',
        default_key='f'
    ),
    18: DiscreteAction(
        index=18,
        name='look_down_20',
        duration=1,
        to_minerl_dict=lambda: _create_camera(20.0, 0.0),
        display_name='LD20',
        default_key='g'
    ),
    19: DiscreteAction(
        index=19,
        name='craft_planks',
        duration=1,
        to_minerl_dict=lambda: {**_create_minerl_noop(), 'craft': 'planks'},
        display_name='PLANKS',
        default_key='7'
    ),
    20: DiscreteAction(
        index=20,
        name='make_table',
        duration=1,
        to_minerl_dict=lambda: {**_create_minerl_noop(), 'place': 'crafting_table'},
        display_name='TABLE',
        default_key='8'
    ),
    21: DiscreteAction(
        index=21,
        name='craft_sticks',
        duration=1,
        to_minerl_dict=lambda: {**_create_minerl_noop(), 'nearbyCraft': 'stick'},
        display_name='STICKS',
        default_key='9'
    ),
    22: DiscreteAction(
        index=22,
        name='craft_axe',
        duration=1,
        to_minerl_dict=lambda: {**_create_minerl_noop(), 'nearbyCraft': 'wooden_axe'},
        display_name='AXE',
        default_key='0'
    ),
    23: DiscreteAction(
        index=23,
        name='craft_entire_axe',
        duration=4,  # Multi-step macro
        to_minerl_dict=lambda: {**_create_minerl_noop(), 'craft': 'planks'},  # First step
        display_name='FULLAXE',
        default_key='['
    ),
    24: DiscreteAction(
        index=24,
        name='attack_5',
        duration=5,  # Attack for 5 agent steps
        to_minerl_dict=_create_attack,
        display_name='ATK5',
        default_key='j'
    ),
    25: DiscreteAction(
        index=25,
        name='attack_10',
        duration=10,  # Attack for 10 agent steps
        to_minerl_dict=_create_attack,
        display_name='ATK10',
        default_key='h'
    ),
}


def get_enabled_actions(enabled_indices: list) -> Dict[int, DiscreteAction]:
    """
    Get subset of actions based on enabled indices from config.

    Args:
        enabled_indices: List of action indices to enable

    Returns:
        Dict mapping index to DiscreteAction
    """
    return {idx: DISCRETE_ACTION_POOL[idx] for idx in enabled_indices}


def build_key_mapping(enabled_actions: Dict[int, DiscreteAction]) -> Dict[str, int]:
    """
    Build pygame key -> action index mapping.

    Args:
        enabled_actions: Dict of enabled actions

    Returns:
        Dict mapping pygame key string to action index
    """
    import pygame

    # Map string key names to pygame constants
    KEY_NAME_TO_PYGAME = {
        'w': pygame.K_w,
        's': pygame.K_s,
        'a': pygame.K_a,
        'd': pygame.K_d,
        'space': pygame.K_SPACE,
        'q': pygame.K_q,
        'e': pygame.K_e,
        'r': pygame.K_r,
        't': pygame.K_t,
        'f': pygame.K_f,
        'g': pygame.K_g,
        'h': pygame.K_h,
        'j': pygame.K_j,
        'k': pygame.K_k,
        'n': pygame.K_n,
        '1': pygame.K_1,
        '2': pygame.K_2,
        '3': pygame.K_3,
        '4': pygame.K_4,
        '5': pygame.K_5,
        '6': pygame.K_6,
        '7': pygame.K_7,
        '8': pygame.K_8,
        '9': pygame.K_9,
        '0': pygame.K_0,
        '[': pygame.K_LEFTBRACKET,
    }

    key_mapping = {}
    for action in enabled_actions.values():
        if action.default_key in KEY_NAME_TO_PYGAME:
            pygame_key = KEY_NAME_TO_PYGAME[action.default_key]
            key_mapping[pygame_key] = action.index

    return key_mapping


if __name__ == "__main__":
    print("✅ Discrete Action Definitions")
    print(f"Total actions: {len(DISCRETE_ACTION_POOL)}")

    # Test action creation
    print("\nTesting action creation:")
    for idx in [0, 1, 6, 25]:
        action = DISCRETE_ACTION_POOL[idx]
        minerl_dict = action.to_minerl_dict()
        print(f"  [{idx}] {action.name:20s} duration={action.duration:2d} key={action.default_key:6s}")

    # Test enabled subset
    print("\nTesting enabled subset [1, 8, 12, 15, 17, 25]:")
    enabled = get_enabled_actions([1, 8, 12, 15, 17, 25])
    for action in enabled.values():
        print(f"  {action.display_name:8s} ({action.name})")

    print("\n✅ All tests passed!")
