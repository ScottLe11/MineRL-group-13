"""
Discrete Action Recorder for MineRL Tree-Chopping.

Human plays with discrete keyboard controls:
- One action at a time
- Action queuing (buffer one next action)
- Macro actions (attack_10 executes for 10 steps)
- Visual feedback for current action state

Records discrete action indices directly to pkl files.
No post-processing needed - actions are already discrete!
"""

import time
import pygame
import gym
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from wrappers import StackAndProcessWrapper, RewardWrapper, ObservationWrapper, FrameSkipWrapper
from wrappers.recorder import TrajectoryRecorder
from wrappers.discrete_actions import DISCRETE_ACTION_POOL, get_enabled_actions, build_key_mapping
from recording.action_queue import ActionQueue
from utils.config import load_config


# Display settings
WINDOW_W, WINDOW_H = 960, 540
UI_HEIGHT = 120  # Space for UI at bottom
GAME_HEIGHT = WINDOW_H - UI_HEIGHT


def get_frame(env):
    """Recursively search wrapper stack to retrieve raw color frame."""
    e = env
    for _ in range(6):
        if hasattr(e, "get_last_full_frame"):
            return e.get_last_full_frame()
        if hasattr(e, "env"):
            e = e.env
        else:
            break
    return None


def render_ui(screen, queue_state, enabled_actions, key_mapping):
    """
    Render action queue UI at bottom of screen.

    Shows:
    - Current action with progress bar
    - Queued action
    - Key bindings
    """
    # Clear UI area (dark background)
    ui_rect = pygame.Rect(0, GAME_HEIGHT, WINDOW_W, UI_HEIGHT)
    pygame.draw.rect(screen, (20, 20, 30), ui_rect)

    # Fonts
    font_large = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)

    y_offset = GAME_HEIGHT + 10

    # Current action display
    if queue_state['is_busy']:
        # Current action name
        current_text = font_large.render(
            f"EXECUTING: {queue_state['current_name']}",
            True, (100, 255, 100)
        )
        screen.blit(current_text, (10, y_offset))

        # Progress bar
        action_idx = queue_state['current_action']
        action = enabled_actions[action_idx]
        progress = 1.0 - (queue_state['remaining_steps'] / action.duration)

        bar_x, bar_y = 10, y_offset + 40
        bar_w, bar_h = 300, 20

        # Background
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        # Progress
        pygame.draw.rect(screen, (100, 255, 100), (bar_x, bar_y, int(bar_w * progress), bar_h))
        # Border
        pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h), 2)

        # Steps remaining
        steps_text = font_small.render(
            f"{queue_state['remaining_steps']} steps left",
            True, (200, 200, 200)
        )
        screen.blit(steps_text, (bar_x + bar_w + 10, bar_y))

        # Queued action
        if queue_state['queued_action'] is not None:
            queued_text = font_small.render(
                f"QUEUED: {queue_state['queued_name']}",
                True, (255, 255, 100)
            )
            screen.blit(queued_text, (10, y_offset + 70))
        else:
            ready_text = font_small.render(
                "Press key to queue next action",
                True, (150, 150, 150)
            )
            screen.blit(ready_text, (10, y_offset + 70))
    else:
        # Ready state
        ready_text = font_large.render(
            "READY - Press any action key",
            True, (255, 255, 100)
        )
        screen.blit(ready_text, (10, y_offset))

    # Key bindings (right side)
    bindings_x = 400
    bindings_y = y_offset

    bindings_title = font_small.render("Key Bindings:", True, (200, 200, 200))
    screen.blit(bindings_title, (bindings_x, bindings_y))

    # Show first 6 actions
    reverse_mapping = {v: k for k, v in key_mapping.items()}
    y = bindings_y + 25
    for i, (action_idx, action) in enumerate(list(enabled_actions.items())[:6]):
        if action_idx in reverse_mapping:
            key_code = reverse_mapping[action_idx]
            key_name = pygame.key.name(key_code).upper()

            binding_text = font_small.render(
                f"{key_name}: {action.display_name}",
                True, (150, 150, 150)
            )
            screen.blit(binding_text, (bindings_x + (i % 3) * 150, y + (i // 3) * 25))


def flash_screen(screen, color=(255, 0, 0), duration=0.1):
    """Flash screen with color (for error feedback)."""
    overlay = pygame.Surface((WINDOW_W, WINDOW_H))
    overlay.set_alpha(100)
    overlay.fill(color)
    screen.blit(overlay, (0, 0))
    pygame.display.flip()
    time.sleep(duration)


if __name__ == "__main__":
    # Load recording config (or fall back to main config)
    import yaml
    try:
        recording_config_path = Path(__file__).parent / "config" / "recording_config.yaml"
        with open(recording_config_path, 'r') as f:
            recording_config = yaml.safe_load(f)

        # Check if we should use recording config
        if recording_config.get('use_config', True):
            config = recording_config
            print("[Recorder] Using recording_config.yaml")
        else:
            config = load_config()
            print("[Recorder] Using main config.yaml")
    except FileNotFoundError:
        config = load_config()
        print("[Recorder] recording_config.yaml not found, using main config.yaml")

    # Get enabled actions from config
    action_config = config['action_space']
    if action_config['preset'] == 'custom' and action_config.get('enabled_actions'):
        enabled_indices = action_config['enabled_actions']
    else:
        # Default to base 23 actions
        enabled_indices = list(range(23))

    enabled_actions = get_enabled_actions(enabled_indices)

    # Build key mapping from config if available, otherwise use defaults
    if 'key_bindings' in action_config:
        # Custom key bindings from config
        import pygame
        KEY_NAME_TO_CONSTANT = {
            'w': pygame.K_w, 'a': pygame.K_a, 's': pygame.K_s, 'd': pygame.K_d,
            'space': pygame.K_SPACE, 'r': pygame.K_r, 'f': pygame.K_f,
            '1': pygame.K_1, '2': pygame.K_2, '3': pygame.K_3, '4': pygame.K_4,
            'q': pygame.K_q, 'e': pygame.K_e, 'c': pygame.K_c, 'x': pygame.K_x,
        }
        key_mapping = {}
        for action_idx, key_name in action_config['key_bindings'].items():
            if key_name.lower() in KEY_NAME_TO_CONSTANT:
                key_mapping[KEY_NAME_TO_CONSTANT[key_name.lower()]] = action_idx
            else:
                print(f"[WARNING] Unknown key name: {key_name}")
    else:
        # Use default key mappings
        key_mapping = build_key_mapping(enabled_actions)

    print(f"Enabled {len(enabled_actions)} actions:")
    for action in enabled_actions.values():
        print(f"  [{action.index:2d}] {action.name:20s} key={action.default_key:6s} duration={action.duration}")

    # Setup environment
    LOG_DIR_NAME = config['recording'].get('log_dir', "expert_trajectory_discrete")
    Path(LOG_DIR_NAME).mkdir(parents=True, exist_ok=True)

    MAX_STEPS = config['recording'].get('episode_seconds', 600) * 5  # 5 agent steps per second

    base_env = gym.make('MineRLcustom_treechop-v0')
    env = RewardWrapper(base_env,
                        wood_value=config['rewards']['wood_value'],
                        step_penalty=config['rewards']['step_penalty'])
    env = StackAndProcessWrapper(env)
    env = FrameSkipWrapper(env, skip=4)
    env = ObservationWrapper(env, max_episode_steps=MAX_STEPS)
    env = TrajectoryRecorder(env, log_dir=LOG_DIR_NAME)
    env.reset()

    # Initialize action queue
    action_queue = ActionQueue(enabled_actions)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("MineRL Discrete Action Recorder")
    clock = pygame.time.Clock()
    running = True

    print("\n" + "=" * 60)
    print("DISCRETE ACTION RECORDER")
    print("=" * 60)
    print("Controls:")
    print("  - Press action keys to queue actions")
    print("  - ESC: Quit")
    print("  - Actions execute for their full duration")
    print("  - You can queue ONE action while current executes")
    print("=" * 60 + "\n")

    # Main game loop
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key in key_mapping:
                    # Try to queue action
                    action_idx = key_mapping[event.key]
                    success = action_queue.queue_action(action_idx)

                    if not success:
                        # Queue full - flash red
                        flash_screen(screen, color=(255, 0, 0), duration=0.05)
                    else:
                        # Successfully queued
                        action = enabled_actions[action_idx]
                        print(f"Queued: [{action_idx}] {action.name} (duration={action.duration})")

        # Execute current action
        discrete_action_idx = action_queue.step()

        # Convert to MineRL dictionary action
        minerl_action = enabled_actions[discrete_action_idx].to_minerl_dict()

        # Step environment with BOTH discrete index and MineRL dict
        # Recorder will save the discrete index
        step_res = env.step((discrete_action_idx, minerl_action))
        done = (len(step_res) == 5 and (step_res[2] or step_res[3])) or (len(step_res) == 4 and step_res[2])

        if done:
            env.reset()
            action_queue.clear()  # Clear queue on episode reset

        # Render game frame
        frame = get_frame(env)
        if frame is not None:
            # Scale to game area
            surf = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
            surf = pygame.transform.smoothscale(surf, (WINDOW_W, GAME_HEIGHT))
            screen.blit(surf, (0, 0))

        # Render UI
        queue_state = action_queue.get_state()
        render_ui(screen, queue_state, enabled_actions, key_mapping)

        pygame.display.flip()
        clock.tick(30)

    # Cleanup
    print("\nShutting down...")
    stats = action_queue.get_statistics()
    print(f"Total actions executed: {stats['total_actions_executed']}")
    print(f"Queue rejections: {stats['total_queue_rejections']}")

    pygame.quit()
    env.close()
    print("Done!")
