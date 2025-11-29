import time, pygame, gym
import main
from gym.envs.registration import register
from wrappers import StackAndProcessWrapper
from wrappers import RewardWrapper
from wrappers import ObservationWrapper
from wrappers import TrajectoryRecorder
from wrappers import FrameSkipWrapper
from pathlib import Path

WINDOW_W, WINDOW_H = 960, 540
MOUSE_SENS = 0.20
KEYS_DOWN, MOUSE_BTNS = set(), set()
INVENTORY_TAP = DROP_TAP = SWAPHANDS_TAP = 0

def make_action(env, mouse_dx, mouse_dy):
    """Converts pygame input state into a valid MineRL dictionary action."""
    global INVENTORY_TAP, DROP_TAP, SWAPHANDS_TAP
    a = env.action_space.no_op()
    a['forward'] = 1 if pygame.K_w in KEYS_DOWN else 0
    a['back']    = 1 if pygame.K_s in KEYS_DOWN else 0
    a['left']    = 1 if pygame.K_a in KEYS_DOWN else 0
    a['right']   = 1 if pygame.K_d in KEYS_DOWN else 0
    a['jump']    = 1 if pygame.K_SPACE in KEYS_DOWN else 0
    a['sneak']   = 1 if (pygame.K_LSHIFT in KEYS_DOWN or pygame.K_RSHIFT in KEYS_DOWN) else 0
    a['sprint']  = 1 if (pygame.K_LCTRL in KEYS_DOWN or pygame.K_RCTRL in KEYS_DOWN) else 0
    a['attack']    = 1 if 1 in MOUSE_BTNS else 0
    a['use']       = 1 if 3 in MOUSE_BTNS else 0
    a['pickItem']  = 1 if 2 in MOUSE_BTNS else 0
    a['inventory'] = 1 if INVENTORY_TAP else 0
    a['drop']      = 1 if DROP_TAP else 0
    a['swapHands'] = 1 if SWAPHANDS_TAP else 0
    INVENTORY_TAP = DROP_TAP = SWAPHANDS_TAP = 0
    for i, key in enumerate([pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,pygame.K_6,pygame.K_7,pygame.K_8,pygame.K_9], start=1):
        a[f'hotbar.{i}'] = 1 if key in KEYS_DOWN else 0
    a['camera'] = [float(mouse_dy * MOUSE_SENS), float(mouse_dx * MOUSE_SENS)]
    return a

def get_frame(env):
    """Recursively searches the wrapper stack to retrieve the latest raw color frame."""
    e = env
    for _ in range(6):
        if hasattr(e, "get_last_full_frame"): return e.get_last_full_frame()
        if hasattr(e, "env"): e = e.env
        else: break
    return None

if __name__ == "__main__":
    # Environment Setup
    LOG_DIR_NAME = "expert_trajectory"
    Path(LOG_DIR_NAME).mkdir(parents=True, exist_ok=True)
    MAX_STEPS = 125
    base_env = gym.make('MineRLcustom_treechop-v0')
    env = RewardWrapper(base_env)                    # Shape rewards per frame
    env = StackAndProcessWrapper(env)                # Stack 4 frames → (4, 84, 84)
    env = FrameSkipWrapper(env, skip=4)              # Repeat action 4x, accumulate rewards
    env = ObservationWrapper(env, max_episode_steps=MAX_STEPS)  # Add scalars
    env = TrajectoryRecorder(env, log_dir=LOG_DIR_NAME)  # Record transitions  
    env.reset()

    # Pygame Setup 
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("MineRL (pygame)")
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    clock, running = pygame.time.Clock(), True

    # Main Game Loop 
    while running:

        # Reset mouse movement
        mouse_dx = mouse_dy = 0.0

        # Process Inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

            #Hanlde keys being pressed
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_e: INVENTORY_TAP = 1
                elif event.key == pygame.K_q: DROP_TAP = 1
                elif event.key == pygame.K_f: SWAPHANDS_TAP = 1
                else: KEYS_DOWN.add(event.key)
            elif event.type == pygame.KEYUP: KEYS_DOWN.discard(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN: MOUSE_BTNS.add(event.button)
            elif event.type == pygame.MOUSEBUTTONUP:   MOUSE_BTNS.discard(event.button)
            elif event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel; mouse_dx += dx; mouse_dy += dy

        action = make_action(env, mouse_dx, mouse_dy)
        step_res = env.step(action)
        done = (len(step_res)==5 and (step_res[2] or step_res[3])) or (len(step_res)==4 and step_res[2])
        if done: env.reset()

        # Convert the pixels from MineRl to Pygame screen
        frame = get_frame(env)
        if frame is not None:
            surf = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
            surf = pygame.transform.smoothscale(surf, (WINDOW_W, WINDOW_H))
            screen.blit(surf, (0, 0)); pygame.display.flip()
        clock.tick(30)
    pygame.quit(); env.close()
