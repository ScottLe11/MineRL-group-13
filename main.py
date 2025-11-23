import gym
from gym.envs.registration import register
import numpy as np

from treechop_spec import Treechop, handlers
from wrappers import StackAndProcessWrapper, SimpleActionWrapper, HoldAttackWrapper

# Define custom class, inheriting from 'treechop_spec'
class custom_treechop(Treechop):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLcustom_treechop-v0'
        super().__init__(*args, **kwargs)

    def create_agent_start(self) -> list:
        base_handlers = super().create_agent_start()
        base_handlers.append(
            handlers.AgentStartNear([
                dict(type="log", distance=5)
            ])
        )
        return base_handlers

def make_fist_treechop_env():
    spec = custom_treechop(resolution=(640, 360))
    return spec.make()


register(
    id='MineRLcustom_treechop-v0',
    entry_point=make_fist_treechop_env,
    max_episode_steps=1510
)


if __name__ == "__main__":
    print("Creating environment...")
    base_env = gym.make('MineRLcustom_treechop-v0')

    env_vision = StackAndProcessWrapper(base_env)
    env_hold   = HoldAttackWrapper(
        env_vision,
        hold_steps=35,       
        lock_aim=True,      
        pass_through_move=False,  
        yaw_per_tick=0.0,    
        fwd_jump_ticks=0      
    )
    env = SimpleActionWrapper(env_hold)

    print("Resetting the environment...")
    obs = env.reset()

    print("Testing for 1510 steps (60 seconds)...")
    done = False
    step_count = 0
    NUM_ACTIONS = env.action_space.n
    # Test prob distribution of original 7-action 
    BASE = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]  

    if NUM_ACTIONS == 7:
        FIXED_PROBABILITIES = BASE
    elif NUM_ACTIONS == 8:
        P_MACRO = 1.00  
        if P_MACRO >= 1.0:
            probs = [0.0] * len(BASE)
        else:
            scale = (1.0 - P_MACRO) / sum(BASE)      
            probs = [p * scale for p in BASE]    
        macro_idx = getattr(env, "PIPELINE", 6)  
        probs.insert(macro_idx, P_MACRO)          
        FIXED_PROBABILITIES = probs
    else:
        # even prob distriution
        FIXED_PROBABILITIES = [1.0 / NUM_ACTIONS] * NUM_ACTIONS
    
    try:
        while True: 
            # The human provides the action, which the wrapper chain processes.
            # We must use env.action_space.sample() to get a valid discrete action index (0-5)
            # that is then immediately replaced by the human input coming from the interactor.
            action = np.random.choice(range(NUM_ACTIONS), p=FIXED_PROBABILITIES)
            env.render()
            # The wrapped environment's step function handles the human input and recording
            obs, reward, done, info = env.step(action)
            
            if done:
                print(f"\nEpisode ended. Total Reward: {reward}")
                # Resetting triggers the save of the last episode
                obs = env.reset() # Keep the loop running for the next episode

    except KeyboardInterrupt:
        print("\nInterrupt received. Saving final trajectory and closing.")

    finally:
        env.close()