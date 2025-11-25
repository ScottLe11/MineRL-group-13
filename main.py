import gym
from gym.envs.registration import register
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

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

def make_single_wrapped_env():
    # This function should contain all the steps from your main.py
    base_env = gym.make('MineRLcustom_treechop-v0')
    env_vision = StackAndProcessWrapper(base_env)
    env_hold   = HoldAttackWrapper(
        env_vision,
        hold_steps=35, lock_aim=True,
        pass_through_move=False, yaw_per_tick=0.0, fwd_jump_ticks=0
    )
    # Assuming SimpleActionWrapper is the final wrapper before the agent sees it
    env = SimpleActionWrapper(env_hold)
    return env


if __name__ == "__main__":
    print("Creating environment...")   
    NUM_ENVS = 6
    env = SubprocVecEnv([make_single_wrapped_env] * NUM_ENVS)

    ### Start of Algo Slotting - example format

    ## hyperparameters = {}

    ## model = PPO()

    ## model.learn()

    ## model.save()

    env.close()
    print("Training finished and environment closed.")
