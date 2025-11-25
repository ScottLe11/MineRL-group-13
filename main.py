import gym
from gym.envs.registration import register
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from treechop_spec import Treechop, handlers
from wrappers import StackAndProcessWrapper, SimpleActionWrapper, HoldAttackWrapper

# Curriculum config - modify these to change starting conditions
CURRICULUM_CONFIG = {
    'with_logs': 0,    # Number of starting logs (0-10)
    'with_axe': False, # Start with wooden axe equipped
}


# Define custom class, inheriting from 'treechop_spec'
class custom_treechop(Treechop):
    """Custom treechop environment with configurable starting conditions."""
    
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLcustom_treechop-v0'
        super().__init__(*args, **kwargs)

    def create_agent_start(self) -> list:
        """Override to use curriculum config instead of hardcoded values."""
        # Get base handlers from HumanControlEnvSpec (skip Treechop's hardcoded inventory)
        from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
        base_handlers = HumanControlEnvSpec.create_agent_start(self)
        
        # Build inventory from curriculum config
        inventory = []
        
        with_logs = CURRICULUM_CONFIG.get('with_logs', 0)
        with_axe = CURRICULUM_CONFIG.get('with_axe', False)
        
        if with_logs > 0:
            inventory.append(dict(type="oak_log", quantity=with_logs))
        
        if with_axe:
            inventory.append(dict(type="wooden_axe", quantity=1))
        
        if inventory:
            base_handlers.append(handlers.SimpleInventoryAgentStart(inventory))
        
        # Optionally spawn near trees
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
