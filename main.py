# main.py
import gym
from gym.envs.registration import register

from treechop_spec import Treechop, handlers
from wrappers import StackAndProcessWrapper


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
                dict(type="log", distance=3)
            ])
        )
        return base_handlers

    def create_observation_space_handlers(self) -> list:
        base_handlers = super().create_observation_space_handlers()
        base_handlers.append(
            handlers.EquippedItemObservation(
                ['air', 'log', 'iron_axe', 'crafting_table', 'other'],
                mainhand=True, offhand=False, armor=False
            )
        )
        return base_handlers

    def create_rewardables(self) -> list:
        return [
            handlers.RewardForCollectingItems([
                dict(type="log", amount=1, reward=1.0),
                dict(type="crafting_table", amount=1, reward=10.0)
            ])
        ]


def make_fist_treechop_env():
    spec = custom_treechop(resolution=(640, 360))
    return spec.make()


register(
    id='MineRLcustom_treechop-v0',
    entry_point=make_fist_treechop_env,
    max_episode_steps=400
)


if __name__ == "__main__":
    print("Creating environment...")
    base_env = gym.make('MineRLcustom_treechop-v0')

    env = StackAndProcessWrapper(base_env)

    print("Resetting the environment...")
    obs = env.reset()

    print("Testing for 400 steps (20 seconds)...")
    done = False
    step_count = 0

    while not done:
        action = env.action_space.no_op()
        action['forward'] = 1
        if step_count % 10 == 0:
            action['attack'] = 1

        env.render()
        obs, reward, done, _ = env.step(action)
        step_count += 1

    env.close()
