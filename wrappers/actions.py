import gym
from gym.spaces import Discrete
from gym import ActionWrapper
from crafting import craft_pipeline_make_and_equip_axe, GuiClicker

class SimpleActionWrapper(ActionWrapper):
    def __init__(self, env, logs_to_convert=3, gui_size=(640, 360)):
        super().__init__(env)
        self.logs_to_convert = logs_to_convert
        self.gui_size = gui_size

        self.actions = [
            env.action_space.no_op(),
            {'forward':1},
            {'back':1},
            {'right':1},
            {'left':1},
            {'jump':1},
            {'attack':1},
        ]

        self.PIPELINE = len(self.actions)
        self.action_space = Discrete(self.PIPELINE + 1)

    def action(self, action_index:int):
        if action_index == self.PIPELINE:
            return self.env.action_space.no_op()
        full_action = self.env.action_space.no_op()
        for k, v in self.actions[action_index].items():
            full_action[k] = v
        return full_action

    def step(self, action_index: int):
        if action_index != self.PIPELINE:
            return super().step(action_index)

        suppressor = None
        cur = self.env
        while hasattr(cur, "env"):
            if hasattr(cur, "set_hold_suppressed"):
                suppressor = cur
                break
            cur = cur.env
        if suppressor is not None:
            suppressor.set_hold_suppressed(True)

        class _Tracer(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.last = None
                self.R = 0.0
            def step(self, a):
                o, r, d, i = self.env.step(a)
                self.last = (o, r, d, i)
                self.R += float(r)
                return o, r, d, i

        try:
            tracer = _Tracer(self.env)
            helper = GuiClicker(tracer)
            craft_pipeline_make_and_equip_axe(
                tracer, helper,
                logs_to_convert=self.logs_to_convert,
                width=self.gui_size[0], height=self.gui_size[1],
            )

            if tracer.last is None: 
                tracer.last = tracer.step(tracer.action_space.no_op())

            o, _, d, i = tracer.last
            i = dict(i or {})
            i["macro"] = "craft_pipeline_make_and_equip_axe"
            i["macro_steps_reward"] = tracer.R
            return o, tracer.R, d, i
        finally:
            if suppressor is not None:
                suppressor.set_hold_suppressed(False)
