import gym, numpy as np

class HoldAttackWrapper(gym.Wrapper):
    """Extends a single attack action into a held attack sequence while preventing interference with GUI interactions."""
    def __init__(self, env, hold_steps=35, yaw_per_tick=0.0, fwd_jump_ticks=0,
                 lock_aim=True, pass_through_move=False, gui_cooldown=4):
        super().__init__(env)
        self.hold_steps = hold_steps
        self.yaw_per_tick = yaw_per_tick
        self.fwd_jump_ticks = fwd_jump_ticks
        self.lock_aim = lock_aim
        self.pass_through_move = pass_through_move
        self.gui_cooldown = gui_cooldown

        self._attack_left = 0
        self._gui_open = False
        self._hold_block = 0     
        self._suppress = False   

    def set_hold_suppressed(self, flag: bool = True):
        """Enables or disables a suppression flag that disables attack holding, this is for crafting."""
        self._suppress = bool(flag)
        self._attack_left = 0
        if flag:
            self._hold_block = max(self._hold_block, self.gui_cooldown)
        else:
            self._hold_block = 0

    def reset(self, *args, **kwargs):
        """Resets the environment and clears internal state related to attack holding and GUI suppression."""
        out = self.env.reset(*args, **kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        self._gui_open = bool(obs.get("isGuiOpen", False)) if isinstance(obs, dict) else False
        self._attack_left = 0
        self._hold_block = 0
        self._suppress = False
        return out

    @staticmethod
    def _any_hotbar(action):
        """Checks if any hotbar selection key is active in the given action."""
        return any((k.startswith("hotbar.") and action.get(k, 0) == 1) for k in action.keys())

    def _is_gui_intent(self, action):
        """Determines if the action implies an interaction with the GUI, such as crafting."""
        if action.get("inventory", 0) == 1 or action.get("use", 0) == 1 or self._any_hotbar(action):
            return True
        cam = action.get("camera", None)
        cam_nonzero = isinstance(cam, np.ndarray) and (abs(cam[0]) > 1e-6 or abs(cam[1]) > 1e-6)
        moving = any(action.get(k, 0) == 1 for k in ("forward","back","left","right","jump","sprint"))
        if cam_nonzero and (self._gui_open or self._hold_block > 0) and not moving:
            return True
        return False

    def step(self, action):
        """Executes the environment step, managing attack holding logic, GUI cooldowns, and suppression state."""
        if self._suppress:
            obs, reward, done, info = self.env.step(action)
            if isinstance(obs, dict):
                self._gui_open = bool(obs.get("isGuiOpen", self._gui_open))
            return obs, reward, done, info

        if self._is_gui_intent(action):
            self._attack_left = 0
            self._hold_block = self.gui_cooldown
            obs, reward, done, info = self.env.step(action)
            if isinstance(obs, dict):
                self._gui_open = bool(obs.get("isGuiOpen", self._gui_open))
                if self._gui_open:
                    self._hold_block = self.gui_cooldown
            return obs, reward, done, info

        if self._gui_open:
            self._attack_left = 0
            self._hold_block = max(self._hold_block, 1)
        else:
            if self._hold_block > 0:
                self._hold_block -= 1
            if self._hold_block == 0 and self._attack_left == 0 and action.get('attack', 0) == 1:
                self._attack_left = self.hold_steps

        if self._attack_left > 0 and self._hold_block == 0:
            base = self.env.action_space.no_op()
            base['attack'] = 1

            if self.yaw_per_tick != 0.0:
                base['camera'] = np.array([0.0, self.yaw_per_tick], dtype=np.float32)
            elif self.lock_aim:
                base['camera'] = np.array([0.0, 0.0], dtype=np.float32)

            if self.fwd_jump_ticks > 0 and self._attack_left > self.hold_steps - self.fwd_jump_ticks:
                base['forward'] = 1
                base['jump'] = 1

            if self.pass_through_move:
                for k in ('forward','back','left','right','jump','sprint','sneak'):
                    if action.get(k, 0) == 1:
                        base[k] = 1

            action = base
            self._attack_left -= 1

        obs, reward, done, info = self.env.step(action)
        if isinstance(obs, dict):
            prev_gui = self._gui_open
            self._gui_open = bool(obs.get("isGuiOpen", self._gui_open))
            if self._gui_open and not prev_gui:
                self._attack_left = 0
                self._hold_block = self.gui_cooldown
        return obs, reward, done, info
