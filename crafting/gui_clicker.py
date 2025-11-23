from dataclasses import dataclass
import numpy as np
from minerl.herobraine.hero.mc import CAMERA_SCALER

@dataclass
class Cursor:
    x: float
    y: float

class GuiClicker:
    """ Helper for GUI pointer movement and clicks, plus a few keypress helpers. """
    def __init__(self, env, width=640, height=360):
        self.env = env
        self.w, self.h = width, height
        self.cursor = Cursor(width // 2, height // 2)

    def _step(self, ddx_px=0, ddy_px=0, left=0, right=0, hold=1, render=True):
        """Executes a single simulation step with optional camera movement and click actions."""
        for _ in range(max(1, hold)):
            cam = np.array([ddy_px * CAMERA_SCALER, ddx_px * CAMERA_SCALER], dtype=np.float32)
            act = self.env.action_space.no_op()
            act["camera"] = cam
            if left:  act["attack"] = 1
            if right: act["use"] = 1
            if render: self.env.render()
            self.env.step(act)

    def move_to(self, x, y, max_step=60, render=True):
        """Moves the virtual cursor to a specific screen coordinate, doing it over multiple steps if necessary."""
        dx, dy = x - self.cursor.x, y - self.cursor.y
        steps = max(1, int(max(abs(dx), abs(dy)) // max_step))
        for _ in range(steps):
            self._step(dx/steps, dy/steps, render=render)
            self.cursor.x += dx/steps
            self.cursor.y += dy/steps

    def left_click(self, render=True, hold=2):   
        """Simulates a left mouse click at the current cursor position."""
        self._step(left=1,  hold=hold, render=render)

    def right_click(self, render=True, hold=2):  
        """Simulates a right mouse click at the current cursor position."""
        self._step(right=1, hold=hold, render=render)

    def select_hotbar(self, idx1to9, render=True):
        """Selects a specific hotbar slot (1-9) in the game."""
        act = self.env.action_space.no_op()
        act[f"hotbar.{idx1to9}"] = 1
        if render: self.env.render()
        self.env.step(act)

    def toggle_inventory(self, render=True):
        """Toggles the inventory screen and resets the cursor to the center."""
        act = self.env.action_space.no_op()
        act["inventory"] = 1
        if render: self.env.render()
        self.env.step(act)
        self.cursor = Cursor(self.w // 2, self.h // 2)