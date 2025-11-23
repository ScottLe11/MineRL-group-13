import gym, cv2, numpy as np
from collections import deque

class StackAndProcessWrapper(gym.Wrapper):
    """ 
    This wrapper preprocesses the environment's visual observations by resizing and converting frames to 
    grayscale, then stacking the four most recent frames to provide the agent with context and motion cues.
    """
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.frame_stack = deque(maxlen=4)
        self.last_full_obs = None

        new_pov_space = gym.spaces.Box(
            low=0, high=255, shape=(4, shape[0], shape[1]), dtype=np.uint8
        )
        self.observation_space.spaces['pov'] = new_pov_space

    def _preprocess(self, frame):
        """Converts frame to grayscale and resizes it."""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, self.shape, interpolation=cv2.INTER_AREA)

    def _get_stacked_obs(self):
        """Stacks the frames in the deque into a single numpy array."""
        return np.stack(self.frame_stack, axis=0)

    def reset(self):
        """Resets env, saves the original frame, and populates the stack."""
        obs = self.env.reset()
        self.last_full_obs = obs['pov']
        f = self._preprocess(obs['pov'])
        for _ in range(4): 
            self.frame_stack.append(f)
        obs['pov'] = self._get_stacked_obs()
        return obs

    def step(self, action):
        """Takes a step, saves the original frame, and stacks the new one."""
        obs, reward, done, info = self.env.step(action)
        self.last_full_obs = obs['pov']
        self.frame_stack.append(self._preprocess(obs['pov']))
        obs['pov'] = self._get_stacked_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        """Displays the color frame."""
        if self.last_full_obs is not None:
            cv2.imshow("MineRL Render", self.last_full_obs[:, :, ::-1])
            cv2.waitKey(1)

    def get_last_full_frame(self):
        """Expose the latest color frame for external UIs (e.g., pygame)."""
        return self.last_full_obs

    def close(self):
        """Closes the base env and our custom render window."""
        cv2.destroyWindow("MineRL Render")
        self.env.close()
