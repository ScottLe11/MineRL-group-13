import gym
from gym import Wrapper
import time
import pickle
import os

class TrajectoryRecorder(Wrapper):
    def __init__(self, env, log_dir = "expert_trajectory"):
        super().__init__(env)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.trajectory = []
        self.current_obs = None
        self.start_time = None
        print(f"[Recorder] Data will be saved to: {self.log_dir}")
    
    
    def reset(self, **kwargs):
        # Save previous trajectory if it exists (i.e., episode just finished)
        if self.trajectory:
            self._save_trajectory()
            
        self.trajectory = []
        self.start_time = time.time()
        
        out = self.env.reset(**kwargs)

        # Handle 1, 2, or more-than-2 return values 
        if isinstance(out, tuple):
            if len(out) == 1:
                obs, info = out[0], {}
            else:
                # Take the first two as (obs, info), ignore any extras
                obs, info = out[0], out[1]
        else:
            obs, info = out, {}

        self.current_obs = obs # Store initial observation
        return obs, info

    def step(self, action):
        """Executes the environment step, records the transition data to the buffer, and updates the current observation."""
        res = self.env.step(action)
        if isinstance(res, tuple) and len(res) == 5:
            obs, reward, terminated, truncated, info = res
        else:
            obs, reward, done, info = res
            terminated = bool(done)
            truncated = False

        self.trajectory.append({
            'state': self.current_obs,
            'action': action,
            'reward': reward,
            'next_state': obs,
            'terminated': bool(terminated),
            'truncated': bool(truncated),
            'info': info,
        })

        self.current_obs = obs
        return res

    def _save_trajectory(self):
        """Saves the current episode's data to a file."""
        timestamp = int(self.start_time)
        filename = os.path.join(self.log_dir, f"ep_{timestamp}_{len(self.trajectory)}_steps.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump(self.trajectory, f)
            
        print(f"[Recorder] SUCCESS: Saved trajectory with {len(self.trajectory)} steps to {filename}")

    def close(self):
        if self.trajectory:
            self._save_trajectory()
        self.env.close()