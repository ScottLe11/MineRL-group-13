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
        self.episode_counter = 0  # Counter to ensure unique filenames
        print(f"[Recorder] Data will be saved to: {self.log_dir}")
    
    
    def reset(self, **kwargs):
        # Save previous trajectory if it exists (i.e., episode just finished)
        if self.trajectory:
            self._save_trajectory()

        self.trajectory = []
        self.start_time = time.time()
        self.episode_counter += 1
        
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
        """
        Executes the environment step, records the transition data to the buffer, and updates the current observation.

        Args:
            action: Can be either:
                - Dict (legacy MineRL action dictionary)
                - Tuple[int, dict] (discrete_idx, minerl_dict) for discrete action recording

        The discrete_idx (if provided) is stored directly in the trajectory for simplified parsing.
        """
        # Extract discrete action index if provided (new discrete recording format)
        if isinstance(action, tuple) and len(action) == 2:
            discrete_idx, minerl_dict = action
            action_to_save = discrete_idx  # Save discrete index
            action_to_execute = minerl_dict  # Execute MineRL dict
        else:
            # Legacy format: action is MineRL dict
            action_to_save = action
            action_to_execute = action

        res = self.env.step(action_to_execute)
        if isinstance(res, tuple) and len(res) == 5:
            obs, reward, terminated, truncated, info = res
        else:
            obs, reward, done, info = res
            terminated = bool(done)
            truncated = False

        self.trajectory.append({
            'state': self.current_obs,
            'action': action_to_save,  # Store discrete index (or dict for legacy)
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
        # Use millisecond precision + counter to ensure unique filenames
        timestamp_ms = int(self.start_time * 1000)
        filename = os.path.join(
            self.log_dir,
            f"ep_{self.episode_counter:04d}_{timestamp_ms}_{len(self.trajectory)}_steps.pkl"
        )
        
        with open(filename, 'wb') as f:
            pickle.dump(self.trajectory, f)
            
        print(f"[Recorder] SUCCESS: Saved trajectory with {len(self.trajectory)} steps to {filename}")

    def close(self):
        if self.trajectory:
            self._save_trajectory()
        self.env.close()