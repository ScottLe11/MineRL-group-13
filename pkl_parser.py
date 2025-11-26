import numpy as np
import pickle
import os

def load_all_trajectories(log_dir="expert_trajectory"):
    """Loads and combines all trajectories from the specified directory."""
    all_trajectories = []
    
    # Get all .pkl files in the directory
    for filename in os.listdir(log_dir):
        if filename.endswith(".pkl"):
            filepath = os.path.join(log_dir, filename)
            
            # Use 'rb' (read binary) mode to load the pickle file
            with open(filepath, 'rb') as f:
                # The file content is the list of dictionaries saved by your recorder
                trajectory = pickle.load(f)
                all_trajectories.extend(trajectory)
                
    print(f"Successfully loaded and combined {len(os.listdir(log_dir))} files.")
    print(f"Total number of transitions (steps): {len(all_trajectories)}")
    return all_trajectories

# Example usage:
raw_transitions = load_all_trajectories()

def extract_bc_data(raw_transitions):
    """
    Extracts all image and scalar observations, actions, rewards, and dones.
    Uses robust key checking to skip corrupted transitions gracefully.
    """
    obs_pov_list = []
    obs_time_list = []
    obs_yaw_list = []
    obs_pitch_list = []
    
    action_list = []
    reward_list = []
    done_list = []
    
    # These are the four keys required by the DQN network
    REQUIRED_OBS_KEYS = ['pov', 'time_left', 'yaw', 'pitch']
    
    for i, transition in enumerate(raw_transitions):
        try:
            state_dict = transition.get('state', None)
            
            # 1. Basic sanity check
            if not isinstance(state_dict, dict):
                 print(f"[WARNING] Skipping record at index {i}. 'state' is corrupted (not a dict).")
                 continue

            # 2. Check for all 4 required observation keys
            if not all(k in state_dict for k in REQUIRED_OBS_KEYS):
                 print(f"[WARNING] Skipping record at index {i}. Missing scalar keys.")
                 continue

            # 3. Safely Extract Observations (assumes scalars are shape (1,))
            obs_pov_list.append(state_dict['pov'])
            obs_time_list.append(float(state_dict['time_left'][0]))
            obs_yaw_list.append(float(state_dict['yaw'][0]))
            obs_pitch_list.append(float(state_dict['pitch'][0]))

            # 4. Extract Action/Reward/Done
            raw_action = transition['action']
            # We assume the action is already a discrete index (0-22)
            # If recorder saves dict, the mapping logic should be here.
            action_list.append(raw_action) 
            reward_list.append(transition['reward'])
            
            is_done = transition.get('terminated', False) or transition.get('truncated', False)
            done_list.append(is_done)

        except IndexError as e:
            # Catches if the scalar array exists but is the wrong shape (e.g., empty array)
            print(f"[WARNING] Skipping record at index {i}. Corrupted Scalar Shape: {e}")
            continue
        except Exception as e:
            # Final catch-all for other issues (e.g., missing 'action' key)
            print(f"[WARNING] Skipping record at index {i} due to generic error: {e}")
            continue
            
    if not obs_pov_list:
        raise ValueError("No valid observations were extracted. Data extraction failed.")

    # Convert lists to NumPy arrays
    observations = np.stack(obs_pov_list)
    # ... (rest of array creation)
    actions = np.array(action_list, dtype=np.int64)
    rewards = np.array(reward_list, dtype=np.float32)
    dones = np.array(done_list, dtype=np.bool_)
    
    # Re-package the output to match the original structure, plus scalars
    return {
        'obs_pov': observations,
        'obs_time': np.array(obs_time_list, dtype=np.float32),
        'obs_yaw': np.array(obs_yaw_list, dtype=np.float32),
        'obs_pitch': np.array(obs_pitch_list, dtype=np.float32),
        'actions': actions,
        'rewards': rewards,
        'dones': dones
    }


if __name__ == "__main__":
    raw_transitions = load_all_trajectories()

    # 2. Extract and format the observations and actions into NumPy arrays
    try:
        observations, actions = extract_bc_data(raw_transitions)
        
        # 3. Save the processed data to a compressed NumPy file (.npz)
        OUTPUT_FILENAME = "bc_expert_data.npz"
        np.savez_compressed(OUTPUT_FILENAME, obs=observations, actions=actions)
        
        print(f"\n SUCCESS: Processed data saved to {OUTPUT_FILENAME}")
        print("This file is ready for training.")

    except ValueError as e:
        print(f"\n FAILURE: Extraction failed.")
        print(f"Error: {e}")
        print("Check your observation dictionary keys or trajectory file integrity.")

