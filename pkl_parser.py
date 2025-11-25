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

def _get_discrete_action_index(action_dict):
    """
    Converts a raw MineRL action dictionary to the discrete index used by SimpleActionWrapper.
    Assumes only one key is active, or defaults to the macro index (7).
    """
    # 0: no_op
    if not any(action_dict.values()):
        return 0
    
    # Primitives based on SimpleActionWrapper's list (0-6)
    # The list is: [noop, forward, back, right, left, jump, attack]
    if action_dict.get('forward', 0) == 1: return 1
    if action_dict.get('back', 0) == 1: return 2
    if action_dict.get('right', 0) == 1: return 3
    if action_dict.get('left', 0) == 1: return 4
    if action_dict.get('jump', 0) == 1: return 5
    if action_dict.get('attack', 0) == 1: return 6

    # 7: Macro/Other Actions (We assume the last index is the pipeline macro)
    return 7

def extract_bc_data(raw_transitions):
    """
    Extracts the image observation ('pov') and actions for Behavior Cloning.
    """
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    
    for i, transition in enumerate(raw_transitions):
        try:
            state_dict = transition.get('state', None)
            
            # CRITICAL ROBUSTNESS CHECK: state_dict MUST be a dictionary
            if not isinstance(state_dict, dict):
                print(f"[WARNING] Skipping corrupted record at index {i}. 'state' is type {type(state_dict).__name__} (expected dict).")
                continue
                
            if 'pov' in state_dict:
                obs_list.append(state_dict['pov'])
            else:
                print(f"[WARNING] Skipping record at index {i}. 'state' dict missing 'pov' key.")
                continue # Skip transition if POV is missing

            raw_action = transition['action']
            discrete_action = _get_discrete_action_index(raw_action)
            action_list.append(discrete_action)
            #action_list.append(transition['action'])
            reward_list.append(transition['reward'])
            
            # Handle both 'terminated' and 'truncated' or combined 'done' state
            is_done = transition.get('terminated', False) or transition.get('truncated', False)
            done_list.append(is_done)

        except Exception as e:
            # Catch other potential errors (like missing 'action' key)
            print(f"[WARNING] Skipping record at index {i} due to generic error: {e}")
            continue

    if not obs_list:
        raise ValueError("No valid observations were extracted. Data extraction failed.")

    observations = np.stack(obs_list)
    actions = np.array(action_list)
    rewards = np.array(reward_list, dtype=np.float32)
    dones = np.array(done_list, dtype=np.bool_)
    
    # Sanity check that all arrays have the same length
    if not (len(observations) == len(actions) == len(rewards) == len(dones)):
        print(f"[ERROR] Array length mismatch: {len(observations)} obs vs {len(actions)} actions.")
        # This should be prevented by the loops above, but useful to keep for safety.

    return observations, actions, rewards, dones


if __name__ == "__main__":
    raw_transitions = load_all_trajectories()

    # 2. Extract and format the observations and actions into NumPy arrays
    try:
        observations, actions, rewards, dones = extract_bc_data(raw_transitions)
        
        # 3. Save the processed data to a compressed NumPy file (.npz)
        OUTPUT_FILENAME = "bc_expert_data.npz"
        np.savez_compressed(OUTPUT_FILENAME, obs=observations, actions=actions, rewards=rewards,
            dones=dones)
        
        print(f"\n SUCCESS: Processed data saved to {OUTPUT_FILENAME}")
        print("This file is ready for training.")

    except ValueError as e:
        print(f"\n FAILURE: Extraction failed.")
        print(f"Error: {e}")
        print("Check your observation dictionary keys or trajectory file integrity.")

