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
    Extracts the image observation ('pov') and actions for Behavior Cloning.
    """
    obs_list = []
    action_list = []
    
    for transition in raw_transitions:
        
        # The 'state' is a dictionary. We extract the 'pov' key which holds the image.
        state_dict = transition['state']
        if 'pov' in state_dict:
            image_observation = state_dict['pov']
            obs_list.append(image_observation)
        else:
            # Handle the case where the image key might be different or missing
            raise ValueError("Observation dictionary is missing the 'pov' key.")
            
        action_list.append(transition['action'])

    observations = np.stack(obs_list)
    actions = np.array(action_list)
    
    print(f"Observations shape (should be 4D): {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    
    return observations, actions


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

