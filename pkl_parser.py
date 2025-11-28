import numpy as np
import pickle
import os
from collections import OrderedDict

# Full pool action indices (from wrappers/actions.py)
FULL_ACTION_POOL = {
    'noop': 0, 'forward': 1, 'back': 2, 'right': 3, 'left': 4, 'jump': 5, 'attack': 6,
    'turn_left_30': 7, 'turn_left_45': 8, 'turn_left_60': 9, 'turn_left_90': 10,
    'turn_right_30': 11, 'turn_right_45': 12, 'turn_right_60': 13, 'turn_right_90': 14,
    'look_up_12': 15, 'look_up_20': 16, 'look_down_12': 17, 'look_down_20': 18,
    'craft_planks': 19, 'make_table': 20, 'craft_sticks': 21, 'craft_axe': 22,
    'craft_entire_axe': 23, 'attack_5': 24, 'attack_10': 25,
}

# High-priority low-level keys mapped to their original index (0-25)
PRIMITIVE_MAP = OrderedDict([
    ('attack', 6), 
    ('jump', 5),
    ('forward', 1),
    ('back', 2),
    ('left', 4),
    ('right', 3),
])


def discretize_action_factory(enabled_actions: list) -> callable:
    """
    Creates a discrete action function that maps the recorded 
    low-level action dictionary to one of the enabled indices (0 to N-1).
    
    Args:
        enabled_actions: List of original action indices (0-25) defined in config.
                         e.g., [1, 8, 9, 12, 13, 15, 17, 25]
                         
    Returns:
        A function: (action_dict) -> mapped_discrete_index (0 to 7)
    """
    # Create reverse map: Original_Index (e.g., 1) -> Mapped_Index (e.g., 0)
    reverse_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(enabled_actions)}
    
    # ----------------------------------------------------------------------
    # BASE 23 CHECK (Only necessary if we want to run the full set)
    # The logic below defaults to the largest supported action if an unsupported 
    # action is chosen. This is simpler than full dynamic priority.
    # ----------------------------------------------------------------------
    
    def discretize_action(action_dict: dict) -> int:
        """
        Maps a low-level action dictionary to a discrete index (0 to N-1) 
        based on the provided enabled_actions list.
        """
        
        # --- PRIORITY 1: ATTACK ---
        if action_dict.get('attack', 0) == 1:
            # Prefer the strongest available attack macro if a primitive attack (6) is used.
            if FULL_ACTION_POOL['attack_10'] in enabled_actions:
                return reverse_map[FULL_ACTION_POOL['attack_10']] # Index 25 -> Mapped Index 7
            elif FULL_ACTION_POOL['attack'] in enabled_actions:
                return reverse_map[FULL_ACTION_POOL['attack']] # Index 6 -> Mapped Index N
            # If no attack is enabled, fall through to movement/camera
            
        # --- PRIORITY 2: MOVEMENT ---
        # Note: In your custom list, only 'forward' (1) is available.
        if FULL_ACTION_POOL['forward'] in enabled_actions and action_dict.get('forward', 0) == 1:
            return reverse_map[FULL_ACTION_POOL['forward']] # Index 1 -> Mapped Index 0
            
        # --- PRIORITY 3: CAMERA ---
        camera_array = action_dict.get('camera', [0.0, 0.0])
        pitch_delta = camera_array[0]
        yaw_delta = camera_array[1]
        
        # Use thresholds matching the high-angle primitives
        YAW_THRESHOLD = 10.0
        PITCH_THRESHOLD = 3.0

        # Yaw (Left/Right) - Prioritize 60 degree turns (Index 9, 13)
        if yaw_delta < -YAW_THRESHOLD:
            if FULL_ACTION_POOL['turn_left_60'] in enabled_actions:
                return reverse_map[FULL_ACTION_POOL['turn_left_60']] # Index 9 -> Mapped Index 2
            elif FULL_ACTION_POOL['turn_left_45'] in enabled_actions:
                return reverse_map[FULL_ACTION_POOL['turn_left_45']] # Index 8 -> Mapped Index 1
        
        if yaw_delta > YAW_THRESHOLD:
            if FULL_ACTION_POOL['turn_right_60'] in enabled_actions:
                return reverse_map[FULL_ACTION_POOL['turn_right_60']] # Index 13 -> Mapped Index 4
            elif FULL_ACTION_POOL['turn_right_45'] in enabled_actions:
                return reverse_map[FULL_ACTION_POOL['turn_right_45']] # Index 12 -> Mapped Index 3
                
        # Pitch (Up/Down) - Map to the available options (Index 15, 17)
        if pitch_delta < -PITCH_THRESHOLD and FULL_ACTION_POOL['look_up_12'] in enabled_actions:
            return reverse_map[FULL_ACTION_POOL['look_up_12']] # Index 15 -> Mapped Index 5
        if pitch_delta > PITCH_THRESHOLD and FULL_ACTION_POOL['look_down_12'] in enabled_actions:
            return reverse_map[FULL_ACTION_POOL['look_down_12']] # Index 17 -> Mapped Index 6

        # 4. Default: NOOP/FALLBACK
        # Since NOOP (index 0) is not in the custom list, we must map all non-actions 
        # to an enabled action. We will use the lowest priority enabled action, 
        # which in this case is 'forward' (1) if it exists, or the first element.
        if FULL_ACTION_POOL['forward'] in enabled_actions:
             return reverse_map[FULL_ACTION_POOL['forward']]
        
        # If all else fails, use the first enabled action.
        return reverse_map[enabled_actions[0]]

    return discretize_action

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



def extract_bc_data(raw_transitions, config: dict):
    action_config = config['action_space']
    preset = action_config.get('preset', 'base')
    
    if preset == 'custom' and action_config.get('enabled_actions'):
        enabled_actions = action_config['enabled_actions']
    else:
        # Fallback to the full base 23 actions if config is default/missing
        enabled_actions = list(range(23))
        
    discretize = discretize_action_factory(enabled_actions)
    
    """
    Extracts all image and scalar observations, actions, rewards, and dones.
    Uses robust key checking to skip corrupted transitions gracefully.
    """
    obs_pov_list = []
    obs_time_list = []
    obs_yaw_list = []
    obs_pitch_list = []
    obs_place_table_safe_list = []

    action_list = []
    reward_list = []
    done_list = []

    # These are the five keys required by the DQN network
    REQUIRED_OBS_KEYS = ['pov', 'time_left', 'yaw', 'pitch', 'place_table_safe']
    
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
            obs_place_table_safe_list.append(float(state_dict['place_table_safe'][0]))

            # 4. Extract Action/Reward/Done
            raw_action = transition['action']
            if not isinstance(raw_action, dict):
                 # This should ideally not happen if recording was done correctly
                 print(f"[WARNING] Skipping record at index {i}. Action is not a dictionary.")
                 continue
            # We assume the action is already a discrete index (0-22)
            # If recorder saves dict, the mapping logic should be here.
            discrete_action_mapped_index = discretize(raw_action)
            action_list.append(discrete_action_mapped_index)
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
    
    actions = np.array(action_list, dtype=np.int64)
    rewards = np.array(reward_list, dtype=np.float32)
    dones = np.array(done_list, dtype=np.bool_)
    
    # Re-package the output to match the original structure, plus scalars
    return {
        'obs_pov': observations,
        'obs_time': np.array(obs_time_list, dtype=np.float32),
        'obs_yaw': np.array(obs_yaw_list, dtype=np.float32),
        'obs_pitch': np.array(obs_pitch_list, dtype=np.float32),
        'obs_place_table_safe': np.array(obs_place_table_safe_list, dtype=np.float32),
        'actions': actions,
        'rewards': rewards,
        'dones': dones
    }


if __name__ == "__main__":
    mock_config = {'action_space': {'preset': 'custom', 'enabled_actions': [1, 8, 9, 12, 13, 15, 17, 25]}}
    discretizer = discretize_action_factory(mock_config['action_space']['enabled_actions'])
    
    # Test cases:
    # Action 25 (attack_10) -> Mapped Index 7
    # Action 1 (forward) -> Mapped Index 0
    # Camera turn > 10.0 -> Mapped Index 2 or 4
    
    assert discretizer({'attack': 1}) == 7, "Attack should map to attack_10 (Mapped Index 7)"
    assert discretizer({'forward': 1}) == 0, "Forward should map to Mapped Index 0"
    assert discretizer({'camera': [0, 20.0]}) == 4, "Large yaw right should map to turn_right_60 (Mapped Index 4)"
    print("✅ Discretizer logic validated.")
    
    # Example usage:
    try:
        raw_transitions = load_all_trajectories()

        # 2. Extract and format the observations and actions into NumPy arrays
        data = extract_bc_data(raw_transitions, config=mock_config)
            
        # 3. Save the processed data to a compressed NumPy file (.npz)
        OUTPUT_FILENAME = "bc_expert_data.npz"
        # Combine all required keys into the output file
        np.savez_compressed(OUTPUT_FILENAME, **data)
            
        print(f"\n SUCCESS: Processed data saved to {OUTPUT_FILENAME}")
        print(f" \t Total discrete actions mapped: {len(data['actions'])}")
        print("This file is ready for Behavioral Cloning (BC) training.")

    except ValueError as e:
        print(f"\n FAILURE: Extraction failed.")
        print(f"Error: {e}")

