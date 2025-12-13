# utils/run_grad_cam.py
import torch
import cv2
import sys
import os

# Add project root to path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from networks.dqn_network import DQNNetwork
from utils.visualization import generate_grad_cam_overlay
from utils.env_factory import create_env
from utils.config import load_config

def main():
    """
    Generates and saves a Grad-CAM visualization for a trained DQN model.
    This script serves as a simple way to test the Grad-CAM functionality.
    """
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIG_PATH = "config/config.yaml"
    
    # Load configuration
    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{CONFIG_PATH}'")
        return

    # --- Model and Checkpoint ---
    MODEL_CHECKPOINT_PATH = config['training'].get('grad_cam_checkpoint_path', "checkpoints/best_model_dqn.pt")
    
    # --- Target Action ---
    # The action to generate the heatmap for (e.g., 'attack')
    # The index must match the model's action space.
    ATTACK_ACTION_INDEX = config['grad_cam'].get('attack_action_index', 6)

    # 1. Initialize Model
    num_actions = config['agent']['num_actions']
    model = DQNNetwork(
        num_actions=num_actions,
        input_channels=config['agent']['input_channels'],
        num_scalars=config['agent']['num_scalars'],
        cnn_architecture=config['agent']['cnn_architecture'],
        attention_type=config['agent']['attention_type']
    ).to(DEVICE)
    
    try:
        # We need the full checkpoint to load the agent state if necessary,
        # but only the network state dict is used for the model.
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['q_network_state_dict'])
        print(f"✅ Successfully loaded model from {MODEL_CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Model checkpoint not found at '{MODEL_CHECKPOINT_PATH}'.")
        print("   Please update the 'grad_cam_checkpoint_path' in your config.yaml.")
        return
    except KeyError:
        print(f"❌ Error: 'q_network_state_dict' not found in the checkpoint.")
        print("   The checkpoint may be corrupted or from a different agent type.")
        return

    model.eval()

    # 2. Select the Target Layer for Grad-CAM
    try:
        target_layer = model.cnn.conv[4] # Last conv layer in the 'medium' architecture
        print(f"✅ Target layer for Grad-CAM: {target_layer}")
    except (AttributeError, IndexError) as e:
        print(f"❌ Error selecting target layer: {e}")
        print("   Please inspect your model's architecture and set `target_layer` correctly.")
        return

    # 3. Get a Raw Observation from the Environment
    print("\nStarting MineRL environment to get a single observation...")
    # We create a base, unwrapped environment to get a single raw frame.
    # The visualization function will handle the necessary preprocessing.
    try:
        raw_env = create_env(config, wrap=False)
        obs = raw_env.reset()
        raw_env.close()
        print("✅ Environment created and observation received.")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print("   Check your MineRL installation and environment configuration.")
        return

    # 4. Generate Grad-CAM Overlay using the utility function
    print(f"Generating heatmap for the 'attack' action (index: {ATTACK_ACTION_INDEX})...")
    try:
        overlay_image = generate_grad_cam_overlay(
            model=model,
            target_layer=target_layer,
            obs=obs, # Pass the raw observation
            target_action_index=ATTACK_ACTION_INDEX,
            device=DEVICE
        )
    except Exception as e:
        print(f"❌ Error during Grad-CAM generation: {e}")
        return

    # 5. Save the Visualization
    output_filename = config['grad_cam'].get('output_filename', "grad_cam_visualization.jpg")
    # Convert RGB (from function) to BGR for OpenCV's imwrite
    cv2.imwrite(output_filename, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
    print(f"\n✅ Successfully saved Grad-CAM visualization to '{output_filename}'")
    print("   You can now view this image to see the model's attention.")

if __name__ == '__main__':
    main()
