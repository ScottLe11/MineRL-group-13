"""
Visualize both GradCAM and CBAM Attention (Channel + Spatial) for a trained PPO model.

This script:
1. Loads the trained model checkpoint
2. Collects sample observations from the environment
3. Generates GradCAM heatmap
4. Extracts CBAM attention weights (channel + spatial)
5. Creates comprehensive visualizations showing all attention mechanisms
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from utils.agent_factory import create_agent
from utils.visualization import GradCAM, create_visual_overlay
from utils.env_factory import create_env


def load_model_and_config(checkpoint_path, config_path='config/config.yaml'):
    """Load the trained model and config."""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Calculate num_actions from action_space config
    from utils.env_factory import parse_action_space_config
    enabled_actions = parse_action_space_config(config['action_space'])
    num_actions = len(enabled_actions)

    # Handle device="auto" by setting to None (agent will auto-detect)
    if config.get('device') == 'auto':
        config['device'] = None

    print(f"Loading model from {checkpoint_path}")
    agent = create_agent(config, num_actions)

    checkpoint = torch.load(checkpoint_path, map_location=agent.device)

    # Handle different checkpoint formats
    if 'network' in checkpoint:
        # Check if network contains policy_state_dict
        if isinstance(checkpoint['network'], dict) and 'policy_state_dict' in checkpoint['network']:
            state_dict = checkpoint['network']['policy_state_dict']
        else:
            state_dict = checkpoint['network']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        raise KeyError(f"Checkpoint keys: {list(checkpoint.keys())}")

    # PPO uses .policy, DQN uses .q_network
    if hasattr(agent, 'policy'):
        agent.policy.load_state_dict(state_dict)
        agent.policy.eval()
        model = agent.policy
    elif hasattr(agent, 'q_network'):
        agent.q_network.load_state_dict(state_dict)
        agent.q_network.eval()
        model = agent.q_network
    else:
        raise AttributeError("Agent has neither 'policy' nor 'q_network' attribute")

    print(f"  Model loaded successfully!")
    print(f"  Episode: {checkpoint.get('episode', 'unknown')}")
    print(f"  Architecture: {config['network']['architecture']}")
    print(f"  Attention: {config['network']['attention']}")

    return agent, config, model


def get_target_conv_layer(model):
    """Get the last convolutional layer from the model for GradCAM."""
    # For PPO model with CNN backbone
    if hasattr(model, 'cnn') and hasattr(model.cnn, 'conv'):
        # Find last conv layer
        last_conv = None
        for module in reversed(list(model.cnn.conv.modules())):
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                break
        return last_conv
    return None


def collect_sample_observations(env, agent, num_samples=5):
    """Collect sample observations from the environment."""
    print(f"\nCollecting {num_samples} sample observations...")
    observations = []

    for i in range(num_samples):
        obs = env.reset()
        # Take a few random steps to get interesting frames
        for _ in range(np.random.randint(5, 20)):
            # PPO agent returns (action, log_prob, value)
            if hasattr(agent, 'select_action'):
                result = agent.select_action(obs)
                action = result[0] if isinstance(result, tuple) else result
            else:
                action = env.action_space.sample()

            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        observations.append(obs)
        print(f"  Collected sample {i+1}/{num_samples}")

    return observations


def generate_gradcam_heatmap(model, target_layer, obs, target_action_index, device):
    """Generate GradCAM heatmap for a given observation."""
    # Prepare input tensor
    pov = obs['pov']
    if isinstance(pov, np.ndarray):
        pov = torch.from_numpy(pov).to(device)

    if pov.dim() == 3:
        pov = pov.unsqueeze(0)

    if pov.max() > 1.0:
        pov = pov.float() / 255.0

    # Create observation dict for model
    obs_dict = {
        'pov': pov,
        'time_left': torch.tensor([obs.get('time_left', 1.0)]).to(device),
        'yaw': torch.tensor([obs.get('yaw', 0.0)]).to(device),
        'pitch': torch.tensor([obs.get('pitch', 0.0)]).to(device),
        'place_table_safe': torch.tensor([obs.get('place_table_safe', 0.0)]).to(device),
        'inv_logs': torch.tensor([obs.get('inv_logs', 0.0)]).to(device),
        'inv_planks': torch.tensor([obs.get('inv_planks', 0.0)]).to(device),
        'inv_sticks': torch.tensor([obs.get('inv_sticks', 0.0)]).to(device),
        'inv_table': torch.tensor([obs.get('inv_table', 0.0)]).to(device),
        'inv_axe': torch.tensor([obs.get('inv_axe', 0.0)]).to(device)
    }

    # Generate GradCAM
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    # Forward pass to get action logits
    model.eval()
    with torch.no_grad():
        logits, _ = model(obs_dict)

    if target_action_index is None:
        target_action_index = torch.argmax(logits).item()

    # Generate heatmap
    model.zero_grad()
    logits, _ = model(obs_dict)

    # Backward pass for target action
    one_hot = torch.zeros_like(logits)
    one_hot[0, target_action_index] = 1
    logits.backward(gradient=one_hot, retain_graph=True)

    # Get gradients and feature maps
    if grad_cam.gradients is None or grad_cam.feature_maps is None:
        raise RuntimeError("Could not retrieve gradients or feature maps")

    # Pool gradients
    pooled_gradients = torch.mean(grad_cam.gradients, dim=[2, 3])

    # Weight feature maps
    feature_maps = grad_cam.feature_maps[0].clone()
    for i in range(len(pooled_gradients[0])):
        feature_maps[i, :, :] *= pooled_gradients[0, i]

    # Generate heatmap
    heatmap = torch.mean(feature_maps, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    return heatmap, target_action_index


def extract_attention_weights(model, obs, device):
    """Extract CBAM attention weights (channel + spatial) from the model."""
    # Prepare input
    pov = obs['pov']
    if isinstance(pov, np.ndarray):
        pov = torch.from_numpy(pov).to(device)

    if pov.dim() == 3:
        pov = pov.unsqueeze(0)

    if pov.max() > 1.0:
        pov = pov.float() / 255.0

    obs_dict = {
        'pov': pov,
        'time_left': torch.tensor([obs.get('time_left', 1.0)]).to(device),
        'yaw': torch.tensor([obs.get('yaw', 0.0)]).to(device),
        'pitch': torch.tensor([obs.get('pitch', 0.0)]).to(device),
        'place_table_safe': torch.tensor([obs.get('place_table_safe', 0.0)]).to(device),
        'inv_logs': torch.tensor([obs.get('inv_logs', 0.0)]).to(device),
        'inv_planks': torch.tensor([obs.get('inv_planks', 0.0)]).to(device),
        'inv_sticks': torch.tensor([obs.get('inv_sticks', 0.0)]).to(device),
        'inv_table': torch.tensor([obs.get('inv_table', 0.0)]).to(device),
        'inv_axe': torch.tensor([obs.get('inv_axe', 0.0)]).to(device)
    }

    # Forward pass through CNN with attention
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'cnn') and hasattr(model, 'attention'):
            # Get conv features
            conv_features = model.cnn.conv(obs_dict['pov'])

            # Apply CBAM attention if available
            if hasattr(model.attention, 'channel_attention') and hasattr(model.attention, 'spatial_attention'):
                # CBAM case
                x_channel, channel_weights = model.attention.channel_attention(conv_features)
                x_spatial, spatial_weights = model.attention.spatial_attention(x_channel)

                return {
                    'channel': channel_weights.cpu().numpy(),
                    'spatial': spatial_weights.cpu().numpy(),
                    'conv_features': conv_features.cpu().numpy()
                }
            elif hasattr(model.attention, 'forward'):
                # Try to get attention map from forward pass
                attended, attention_info = model.attention(conv_features)
                if isinstance(attention_info, dict):
                    return {
                        'channel': attention_info.get('channel', None),
                        'spatial': attention_info.get('spatial', None),
                        'conv_features': conv_features.cpu().numpy()
                    }

    return None


def visualize_all(obs, gradcam_heatmap, attention_weights, action_idx, save_path):
    """Create comprehensive visualization showing GradCAM and CBAM attention."""
    # Get the last frame from POV
    pov = obs['pov']
    if isinstance(pov, torch.Tensor):
        pov = pov.cpu().numpy()

    # Get last frame (most recent)
    if pov.shape[0] == 4:  # (4, H, W) - frame stack
        frame = pov[-1]
    else:
        frame = pov[0] if pov.shape[0] == 1 else pov

    # Normalize frame to [0, 255]
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)

    # Determine number of subplots based on available data
    num_plots = 3  # Frame, GradCAM, GradCAM overlay
    if attention_weights is not None:
        if attention_weights.get('channel') is not None:
            num_plots += 1
        if attention_weights.get('spatial') is not None:
            num_plots += 1

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    plot_idx = 0

    # 1. Original frame
    axes[plot_idx].imshow(frame, cmap='gray')
    axes[plot_idx].set_title('Original Frame', fontsize=12, fontweight='bold')
    axes[plot_idx].axis('off')
    plot_idx += 1

    # 2. GradCAM heatmap
    im = axes[plot_idx].imshow(gradcam_heatmap, cmap='jet', vmin=0, vmax=1)
    axes[plot_idx].set_title(f'GradCAM\n(Action {action_idx})', fontsize=12, fontweight='bold')
    axes[plot_idx].axis('off')
    plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
    plot_idx += 1

    # 3. GradCAM overlay
    gradcam_resized = cv2.resize(gradcam_heatmap, (frame.shape[1], frame.shape[0]))
    axes[plot_idx].imshow(frame, cmap='gray')
    axes[plot_idx].imshow(gradcam_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[plot_idx].set_title('GradCAM Overlay', fontsize=12, fontweight='bold')
    axes[plot_idx].axis('off')
    plot_idx += 1

    # 4. Channel Attention (if available)
    if attention_weights is not None and attention_weights.get('channel') is not None:
        channel_attn = attention_weights['channel'][0, :, 0, 0]  # (C,)
        axes[plot_idx].bar(range(len(channel_attn)), channel_attn)
        axes[plot_idx].set_title('Channel Attention Weights', fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('Channel Index')
        axes[plot_idx].set_ylabel('Weight')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    # 5. Spatial Attention (if available)
    if attention_weights is not None and attention_weights.get('spatial') is not None:
        spatial_attn = attention_weights['spatial'][0, 0]  # (H, W)
        im = axes[plot_idx].imshow(spatial_attn, cmap='hot', vmin=0, vmax=1)
        axes[plot_idx].set_title('Spatial Attention Map', fontsize=12, fontweight='bold')
        axes[plot_idx].axis('off')
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
        plot_idx += 1

    plt.suptitle(f'Attention Analysis - Action: {action_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved visualization to {save_path}")

    return save_path


def main():
    # Configuration
    checkpoint_path = 'best_model/continuethisoneplz.pt'
    config_path = 'config/config.yaml'
    output_dir = 'attention_visualizations'
    num_samples = 3

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ATTENTION + GRADCAM VISUALIZATION")
    print("=" * 80)

    # Load model and config
    agent, config, model = load_model_and_config(checkpoint_path, config_path)

    # Get target layer for GradCAM
    target_layer = get_target_conv_layer(model)
    if target_layer is None:
        print("ERROR: Could not find target convolutional layer for GradCAM")
        return

    print(f"\nTarget layer for GradCAM: {target_layer}")

    # Create environment
    print("\nCreating environment...")
    env = create_env(config)

    # Collect sample observations
    observations = collect_sample_observations(env, agent, num_samples)

    # Process each observation
    print(f"\nGenerating visualizations...")
    print("=" * 80)

    for i, obs in enumerate(observations):
        print(f"\nProcessing sample {i+1}/{num_samples}:")

        # Generate GradCAM heatmap
        print("  Generating GradCAM...")
        gradcam_heatmap, action_idx = generate_gradcam_heatmap(
            model,
            target_layer,
            obs,
            target_action_index=None,  # Use best action
            device=agent.device
        )

        # Extract attention weights
        print("  Extracting attention weights...")
        attention_weights = extract_attention_weights(model, obs, agent.device)

        if attention_weights is not None:
            print(f"    ✓ Channel attention shape: {attention_weights['channel'].shape}")
            print(f"    ✓ Spatial attention shape: {attention_weights['spatial'].shape}")
        else:
            print("    ⚠ No attention weights extracted")

        # Create visualization
        save_path = os.path.join(output_dir, f'attention_analysis_sample_{i+1}.png')
        visualize_all(obs, gradcam_heatmap, attention_weights, action_idx, save_path)

    # Clean up
    env.close()

    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to '{output_dir}/'")
    print("=" * 80)


if __name__ == "__main__":
    main()
