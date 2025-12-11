"""
Verify the transferred model checkpoint.

This script loads and inspects the transferred model to ensure it was created correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from networks.policy_network import ActorCriticNetwork


def verify_transferred_model(checkpoint_path: str = "checkpoints/transferred_model.pt"):
    """
    Load and verify the transferred model checkpoint.

    Args:
        checkpoint_path: Path to transferred model checkpoint
    """
    print("=" * 80)
    print("VERIFYING TRANSFERRED MODEL")
    print("=" * 80)

    # Load checkpoint
    print(f"\n[1/4] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Check checkpoint structure
    print(f"\n[2/4] Checking checkpoint structure...")

    # Check for new PPO-compatible format
    if 'network' in checkpoint and 'policy_state_dict' in checkpoint['network']:
        print("  ✓ Found network.policy_state_dict (PPO format)")
        state_dict = checkpoint['network']['policy_state_dict']
    elif 'model_state_dict' in checkpoint:
        print("  ⚠️  Found model_state_dict (old format - won't load in PPO!)")
        state_dict = checkpoint['model_state_dict']
    elif 'policy_state_dict' in checkpoint:
        print("  ✓ Found policy_state_dict (old PPO format)")
        state_dict = checkpoint['policy_state_dict']
    else:
        print("  ✗ No recognized state dict found!")
        return

    if 'transfer_info' in checkpoint:
        print("  ✓ Found transfer_info")
        info = checkpoint['transfer_info']
    else:
        print("  ⚠ No transfer_info found (may be old checkpoint)")
        info = {}

    if 'progress_settings' in checkpoint:
        print("  ✓ Found progress_settings")
    else:
        print("  ⚠ No progress_settings (will start from episode 0)")

    # Print transfer info
    if info:
        print(f"\n[3/4] Transfer Information:")
        print(f"  Source model: {info.get('source_model', 'Unknown')}")
        print(f"  Source config: {info.get('source_config', 'Unknown')}")
        print(f"  Target config: {info.get('target_config', 'Unknown')}")
        print(f"  Old actions: {info.get('old_num_actions', 'Unknown')}")
        print(f"  New actions: {info.get('new_num_actions', 'Unknown')}")

        if 'action_mapping' in info:
            print(f"\n  Action Mapping:")
            for old_pos, new_pos in info['action_mapping'].items():
                print(f"    Old position {old_pos} → New position {new_pos}")

    # Verify state dict keys
    print(f"\n[4/4] Verifying state dict keys...")
    expected_prefixes = ['cnn.', 'attention.', 'scalar_network.', 'actor.', 'critic.']
    found_prefixes = set()

    for key in state_dict.keys():
        for prefix in expected_prefixes:
            if key.startswith(prefix):
                found_prefixes.add(prefix)

    print(f"  Found components:")
    for prefix in expected_prefixes:
        status = "✓" if prefix in found_prefixes else "✗"
        component = prefix.rstrip('.')
        print(f"    {status} {component}")

    # Check specific weights
    print(f"\n  Checking key weights:")

    # CNN
    if 'cnn.conv.0.weight' in state_dict:
        cnn_shape = state_dict['cnn.conv.0.weight'].shape
        print(f"    CNN first layer: {cnn_shape}")

    # Scalar network
    if 'scalar_network.network.0.weight' in state_dict:
        scalar_shape = state_dict['scalar_network.network.0.weight'].shape
        num_scalars = scalar_shape[1]
        print(f"    Scalar network input: {scalar_shape} ({num_scalars} scalars)")

    # Actor
    if 'actor.2.weight' in state_dict:
        actor_shape = state_dict['actor.2.weight'].shape
        num_actions = actor_shape[0]
        print(f"    Actor output: {actor_shape} ({num_actions} actions)")

    # Critic
    if 'critic.2.weight' in state_dict:
        critic_shape = state_dict['critic.2.weight'].shape
        print(f"    Critic output: {critic_shape}")

    # Try loading into a network
    print(f"\n  Testing model loading...")
    try:
        # Get num_actions from transfer_info or from state_dict
        if info and 'new_num_actions' in info:
            num_actions = info['new_num_actions']
        else:
            # Infer from actor output layer
            num_actions = state_dict['actor.2.weight'].shape[0]

        network = ActorCriticNetwork(
            num_actions=num_actions,
            input_channels=4,
            num_scalars=9,
            cnn_architecture='medium',
            attention_type='cbam',
            use_scalar_network=True,
            scalar_hidden_dim=64,
            scalar_output_dim=64
        )
        network.load_state_dict(state_dict)
        print(f"    ✓ Successfully loaded into ActorCriticNetwork")
        print(f"    Total parameters: {sum(p.numel() for p in network.parameters()):,}")
    except Exception as e:
        print(f"    ✗ Failed to load: {e}")
        return

    # Test forward pass
    print(f"\n  Testing forward pass...")
    try:
        batch = {
            'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
            'time_left': torch.rand(2),
            'yaw': torch.rand(2),
            'pitch': torch.rand(2),
            'place_table_safe': torch.rand(2),
            'inv_logs': torch.randint(0, 10, (2,)).float(),
            'inv_planks': torch.randint(0, 20, (2,)).float(),
            'inv_sticks': torch.randint(0, 10, (2,)).float(),
            'inv_table': torch.randint(0, 2, (2,)).float(),
            'inv_axe': torch.randint(0, 2, (2,)).float()
        }

        with torch.no_grad():
            logits, value = network(batch)

        print(f"    ✓ Forward pass successful")
        print(f"    Action logits shape: {logits.shape}")
        print(f"    Value shape: {value.shape}")
        print(f"    Sample action probabilities (softmax):")
        probs = torch.softmax(logits[0], dim=0)
        top5_actions = torch.topk(probs, k=5)
        for i, (prob, idx) in enumerate(zip(top5_actions.values, top5_actions.indices)):
            print(f"      Action {idx.item():2d}: {prob.item():.4f}")

    except Exception as e:
        print(f"    ✗ Forward pass failed: {e}")
        return

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE - MODEL IS READY TO USE!")
    print("=" * 80)
    print("\nTo use this model in training:")
    print("  1. Load the checkpoint in your training script:")
    print(f"     checkpoint = torch.load('{checkpoint_path}')")
    print("     policy.load_state_dict(checkpoint['model_state_dict'])")
    print("  2. Continue training with recommended hyperparameters (see TRANSFER_LEARNING_README.md)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify transferred model checkpoint")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/transferred_model.pt',
        help='Path to transferred model checkpoint'
    )

    args = parser.parse_args()

    verify_transferred_model(args.checkpoint)
