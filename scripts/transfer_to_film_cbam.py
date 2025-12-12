"""
Transfer Learning to FiLM-CBAM Architecture.

Transfers weights from a simple ActorCriticNetwork (CBAM) to FiLMCBAMPolicyNetwork.

Key differences:
- Old: ActorCriticNetwork with CBAM attention
- New: FiLMCBAMPolicyNetwork with FiLM conditioning + CBAM

Transfer strategy:
1. CNN (conv1-3): Direct copy ✓
2. CBAM attention: Direct copy ✓
3. FiLM generator: Initialize fresh (NEW component)
4. conv4: Initialize fresh (NEW layer)
5. Scalar network: Initialize fresh (different architecture)
6. Decision layers (fc1, fc2): Initialize fresh (different dimensions)
7. Actor/Critic heads: Initialize fresh (different architecture)

The transferred knowledge:
- Visual feature extraction (conv1-3)
- Attention mechanism (CBAM)
- This gives the model a head start on understanding the visual environment
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from networks.policy_network import ActorCriticNetwork
from networks.film_cbam_policy_network import FiLMCBAMPolicyNetwork


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_action_mapping(old_actions: list, new_actions: list):
    """Create mapping from old action positions to new action positions."""
    mapping = {}
    for old_pos, old_idx in enumerate(old_actions):
        if old_idx in new_actions:
            new_pos = new_actions.index(old_idx)
            mapping[old_pos] = new_pos
        else:
            print(f"Warning: Old action {old_idx} not found in new action space!")
    return mapping


def transfer_to_film_cbam(
    old_model_path: str,
    old_config_path: str,
    new_config_path: str,
    output_path: str,
    device: str = 'cpu'
):
    """
    Transfer weights from ActorCriticNetwork to FiLMCBAMPolicyNetwork.

    Args:
        old_model_path: Path to old model checkpoint
        old_config_path: Path to old configuration
        new_config_path: Path to new configuration
        output_path: Where to save transferred model
        device: Device to use
    """
    print("=" * 80)
    print("TRANSFER TO FiLM-CBAM ARCHITECTURE")
    print("=" * 80)

    # Load configs
    print("\n[1/6] Loading configurations...")
    old_config = load_config(old_config_path)
    new_config = load_config(new_config_path)

    old_actions = old_config['action_space']['enabled_actions']
    new_actions = new_config['action_space']['enabled_actions']
    print(f"  Old actions: {old_actions} ({len(old_actions)} actions)")
    print(f"  New actions: {new_actions} ({len(new_actions)} actions)")

    # Create action mapping
    print("\n[2/6] Creating action space mapping...")
    action_mapping = create_action_mapping(old_actions, new_actions)
    print(f"  Mapped {len(action_mapping)}/{len(old_actions)} actions")

    # Load old model
    print(f"\n[3/6] Loading old model from {old_model_path}...")
    checkpoint = torch.load(old_model_path, map_location=device, weights_only=False)

    # Get old state dict
    if 'network' in checkpoint and 'policy_state_dict' in checkpoint['network']:
        old_state_dict = checkpoint['network']['policy_state_dict']
    elif 'policy_state_dict' in checkpoint:
        old_state_dict = checkpoint['policy_state_dict']
    elif 'model_state_dict' in checkpoint:
        old_state_dict = checkpoint['model_state_dict']
    else:
        old_state_dict = checkpoint

    # Detect old scalar network configuration
    if 'scalar_network.network.0.weight' in old_state_dict:
        old_num_scalars = old_state_dict['scalar_network.network.0.weight'].shape[1]
    else:
        old_num_scalars = 9

    print(f"  Detected old model uses {old_num_scalars} scalars")
    print(f"  Old state dict has {len(old_state_dict)} keys")

    # Create new FiLM-CBAM network
    print(f"\n[4/6] Creating new FiLM-CBAM network...")
    new_network = FiLMCBAMPolicyNetwork(
        num_actions=len(new_actions),
        input_channels=4,
        num_scalars=9
    ).to(device)

    print(f"  New network created: {sum(p.numel() for p in new_network.parameters()):,} params")

    # Transfer weights
    print(f"\n[5/6] Transferring weights...")
    new_state_dict = new_network.state_dict()

    # 1. Transfer CNN (conv1, conv2, conv3)
    print("  ✓ Transferring CNN (conv1, conv2, conv3)...")
    cnn_layers = ['conv1', 'conv2', 'conv3']
    for layer in cnn_layers:
        for param in ['weight', 'bias']:
            old_key = f'cnn.{layer}.{param}' if 'cnn.' in old_state_dict else f'{layer}.{param}'
            new_key = f'{layer}.{param}'

            # Try different key formats
            if old_key in old_state_dict:
                new_state_dict[new_key] = old_state_dict[old_key].clone()
            elif f'cnn.conv.{cnn_layers.index(layer)*2}.{param}' in old_state_dict:
                # Old format: cnn.conv.0.weight, cnn.conv.2.weight, etc.
                idx = cnn_layers.index(layer) * 2
                old_key = f'cnn.conv.{idx}.{param}'
                new_state_dict[new_key] = old_state_dict[old_key].clone()

    # 2. Transfer CBAM attention
    print("  ✓ Transferring CBAM attention...")
    attention_keys = [k for k in old_state_dict.keys() if k.startswith('attention.')]
    transferred_attention = False
    for key in attention_keys:
        # Map old attention.* to new cbam.*
        new_key = key.replace('attention.', 'cbam.')
        if new_key in new_state_dict:
            new_state_dict[new_key] = old_state_dict[key].clone()
            transferred_attention = True

    if not transferred_attention and len(attention_keys) > 0:
        print(f"    Warning: Found {len(attention_keys)} attention keys but couldn't transfer")
    elif transferred_attention:
        print(f"    Transferred {len(attention_keys)} attention parameters")

    # 3. FiLM generator - Initialize fresh (NEW)
    print("  ⚠ FiLM generator initialized fresh (NEW component)")

    # 4. conv4 - Initialize fresh (NEW)
    print("  ⚠ conv4 initialized fresh (NEW layer)")

    # 5. Scalar network - Initialize fresh (different architecture)
    print("  ⚠ Scalar network initialized fresh (different architecture)")
    print(f"    Old: {old_num_scalars} → 64 → 64")
    print(f"    New: 9 → 32 → 32")

    # 6. Decision layers - Initialize fresh (different dimensions)
    print("  ⚠ Decision layers (fc1, fc2) initialized fresh (different dimensions)")

    # 7. Actor/Critic heads - Initialize fresh (different architecture)
    print("  ⚠ Actor/Critic heads initialized fresh (different architecture)")

    # Load transferred weights
    new_network.load_state_dict(new_state_dict, strict=False)

    # Save transferred model in PPO-compatible format
    print(f"\n[6/6] Saving transferred model to {output_path}...")

    checkpoint = {
        'network': {
            'policy_state_dict': new_network.state_dict()
        },
        'progress_settings': {
            'step_count': 0,
            'update_count': 0,
            'episode_count': 0,
            'action_counts': [0] * len(new_actions),
            'last_actions': [],
            'learning_rate': new_config['ppo']['learning_rate']
        },
        'transfer_info': {
            'source_model': old_model_path,
            'source_config': old_config_path,
            'target_config': new_config_path,
            'architecture': 'FiLM-CBAM',
            'old_num_scalars': old_num_scalars,
            'new_num_scalars': 9,
            'old_num_actions': len(old_actions),
            'new_num_actions': len(new_actions),
            'action_mapping': action_mapping,
            'transferred_components': ['CNN (conv1-3)', 'CBAM attention'],
            'fresh_components': ['FiLM generator', 'conv4', 'scalar_network', 'fc1/fc2', 'actor/critic heads']
        }
    }

    torch.save(checkpoint, output_path)

    print("\n" + "=" * 80)
    print("TRANSFER TO FiLM-CBAM COMPLETE!")
    print("=" * 80)
    print(f"\nTransferred model saved to: {output_path}")
    print("\nWhat was transferred:")
    print("  ✓ CNN backbone (conv1, conv2, conv3) - Visual feature extraction")
    print("  ✓ CBAM attention - Focus mechanism")
    print("\nWhat is NEW (initialized fresh):")
    print("  • FiLM generator - Scalar conditioning")
    print("  • conv4 - Additional spatial reasoning")
    print("  • Scalar network - Different architecture (9→32→32)")
    print("  • Decision layers (fc1, fc2) - Different dimensions")
    print("  • Actor/Critic heads - Different architecture")
    print("\nExpectations:")
    print("  - Model has strong VISUAL understanding (CNN + CBAM)")
    print("  - BUT needs to learn:")
    print("    * How to use FiLM conditioning with scalars")
    print("    * New spatial reasoning (conv4)")
    print("    * All action outputs (fresh actor head)")
    print("    * Value estimation (fresh critic head)")
    print("\nTraining recommendations:")
    print("  - Start with learning rate: 0.00005-0.0001")
    print("  - Higher entropy: 0.03-0.05")
    print("  - Be patient: ~200-300 episodes to learn new components")
    print("=" * 80)

    return new_network


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transfer to FiLM-CBAM architecture")
    parser.add_argument(
        '--old-model',
        type=str,
        default='best_model/best_model_ppo_ep2050.pt',
        help='Path to old model checkpoint'
    )
    parser.add_argument(
        '--old-config',
        type=str,
        default='config/bestsimple.yaml',
        help='Path to old configuration'
    )
    parser.add_argument(
        '--new-config',
        type=str,
        default='config/config.yaml',
        help='Path to new configuration'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoints/transferred_film_cbam.pt',
        help='Output path for transferred model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu, cuda, mps)'
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Run transfer
    transfer_to_film_cbam(
        old_model_path=args.old_model,
        old_config_path=args.old_config,
        new_config_path=args.new_config,
        output_path=args.output,
        device=args.device
    )
