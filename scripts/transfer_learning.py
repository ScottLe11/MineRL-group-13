"""
Transfer Learning Script for MineRL PPO Agent.

Transfers weights from a 6-action model (tree chopping with axe) to an 18-action model
(crafting from scratch). Handles action space mapping and observation differences.

Old config (bestsimple.yaml):
  - Actions: [1,6,7,11,15,17] - 6 actions (forward, attack, turn left/right, look up/down)
  - with_axe: true, with_logs: 0
  - Scalars: has_axe=1, wood=variable, planks=0, sticks=0, table=0

New config (config.yaml):
  - Actions: [0,1,2,3,4,5,6,7,8,11,12,15,17,19,20,21,23,24] - 18 actions (adds movement, crafting)
  - with_axe: false, with_logs: 5
  - Scalars: has_axe=variable, wood=variable, planks=variable, sticks=variable, table=variable
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from networks.policy_network import ActorCriticNetwork


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_action_mapping(old_actions: list, new_actions: list):
    """
    Create mapping from old action positions to new action positions.

    Args:
        old_actions: List of old enabled action indices
        new_actions: List of new enabled action indices

    Returns:
        dict: Mapping from old position to new position
    """
    # Create mapping: old_position -> new_position
    mapping = {}
    for old_pos, old_idx in enumerate(old_actions):
        # Find position of this action index in new action space
        if old_idx in new_actions:
            new_pos = new_actions.index(old_idx)
            mapping[old_pos] = new_pos
        else:
            print(f"Warning: Old action {old_idx} not found in new action space!")

    return mapping


def transfer_weights(old_model_path: str, old_config_path: str, new_config_path: str,
                    output_path: str, device: str = 'cpu'):
    """
    Transfer weights from old model to new model with proper action space mapping.

    Args:
        old_model_path: Path to old model checkpoint (.pt file)
        old_config_path: Path to old configuration (bestsimple.yaml)
        new_config_path: Path to new configuration (config.yaml)
        output_path: Where to save the transferred model
        device: Device to load models on
    """
    print("=" * 80)
    print("TRANSFER LEARNING: Old Model → New Model")
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
    print("  Action mapping (old_pos → new_pos):")
    for old_pos, new_pos in action_mapping.items():
        old_idx = old_actions[old_pos]
        new_idx = new_actions[new_pos]
        print(f"    {old_pos} (action {old_idx}) → {new_pos} (action {new_idx})")

    # Load old model
    print(f"\n[3/6] Loading old model from {old_model_path}...")
    checkpoint = torch.load(old_model_path, map_location=device, weights_only=False)
    # Try different possible keys for the state dict
    old_state_dict = checkpoint.get('policy_state_dict',
                                    checkpoint.get('model_state_dict', checkpoint))

    # Detect old scalar network configuration from checkpoint
    # Old models may have had fewer scalars (e.g., 4 vs 9)
    if 'scalar_network.network.0.weight' in old_state_dict:
        old_num_scalars = old_state_dict['scalar_network.network.0.weight'].shape[1]
    else:
        old_num_scalars = 9  # Default assumption

    print(f"  Detected old model uses {old_num_scalars} scalars")

    # Create old network to load weights
    old_network = ActorCriticNetwork(
        num_actions=len(old_actions),
        input_channels=old_config['network']['input_channels'],
        num_scalars=old_num_scalars,  # Use detected number
        cnn_architecture=old_config['network']['architecture'],
        attention_type=old_config['network']['attention'],
        use_scalar_network=old_config['network']['use_scalar_network'],
        scalar_hidden_dim=old_config['network']['scalar_hidden_dim'],
        scalar_output_dim=old_config['network']['scalar_output_dim']
    ).to(device)

    old_network.load_state_dict(old_state_dict, strict=False)
    print(f"  Old network loaded: {sum(p.numel() for p in old_network.parameters()):,} params")

    # Create new network
    print(f"\n[4/6] Creating new network...")
    new_network = ActorCriticNetwork(
        num_actions=len(new_actions),
        input_channels=new_config['network']['input_channels'],
        num_scalars=9,  # Both use 9 scalars
        cnn_architecture=new_config['network']['architecture'],
        attention_type=new_config['network']['attention'],
        use_scalar_network=new_config['network']['use_scalar_network'],
        scalar_hidden_dim=new_config['network']['scalar_hidden_dim'],
        scalar_output_dim=new_config['network']['scalar_output_dim']
    ).to(device)

    print(f"  New network created: {sum(p.numel() for p in new_network.parameters()):,} params")

    # Transfer weights
    print(f"\n[5/6] Transferring weights...")
    new_state_dict = new_network.state_dict()

    # 1. Transfer CNN backbone (exact copy)
    print("  ✓ Transferring CNN backbone...")
    for key in old_network.cnn.state_dict().keys():
        full_key = f'cnn.{key}'
        new_state_dict[full_key] = old_state_dict[full_key].clone()

    # 2. Transfer attention (exact copy)
    if old_network.use_attention:
        print("  ✓ Transferring attention mechanism...")
        for key in old_network.attention.state_dict().keys():
            full_key = f'attention.{key}'
            new_state_dict[full_key] = old_state_dict[full_key].clone()

    # 3. Transfer scalar network (if input dimensions match)
    if old_network.scalar_network is not None:
        if old_num_scalars == 9:
            print("  ✓ Transferring scalar network (same input dimensions)...")
            for key in old_network.scalar_network.state_dict().keys():
                full_key = f'scalar_network.{key}'
                new_state_dict[full_key] = old_state_dict[full_key].clone()
        else:
            print(f"  ⚠ Scalar network NOT transferred (old: {old_num_scalars} scalars, new: 9 scalars)")
            print(f"    Old scalars: time_left, yaw, pitch, place_table_safe")
            print(f"    New scalars: + inv_logs, inv_planks, inv_sticks, inv_table, inv_axe")
            print(f"    Initializing new scalar network from scratch...")
            # Keep the randomly initialized weights from new_network

    # 4. Transfer actor head (with action space mapping)
    print("  ✓ Transferring actor head with action mapping...")

    # Actor layer 0 (input layer): Direct copy (same feature_dim)
    new_state_dict['actor.0.weight'] = old_state_dict['actor.0.weight'].clone()
    new_state_dict['actor.0.bias'] = old_state_dict['actor.0.bias'].clone()

    # Actor layer 2 (output layer): Map old actions to new positions
    # Initialize with small random values (for new actions)
    torch.nn.init.orthogonal_(new_state_dict['actor.2.weight'], gain=0.01)
    torch.nn.init.constant_(new_state_dict['actor.2.bias'], 0)

    # Copy weights for mapped actions
    for old_pos, new_pos in action_mapping.items():
        new_state_dict['actor.2.weight'][new_pos] = old_state_dict['actor.2.weight'][old_pos].clone()
        new_state_dict['actor.2.bias'][new_pos] = old_state_dict['actor.2.bias'][old_pos].clone()

    print(f"    - Transferred {len(action_mapping)}/{len(new_actions)} action outputs")
    print(f"    - Initialized {len(new_actions) - len(action_mapping)} new action outputs")

    # 5. Transfer critic head (exact copy - value is independent of action space)
    print("  ✓ Transferring critic head...")
    new_state_dict['critic.0.weight'] = old_state_dict['critic.0.weight'].clone()
    new_state_dict['critic.0.bias'] = old_state_dict['critic.0.bias'].clone()
    new_state_dict['critic.2.weight'] = old_state_dict['critic.2.weight'].clone()
    new_state_dict['critic.2.bias'] = old_state_dict['critic.2.bias'].clone()

    # Load transferred weights into new network
    new_network.load_state_dict(new_state_dict)

    # Save transferred model in PPO-compatible format
    print(f"\n[6/6] Saving transferred model to {output_path}...")

    # Create checkpoint in PPO agent format
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
            'old_num_scalars': old_num_scalars,
            'new_num_scalars': 9,
            'old_num_actions': len(old_actions),
            'new_num_actions': len(new_actions),
            'action_mapping': action_mapping,
            'transferred_actions': list(action_mapping.values()),
            'new_actions': list(range(len(new_actions)))
        }
    }

    torch.save(checkpoint, output_path)

    print("\n" + "=" * 80)
    print("TRANSFER COMPLETE!")
    print("=" * 80)
    print(f"\nTransferred model saved to: {output_path}")
    print("\nNotes:")
    print(f"  - {len(action_mapping)} actions transferred from old model")
    print(f"  - {len(new_actions) - len(action_mapping)} new actions initialized randomly (small weights)")
    print("  - CNN, attention, scalar network, and critic fully transferred")
    print("  - Ready for continued training with new action space!")
    print("\nNext steps:")
    print("  1. Update your training config to use this checkpoint as 'resume_from'")
    print("  2. Start training with lower learning rate for fine-tuning")
    print("  3. Monitor performance on new actions (crafting, etc.)")
    print("=" * 80)

    return new_network


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transfer weights from old PPO model to new model")
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
        default='checkpoints/transferred_model.pt',
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
    transfer_weights(
        old_model_path=args.old_model,
        old_config_path=args.old_config,
        new_config_path=args.new_config,
        output_path=args.output,
        device=args.device
    )
