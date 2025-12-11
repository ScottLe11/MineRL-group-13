"""
Prepare transferred model for PPO training.

This script renames the transferred model checkpoint so it can be loaded
by the PPO training script at episode 1.
"""

import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def prepare_transfer_checkpoint():
    """
    Copy transferred_model.pt to final_model_bc_ppo.pt so training loads it.
    """
    source = project_root / "checkpoints" / "transferred_model.pt"
    target = project_root / "checkpoints" / "final_model_bc_ppo.pt"

    if not source.exists():
        print(f"❌ Error: Transferred model not found at {source}")
        print("   Run transfer_learning.py first!")
        return False

    # Backup existing BC checkpoint if it exists
    if target.exists():
        backup = project_root / "checkpoints" / "final_model_bc_ppo_backup.pt"
        print(f"⚠️  Backing up existing BC checkpoint to {backup.name}")
        shutil.copy(target, backup)

    # Copy transferred model
    print(f"📋 Copying {source.name} → {target.name}")
    shutil.copy(source, target)

    # Verify it's loadable
    print(f"\n🔍 Verifying checkpoint...")
    try:
        checkpoint = torch.load(target, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Found model_state_dict with {len(state_dict)} keys")

            # Check for key components
            has_cnn = any(k.startswith('cnn.') for k in state_dict.keys())
            has_actor = any(k.startswith('actor.') for k in state_dict.keys())
            has_critic = any(k.startswith('critic.') for k in state_dict.keys())

            print(f"   CNN: {'✓' if has_cnn else '✗'}")
            print(f"   Actor: {'✓' if has_actor else '✗'}")
            print(f"   Critic: {'✓' if has_critic else '✗'}")

            # Convert to policy_state_dict format if needed
            checkpoint['policy_state_dict'] = checkpoint['model_state_dict']
            torch.save(checkpoint, target)
            print(f"\n✅ Converted to policy_state_dict format")

        elif 'policy_state_dict' in checkpoint:
            print(f"✅ Already in policy_state_dict format")
        else:
            print(f"⚠️  Warning: No recognized state dict key found")

        if 'transfer_info' in checkpoint:
            info = checkpoint['transfer_info']
            print(f"\n📊 Transfer Info:")
            print(f"   Source: {info.get('source_model', 'Unknown')}")
            print(f"   Actions: {info.get('old_num_actions', '?')} → {info.get('new_num_actions', '?')}")
            print(f"   Transferred actions: {len(info.get('action_mapping', {}))}")

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

    print(f"\n" + "=" * 80)
    print("✅ SUCCESS! Transferred model is ready for training")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Make sure config.yaml has the correct action space (18 actions)")
    print("2. Run training:")
    print("   conda run -n minerl-env python scripts/train.py --config config/config.yaml --algorithm ppo")
    print("\n⚠️  IMPORTANT:")
    print("   - Start with LOWER learning rate (0.00003-0.00005)")
    print("   - Use HIGHER exploration (epsilon_start=0.5-0.7 for DQN, entropy_coef=0.02-0.05 for PPO)")
    print("   - The model needs 100-200 episodes to learn crafting actions")
    print("   - Monitor action distribution - it should try crafting actions")
    print("=" * 80)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare transferred model for training"
    )

    args = parser.parse_args()

    success = prepare_transfer_checkpoint()
    sys.exit(0 if success else 1)
