"""
Configuration loader for the MineRL DQN agent.
Simple YAML-based config with sensible defaults.
"""

import os
import yaml
import torch
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config/config.yaml.
        
    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        # Default to config/config.yaml relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve device
    config['device'] = get_device(config.get('device', 'auto'))
    
    # Create directories
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints')
    log_dir = config.get('training', {}).get('log_dir', 'runs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return config


def get_device(device_str: str = "auto") -> str:
    """
    Determine the best available device.
    
    Args:
        device_str: One of "auto", "cpu", "cuda", "mps".
        
    Returns:
        Device string for PyTorch.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


def save_config(config: dict, path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    # Convert device back to string for saving
    config_copy = config.copy()
    if 'device' in config_copy and not isinstance(config_copy['device'], str):
        config_copy['device'] = str(config_copy['device'])
    
    with open(path, 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    print("✅ Config Loader Test")
    
    # Test device detection
    device = get_device("auto")
    print(f"  Auto-detected device: {device}")
    
    # Test loading default config
    try:
        config = load_config()
        print(f"  Loaded config successfully")
        print(f"  Environment: {config['environment']['name']}")
        print(f"  Learning rate: {config['dqn']['learning_rate']}")
        print(f"  Device: {config['device']}")
        print("\n✅ Config loader validated!")
    except FileNotFoundError as e:
        print(f"  Config file not found (expected if running from wrong directory)")
        print(f"  Error: {e}")

