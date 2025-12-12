"""
Agent factory for creating and configuring RL agents.

This module handles:
- Agent instantiation from config
- Network architecture selection
- Hyperparameter configuration
"""

from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from networks.cnn import get_architecture_info


def create_dqn_agent(config: dict, num_actions: int) -> DQNAgent:
    """
    Create a DQN agent from configuration.

    Args:
        config: Full configuration dictionary
        num_actions: Number of discrete actions in the environment

    Returns:
        Configured DQNAgent instance
    """
    dqn_config = config['dqn']
    network_config = config['network']
    target_config = dqn_config.get('target_update', {})
    per_config = dqn_config.get('prioritized_replay', {})

    # Log network architecture choice
    arch_name = network_config.get('architecture', 'small')
    attention_type = network_config.get('attention', 'none')
    use_scalar_network = network_config.get('use_scalar_network', False)
    scalar_hidden_dim = network_config.get('scalar_hidden_dim', 64)
    scalar_output_dim = network_config.get('scalar_output_dim', 64)
    use_fusion_layer = network_config.get('use_fusion_layer', False)
    fusion_hidden_dim = network_config.get('fusion_hidden_dim', 128)

    if arch_name == 'film_cbam':
        print(f"Network architecture: film_cbam (FiLM + CBAM + Conv4 + Dual Fusion)")
        print(f"  - Visual: Conv1-3 (Medium) → FiLM → CBAM → Conv4 → Flatten (2,048)")
        print(f"  - Scalar: 9 → 32 → 32")
        print(f"  - Fusion: 2,080 → 512 → 128")
        print(f"  - Head: Dueling from 128 features")
    else:
        arch_info = get_architecture_info().get(arch_name, {})
        print(f"Network architecture: {arch_name} ({arch_info.get('params', 'unknown'):,} params)")
        print(f"Attention mechanism: {attention_type}")
        print(f"Scalar network: {'enabled' if use_scalar_network else 'disabled'}" +
              (f" (hidden={scalar_hidden_dim}, output={scalar_output_dim})" if use_scalar_network else ""))
        print(f"Fusion layer: {'enabled' if use_fusion_layer else 'disabled'}" +
              (f" (dim={fusion_hidden_dim})" if use_fusion_layer else ""))

    # Log PER and target update settings
    use_per = per_config.get('enabled', False)
    target_method = target_config.get('method', 'soft')
    print(f"Prioritized replay: {'enabled' if use_per else 'disabled'}")
    print(f"Target updates: {target_method}")

    agent = DQNAgent(
        num_actions=num_actions,
        input_channels=network_config['input_channels'],
        num_scalars=9,  # time_left, yaw, pitch, place_table_safe, inv_logs, inv_planks, inv_sticks, inv_table, inv_axe
        learning_rate=dqn_config['learning_rate'],
        gamma=dqn_config['gamma'],
        # Target update settings
        tau=target_config.get('tau', 0.005),
        target_update_method=target_method,
        soft_update_freq=target_config.get('soft_update_freq', 1),
        hard_update_freq=target_config.get('hard_update_freq', 1000),
        # Exploration settings
        epsilon_start=dqn_config['exploration']['epsilon_start'],
        epsilon_end=dqn_config['exploration']['epsilon_end'],
        epsilon_decay_steps=dqn_config['exploration']['epsilon_decay_steps'],
        # Replay buffer settings
        buffer_capacity=dqn_config['replay_buffer']['capacity'],
        buffer_min_size=dqn_config['replay_buffer']['min_size'],
        batch_size=dqn_config['batch_size'],
        # Training settings
        max_grad_norm=dqn_config.get('gradient_clip', 10.0),
        # Prioritized Experience Replay
        use_per=use_per,
        per_alpha=per_config.get('alpha', 0.6),
        per_beta_start=per_config.get('beta_start', 0.4),
        per_beta_end=per_config.get('beta_end', 1.0),
        # Network architecture
        cnn_architecture=arch_name,
        attention_type=attention_type,
        # Scalar network settings
        use_scalar_network=use_scalar_network,
        scalar_hidden_dim=scalar_hidden_dim,
        scalar_output_dim=scalar_output_dim,
        # Fusion layer settings
        use_fusion_layer=use_fusion_layer,
        fusion_hidden_dim=fusion_hidden_dim,
        device=config['device']
    )

    return agent


def create_ppo_agent(config: dict, num_actions: int) -> PPOAgent:
    """
    Create a PPO agent from configuration.

    Args:
        config: Full configuration dictionary
        num_actions: Number of discrete actions in the environment

    Returns:
        Configured PPOAgent instance
    """
    ppo_config = config['ppo']
    network_config = config['network']

    # Log network architecture choice (now PPO supports configurable CNNs!)
    arch_name = network_config.get('architecture', 'small')
    attention_type = network_config.get('attention', 'none')
    use_scalar_network = network_config.get('use_scalar_network', False)
    scalar_hidden_dim = network_config.get('scalar_hidden_dim', 64)
    scalar_output_dim = network_config.get('scalar_output_dim', 64)
    use_fusion_layer = network_config.get('use_fusion_layer', False)
    fusion_hidden_dim = network_config.get('fusion_hidden_dim', 128)

    if arch_name == 'film_cbam':
        print(f"Network architecture: film_cbam (FiLM + CBAM + Conv4 + Dual Fusion)")
        print(f"  - Visual: Conv1-3 (Medium) → FiLM → CBAM → Conv4 → Flatten (2,048)")
        print(f"  - Scalar: 9 → 32 → 32")
        print(f"  - Fusion: 2,048 + 32 = 2,080 → 512 → 128")
        print(f"  - Head: Actor-Critic from 128 features")
    else:
        arch_info = get_architecture_info().get(arch_name, {})
        print(f"Network architecture: {arch_name} ({arch_info.get('params', 'unknown'):,} params)")
        print(f"Attention mechanism: {attention_type}")
        print(f"Scalar network: {'enabled' if use_scalar_network else 'disabled'}" +
              (f" (hidden={scalar_hidden_dim}, output={scalar_output_dim})" if use_scalar_network else ""))
        print(f"Fusion layer: {'enabled' if use_fusion_layer else 'disabled'}" +
              (f" (dim={fusion_hidden_dim})" if use_fusion_layer else ""))

    agent = PPOAgent(
        num_actions=num_actions,
        input_channels=network_config['input_channels'],
        num_scalars=9,  # time_left, yaw, pitch, place_table_safe, inv_logs, inv_planks, inv_sticks, inv_table, inv_axe
        learning_rate=ppo_config['learning_rate'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_epsilon=ppo_config['clip_epsilon'],
        entropy_coef=ppo_config['entropy_coef'],
        value_coef=ppo_config['value_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        n_steps=ppo_config['n_steps'],
        n_epochs=ppo_config['n_epochs'],
        batch_size=ppo_config['batch_size'],
        # Network architecture (now configurable!)
        cnn_architecture=arch_name,
        attention_type=attention_type,
        # Scalar network settings
        use_scalar_network=use_scalar_network,
        scalar_hidden_dim=scalar_hidden_dim,
        scalar_output_dim=scalar_output_dim,
        # Fusion layer settings
        use_fusion_layer=use_fusion_layer,
        fusion_hidden_dim=fusion_hidden_dim,
        device=config['device']
    )

    return agent


def create_agent(config: dict, num_actions: int):
    """
    Create an agent based on the algorithm specified in config.

    Args:
        config: Full configuration dictionary
        num_actions: Number of discrete actions in the environment

    Returns:
        Configured agent instance (DQNAgent or PPOAgent)
    """
    algorithm = config.get('algorithm', 'dqn').lower()

    if algorithm == 'dqn':
        return create_dqn_agent(config, num_actions)
    elif algorithm == 'ppo':
        return create_ppo_agent(config, num_actions)
    elif algorithm == 'bc_dqn':
        # BC with DQN - same agent as 'dqn', different training (simple supervised loss)
        return create_dqn_agent(config, num_actions)
    elif algorithm == 'bc_ppo':
        # BC with PPO - same agent as 'ppo', different training (simple supervised loss)
        return create_ppo_agent(config, num_actions)
    elif algorithm == 'dqfd':
        # DQfD - same agent as 'dqn', different training (multi-objective loss)
        # Training uses: supervised + TD + margin + L2 losses
        return create_dqn_agent(config, num_actions)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'dqn', 'ppo', 'bc_dqn', 'bc_ppo', or 'dqfd'")
