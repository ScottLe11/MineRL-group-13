"""
Shared training helper functions for DQN and PPO trainers.

This module contains common utilities used by both training algorithms:
- Environment reset with retry logic
- Configuration summary printing
- Episode statistics logging
- Checkpoint saving
"""

import os
import time
import socket
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from typing import Dict

from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from networks.cnn import get_architecture_info

# Constants
AGENT_STEPS_PER_SECOND = 5  # Each agent step = 4 frames at 20 ticks/sec = 0.2s


def safe_env_reset(env, max_retries: int = 3, retry_delay: float = 3.0, recreate_fn=None):
    """
    Safely reset environment with retry logic for MineRL socket timeouts.

    Args:
        env: The environment to reset
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries (doubles each retry)
        recreate_fn: Optional function to recreate environment on persistent failures

    Returns:
        obs: Initial observation from env.reset()

    Raises:
        Exception: If all retries fail and no recreate_fn provided

    Note:
        When recreate_fn is provided and used, the old env is closed and a new
        env is created. The caller should use the env from their scope, which
        will need to be reassigned if recreation occurs. Consider using the
        trainers' env_recreation_interval mechanism instead.
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            obs = env.reset()
            if attempt > 0:
                print(f"✅ Environment reset succeeded on attempt {attempt + 1}")
            return obs
        except (socket.timeout, TimeoutError, OSError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"⚠️  Environment reset failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                print(f"   Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Environment reset failed after {max_retries} attempts")
                if recreate_fn:
                    print("🔄 Attempting to recreate environment...")
                    print("⚠️  WARNING: Using recreate_fn from safe_env_reset is not recommended.")
                    print("   The caller's env reference will not be updated.")
                    print("   Consider lowering env_recreation_interval in config instead.")
                    raise last_exception
                else:
                    print("💡 HINT: Lower env_recreation_interval in config to prevent timeouts")
                    print("   Current interval triggers recreation every N episodes")
                    raise

    # Should never reach here, but just in case
    raise RuntimeError(f"Unexpected error in safe_env_reset: {last_exception}")


def print_config_summary(config: dict, agent, env_config: dict):
    """Print training configuration summary."""
    algorithm = config.get('algorithm', 'dqn')
    network_config = config['network']
    arch_name = network_config.get('architecture', 'small')
    arch_info = get_architecture_info().get(arch_name, {})
    curriculum = env_config.get('curriculum', {})
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps = episode_seconds * AGENT_STEPS_PER_SECOND

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Episode length: {episode_seconds} seconds ({max_steps} agent steps)")
    print(f"Network: {arch_name} ({arch_info.get('params', '?'):,} params)")

    # Algorithm-specific config
    if algorithm == 'dqn':
        dqn_config = config['dqn']
        per_config = dqn_config.get('prioritized_replay', {})
        target_config = dqn_config.get('target_update', {})
        use_per = per_config.get('enabled', False)
        target_method = target_config.get('method', 'soft')
        print(f"Replay buffer: {dqn_config['replay_buffer']['capacity']:,} capacity, {dqn_config['replay_buffer']['min_size']:,} min")
        print(f"PER: {'enabled (α={}, β={}->{})'.format(per_config.get('alpha', 0.6), per_config.get('beta_start', 0.4), per_config.get('beta_end', 1.0)) if use_per else 'disabled'}")
        print(f"Target update: {target_method}" + (f" (τ={target_config.get('tau', 0.005)})" if target_method == 'soft' else f" (every {target_config.get('hard_update_freq', 1000)} steps)"))
    elif algorithm == 'ppo':
        ppo_config = config['ppo']
        print(f"Rollout size: {ppo_config['n_steps']} steps")
        print(f"Clip epsilon: {ppo_config['clip_epsilon']}")
        print(f"Entropy coef: {ppo_config['entropy_coef']}")

    print(f"Curriculum: with_logs={curriculum.get('with_logs', 0)}, with_axe={curriculum.get('with_axe', False)}")
    print(f"{'='*60}\n")


def log_episode_stats(episode: int, num_episodes: int, global_step: int,
                     episode_wood: int, recent_wood: list, agent,
                     env, obs: dict, log_freq: int):
    """Print episode statistics and Q-values/action stats."""
    if episode % log_freq != 0:
        return

    avg_wood = np.mean(recent_wood) if recent_wood else 0
    success_rate = sum(1 for w in recent_wood if w > 0) / len(recent_wood) * 100 if recent_wood else 0

    # Build base stats string
    stats_str = (f"Episode {episode}/{num_episodes} | "
                f"Steps: {global_step} | "
                f"Wood: {episode_wood} | "
                f"Avg(50): {avg_wood:.2f} | "
                f"Success: {success_rate:.0f}%")

    # Add epsilon or buffer info depending on algorithm
    if hasattr(agent, 'get_epsilon'):
        stats_str += f" | ε: {agent.get_epsilon():.3f}"

    if hasattr(agent, 'replay_buffer'):
        stats_str += f" | Buffer: {len(agent.replay_buffer)}"
    elif hasattr(agent, 'buffer'):
        stats_str += f" | Buffer: {len(agent.buffer.observations)}"

    print(stats_str)

    # Print Q-values for DQN agents
    if hasattr(agent, 'get_q_values'):
        try:
            q_values = agent.get_q_values(obs)

            # Get action names from environment
            action_names = []
            if hasattr(env, 'action_names'):
                action_names = env.action_names
            else:
                from wrappers.actions import ACTION_NAMES_POOL
                action_names = ACTION_NAMES_POOL[:agent.num_actions]

            # Get top 5 actions by Q-value
            top_indices = np.argsort(q_values)[-5:][::-1]  # Descending order
            top_q_str = ", ".join([
                f"{action_names[idx] if idx < len(action_names) else f'a{idx}'}:{q_values[idx]:.2f}"
                for idx in top_indices
            ])
            print(f"  [Top Q-values] {top_q_str}")
        except Exception as e:
            print(f"  [Top Q-values] Error: {type(e).__name__}: {e}")

    # Print action statistics
    if hasattr(agent, 'get_action_stats'):
        stats = agent.get_action_stats()
        if stats:
            print(f"  [Action Stats] Last 100: {stats['last_100_unique']}/{len(stats['last_100_actions'])} unique")

            # Print top 3 most frequent actions WITH NAMES
            top_actions = sorted(enumerate(stats['action_frequencies']), key=lambda x: x[1], reverse=True)[:3]

            # Get action names from environment
            action_names = []
            if hasattr(env, 'action_names'):
                action_names = env.action_names
            else:
                from wrappers.actions import ACTION_NAMES_POOL
                action_names = ACTION_NAMES_POOL[:agent.num_actions]

            top_str = ", ".join([
                f"{action_names[idx] if idx < len(action_names) else f'a{idx}'}:{freq*100:.1f}%"
                for idx, freq in top_actions
            ])
            print(f"  [Top Actions] {top_str}")


def save_checkpoint(agent, config: dict, episode: int, final: bool = False,
                   best: bool = False, save_buffer: bool = True):
    """
    Save a training checkpoint (algorithm-agnostic).

    Args:
        agent: Agent to save (DQN or PPO)
        config: Configuration dict
        episode: Current episode number
        final: Whether this is the final checkpoint
        best: Whether this is the best model so far
        save_buffer: Whether to save replay buffer (DQN only, large file)
    """
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    algorithm = config.get('algorithm', 'dqn')

    if final:
        path = os.path.join(checkpoint_dir, f"final_model_{algorithm}.pt")
    elif best:
        path = os.path.join(checkpoint_dir, f"best_model_{algorithm}.pt")
    else:
        path = os.path.join(checkpoint_dir, f"checkpoint_{algorithm}_ep{episode}.pt")

    # Build checkpoint dict based on algorithm
    if isinstance(agent, DQNAgent):
        checkpoint = {
            'episode': episode,
            'step_count': agent.step_count,
            'train_count': agent.train_count,
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'action_counts': agent.action_counts,
            'last_actions': agent.last_actions,
        }

        # Optionally save replay buffer (can be large!)
        if save_buffer and len(agent.replay_buffer) > 0:
            if hasattr(agent.replay_buffer, 'get_all_experiences'):
                # PrioritizedReplayBuffer
                checkpoint['replay_buffer'] = agent.replay_buffer.get_all_experiences()
            elif hasattr(agent.replay_buffer, 'buffer'):
                # Regular ReplayBuffer
                checkpoint['replay_buffer'] = list(agent.replay_buffer.buffer)
            else:
                print("⚠️  Warning: Unknown replay buffer type, skipping buffer save")
                save_buffer = False

            if save_buffer:
                buffer_size_mb = len(agent.replay_buffer) * 84 * 84 * 4 * 4 / (1024 * 1024)
                print(f"💾 Saving with replay buffer (~{buffer_size_mb:.1f}MB)...")

    elif isinstance(agent, PPOAgent):
        checkpoint = {
            'episode': episode,
            'step_count': agent.step_count,
            'update_count': agent.update_count,
            'policy_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
        }

        # Add action tracking if available
        if hasattr(agent, 'action_counts'):
            checkpoint['action_counts'] = agent.action_counts
        if hasattr(agent, 'last_actions'):
            checkpoint['last_actions'] = agent.last_actions
    else:
        raise ValueError(f"Unknown agent type: {type(agent)}")

    torch.save(checkpoint, path)
    print(f"💾 Saved: {path}")


#recording training
def load_bc_data(filename: str) -> Dict[str, torch.Tensor]:
    """
    Loads processed expert data and converts it to PyTorch tensors.

    Args:
        filename: Path to the .npz file created by pkl_parser.py

    Returns:
        Dictionary of PyTorch tensors ready for DataLoader.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Expert data file not found: {filename}. Please run pkl_parser.py first.")

    data = np.load(filename)
    print(f"Loaded {len(data['actions'])} transitions from {filename}")

    # Convert NumPy arrays to PyTorch tensors
    tensors = {
        'pov': torch.tensor(data['obs_pov'], dtype=torch.uint8).float().div(255.0), # Normalize POV here
        'time': torch.tensor(data['obs_time'], dtype=torch.float32),
        'yaw': torch.tensor(data['obs_yaw'], dtype=torch.float32),
        'pitch': torch.tensor(data['obs_pitch'], dtype=torch.float32),
        'place_table_safe': torch.tensor(data['obs_place_table_safe'], dtype=torch.float32),
        'actions': torch.tensor(data['actions'], dtype=torch.long)
    }

    # Ensure scalar tensors have correct shape (N, 1)
    for k in ['time', 'yaw', 'pitch', 'place_table_safe']:
        if tensors[k].dim() == 1:
            tensors[k] = tensors[k].unsqueeze(1)

    return tensors


def train_bc(config: dict, env, agent, logger):
    """
    Behavioral Cloning (BC) training loop (Supervised Learning).

    Args:
        config: Configuration dict
        env: MineRL environment (used only for initialization/cleanup)
        agent: DQNAgent (used as the policy network)
        logger: TensorBoard logger
    """
    bc_config = config.get('bc', {})
    training_config = config['training']
    device = agent.device
    
    # 1. Load Data
    data_path = bc_config.get('data_path', 'bc_expert_data.npz')
    print(f"\n📦 Starting Behavioral Cloning (BC) using data from: {data_path}")
    expert_tensors = load_bc_data(data_path)
    
    # 2. Setup DataLoader
    dataset = TensorDataset(
        expert_tensors['pov'].to(device),
        expert_tensors['time'].to(device),
        expert_tensors['yaw'].to(device),
        expert_tensors['pitch'].to(device),
        expert_tensors['place_table_safe'].to(device),
        expert_tensors['actions'].to(device)
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=bc_config.get('batch_size', 32), 
        shuffle=True
    )
    num_epochs = training_config.get('num_episodes', 100)
    
    # 3. Use DQN Network as Policy and Setup Optimizer/Loss (no target network needed)
    # The DQNAgent object (agent) already contains the Q-Network, which we will treat as the policy network.
    
    optimizer = optim.Adam(agent.q_network.parameters(), lr=bc_config.get('learning_rate', 1e-4))
    
    global_step = 0
    
    print(f"Training for {num_epochs} epochs with {len(dataloader)} batches per epoch.")

    # 4. Main BC Training Loop
    for epoch in range(1, num_epochs + 1):
        total_epoch_loss = 0
        num_batches = 0
        
        for pov_batch, time_batch, yaw_batch, pitch_batch, place_table_safe_batch, action_batch in dataloader:
            # Construct observation dictionary for the DQN network
            obs_batch = {
                'pov': pov_batch,
                'time': time_batch,
                'yaw': yaw_batch,
                'pitch': pitch_batch,
                'place_table_safe': place_table_safe_batch
            }
            
            # Forward pass: get Q-values (logits)
            # We use the Q-network's output as the action preference scores (logits)
            q_values = agent.q_network(obs_batch)
            
            # Loss: Cross-Entropy between Q-value logits and expert actions
            # F.cross_entropy expects logits (not softmaxed probabilities)
            loss = F.cross_entropy(q_values, action_batch)
            
            # Gradient clipping (max_grad_norm)
            max_grad_norm = bc_config.get('gradient_clip', 10.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            total_epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Log metrics every N steps
            if global_step % 50 == 0:
                avg_loss = total_epoch_loss / num_batches
                logger.log_training_step(
                    step=global_step, 
                    loss=avg_loss, 
                    q_mean=q_values.mean().item()
                )

        # Log end of epoch metrics
        avg_loss = total_epoch_loss / num_batches
        logger.log_scalar("bc/loss_epoch", avg_loss, epoch)
        
        # Console logging
        print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Global Steps: {global_step}")

        # Save checkpoint
        if epoch % training_config.get('save_freq', 50) == 0:
            save_checkpoint(agent, config, epoch, save_buffer=False)
            
    # Final save
    save_checkpoint(agent, config, num_epochs, final=True, save_buffer=False)
    
    print(f"\n{'='*60}")
    print(f"BEHAVIORAL CLONING COMPLETE")
    print(f"{'='*60}")
    print(f"Total epochs: {num_epochs}")
    print(f"Total steps (batches): {global_step}")
    print(f"{'='*60}")

    # No environment usage during BC, so just return
    return env





def train(config: dict, render: bool = False):
    """
    Main training entry point - provides common infrastructure and routes to algorithm.

    Args:
        config: Configuration dictionary
        render: If True, render the Minecraft window during training
    """
    # Setup
    set_seed(config.get('seed'))
    device = config['device']
    print(f"Training on device: {device}")

    # Create environment
    env = create_env(config)
    print(f"Environment created: {config['environment']['name']}")
    print(f"Action space: {env.action_space}")

    # Create agent
    agent = create_agent(config, num_actions=env.action_space.n)

    algorithm = config.get('algorithm', 'dqn')
    
    if algorithm == 'dqn':
        # Check if BC checkpoint exists to load weights
        checkpoint_dir = config['training']['checkpoint_dir']
        bc_checkpoint_path = os.path.join(checkpoint_dir, "final_model_bc.pt")
        
        if os.path.exists(bc_checkpoint_path):
            print(f"\n🧠 Loading BC pre-trained weights: {bc_checkpoint_path}")
            
            # Load checkpoint data
            checkpoint = torch.load(bc_checkpoint_path, map_location=device)
            
            # Load Q-Network weights from BC checkpoint
            if 'q_network_state_dict' in checkpoint:
                 agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                 # Initialize target network with pre-trained weights
                 agent.target_network.load_state_dict(checkpoint['q_network_state_dict'])
                 print("✅ Successfully loaded weights into Q-Network and Target Network.")
            else:
                 print("⚠️  Warning: BC checkpoint found but 'q_network_state_dict' key is missing.")
    
    # Create logger
    algorithm = config.get('algorithm', 'dqn')
    logger = Logger(
        log_dir=config['training']['log_dir'],
        experiment_name=f"treechop_{algorithm}_{config.get('seed', 'noseed')}"
    )

    # Print configuration summary
    print_config_summary(config, agent, config['environment'])

    # Route to algorithm-specific training loop
    if algorithm == 'dqn':
        from trainers.train_dqn import train_dqn
        env = train_dqn(config, env, agent, logger, render)
    elif algorithm == 'ppo':
        from trainers.train_ppo import train_ppo
        env = train_ppo(config, env, agent, logger, render)
    elif algorithm == 'bc':
        # Behavioral Cloning uses the DQN network architecture as its policy
        # NOTE: Environment is NOT used during BC training epochs, only for init.
        # But we pass it along for consistent cleanup.
        env = train_bc(config, env, agent, logger)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'dqn' or 'ppo'.")

    # Cleanup
    logger.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train MineRL Tree-Chopping RL Agent")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--render', action='store_true',
                        help='Render the Minecraft window during training')
    args = parser.parse_args()

    # Load config
    default_config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    #config = load_config(args.config)
    config = load_config(args.config if args.config else default_config_path)

    print("=" * 60)
    print("MineRL Tree-Chopping RL Training")
    print("=" * 60)
    print(f"Config: {args.config or 'config/config.yaml'}")
    print(f"Algorithm: {config.get('algorithm', 'dqn').upper()}")
    print(f"Device: {config['device']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Episode length: {config['environment']['episode_seconds']}s")
    print(f"Render: {args.render}")
    print("=" * 60)

    # Train
    train(config, render=args.render)


if __name__ == "__main__":
    main()
