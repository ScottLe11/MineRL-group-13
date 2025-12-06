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

    # Print policy info for PPO agents
    if hasattr(agent, 'get_policy_info'):
        try:
            policy_info = agent.get_policy_info(obs)

            # Get action names from environment
            action_names = []
            if hasattr(env, 'action_names'):
                action_names = env.action_names
            else:
                from wrappers.actions import ACTION_NAMES_POOL
                action_names = ACTION_NAMES_POOL[:agent.num_actions]

            # Print entropy and value
            entropy = policy_info['entropy']
            value = policy_info['value']
            print(f"  [Policy] Entropy: {entropy:.3f}, Value: {value:.2f}")

            # Print top 5 actions by probability
            top_actions = policy_info['top_actions']
            top_probs_str = ", ".join([
                f"{action_names[idx] if idx < len(action_names) else f'a{idx}'}:{prob*100:.1f}%"
                for idx, prob in top_actions
            ])
            print(f"  [Top Probs] {top_probs_str}")
        except Exception as e:
            print(f"  [Policy Info] Error: {type(e).__name__}: {e}")

    # Print action statistics
    if hasattr(agent, 'get_action_stats'):
        stats = agent.get_action_stats()
        if stats:
            print(f"  [Action Stats] Last {agent.num_actions}: {stats['last_100_unique']}/{agent.num_actions} unique")

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
        'time_left': torch.tensor(data['obs_time'], dtype=torch.float32),
        'yaw': torch.tensor(data['obs_yaw'], dtype=torch.float32),
        'pitch': torch.tensor(data['obs_pitch'], dtype=torch.float32),
        'place_table_safe': torch.tensor(data['obs_place_table_safe'], dtype=torch.float32),
        'actions': torch.tensor(data['actions'], dtype=torch.long),
        'rewards': torch.tensor(data['rewards'], dtype=torch.float32),
        'dones': torch.tensor(data['dones'], dtype=torch.bool)
    }

    # Ensure scalar tensors have correct shape (N, 1)
    for k in ['time_left', 'yaw', 'pitch', 'place_table_safe']:
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
        expert_tensors['time_left'].to(device),
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
        
        for pov_batch, time_left_batch, yaw_batch, pitch_batch, place_table_safe_batch, action_batch in dataloader:
            # Construct observation dictionary for the DQN network
            obs_batch = {
                'pov': pov_batch,
                'time_left': time_left_batch,
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

    # NOTE: We'll save the final checkpoint AFTER pre-filling the buffer
    # This way the saved checkpoint includes expert demonstrations

    print(f"\n{'='*60}")
    print(f"BEHAVIORAL CLONING COMPLETE")
    print(f"{'='*60}")
    print(f"Total epochs: {num_epochs}")
    print(f"Total steps (batches): {global_step}")
    print(f"{'='*60}")

    # Pre-fill replay buffer with expert demonstrations
    print(f"\n{'='*60}")
    print(f"PRE-FILLING REPLAY BUFFER WITH EXPERT DEMONSTRATIONS")
    print(f"{'='*60}\n")

    # Reload expert data (we already have it in expert_tensors, but let's be explicit)
    num_transitions = len(expert_tensors['actions'])

    # Construct (state, action, reward, next_state, done) tuples
    # States are at indices [0, 1, 2, ..., N-1]
    # Next states are at indices [1, 2, 3, ..., N]
    # We can't create a transition for the last state (no next_state), so we use N-1 transitions

    num_valid_transitions = num_transitions - 1
    transitions_added = 0

    print(f"Adding {num_valid_transitions} expert transitions to replay buffer...")

    for i in range(num_valid_transitions):
        # Current state
        state = {
            'pov': expert_tensors['pov'][i].unsqueeze(0),      # Add batch dim
            'time_left': expert_tensors['time_left'][i].unsqueeze(0),
            'yaw': expert_tensors['yaw'][i].unsqueeze(0),
            'pitch': expert_tensors['pitch'][i].unsqueeze(0),
            'place_table_safe': expert_tensors['place_table_safe'][i].unsqueeze(0)
        }

        # Next state
        next_state = {
            'pov': expert_tensors['pov'][i+1].unsqueeze(0),
            'time_left': expert_tensors['time_left'][i+1].unsqueeze(0),
            'yaw': expert_tensors['yaw'][i+1].unsqueeze(0),
            'pitch': expert_tensors['pitch'][i+1].unsqueeze(0),
            'place_table_safe': expert_tensors['place_table_safe'][i+1].unsqueeze(0)
        }

        action = int(expert_tensors['actions'][i].item())
        reward = float(expert_tensors['rewards'][i].item())
        done = bool(expert_tensors['dones'][i].item())

        # Add to replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)
        transitions_added += 1

        # Log progress every 10%
        if (i + 1) % max(1, num_valid_transitions // 10) == 0:
            progress = (i + 1) / num_valid_transitions * 100
            print(f"  Progress: {progress:.1f}% ({i+1}/{num_valid_transitions} transitions)")

    print(f"\n✅ Successfully added {transitions_added} expert transitions to replay buffer")
    print(f"   Replay buffer size: {len(agent.replay_buffer)} / {agent.replay_buffer.capacity}")
    print(f"   Ready for DQN fine-tuning!\n")
    print(f"{'='*60}\n")

    # Final save with replay buffer
    print("💾 Saving final checkpoint with expert demonstrations in replay buffer...")
    save_checkpoint(agent, config, num_epochs, final=True, save_buffer=True)

    # No environment usage during BC, so just return
    return env


def train_bc_ppo(config: dict, env, agent, logger):
    """
    Behavioral Cloning (BC) training loop for PPO (Supervised Learning).

    Trains the PPO policy network to imitate expert demonstrations.
    Similar to train_bc but for PPO's ActorCriticNetwork instead of Q-Network.

    Args:
        config: Configuration dict
        env: MineRL environment (used only for initialization/cleanup)
        agent: PPOAgent (used as the policy network)
        logger: TensorBoard logger
    """
    bc_config = config.get('bc', {})
    training_config = config['training']
    device = agent.device

    # 1. Load Data
    data_path = bc_config.get('data_path', 'bc_expert_data.npz')
    print(f"\n📦 Starting Behavioral Cloning (BC) for PPO using data from: {data_path}")
    expert_tensors = load_bc_data(data_path)

    # 2. Setup DataLoader
    dataset = TensorDataset(
        expert_tensors['pov'].to(device),
        expert_tensors['time_left'].to(device),
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

    # 3. Use PPO Policy Network and Setup Optimizer/Loss
    # The PPOAgent has a policy (ActorCriticNetwork) with actor and critic heads
    # We'll train the actor head to match expert actions

    optimizer = optim.Adam(agent.policy.parameters(), lr=bc_config.get('learning_rate', 1e-4))

    global_step = 0

    print(f"Training for {num_epochs} epochs with {len(dataloader)} batches per epoch.")

    # 4. Main BC Training Loop
    for epoch in range(1, num_epochs + 1):
        total_epoch_loss = 0
        num_batches = 0

        for pov_batch, time_left_batch, yaw_batch, pitch_batch, place_table_safe_batch, action_batch in dataloader:
            # Construct observation dictionary for the PPO policy network
            obs_batch = {
                'pov': pov_batch,
                'time_left': time_left_batch,
                'yaw': yaw_batch,
                'pitch': pitch_batch,
                'place_table_safe': place_table_safe_batch
            }

            # Forward pass: get action logits from the policy network
            # ActorCriticNetwork returns (action_logits, value)
            # We only need action_logits for BC
            action_logits, _ = agent.policy(obs_batch)

            # Loss: Cross-Entropy between policy logits and expert actions
            loss = F.cross_entropy(action_logits, action_batch)

            # Gradient clipping (max_grad_norm)
            max_grad_norm = bc_config.get('gradient_clip', 10.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=max_grad_norm)
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
                    q_mean=action_logits.mean().item()
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
    print(f"BEHAVIORAL CLONING (PPO) COMPLETE")
    print(f"{'='*60}")
    print(f"Total epochs: {num_epochs}")
    print(f"Total steps (batches): {global_step}")
    print(f"{'='*60}")

    # No environment usage during BC, so just return
    return env


def train_dqfd(config: dict, env, agent, logger):
    """
    Deep Q-Learning from Demonstrations (DQfD) - Imitation Learning for DQN.

    DQfD is a more sophisticated imitation learning approach than behavioral cloning.
    It combines multiple loss terms:
    1. Supervised loss (cross-entropy on expert actions)
    2. TD loss (Bellman error for value learning)
    3. Large margin classification loss (ensure expert actions have highest Q-values)
    4. L2 regularization (prevent overfitting)

    This pre-training phase learns from expert demonstrations before RL fine-tuning.

    Args:
        config: Full configuration dictionary
        env: Environment (not used during pre-training, but kept for consistency)
        agent: DQN agent instance
        logger: TensorBoard logger

    Returns:
        env: Environment (unchanged)
    """
    print(f"\n{'='*60}")
    print(f"STARTING DQfD PRE-TRAINING (Imitation Learning)")
    print(f"{'='*60}\n")

    # Configuration
    bc_config = config.get('bc', {})
    training_config = config['training']
    device = config['device']

    # DQfD hyperparameters
    lambda_1 = bc_config.get('lambda_supervised', 1.0)    # Supervised loss weight
    lambda_2 = bc_config.get('lambda_td', 1.0)            # TD loss weight
    lambda_3 = bc_config.get('lambda_margin', 1.0)        # Margin loss weight
    lambda_4 = bc_config.get('lambda_l2', 1e-5)           # L2 regularization weight
    margin = bc_config.get('margin', 0.8)                 # Margin for classification loss
    gamma = config['dqn']['gamma']

    print(f"DQfD Hyperparameters:")
    print(f"  λ_supervised: {lambda_1}")
    print(f"  λ_td: {lambda_2}")
    print(f"  λ_margin: {lambda_3}")
    print(f"  λ_l2: {lambda_4}")
    print(f"  margin: {margin}")
    print(f"  gamma: {gamma}\n")

    # 1. Load expert data
    data_path = bc_config.get('data_path', 'bc_expert_data.npz')
    print(f"Loading expert data from: {data_path}")
    expert_tensors = load_bc_data(data_path)
    print(f"✓ Loaded {len(expert_tensors['actions'])} expert transitions\n")

    # 2. Create DataLoader
    dataset = TensorDataset(
        expert_tensors['pov'].to(device),
        expert_tensors['time_left'].to(device),
        expert_tensors['yaw'].to(device),
        expert_tensors['pitch'].to(device),
        expert_tensors['place_table_safe'].to(device),
        expert_tensors['actions'].to(device),
        expert_tensors['rewards'].to(device),
        expert_tensors['dones'].to(device)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=bc_config.get('batch_size', 32),
        shuffle=True
    )
    num_epochs = training_config.get('num_episodes', 100)

    # 3. Setup optimizer (use DQN's optimizer settings)
    optimizer = optim.Adam(
        agent.q_network.parameters(),
        lr=bc_config.get('learning_rate', 1e-4)
    )

    global_step = 0
    print(f"Training for {num_epochs} epochs with {len(dataloader)} batches per epoch.\n")

    # 4. DQfD Pre-training Loop
    for epoch in range(1, num_epochs + 1):
        total_epoch_loss = 0
        total_supervised_loss = 0
        total_td_loss = 0
        total_margin_loss = 0
        total_l2_loss = 0
        num_batches = 0

        for batch_data in dataloader:
            # Unpack batch (no next_state in current data, so we approximate)
            pov_batch, time_left_batch, yaw_batch, pitch_batch, place_safe_batch, action_batch, reward_batch, done_batch = batch_data

            # Construct observation dictionary
            obs_batch = {
                'pov': pov_batch,
                'time_left': time_left_batch,
                'yaw': yaw_batch,
                'pitch': pitch_batch,
                'place_table_safe': place_safe_batch
            }

            # Forward pass: get current Q-values
            q_values = agent.q_network(obs_batch)  # Shape: (batch_size, num_actions)

            # --- LOSS 1: Supervised Loss (Cross-Entropy) ---
            # Maximize Q-value of expert action
            loss_supervised = F.cross_entropy(q_values, action_batch)

            # --- LOSS 2: TD Loss (1-step Bellman error) ---
            # For expert demonstrations, we compute TD error
            # Q(s,a) should match r + γ * max_a' Q(s',a')
            # Since we don't have next_state in current data format, we skip this for now
            # In a full implementation, we'd need (s, a, r, s', done) tuples
            loss_td = torch.tensor(0.0, device=device)

            # --- LOSS 3: Large Margin Classification Loss ---
            # Ensure expert action has highest Q-value with a margin
            # J_E(Q) = max_{a ≠ a_E} [Q(s,a) + l(a_E, a)] - Q(s, a_E)
            # where l(a_E, a) = margin if a ≠ a_E, else 0
            batch_size = q_values.shape[0]
            num_actions = q_values.shape[1]

            # Get Q-value of expert action
            q_expert = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)

            # Add margin to all actions except expert action
            margin_mask = torch.ones_like(q_values) * margin
            margin_mask.scatter_(1, action_batch.unsqueeze(1), 0.0)  # No margin for expert action
            q_values_with_margin = q_values + margin_mask

            # Max Q-value among non-expert actions (with margin)
            q_max_with_margin = q_values_with_margin.max(dim=1)[0]

            # Margin loss: penalize if non-expert action has higher Q-value
            loss_margin = F.relu(q_max_with_margin - q_expert).mean()

            # --- LOSS 4: L2 Regularization ---
            loss_l2 = sum(p.pow(2).sum() for p in agent.q_network.parameters())

            # --- COMBINED LOSS ---
            loss = (lambda_1 * loss_supervised +
                    lambda_2 * loss_td +
                    lambda_3 * loss_margin +
                    lambda_4 * loss_l2)

            # Gradient descent
            max_grad_norm = bc_config.get('gradient_clip', 10.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            # Update target network (soft update)
            if hasattr(agent, 'soft_update_target'):
                agent.soft_update_target()

            # Accumulate losses
            total_epoch_loss += loss.item()
            total_supervised_loss += loss_supervised.item()
            total_td_loss += loss_td.item()
            total_margin_loss += loss_margin.item()
            total_l2_loss += loss_l2.item()
            num_batches += 1
            global_step += 1

            # Log metrics every N steps
            if global_step % 50 == 0:
                logger.log_training_step(
                    step=global_step,
                    loss=total_epoch_loss / num_batches,
                    q_mean=q_values.mean().item()
                )

        # Epoch summary
        avg_loss = total_epoch_loss / num_batches
        avg_supervised = total_supervised_loss / num_batches
        avg_td = total_td_loss / num_batches
        avg_margin = total_margin_loss / num_batches
        avg_l2 = total_l2_loss / num_batches

        logger.log_scalar("dqfd/total_loss", avg_loss, epoch)
        logger.log_scalar("dqfd/supervised_loss", avg_supervised, epoch)
        logger.log_scalar("dqfd/td_loss", avg_td, epoch)
        logger.log_scalar("dqfd/margin_loss", avg_margin, epoch)
        logger.log_scalar("dqfd/l2_loss", avg_l2, epoch)

        # Console logging
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Total: {avg_loss:.4f} | "
              f"Supervised: {avg_supervised:.4f} | "
              f"Margin: {avg_margin:.4f} | "
              f"L2: {avg_l2:.6f}")

        # Save checkpoint
        if epoch % training_config.get('save_freq', 50) == 0:
            save_checkpoint(agent, config, epoch, save_buffer=False)

    # Final save
    save_checkpoint(agent, config, num_epochs, final=True, save_buffer=False)

    print(f"\n{'='*60}")
    print(f"DQfD PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total epochs: {num_epochs}")
    print(f"Total steps: {global_step}")
    print(f"The agent can now be fine-tuned with standard DQN (set algorithm='dqn')")
    print(f"{'='*60}\n")

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
    elif algorithm == 'bc_dqn':
        # Behavioral Cloning with DQN (simple supervised learning - cross-entropy only)
        # Same agent as 'dqn', different training
        # NOTE: Environment is NOT used during BC training epochs, only for init.
        env = train_bc(config, env, agent, logger)
    elif algorithm == 'bc_ppo':
        # Behavioral Cloning with PPO (simple supervised learning - cross-entropy only)
        # Same agent as 'ppo', different training
        # NOTE: Environment is NOT used during BC training epochs, only for init.
        env = train_bc_ppo(config, env, agent, logger)
    elif algorithm == 'dqfd':
        # Deep Q-Learning from Demonstrations (advanced imitation learning)
        # Same agent as 'dqn', different training (multi-objective loss)
        # Combines supervised + TD + margin + L2 losses
        # NOTE: Environment is NOT used during pre-training epochs, only for init.
        env = train_dqfd(config, env, agent, logger)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'dqn', 'ppo', 'bc_dqn', 'bc_ppo', or 'dqfd'.")

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
