#!/usr/bin/env python3
"""
PPO-specific training loop.

This module contains ONLY PPO-specific logic:
- Rollout buffer collection
- Policy sampling
- PPO update with clipped objective

Common infrastructure (checkpointing, logging, env recreation) is in train.py.
"""

import numpy as np
import os
import torch
import glob

def cleanup_checkpoints(checkpoint_dir, algorithm, current_ep, keep_last=2):
    """Prunes old checkpoints, keeping last 2 and ignores best_models."""
    pattern = os.path.join(checkpoint_dir, f"checkpoint_{algorithm}_ep*.pt")
    files = glob.glob(pattern)
    def get_ep(f):
        try: return int(f.split('_ep')[-1].replace('.pt', ''))
        except: return 0
    files.sort(key=get_ep)
    if len(files) > keep_last:
        files_to_remove = files[:-keep_last]
        for f in files_to_remove:
            try: os.remove(f); 
            except OSError as e: print(f"Error deleting {f}: {e}")

def get_latest_best_model(checkpoint_dir, algorithm):
    """Finds the existing best model to track it for deletion."""
    pattern = os.path.join(checkpoint_dir, f"best_model_{algorithm}_ep*.pt")
    files = glob.glob(pattern)
    if not files: return None
    files.sort(key=os.path.getctime) 
    return files[-1]

def train_ppo(config: dict, env, agent, logger, render: bool = False):
    """
    PPO-specific training loop (on-policy with rollout buffer).

    Args:
        config: Configuration dict
        env: MineRL environment
        agent: PPO agent
        logger: TensorBoard logger
        render: Whether to render environment

    Returns:
        env: Updated environment (may be recreated)
    """
    from trainers.helpers import safe_env_reset, save_checkpoint, log_episode_stats

    # Training parameters
    num_episodes = config['training']['num_episodes']
    log_freq = config['training']['log_freq']
    save_freq = config['training']['save_freq']
    env_recreation_interval = config['training'].get('env_recreation_interval', 50)

    # Episode settings
    env_config = config['environment']
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps_per_episode = episode_seconds * 5

    algorithm = 'ppo'

    # PPO specific
    ppo_config = config['ppo']
    n_steps = ppo_config['n_steps']

    # Training state
    global_step = agent.step_count
    best_avg_wood = 0
    recent_wood = []
    
    checkpoint_dir = config['training']['checkpoint_dir']
    last_best_model_path = get_latest_best_model(checkpoint_dir, algorithm)
    
    if hasattr(agent, 'episode_count') and agent.episode_count > 0:
        episode = agent.episode_count + 1
    else:
        episode = (global_step // max_steps_per_episode) + 1

    print(f"Starting/Resuming from Episode {episode} ")
    
    if episode == 1:
        # The BC PPO checkpoint is saved using the algorithm name 'bc_ppo' (as per your helpers file)
        bc_checkpoint_path = os.path.join(checkpoint_dir, "final_model_bc_ppo.pt")

        if os.path.exists(bc_checkpoint_path):
            print(f"\n🧠 Loading BC pre-trained weights for PPO: {bc_checkpoint_path}")
            
            # Load checkpoint data
            device = agent.device
            checkpoint = torch.load(bc_checkpoint_path, map_location=device)
            
            # 2. Load the weights into the PPO agent's policy network
            if 'policy_state_dict' in checkpoint:
                agent.policy.load_state_dict(checkpoint['policy_state_dict'])
                print("✅ Successfully loaded policy weights for PPO.")
            else:
                print("⚠️  Warning: BC PPO checkpoint found but 'policy_state_dict' key is missing.")
        else:
            print("⚠️  Warning: BC PPO checkpoint not found. Starting PPO from scratch.")
            
    # Main training loop
    while episode <= num_episodes:
        if hasattr(agent, 'episode_count'):
            agent.episode_count = episode

        # Periodically recreate environment to prevent memory leaks
        if episode > 1 and (episode - 1) % env_recreation_interval == 0:
            print(f"\n🔄 Recreating environment at episode {episode}...")
            env.close()
            from utils.env_factory import create_env
            env = create_env(config)
            print("✓ Environment recreated\n")

        obs = safe_env_reset(env, max_retries=3, retry_delay=2.0)
        episode_reward = 0
        episode_wood = 0
        step_in_episode = 0
        episode += 1

        # Run episode and collect rollout
        while step_in_episode < max_steps_per_episode:
            # PPO-SPECIFIC: Select action from policy (no epsilon-greedy)
            action, log_prob, value = agent.select_action(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)

            # Handle MineRL socket timeout errors
            if 'error' in info:
                print(f"⚠️  MineRL step error in episode {episode}: {info.get('error', 'unknown')}")
                print(f"   Terminating episode early (step {step_in_episode})")
                done = True  # Force episode termination
                # Don't store this transition - it's corrupted

            if render: 
                env.render()
            if done and 'error' in info: 
                # Skip storing corrupted transition, just break
                break
            
            # PPO-SPECIFIC: Store transition in rollout buffer
            agent.store_transition(obs, action, log_prob, reward, value, done)

            # Update counters
            if 'macro_steps' in info:
                steps_used = max(1, info['macro_steps'] // 4)
            else:
                steps_used = 1

            obs = next_obs
            episode_reward += reward
            episode_wood = info.get('wood_count', 0)
            step_in_episode += steps_used
            global_step += steps_used

            # PPO-SPECIFIC: Update when rollout buffer is full
            if len(agent.buffer.observations) >= n_steps:
                # Update policy (agent.update() will compute last_value from obs)
                update_metrics = agent.update(obs)

                # Log update metrics
                if update_metrics:
                    logger.log_training_step(
                        step=global_step,
                        loss=update_metrics.get('total_loss', 0),
                        policy_loss=update_metrics.get('policy_loss', 0),
                        value_loss=update_metrics.get('value_loss', 0),
                        entropy=update_metrics.get('entropy', 0)
                    )

            if done: 
                break

        # Track wood
        recent_wood.append(episode_wood)
        if len(recent_wood) > 50:
            recent_wood.pop(0)

        # Log episode
        logger.log_episode(
            episode_reward=episode_reward,
            episode_length=step_in_episode,
            wood_collected=episode_wood,
            epsilon=0.0  # PPO doesn't use epsilon
        )

        # Console logging (uses common function)
        log_episode_stats(episode, num_episodes, global_step, episode_wood,
                         recent_wood, agent, env, obs, log_freq)
        
        # Save checkpoint (uses common function)
        if episode % save_freq == 0:
            # Standard Save using checkpoint_ prefix
            save_path = os.path.join(checkpoint_dir, f"checkpoint_{algorithm}_ep{episode}.pt")
            agent.save(save_path)

            avg_wood = np.mean(recent_wood) if recent_wood else 0
            if avg_wood > best_avg_wood:
                best_avg_wood = avg_wood
                
                # Delete previous best
                if last_best_model_path and os.path.exists(last_best_model_path):
                    try:
                        os.remove(last_best_model_path)
                    except OSError as e:
                        print(f"Error removing previous best: {e}")

                # Save new best
                best_path = os.path.join(checkpoint_dir, f"best_model_{algorithm}_ep{episode}.pt")
                agent.save(best_path)
                last_best_model_path = best_path
            
            cleanup_checkpoints(checkpoint_dir, algorithm, episode, keep_last=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {global_step}")
    print(f"Best avg wood (50 ep): {best_avg_wood:.2f}")
    print(f"{'='*60}")

    return env