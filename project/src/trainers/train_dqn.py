#!/usr/bin/env python3
"""
DQN-specific training loop.

This module contains ONLY DQN-specific logic:
- Replay buffer experience storage
- Epsilon-greedy exploration
- Q-network training

Common infrastructure (checkpointing, logging, env recreation) is in train.py.
"""

import numpy as np
import os
import glob

def cleanup_checkpoints(checkpoint_dir, algorithm, current_ep, keep_last=2):
    """Keeps only the 'keep_last' most recent checkpoints. Preserves 'best_model'."""
    pattern = os.path.join(checkpoint_dir, f"checkpoint_{algorithm}_ep*.pt")
    files = glob.glob(pattern)
    
    def get_ep(f):
        try:
            return int(f.split('_ep')[-1].replace('.pt', ''))
        except:
            return 0
    files.sort(key=get_ep)
    
    if len(files) > keep_last:
        files_to_remove = files[:-keep_last]
        for f in files_to_remove:
            try:
                os.remove(f)
            except OSError as e:
                print(f"Error deleting {f}: {e}")

def get_latest_best_model(checkpoint_dir, algorithm):
    """Finds the existing best model to track it for deletion."""
    pattern = os.path.join(checkpoint_dir, f"best_model_{algorithm}_ep*.pt")
    files = glob.glob(pattern)
    if not files: return None
    files.sort(key=os.path.getctime) 
    return files[-1]

def train_dqn(config: dict, env, agent, logger, render: bool = False):
    """
    DQN-specific training loop (off-policy with replay buffer).

    Args:
        config: Configuration dict
        env: MineRL environment
        agent: DQN agent
        logger: TensorBoard logger
        render: Whether to render environment

    Returns:
        env: Updated environment (may be recreated)
    """
    from trainers.helpers import safe_env_reset, save_checkpoint, log_episode_stats

    # Training parameters
    num_episodes = config['training']['num_episodes']
    train_freq = config['training']['train_freq']
    log_freq = config['training']['log_freq']
    save_freq = config['training']['save_freq']
    env_recreation_interval = config['training'].get('env_recreation_interval', 50)
    checkpoint_dir = config['training']['checkpoint_dir']
    algorithm = 'dqn'

    # Episode settings
    env_config = config['environment']
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps_per_episode = episode_seconds * 5
    starting_wood = env_config.get('curriculum', {}).get('with_logs', 0)

    # Training state
    global_step = agent.step_count
    best_avg_wood = 0
    recent_wood = [] # Track last 50 episodes
    
    last_best_model_path = get_latest_best_model(checkpoint_dir, algorithm)

    # Calculate Start Episode
    if hasattr(agent, 'episode_count') and agent.episode_count > 0:
        start_episode = agent.episode_count + 1
    else:
        start_episode = (global_step // max_steps_per_episode) + 1

    print(f"Starting/Resuming from Episode {start_episode}")

    for episode in range(start_episode, num_episodes + 1):
        if hasattr(agent, 'episode_count'):
            agent.episode_count = episode

        # Main training loop
        if episode > 1 and (episode - 1) % env_recreation_interval == 0:
            # Periodically recreate environment to prevent memory leaks
            print(f"\n🔄 Recreating environment at episode {episode}...")
            env.close()
            from utils.env_factory import create_env
            env = create_env(config)
            print("✓ Environment recreated\n")

        # Safe reset with retry logic
        obs = safe_env_reset(env, max_retries=3, retry_delay=2.0)
        episode_reward = 0
        episode_wood = 0
        step_in_episode = 0

        # Run episode
        while step_in_episode < max_steps_per_episode:
            # DQN-SPECIFIC: Select action with epsilon-greedy
            action = agent.select_action(obs, explore=True)

            # Take step
            next_obs, reward, done, info = env.step(action)

            # Handle MineRL socket timeout errors
            if 'error' in info:
                print(f"⚠️  MineRL step error in episode {episode}: {info.get('error', 'unknown')}")
                print(f"   Terminating episode early (step {step_in_episode})")
                done = True  # Force episode termination
                # Don't store this experience - it's corrupted

            if render: 
                env.render()

            if done and 'error' in info: 
                # Skip storing corrupted experience, just break
                break
            
            # DQN-SPECIFIC: Store experience in replay buffer
            state = {
                'pov': obs['pov'],
                'time': float(obs['time_left'][0]) if hasattr(obs['time_left'], '__getitem__') else float(obs['time_left']),
                'yaw': float(obs['yaw'][0]) if hasattr(obs['yaw'], '__getitem__') else float(obs['yaw']),
                'pitch': float(obs['pitch'][0]) if hasattr(obs['pitch'], '__getitem__') else float(obs['pitch']),
                'place_table_safe': float(obs['place_table_safe'][0]) if hasattr(obs['place_table_safe'], '__getitem__') else float(obs['place_table_safe']),
            }
            next_state = {
                'pov': next_obs['pov'],
                'time': float(next_obs['time_left'][0]) if hasattr(next_obs['time_left'], '__getitem__') else float(next_obs['time_left']),
                'yaw': float(next_obs['yaw'][0]) if hasattr(next_obs['yaw'], '__getitem__') else float(next_obs['yaw']),
                'pitch': float(next_obs['pitch'][0]) if hasattr(next_obs['pitch'], '__getitem__') else float(next_obs['pitch']),
                'place_table_safe': float(next_obs['place_table_safe'][0]) if hasattr(next_obs['place_table_safe'], '__getitem__') else float(next_obs['place_table_safe']),
            }
            agent.store_experience(state, action, reward, next_state, done)

            # DQN-SPECIFIC: Train from replay buffer every train_freq steps
            if global_step % train_freq == 0 and agent.replay_buffer.is_ready():
                train_metrics = agent.train_step()

                # Log training metrics periodically
                if train_metrics and global_step % (train_freq * 100) == 0:
                    logger.log_training_step(
                        step=global_step,
                        loss=train_metrics.get('loss', 0),
                        q_mean=train_metrics.get('q_mean', 0),
                        td_error=train_metrics.get('td_error_mean', 0),
                        per_beta=train_metrics.get('per_beta', None)
                    )

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

            if done:
                break

        # Track wood for success rate
        recent_wood.append(episode_wood)
        if len(recent_wood) > 50: 
            recent_wood.pop(0)

        # Log episode
        logger.log_episode(
            episode_reward=episode_reward,
            episode_length=step_in_episode,
            wood_collected=episode_wood,
            epsilon=agent.get_epsilon(),
            episode_num=episode
        )

        # Console logging (uses common function)
        log_episode_stats(episode, num_episodes, global_step, episode_wood,
                         recent_wood, agent, env, obs, log_freq, starting_wood)
        
        # --- Grad-CAM Visualization ---
        grad_cam_freq = config['training'].get('grad_cam_freq', 0)
        if grad_cam_freq > 0 and episode % grad_cam_freq == 0:
            try:
                # We need the raw observation from before the wrappers processed it.
                # The `get_last_full_frame` method is on the StackAndProcessWrapper.
                # Assuming the wrapper stack allows access to it.
                raw_pov = env.get_last_full_frame()
                if raw_pov is not None:
                    from utils.visualization import generate_grad_cam_overlay
                    
                    # Create a dictionary for the observation as expected by the function
                    raw_obs_for_cam = {'pov': raw_pov}
                    
                    # Get the necessary components for Grad-CAM
                    model = agent.q_network
                    target_layer = model.cnn.conv[4] # Assuming 'medium' CNN arch
                    attack_action_index = config['grad_cam']['attack_action_index']
                    device = agent.device

                    # Generate the overlay
                    overlay_image = generate_grad_cam_overlay(
                        model, target_layer, raw_obs_for_cam, attack_action_index, device
                    )
                    
                    # Log the image to TensorBoard
                    logger.log_image("Grad-CAM/Attack_Action", overlay_image, episode)
                    print(f"📸 Generated Grad-CAM for episode {episode}.")

                    # Save the image to a file
                    import cv2
                    output_dir = "grad_cam_images"
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.join(output_dir, f"grad_cam_ep{episode}.jpg")
                    # Convert RGB to BGR for cv2.imwrite
                    cv2.imwrite(filename, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
                    print(f"   Saved Grad-CAM image to {filename}")

            except Exception as e:
                print(f"⚠️ Could not generate Grad-CAM visualization: {e}")

        # Save checkpoint (uses common function)
        if episode % save_freq == 0:
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