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

    # Episode settings
    env_config = config['environment']
    episode_seconds = env_config.get('episode_seconds', 20)
    max_steps_per_episode = episode_seconds * 5

    # Training state
    global_step = 0
    best_avg_wood = 0
    recent_wood = []  # Track last 50 episodes

    # Main training loop
    for episode in range(1, num_episodes + 1):
        # Periodically recreate environment to prevent memory leaks
        if episode > 1 and episode % env_recreation_interval == 0:
            print(f"\n🔄 Recreating environment at episode {episode} (prevent memory leaks)...")
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
            epsilon=agent.get_epsilon()
        )

        # Console logging (uses common function)
        log_episode_stats(episode, num_episodes, global_step, episode_wood,
                         recent_wood, agent, env, obs, log_freq)

        # Save checkpoint (uses common function)
        if episode % save_freq == 0:
            save_checkpoint(agent, config, episode)

            # Save best model
            avg_wood = np.mean(recent_wood) if recent_wood else 0
            if avg_wood > best_avg_wood:
                best_avg_wood = avg_wood
                save_checkpoint(agent, config, episode, best=True)

    # Final save
    save_checkpoint(agent, config, num_episodes, final=True)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {global_step}")
    print(f"Best avg wood (50 ep): {best_avg_wood:.2f}")
    print(f"{'='*60}")

    return env
