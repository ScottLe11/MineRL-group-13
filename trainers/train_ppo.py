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

    # PPO specific
    ppo_config = config['ppo']
    n_steps = ppo_config['n_steps']

    # Training state
    global_step = 0
    best_avg_wood = 0
    recent_wood = []
    episode = 0

    # Main training loop
    while episode < num_episodes:
        # Periodically recreate environment to prevent memory leaks
        if episode > 0 and episode % env_recreation_interval == 0:
            print(f"\n🔄 Recreating environment at episode {episode} (prevent memory leaks)...")
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

            if render:
                env.render()

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
            save_checkpoint(agent, config, episode)

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
