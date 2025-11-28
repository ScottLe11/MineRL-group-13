"""
Integration tests for MineRL Tree-Chopping RL Agent.

Tests the integration between:
- Environment creation and wrappers
- Agents (DQN and PPO)
- Networks (CNN architectures, attention)
- Training loops
- Checkpoint saving/loading
"""

import pytest
import os
import sys
import tempfile
import shutil
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.env_factory import create_env
from utils.agent_factory import create_agent
from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from networks.dqn_network import DQNNetwork
from networks.policy_network import ActorCriticNetwork
from trainers.helpers import save_checkpoint


class TestEnvironmentIntegration:
    """Test environment creation and wrapper integration."""

    def setup_method(self):
        """Setup test configuration."""
        self.config = {
            'environment': {
                'name': 'MineRLcustom_treechop-v0',
                'episode_seconds': 5,  # Short for testing
                'frame_shape': [84, 84],
                'frame_stack': 4,
                'curriculum': {
                    'spawn_type': 'random',
                    'with_logs': 0,
                    'with_axe': True
                }
            },
            'action_space': {
                'preset': 'base'
            },
            'rewards': {
                'wood_value': 5.0,
                'step_penalty': -0.001
            },
            'device': 'cpu',
            'seed': 42
        }

    def test_environment_creation(self):
        """Test environment can be created with all wrappers."""
        env = create_env(self.config)
        assert env is not None
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        env.close()

    def test_environment_reset(self):
        """Test environment reset returns correct observation format."""
        env = create_env(self.config)
        obs = env.reset()

        # Check observation is a dict with required keys
        assert isinstance(obs, dict)
        assert 'pov' in obs
        assert 'time' in obs  # Fixed: was 'time_left'
        assert 'yaw' in obs
        assert 'pitch' in obs

        # Check observation shapes and types
        assert obs['pov'].shape == (4, 84, 84)  # Stacked frames
        assert obs['pov'].dtype == np.uint8
        assert obs['time'].shape == (1,)
        assert obs['time'].dtype == np.float32
        assert obs['yaw'].shape == (1,)
        assert obs['pitch'].shape == (1,)

        env.close()

    def test_environment_step(self):
        """Test environment step returns correct format."""
        env = create_env(self.config)
        obs = env.reset()

        # Take a step
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)

        # Check types
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        # Check observation format matches reset
        assert 'pov' in next_obs
        assert 'time' in next_obs
        assert next_obs['pov'].shape == obs['pov'].shape

        env.close()


class TestDQNIntegration:
    """Test DQN agent integration with environment and network."""

    def setup_method(self):
        """Setup test configuration and environment."""
        self.config = {
            'environment': {
                'name': 'MineRLcustom_treechop-v0',
                'episode_seconds': 5,
                'frame_shape': [84, 84],
                'frame_stack': 4,
                'curriculum': {'spawn_type': 'random', 'with_logs': 0, 'with_axe': True}
            },
            'action_space': {'preset': 'base'},
            'rewards': {'wood_value': 5.0, 'step_penalty': -0.001},
            'network': {
                'input_channels': 4,
                'architecture': 'tiny',  # Tiny for faster tests
                'attention': 'none'
            },
            'algorithm': 'dqn',
            'dqn': {
                'learning_rate': 0.0001,
                'gamma': 0.99,
                'batch_size': 32,
                'gradient_clip': 10,
                'replay_buffer': {'capacity': 1000, 'min_size': 100},
                'exploration': {'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay_steps': 1000},
                'target_update': {'method': 'hard', 'hard_update_freq': 100},
                'prioritized_replay': {'enabled': False}
            },
            'training': {'checkpoint_dir': 'test_checkpoints'},
            'device': 'cpu',
            'seed': 42
        }
        self.env = create_env(self.config)

    def teardown_method(self):
        """Cleanup."""
        self.env.close()

    def test_dqn_agent_creation(self):
        """Test DQN agent can be created."""
        agent = create_agent(self.config, num_actions=self.env.action_space.n)
        assert isinstance(agent, DQNAgent)
        assert agent.num_actions == self.env.action_space.n

    def test_dqn_network_forward_pass(self):
        """Test DQN network can process real observations."""
        agent = create_agent(self.config, num_actions=self.env.action_space.n)
        obs = self.env.reset()

        # Test get_q_values
        q_values = agent.get_q_values(obs)
        assert isinstance(q_values, np.ndarray)
        assert q_values.shape == (self.env.action_space.n,)

    def test_dqn_action_selection(self):
        """Test DQN agent can select actions."""
        agent = create_agent(self.config, num_actions=self.env.action_space.n)
        obs = self.env.reset()

        # Test greedy action selection
        action = agent.select_action(obs, explore=False)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < self.env.action_space.n

        # Test exploration
        action = agent.select_action(obs, explore=True)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < self.env.action_space.n

    def test_dqn_training_step(self):
        """Test DQN agent can collect experiences and train."""
        agent = create_agent(self.config, num_actions=self.env.action_space.n)
        obs = self.env.reset()

        # Collect some experiences
        for _ in range(150):  # More than min_size
            action = agent.select_action(obs, explore=True)
            next_obs, reward, done, info = self.env.step(action)

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)

            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

        # Should be able to train now
        assert len(agent.replay_buffer) >= self.config['dqn']['replay_buffer']['min_size']

        # Perform training step
        metrics = agent.train_step()
        assert metrics is not None
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)


class TestPPOIntegration:
    """Test PPO agent integration with environment and network."""

    def setup_method(self):
        """Setup test configuration and environment."""
        self.config = {
            'environment': {
                'name': 'MineRLcustom_treechop-v0',
                'episode_seconds': 5,
                'frame_shape': [84, 84],
                'frame_stack': 4,
                'curriculum': {'spawn_type': 'random', 'with_logs': 0, 'with_axe': True}
            },
            'action_space': {'preset': 'base'},
            'rewards': {'wood_value': 5.0, 'step_penalty': -0.001},
            'network': {
                'input_channels': 4,
                'architecture': 'tiny',
                'attention': 'none'
            },
            'algorithm': 'ppo',
            'ppo': {
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_steps': 128,  # Small for testing
                'n_epochs': 3,
                'batch_size': 32
            },
            'training': {'checkpoint_dir': 'test_checkpoints'},
            'device': 'cpu',
            'seed': 42
        }
        self.env = create_env(self.config)

    def teardown_method(self):
        """Cleanup."""
        self.env.close()

    def test_ppo_agent_creation(self):
        """Test PPO agent can be created."""
        agent = create_agent(self.config, num_actions=self.env.action_space.n)
        assert isinstance(agent, PPOAgent)
        assert agent.num_actions == self.env.action_space.n

    def test_ppo_network_forward_pass(self):
        """Test PPO policy network can process real observations."""
        agent = create_agent(self.config, num_actions=self.env.action_space.n)
        obs = self.env.reset()

        # Test select_action
        action, log_prob, value = agent.select_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < self.env.action_space.n
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_ppo_rollout_collection(self):
        """Test PPO agent can collect rollouts."""
        agent = create_agent(self.config, num_actions=self.env.action_space.n)
        obs = self.env.reset()
        n_steps = self.config['ppo']['n_steps']

        # Collect rollout
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            agent.store_transition(obs, action, log_prob, reward, value, done)

            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

        # Buffer should be full
        assert len(agent.buffer.observations) >= n_steps

    def test_ppo_update(self):
        """Test PPO agent can perform policy update."""
        agent = create_agent(self.config, num_actions=self.env.action_space.n)
        obs = self.env.reset()
        n_steps = self.config['ppo']['n_steps']

        # Collect full rollout
        for _ in range(n_steps + 10):  # Extra to ensure full buffer
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            agent.store_transition(obs, action, log_prob, reward, value, done)

            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

        # Perform update
        metrics = agent.update(obs)
        assert metrics is not None
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics


class TestNetworkArchitectures:
    """Test different network architectures and attention mechanisms."""

    def test_all_cnn_architectures_dqn(self):
        """Test all CNN architectures work with DQN network."""
        architectures = ['tiny', 'small', 'medium', 'wide', 'deep']

        for arch in architectures:
            network = DQNNetwork(
                num_actions=23,
                input_channels=4,
                cnn_architecture=arch,
                attention_type='none'
            )

            # Test forward pass
            obs = {
                'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32),
                'time': torch.tensor([[0.5], [0.3]], dtype=torch.float32),
                'yaw': torch.tensor([[0.0], [0.1]], dtype=torch.float32),
                'pitch': torch.tensor([[0.0], [-0.1]], dtype=torch.float32)
            }
            q_values = network(obs)
            assert q_values.shape == (2, 23)

    def test_all_cnn_architectures_ppo(self):
        """Test all CNN architectures work with PPO policy network."""
        architectures = ['tiny', 'small', 'medium', 'wide', 'deep']

        for arch in architectures:
            network = ActorCriticNetwork(
                num_actions=23,
                input_channels=4,
                cnn_architecture=arch,
                attention_type='none'
            )

            # Test forward pass
            obs = {
                'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32),
                'time': torch.tensor([[0.5], [0.3]], dtype=torch.float32),
                'yaw': torch.tensor([[0.0], [0.1]], dtype=torch.float32),
                'pitch': torch.tensor([[0.0], [-0.1]], dtype=torch.float32)
            }
            action_logits, value = network(obs)
            assert action_logits.shape == (2, 23)
            assert value.shape == (2,)

    def test_attention_mechanisms(self):
        """Test different attention mechanisms work."""
        attention_types = ['none', 'spatial', 'cbam', 'treechop_bias']

        for attn in attention_types:
            network = DQNNetwork(
                num_actions=23,
                input_channels=4,
                cnn_architecture='small',
                attention_type=attn
            )

            obs = {
                'pov': torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32),
                'time': torch.tensor([[0.5]], dtype=torch.float32),
                'yaw': torch.tensor([[0.0]], dtype=torch.float32),
                'pitch': torch.tensor([[0.0]], dtype=torch.float32)
            }
            q_values = network(obs)
            assert q_values.shape == (1, 23)


class TestCheckpointIntegration:
    """Test checkpoint saving and loading."""

    def setup_method(self):
        """Setup temp directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'environment': {
                'name': 'MineRLcustom_treechop-v0',
                'episode_seconds': 5,
                'frame_shape': [84, 84],
                'frame_stack': 4,
                'curriculum': {'spawn_type': 'random', 'with_logs': 0, 'with_axe': True}
            },
            'action_space': {'preset': 'base'},
            'rewards': {'wood_value': 5.0, 'step_penalty': -0.001},
            'network': {'input_channels': 4, 'architecture': 'tiny', 'attention': 'none'},
            'dqn': {
                'learning_rate': 0.0001,
                'gamma': 0.99,
                'batch_size': 32,
                'gradient_clip': 10,
                'replay_buffer': {'capacity': 1000, 'min_size': 100},
                'exploration': {'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay_steps': 1000},
                'target_update': {'method': 'hard', 'hard_update_freq': 100},
                'prioritized_replay': {'enabled': False}
            },
            'ppo': {
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_steps': 128,
                'n_epochs': 3,
                'batch_size': 32
            },
            'training': {'checkpoint_dir': self.temp_dir},
            'device': 'cpu',
            'seed': 42
        }

    def teardown_method(self):
        """Cleanup temp directory."""
        shutil.rmtree(self.temp_dir)

    def test_dqn_checkpoint_save_load(self):
        """Test DQN checkpoint saving and loading."""
        self.config['algorithm'] = 'dqn'
        agent = create_agent(self.config, num_actions=23)

        # Save checkpoint
        save_checkpoint(agent, self.config, episode=10, save_buffer=False)

        # Check file exists
        checkpoint_path = os.path.join(self.temp_dir, 'checkpoint_dqn_ep10.pt')
        assert os.path.exists(checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        assert 'episode' in checkpoint
        assert checkpoint['episode'] == 10
        assert 'q_network_state_dict' in checkpoint
        assert 'target_network_state_dict' in checkpoint

    def test_ppo_checkpoint_save_load(self):
        """Test PPO checkpoint saving and loading."""
        self.config['algorithm'] = 'ppo'
        agent = create_agent(self.config, num_actions=23)

        # Save checkpoint
        save_checkpoint(agent, self.config, episode=20)

        # Check file exists
        checkpoint_path = os.path.join(self.temp_dir, 'checkpoint_ppo_ep20.pt')
        assert os.path.exists(checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        assert 'episode' in checkpoint
        assert checkpoint['episode'] == 20
        assert 'policy_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint


class TestEndToEndIntegration:
    """Test complete training episodes end-to-end."""

    def test_dqn_full_episode(self):
        """Test DQN agent can complete a full episode."""
        config = {
            'environment': {
                'name': 'MineRLcustom_treechop-v0',
                'episode_seconds': 5,  # Short episode
                'frame_shape': [84, 84],
                'frame_stack': 4,
                'curriculum': {'spawn_type': 'random', 'with_logs': 0, 'with_axe': True}
            },
            'action_space': {'preset': 'base'},
            'rewards': {'wood_value': 5.0, 'step_penalty': -0.001},
            'network': {'input_channels': 4, 'architecture': 'tiny', 'attention': 'none'},
            'algorithm': 'dqn',
            'dqn': {
                'learning_rate': 0.0001,
                'gamma': 0.99,
                'batch_size': 32,
                'gradient_clip': 10,
                'replay_buffer': {'capacity': 1000, 'min_size': 100},
                'exploration': {'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay_steps': 1000},
                'target_update': {'method': 'hard', 'hard_update_freq': 100},
                'prioritized_replay': {'enabled': False}
            },
            'device': 'cpu',
            'seed': 42
        }

        env = create_env(config)
        agent = create_agent(config, num_actions=env.action_space.n)

        obs = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 5 * 5  # 5 seconds * 5 steps/sec

        while steps < max_steps:
            action = agent.select_action(obs, explore=True)
            next_obs, reward, done, info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, done)

            episode_reward += reward
            steps += 1
            obs = next_obs

            if done:
                break

        # Episode completed successfully
        assert steps > 0
        assert isinstance(episode_reward, (float, np.floating))

        env.close()

    def test_ppo_full_episode(self):
        """Test PPO agent can complete a full episode."""
        config = {
            'environment': {
                'name': 'MineRLcustom_treechop-v0',
                'episode_seconds': 5,
                'frame_shape': [84, 84],
                'frame_stack': 4,
                'curriculum': {'spawn_type': 'random', 'with_logs': 0, 'with_axe': True}
            },
            'action_space': {'preset': 'base'},
            'rewards': {'wood_value': 5.0, 'step_penalty': -0.001},
            'network': {'input_channels': 4, 'architecture': 'tiny', 'attention': 'none'},
            'algorithm': 'ppo',
            'ppo': {
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_steps': 128,
                'n_epochs': 3,
                'batch_size': 32
            },
            'device': 'cpu',
            'seed': 42
        }

        env = create_env(config)
        agent = create_agent(config, num_actions=env.action_space.n)

        obs = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 5 * 5

        while steps < max_steps:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            agent.store_transition(obs, action, log_prob, reward, value, done)

            episode_reward += reward
            steps += 1
            obs = next_obs

            if done:
                break

        # Episode completed successfully
        assert steps > 0
        assert isinstance(episode_reward, (float, np.floating))

        env.close()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
