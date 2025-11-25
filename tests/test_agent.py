"""
Tests for the DQN agent and replay buffer.

Run with: pytest tests/test_agent.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.replay_buffer import ReplayBuffer
from agent.dqn import DQNAgent, EpsilonSchedule


class TestReplayBuffer:
    """Tests for the ReplayBuffer."""
    
    def test_initialization(self):
        """Buffer should initialize with correct parameters."""
        buffer = ReplayBuffer(capacity=100, min_size=10)
        
        assert len(buffer) == 0
        assert buffer.is_ready() is False
    
    def test_add_experiences(self):
        """Buffer should store experiences correctly."""
        buffer = ReplayBuffer(capacity=100, min_size=10)
        
        for i in range(15):
            state = {'pov': np.zeros((4, 64, 64)), 'time': 0.5, 'yaw': 0.0, 'pitch': 0.0}
            buffer.add(state, action=i % 8, reward=-0.001, next_state=state, done=False)
        
        assert len(buffer) == 15
        assert buffer.is_ready() is True
    
    def test_capacity_limit(self):
        """Buffer should not exceed capacity."""
        buffer = ReplayBuffer(capacity=10, min_size=5)
        
        for i in range(20):
            state = {'pov': np.zeros((4, 64, 64)), 'time': 0.5, 'yaw': 0.0, 'pitch': 0.0}
            buffer.add(state, action=0, reward=0, next_state=state, done=False)
        
        assert len(buffer) == 10  # Should be capped at capacity
    
    def test_sampling(self):
        """Buffer should return correct batch structure."""
        buffer = ReplayBuffer(capacity=100, min_size=10)
        
        for i in range(20):
            state = {'pov': np.random.rand(4, 64, 64), 'time': 0.5, 'yaw': float(i), 'pitch': 0.0}
            next_state = {'pov': np.random.rand(4, 64, 64), 'time': 0.4, 'yaw': float(i+1), 'pitch': 0.0}
            buffer.add(state, action=i % 8, reward=-0.001, next_state=next_state, done=(i == 19))
        
        states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)
        
        assert len(states) == 5
        assert len(actions) == 5
        assert len(rewards) == 5
        assert len(next_states) == 5
        assert len(dones) == 5
    
    def test_sample_not_ready(self):
        """Sampling from not-ready buffer should raise error."""
        buffer = ReplayBuffer(capacity=100, min_size=10)
        
        for i in range(5):
            state = {'pov': np.zeros((4, 64, 64)), 'time': 0.5, 'yaw': 0.0, 'pitch': 0.0}
            buffer.add(state, action=0, reward=0, next_state=state, done=False)
        
        assert buffer.is_ready() is False
        
        with pytest.raises(ValueError, match="not ready"):
            buffer.sample(batch_size=3)


class TestEpsilonSchedule:
    """Tests for epsilon-greedy exploration schedule."""
    
    def test_initial_epsilon(self):
        """Epsilon should start at start value."""
        schedule = EpsilonSchedule(start=1.0, end=0.1, decay_steps=1000)
        
        assert schedule.get_epsilon(0) == 1.0
    
    def test_final_epsilon(self):
        """Epsilon should end at end value."""
        schedule = EpsilonSchedule(start=1.0, end=0.1, decay_steps=1000)
        
        assert schedule.get_epsilon(1000) == 0.1
        assert schedule.get_epsilon(2000) == 0.1  # Should stay at end
    
    def test_linear_decay(self):
        """Epsilon should decay linearly."""
        schedule = EpsilonSchedule(start=1.0, end=0.0, decay_steps=100)
        
        assert abs(schedule.get_epsilon(50) - 0.5) < 0.01
        assert abs(schedule.get_epsilon(25) - 0.75) < 0.01
        assert abs(schedule.get_epsilon(75) - 0.25) < 0.01


class TestDQNAgent:
    """Tests for the DQN agent."""
    
    @pytest.fixture
    def agent_config(self):
        """Create a minimal config for testing."""
        return {
            'device': 'cpu',
            'network': {
                'input_channels': 4,
            },
            'dqn': {
                'num_actions': 8,
                'learning_rate': 0.001,
                'batch_size': 4,
                'gamma': 0.99,
                'replay_buffer': {
                    'capacity': 100,
                    'min_size': 10,
                },
                'exploration': {
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.1,
                    'epsilon_decay_steps': 100,
                },
                'gradient_clip': 1.0,
                'target_update': {
                    'tau': 0.005,
                }
            }
        }
    
    def test_agent_initialization(self, agent_config):
        """Agent should initialize correctly."""
        agent = DQNAgent(agent_config)
        
        assert agent.q_network is not None
        assert agent.target_network is not None
        assert agent.replay_buffer is not None
    
    def test_action_selection_exploration(self, agent_config):
        """Agent should explore at high epsilon."""
        agent = DQNAgent(agent_config)
        
        obs = {
            'pov': np.zeros((4, 64, 64), dtype=np.uint8),
            'time': 1.0,
            'yaw': 0.0,
            'pitch': 0.0,
        }
        
        # At step 0, epsilon=1.0, should always explore
        actions = [agent.select_action(obs, current_step=0) for _ in range(100)]
        unique_actions = set(actions)
        
        # Should see multiple different actions due to random exploration
        assert len(unique_actions) > 1, "Should explore different actions"
    
    def test_experience_storage(self, agent_config):
        """Agent should store experiences in replay buffer."""
        agent = DQNAgent(agent_config)
        
        state = {'pov': np.zeros((4, 64, 64)), 'time': 1.0, 'yaw': 0.0, 'pitch': 0.0}
        next_state = {'pov': np.zeros((4, 64, 64)), 'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
        
        agent.store_experience(state, action=0, reward=-0.001, next_state=next_state, done=False)
        
        assert len(agent.replay_buffer) == 1
    
    def test_training_step(self, agent_config):
        """Agent should perform training step when buffer is ready."""
        agent = DQNAgent(agent_config)
        
        # Fill buffer
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8), 
                     'time': 1.0 - i*0.01, 'yaw': 0.0, 'pitch': 0.0}
            next_state = {'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8), 
                          'time': 0.99 - i*0.01, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, action=i % 8, reward=-0.001, next_state=next_state, done=False)
        
        # Training step should return metrics
        metrics = agent.train_step()
        
        assert 'loss' in metrics
        assert 'q_mean' in metrics
        assert metrics['loss'] >= 0
    
    def test_no_training_when_buffer_not_ready(self, agent_config):
        """Agent should not train when buffer has insufficient experiences."""
        agent = DQNAgent(agent_config)
        
        # Add only a few experiences
        for i in range(5):
            state = {'pov': np.zeros((4, 64, 64)), 'time': 1.0, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, action=0, reward=0, next_state=state, done=False)
        
        metrics = agent.train_step()
        
        assert metrics == {}  # Empty dict when buffer not ready


class TestAgentIntegration:
    """Integration tests for the agent."""
    
    def test_full_episode_simulation(self):
        """Simulate a full episode with the agent."""
        config = {
            'device': 'cpu',
            'network': {'input_channels': 4},
            'dqn': {
                'num_actions': 8,
                'learning_rate': 0.001,
                'batch_size': 4,
                'gamma': 0.99,
                'replay_buffer': {'capacity': 100, 'min_size': 10},
                'exploration': {'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay_steps': 50},
                'gradient_clip': 1.0,
                'target_update': {'tau': 0.005}
            }
        }
        
        agent = DQNAgent(config)
        
        # Simulate episode
        obs = {'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
               'time': 1.0, 'yaw': 0.0, 'pitch': 0.0}
        
        total_reward = 0
        for step in range(50):
            action = agent.select_action(obs, current_step=step)
            
            # Simulate environment response
            next_obs = {'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
                        'time': max(0, 1.0 - step * 0.02), 'yaw': 0.0, 'pitch': 0.0}
            reward = -0.001
            done = (step == 49)
            
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Try training
            if agent.replay_buffer.is_ready():
                metrics = agent.train_step()
                assert metrics != {}  # Should get metrics when buffer is ready
            
            obs = next_obs
            total_reward += reward
        
        # Should have completed without errors
        assert len(agent.replay_buffer) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


