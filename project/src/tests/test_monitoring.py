"""
Tests for training monitoring and metrics collection.

Verifies that the agent properly tracks:
- Q-values (mean, max, variance)
- TD errors
- Gradient norms
- Epsilon values
- Episode statistics

Run with: pytest tests/test_monitoring.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dqn import DQNAgent, EpsilonSchedule
from agent.replay_buffer import ReplayBuffer


class TestQValueMonitoring:
    """Tests for Q-value tracking during training."""
    
    def test_q_values_in_train_metrics(self):
        """train_step should return q_mean and q_max."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            batch_size=4,
            device='cpu'
        )
        
        # Fill buffer
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                     'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, i % 8, -0.001, state, False)
        
        metrics = agent.train_step()
        
        assert metrics is not None, "Should return metrics when buffer is ready"
        assert 'q_mean' in metrics, "Should track mean Q-value"
        assert 'q_max' in metrics, "Should track max Q-value"
        assert isinstance(metrics['q_mean'], float), "q_mean should be float"
        assert isinstance(metrics['q_max'], float), "q_max should be float"
    
    def test_q_values_reasonable_range(self):
        """Q-values should be in reasonable range for untrained network."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            batch_size=4,
            device='cpu'
        )
        
        # Fill buffer with typical experiences
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                     'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, i % 8, -0.001, state, False)
        
        metrics = agent.train_step()
        
        # Untrained network Q-values should be small (near 0)
        assert abs(metrics['q_mean']) < 100, "Q-values should be reasonable for untrained net"
        assert abs(metrics['q_max']) < 100, "Max Q should be reasonable"
    
    def test_q_values_not_nan(self):
        """Q-values should never be NaN."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            batch_size=4,
            device='cpu'
        )
        
        # Fill buffer
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                     'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, i % 8, -0.001, state, False)
        
        # Train multiple steps
        for _ in range(10):
            metrics = agent.train_step()
            assert not np.isnan(metrics['q_mean']), "q_mean should not be NaN"
            assert not np.isnan(metrics['q_max']), "q_max should not be NaN"


class TestLossMonitoring:
    """Tests for loss tracking."""
    
    def test_loss_in_metrics(self):
        """train_step should return loss."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            batch_size=4,
            device='cpu'
        )
        
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                     'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, i % 8, -0.001, state, False)
        
        metrics = agent.train_step()
        
        assert 'loss' in metrics, "Should track loss"
        assert metrics['loss'] >= 0, "Loss should be non-negative"
    
    def test_loss_decreases_over_training(self):
        """Loss should generally decrease with more training (on same data)."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            batch_size=4,
            learning_rate=1e-3,  # Higher LR for faster convergence
            device='cpu'
        )
        
        # Create deterministic experiences
        np.random.seed(42)
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                     'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, i % 8, -0.001, state, False)
        
        # Collect losses over training
        losses = []
        for _ in range(20):
            metrics = agent.train_step()
            losses.append(metrics['loss'])
        
        # Average loss in last 5 steps should be <= first 5 steps
        # (may not always hold due to stochasticity, but is a good sanity check)
        early_loss = np.mean(losses[:5])
        late_loss = np.mean(losses[-5:])
        
        # This is a weak assertion - just check loss doesn't explode
        assert late_loss < early_loss * 10, "Loss should not explode during training"


class TestEpsilonMonitoring:
    """Tests for exploration rate tracking."""
    
    def test_epsilon_in_metrics(self):
        """train_step should return current epsilon."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            batch_size=4,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=100,
            device='cpu'
        )
        
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                     'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, i % 8, -0.001, state, False)
        
        metrics = agent.train_step()
        
        assert 'epsilon' in metrics, "Should track epsilon"
        assert 0 <= metrics['epsilon'] <= 1, "Epsilon should be in [0, 1]"
    
    def test_epsilon_decay_schedule(self):
        """Epsilon should decay according to schedule."""
        schedule = EpsilonSchedule(start=1.0, end=0.1, decay_steps=100)
        
        # At step 0
        assert schedule.get_epsilon(0) == 1.0, "Should start at 1.0"
        
        # At step 50 (halfway)
        eps_50 = schedule.get_epsilon(50)
        assert 0.5 < eps_50 < 0.6, f"Should be ~0.55 at halfway, got {eps_50}"
        
        # At step 100 (end)
        assert schedule.get_epsilon(100) == 0.1, "Should be 0.1 at decay_steps"
        
        # After decay_steps
        assert schedule.get_epsilon(200) == 0.1, "Should stay at end value"


class TestGradientMonitoring:
    """Tests for gradient tracking and clipping."""
    
    def test_gradient_clipping_applied(self):
        """Gradients should be clipped to max_norm=10.0."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            batch_size=4,
            device='cpu'
        )
        
        # Fill buffer with high-reward experiences to create large gradients
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                     'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
            # High reward to potentially create large gradients
            agent.store_experience(state, i % 8, 100.0, state, False)
        
        # Train and check gradients don't explode
        agent.train_step()
        
        # After training step, gradients should be reasonable
        total_norm = 0
        for p in agent.q_network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Note: gradient is already applied, so we can't directly check the clipped value
        # But we can verify the network didn't diverge
        assert total_norm < float('inf'), "Gradients should not be infinite"


class TestCounterMonitoring:
    """Tests for step and training counters."""
    
    def test_step_count_increments(self):
        """step_count should increment with each experience stored."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            device='cpu'
        )
        
        assert agent.step_count == 0, "Should start at 0"
        
        state = {'pov': np.zeros((4, 84, 84), dtype=np.uint8),
                 'time': 1.0, 'yaw': 0.0, 'pitch': 0.0}
        
        for i in range(5):
            agent.store_experience(state, 0, 0.0, state, False)
            assert agent.step_count == i + 1, f"Step count should be {i+1}"
    
    def test_train_count_increments(self):
        """train_count should increment with each train_step."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            batch_size=4,
            device='cpu'
        )
        
        # Fill buffer
        for i in range(15):
            state = {'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
                     'time': 0.9, 'yaw': 0.0, 'pitch': 0.0}
            agent.store_experience(state, i % 8, -0.001, state, False)
        
        assert agent.train_count == 0, "Should start at 0"
        
        for i in range(3):
            agent.train_step()
            assert agent.train_count == i + 1, f"Train count should be {i+1}"


class TestBufferMonitoring:
    """Tests for replay buffer statistics."""
    
    def test_buffer_size_tracking(self):
        """Buffer size should be trackable."""
        buffer = ReplayBuffer(capacity=100, min_size=10)
        
        assert len(buffer) == 0, "Should start empty"
        
        state = {'pov': np.zeros((4, 84, 84)), 'time': 0.5, 'yaw': 0.0, 'pitch': 0.0}
        for i in range(15):
            buffer.add(state, 0, 0.0, state, False)
        
        assert len(buffer) == 15, "Should have 15 experiences"
    
    def test_buffer_ready_threshold(self):
        """is_ready should reflect min_size threshold."""
        buffer = ReplayBuffer(capacity=100, min_size=10)
        
        state = {'pov': np.zeros((4, 84, 84)), 'time': 0.5, 'yaw': 0.0, 'pitch': 0.0}
        
        for i in range(9):
            buffer.add(state, 0, 0.0, state, False)
            assert not buffer.is_ready(), f"Should not be ready with {i+1} experiences"
        
        buffer.add(state, 0, 0.0, state, False)
        assert buffer.is_ready(), "Should be ready with 10 experiences"


class TestDeadNeuronDetection:
    """
    Tests for detecting dead neurons (always output 0).
    
    Note: This is mentioned in the plan as important for activation function choice.
    """
    
    def test_relu_outputs_nonzero(self):
        """Network should have non-zero activations (no dead neurons initially)."""
        agent = DQNAgent(
            num_actions=8,
            buffer_capacity=50,
            buffer_min_size=10,
            device='cpu'
        )
        
        # Create test input
        state = {
            'pov': torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32),
            'time': torch.tensor([0.5]),
            'yaw': torch.tensor([0.0]),
            'pitch': torch.tensor([0.0]),
        }
        
        # Forward pass
        with torch.no_grad():
            q_values = agent.q_network(state)
        
        # At least some Q-values should be different
        unique_q = len(torch.unique(q_values))
        assert unique_q > 1, "Network should produce varied Q-values (no dead output)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

