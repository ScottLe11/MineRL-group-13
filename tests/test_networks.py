"""
Tests for neural network components.

Run with: pytest tests/test_networks.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.cnn import SmallCNN
from networks.dueling_head import DuelingHead
from networks.dqn_network import DQNNetwork


class TestSmallCNN:
    """Tests for the SmallCNN feature extractor."""
    
    def test_output_shape(self):
        """CNN should output (batch, 512) features."""
        cnn = SmallCNN(input_channels=4)
        x = torch.randn(2, 4, 84, 84)
        output = cnn(x)
        
        assert output.shape == (2, 512), f"Expected (2, 512), got {output.shape}"
    
    def test_single_sample(self):
        """CNN should handle batch size 1."""
        cnn = SmallCNN(input_channels=4)
        x = torch.randn(1, 4, 84, 84)
        output = cnn(x)
        
        assert output.shape == (1, 512)
    
    def test_different_input_channels(self):
        """CNN should accept different input channel counts."""
        for channels in [1, 3, 4, 8]:
            cnn = SmallCNN(input_channels=channels)
            x = torch.randn(2, channels, 84, 84)
            output = cnn(x)
            
            assert output.shape == (2, 512), f"Failed for {channels} channels"
    
    def test_gradient_flow(self):
        """Gradients should flow through the CNN."""
        cnn = SmallCNN(input_channels=4)
        x = torch.randn(2, 4, 84, 84, requires_grad=True)
        output = cnn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestDuelingHead:
    """Tests for the DuelingHead Q-value computation."""
    
    def test_output_shape(self):
        """DuelingHead should output (batch, num_actions)."""
        head = DuelingHead(input_dim=515, num_actions=23)
        x = torch.randn(2, 515)
        output = head(x)
        
        assert output.shape == (2, 23), f"Expected (2, 23), got {output.shape}"
    
    def test_different_action_counts(self):
        """DuelingHead should handle different action space sizes."""
        for num_actions in [8, 15, 23, 50]:
            head = DuelingHead(input_dim=512, num_actions=num_actions)
            x = torch.randn(4, 512)
            output = head(x)
            
            assert output.shape == (4, num_actions)
    
    def test_dueling_property(self):
        """
        Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean(A)).
        The mean of advantages should be ~0 due to centering.
        """
        head = DuelingHead(input_dim=512, num_actions=10)
        x = torch.randn(100, 512)
        
        # Get Q-values
        q_values = head(x)
        
        # The Q-values should vary across actions
        q_std = q_values.std(dim=1).mean()
        assert q_std > 0, "Q-values should vary across actions"


class TestDQNNetwork:
    """Tests for the full DQN network."""
    
    def test_forward_pass(self):
        """Full network should process observation dict correctly."""
        network = DQNNetwork(input_channels=4, num_actions=23)
        
        obs = {
            'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
            'time': torch.tensor([0.8, 0.5], dtype=torch.float32),
            'yaw': torch.tensor([0.0, 45.0], dtype=torch.float32),
            'pitch': torch.tensor([0.0, -10.0], dtype=torch.float32),
        }
        
        q_values = network(obs)
        
        assert q_values.shape == (2, 23), f"Expected (2, 23), got {q_values.shape}"
    
    def test_pov_normalization(self):
        """POV should be normalized to [0, 1] internally."""
        network = DQNNetwork(input_channels=4, num_actions=8)
        
        # Max value POV
        obs = {
            'pov': torch.full((1, 4, 84, 84), 255, dtype=torch.uint8),
            'time': torch.tensor([1.0]),
            'yaw': torch.tensor([0.0]),
            'pitch': torch.tensor([0.0]),
        }
        
        # Should not crash or produce NaN
        q_values = network(obs)
        assert not torch.isnan(q_values).any(), "Q-values should not be NaN"
    
    def test_scalar_dimensions(self):
        """Network should handle both (batch,) and (batch, 1) scalar shapes."""
        network = DQNNetwork(input_channels=4, num_actions=8)
        
        # Shape (batch,)
        obs1 = {
            'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
            'time': torch.tensor([0.5, 0.3]),
            'yaw': torch.tensor([0.0, 45.0]),
            'pitch': torch.tensor([0.0, -10.0]),
        }
        q1 = network(obs1)
        
        # Shape (batch, 1)
        obs2 = {
            'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
            'time': torch.tensor([[0.5], [0.3]]),
            'yaw': torch.tensor([[0.0], [45.0]]),
            'pitch': torch.tensor([[0.0], [-10.0]]),
        }
        q2 = network(obs2)
        
        assert q1.shape == q2.shape == (2, 8)
    
    def test_gradient_flow_full_network(self):
        """Gradients should flow through the entire network."""
        network = DQNNetwork(input_channels=4, num_actions=8)
        
        obs = {
            'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
            'time': torch.tensor([0.5, 0.3], requires_grad=True),
            'yaw': torch.tensor([0.0, 45.0], requires_grad=True),
            'pitch': torch.tensor([0.0, -10.0], requires_grad=True),
        }
        
        q_values = network(obs)
        loss = q_values.sum()
        loss.backward()
        
        # Check gradients exist for network parameters
        for name, param in network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestNetworkIntegration:
    """Integration tests for network components."""
    
    def test_target_network_copy(self):
        """Target network should be copyable from online network."""
        online = DQNNetwork(input_channels=4, num_actions=8)
        target = DQNNetwork(input_channels=4, num_actions=8)
        
        # Copy weights
        target.load_state_dict(online.state_dict())
        
        # Same input should produce same output
        obs = {
            'pov': torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.uint8),
            'time': torch.tensor([0.5]),
            'yaw': torch.tensor([0.0]),
            'pitch': torch.tensor([0.0]),
        }
        
        online.eval()
        target.eval()
        
        with torch.no_grad():
            q_online = online(obs)
            q_target = target(obs)
        
        assert torch.allclose(q_online, q_target), "Copied networks should produce same output"
    
    def test_soft_update(self):
        """Soft update should blend weights correctly."""
        online = DQNNetwork(input_channels=4, num_actions=8)
        target = DQNNetwork(input_channels=4, num_actions=8)
        
        tau = 0.005
        
        # Store original target weights
        original_target_param = list(target.parameters())[0].clone()
        
        # Soft update
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
        
        # Weights should have changed slightly
        new_target_param = list(target.parameters())[0]
        assert not torch.equal(original_target_param, new_target_param), "Weights should change after soft update"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


