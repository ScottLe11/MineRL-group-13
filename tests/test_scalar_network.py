"""
Unit tests for ScalarNetwork and its integration with DQN and PPO networks.

Tests:
1. ScalarNetwork standalone functionality
2. DQNNetwork with/without scalar network
3. ActorCriticNetwork with/without scalar network
4. Parameter counting
5. Forward pass shapes
6. Backward compatibility
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks import ScalarNetwork, DQNNetwork, ActorCriticNetwork


class TestScalarNetwork(unittest.TestCase):
    """Test ScalarNetwork standalone."""

    def test_initialization(self):
        """Test network initialization with different configs."""
        configs = [
            (4, 32, 32),
            (4, 64, 64),
            (3, 128, 64),
            (5, 64, 128),
        ]

        for num_scalars, hidden_dim, output_dim in configs:
            with self.subTest(num_scalars=num_scalars, hidden=hidden_dim, output=output_dim):
                net = ScalarNetwork(num_scalars, hidden_dim, output_dim)
                self.assertEqual(net.num_scalars, num_scalars)
                self.assertEqual(net.hidden_dim, hidden_dim)
                self.assertEqual(net.output_dim, output_dim)
                self.assertEqual(net.get_output_dim(), output_dim)

    def test_forward_pass(self):
        """Test forward pass with different batch sizes."""
        net = ScalarNetwork(num_scalars=4, hidden_dim=64, output_dim=64)

        batch_sizes = [1, 2, 8, 32]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(batch_size, 4)
                output = net(input_tensor)

                self.assertEqual(output.shape, (batch_size, 64))
                self.assertTrue(torch.isfinite(output).all())

    def test_parameter_count(self):
        """Test that parameter count is correct."""
        net = ScalarNetwork(num_scalars=4, hidden_dim=64, output_dim=64)

        # Calculate expected parameters
        # Layer 1: 4 * 64 + 64 (bias) = 320
        # Layer 2: 64 * 64 + 64 (bias) = 4160
        # Total: 4480
        expected_params = (4 * 64 + 64) + (64 * 64 + 64)
        actual_params = sum(p.numel() for p in net.parameters())

        self.assertEqual(actual_params, expected_params)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        net = ScalarNetwork(num_scalars=4, hidden_dim=64, output_dim=64)
        input_tensor = torch.randn(2, 4, requires_grad=True)

        output = net(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(input_tensor.grad)
        for param in net.parameters():
            self.assertIsNotNone(param.grad)


class TestDQNNetworkIntegration(unittest.TestCase):
    """Test DQNNetwork with scalar network integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_actions = 8
        self.num_scalars = 3

        self.obs = {
            'pov': torch.randint(0, 256, (self.batch_size, 4, 84, 84), dtype=torch.float32),
            'time_left': torch.tensor([0.8, 0.5]),
            'yaw': torch.tensor([0.0, 0.5]),
            'pitch': torch.tensor([0.0, -0.2])
        }

    def test_without_scalar_network(self):
        """Test DQN without scalar network (default behavior)."""
        net = DQNNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=False
        )

        # Test forward pass
        q_values = net(self.obs)
        self.assertEqual(q_values.shape, (self.batch_size, self.num_actions))
        self.assertTrue(torch.isfinite(q_values).all())

        # Test that scalar_network is not created
        self.assertIsNone(net.scalar_network)

    def test_with_scalar_network(self):
        """Test DQN with scalar network enabled."""
        net = DQNNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=True,
            scalar_hidden_dim=64,
            scalar_output_dim=64
        )

        # Test forward pass
        q_values = net(self.obs)
        self.assertEqual(q_values.shape, (self.batch_size, self.num_actions))
        self.assertTrue(torch.isfinite(q_values).all())

        # Test that scalar_network is created
        self.assertIsNotNone(net.scalar_network)
        self.assertIsInstance(net.scalar_network, ScalarNetwork)

    def test_parameter_increase(self):
        """Test that scalar network adds correct number of parameters."""
        net_without = DQNNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=False
        )

        net_with = DQNNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=True,
            scalar_hidden_dim=64,
            scalar_output_dim=64
        )

        params_without = sum(p.numel() for p in net_without.parameters())
        params_with = sum(p.numel() for p in net_with.parameters())

        # Should add ~4480 params for scalar network + some extra for head size change
        self.assertGreater(params_with, params_without)
        self.assertLess(params_with - params_without, 100000)  # Reasonable upper bound

    def test_different_scalar_configs(self):
        """Test DQN with different scalar network configurations."""
        configs = [
            (32, 32),
            (64, 64),
            (128, 128),
            (64, 128),
        ]

        for hidden_dim, output_dim in configs:
            with self.subTest(hidden=hidden_dim, output=output_dim):
                net = DQNNetwork(
                    num_actions=self.num_actions,
                    num_scalars=self.num_scalars,
                    use_scalar_network=True,
                    scalar_hidden_dim=hidden_dim,
                    scalar_output_dim=output_dim
                )

                q_values = net(self.obs)
                self.assertEqual(q_values.shape, (self.batch_size, self.num_actions))

    def test_gradient_flow_with_scalar_network(self):
        """Test that gradients flow through scalar network in DQN."""
        net = DQNNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=True
        )

        q_values = net(self.obs)
        loss = q_values.sum()
        loss.backward()

        # Check gradients in scalar network
        for param in net.scalar_network.parameters():
            self.assertIsNotNone(param.grad)


class TestActorCriticNetworkIntegration(unittest.TestCase):
    """Test ActorCriticNetwork (PPO) with scalar network integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_actions = 8
        self.num_scalars = 4

        self.obs = {
            'pov': torch.randint(0, 256, (self.batch_size, 4, 84, 84), dtype=torch.uint8),
            'time_left': torch.tensor([0.8, 0.5]),
            'yaw': torch.tensor([0.0, 0.5]),
            'pitch': torch.tensor([0.0, -0.2]),
            'place_table_safe': torch.tensor([1.0, 0.0])
        }

    def test_without_scalar_network(self):
        """Test PPO without scalar network (default behavior)."""
        net = ActorCriticNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=False
        )

        # Test forward pass
        logits, value = net(self.obs)
        self.assertEqual(logits.shape, (self.batch_size, self.num_actions))
        self.assertEqual(value.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value).all())

        # Test that scalar_network is not created
        self.assertIsNone(net.scalar_network)

    def test_with_scalar_network(self):
        """Test PPO with scalar network enabled."""
        net = ActorCriticNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=True,
            scalar_hidden_dim=64,
            scalar_output_dim=64
        )

        # Test forward pass
        logits, value = net(self.obs)
        self.assertEqual(logits.shape, (self.batch_size, self.num_actions))
        self.assertEqual(value.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value).all())

        # Test that scalar_network is created
        self.assertIsNotNone(net.scalar_network)
        self.assertIsInstance(net.scalar_network, ScalarNetwork)

    def test_get_action_and_value(self):
        """Test get_action_and_value with scalar network."""
        net = ActorCriticNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=True
        )

        action, log_prob, entropy, value = net.get_action_and_value(self.obs)

        self.assertEqual(action.shape, (self.batch_size,))
        self.assertEqual(log_prob.shape, (self.batch_size,))
        self.assertEqual(entropy.shape, (self.batch_size,))
        self.assertEqual(value.shape, (self.batch_size,))

        # Check value ranges
        self.assertTrue((action >= 0).all() and (action < self.num_actions).all())
        self.assertTrue((entropy > 0).all())  # Entropy should be positive

    def test_parameter_increase(self):
        """Test that scalar network adds correct number of parameters."""
        net_without = ActorCriticNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=False
        )

        net_with = ActorCriticNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=True,
            scalar_hidden_dim=64,
            scalar_output_dim=64
        )

        params_without = sum(p.numel() for p in net_without.parameters())
        params_with = sum(p.numel() for p in net_with.parameters())

        # Should add ~4480 params for scalar network + some extra for head size change
        self.assertGreater(params_with, params_without)
        self.assertLess(params_with - params_without, 100000)  # Reasonable upper bound

    def test_gradient_flow_with_scalar_network(self):
        """Test that gradients flow through scalar network in PPO."""
        net = ActorCriticNetwork(
            num_actions=self.num_actions,
            num_scalars=self.num_scalars,
            use_scalar_network=True
        )

        logits, value = net(self.obs)
        loss = logits.sum() + value.sum()
        loss.backward()

        # Check gradients in scalar network
        for param in net.scalar_network.parameters():
            self.assertIsNotNone(param.grad)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility - networks should work identically without scalar network."""

    def test_dqn_default_behavior(self):
        """Test that DQN default behavior is unchanged."""
        # Create network with default parameters
        net = DQNNetwork(num_actions=8)

        # Should not have scalar network by default
        self.assertIsNone(net.scalar_network)
        self.assertFalse(net.use_scalar_network)

    def test_ppo_default_behavior(self):
        """Test that PPO default behavior is unchanged."""
        # Create network with default parameters
        net = ActorCriticNetwork(num_actions=8)

        # Should not have scalar network by default
        self.assertIsNone(net.scalar_network)
        self.assertFalse(net.use_scalar_network)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_scalar(self):
        """Test with single scalar input."""
        net = ScalarNetwork(num_scalars=1, hidden_dim=32, output_dim=32)
        input_tensor = torch.randn(4, 1)
        output = net(input_tensor)
        self.assertEqual(output.shape, (4, 32))

    def test_large_batch_size(self):
        """Test with large batch size."""
        net = ScalarNetwork(num_scalars=4, hidden_dim=64, output_dim=64)
        input_tensor = torch.randn(256, 4)
        output = net(input_tensor)
        self.assertEqual(output.shape, (256, 64))

    def test_different_architectures_with_scalar_network(self):
        """Test scalar network with different CNN architectures."""
        architectures = ['tiny', 'small', 'medium', 'wide', 'deep']

        for arch in architectures:
            with self.subTest(architecture=arch):
                net = DQNNetwork(
                    num_actions=8,
                    cnn_architecture=arch,
                    use_scalar_network=True
                )

                obs = {
                    'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32),
                    'time_left': torch.tensor([0.5, 0.5]),
                    'yaw': torch.tensor([0.0, 0.0]),
                    'pitch': torch.tensor([0.0, 0.0])
                }

                q_values = net(obs)
                self.assertEqual(q_values.shape, (2, 8))


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestScalarNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestDQNNetworkIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestActorCriticNetworkIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit(run_tests())
