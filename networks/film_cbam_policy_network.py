"""
FiLM-CBAM Policy Network for PPO with FiLM conditioning and enhanced spatial reasoning.

This network combines:
- Medium CNN backbone (conv1-3)
- FiLM for early scalar fusion
- CBAM attention
- Conv4 for additional spatial reasoning
- Scalar network for late fusion
- Multi-layer decision network
- Separate actor and critic heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .film import FiLMGenerator, FiLMLayer
from .attention import CBAM


class FiLMCBAMPolicyNetwork(nn.Module):
    """
    FiLM-CBAM Policy Network with both early fusion (FiLM) and late fusion (concatenation).

    Architecture:
        Visual pathway:
            Input: (4, 84, 84)
            Conv1: 32 filters, 8x8, stride 4 -> (32, 20, 20)
            Conv2: 64 filters, 4x4, stride 2 -> (64, 9, 9)
            Conv3: 128 filters, 3x3, stride 1 -> (128, 3, 3)
            FiLM conditioning with scalars
            CBAM attention
            Conv4: 128 filters, 3x3, stride 2, padding 1 -> (128, 4, 4)
            Flatten -> 2,048 features

        Scalar pathway:
            Input: 9 scalars
            FC1: 9 -> 32, ReLU
            FC2: 32 -> 32, ReLU

        Fusion:
            Concatenate: 2,048 + 32 = 2,080
            FC1: 2,080 -> 512, ReLU
            FC2: 512 -> 128, ReLU

        Actor-Critic heads:
            Actor (policy): 128 -> 128 -> num_actions (logits)
            Critic (value): 128 -> 128 -> 1 (state value)
    """

    def __init__(
        self,
        num_actions: int = 23,
        input_channels: int = 4,
        num_scalars: int = 9
    ):
        """
        Args:
            num_actions: Number of discrete actions (default: 23)
            input_channels: Number of stacked frames (default: 4)
            num_scalars: Number of scalar observations (default: 9)
        """
        super().__init__()

        self.num_actions = num_actions
        self.num_scalars = num_scalars

        # === VISUAL PATHWAY ===
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        # === FUSION POINT 1: FiLM ===
        self.film_gen = FiLMGenerator(scalar_dim=num_scalars, film_dim=128, hidden_dim=64)
        self.film_layer = FiLMLayer()

        # === ATTENTION ===
        self.cbam = CBAM(channels=128)

        # === SPATIAL REASONING ===
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # 7x7 -> 4x4

        # === SCALAR NETWORK ===
        self.scalar_network = nn.Sequential(
            nn.Linear(num_scalars, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # === DECISION LAYERS (after concatenation) ===
        # 128 × 4 × 4 = 2,048 visual features
        # + 32 scalar features
        # = 2,080 total
        self.fc1 = nn.Linear(2048 + 32, 512)
        self.fc2 = nn.Linear(512, 128)

        # === ACTOR HEAD (policy) ===
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # === CRITIC HEAD (value) ===
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs: dict, return_attention: bool = False):
        """
        Forward pass through the network.

        Args:
            obs: Dictionary with keys 'pov', 'time_left', 'yaw', 'pitch', 'place_table_safe',
                 'inv_logs', 'inv_planks', 'inv_sticks', 'inv_table', 'inv_axe'
                 OR a tensor for pov only (for simple testing)
            return_attention: If True, return attention maps

        Returns:
            action_logits: (batch, num_actions) tensor (raw logits)
            value: (batch, 1) tensor (state value)
            attention_info: dict with attention maps (only if return_attention=True)
        """
        if isinstance(obs, torch.Tensor):
            # Simple case: just pov tensor, no scalars
            pov = obs
            batch_size = pov.size(0)
            scalars = torch.zeros(batch_size, self.num_scalars, device=pov.device)
        else:
            # Full observation dict
            pov = obs['pov']
            batch_size = pov.size(0)
            device = pov.device

            # Handle scalar dimensions
            time_left = obs.get('time_left', torch.zeros(batch_size, device=device))
            yaw = obs.get('yaw', torch.zeros(batch_size, device=device))
            pitch = obs.get('pitch', torch.zeros(batch_size, device=device))
            place_table_safe = obs.get('place_table_safe', torch.zeros(batch_size, device=device))
            inv_logs = obs.get('inv_logs', torch.zeros(batch_size, device=device))
            inv_planks = obs.get('inv_planks', torch.zeros(batch_size, device=device))
            inv_sticks = obs.get('inv_sticks', torch.zeros(batch_size, device=device))
            inv_table = obs.get('inv_table', torch.zeros(batch_size, device=device))
            inv_axe = obs.get('inv_axe', torch.zeros(batch_size, device=device))

            # Ensure scalars are (batch,) - flatten any shape to 1D
            time_left = time_left.view(-1)
            yaw = yaw.view(-1)
            pitch = pitch.view(-1)
            place_table_safe = place_table_safe.view(-1)
            inv_logs = inv_logs.view(-1)
            inv_planks = inv_planks.view(-1)
            inv_sticks = inv_sticks.view(-1)
            inv_table = inv_table.view(-1)
            inv_axe = inv_axe.view(-1)

            # Stack into single tensor
            scalars = torch.stack([
                time_left, yaw, pitch, place_table_safe,
                inv_logs, inv_planks, inv_sticks, inv_table, inv_axe
            ], dim=1)  # (batch, 9)

        # Normalize visual input
        if pov.max() > 1.0:
            pov = pov.float() / 255.0

        # === VISUAL PROCESSING ===
        x = F.relu(self.conv1(pov))    # (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))      # (batch, 64, 9, 9)
        x = F.relu(self.conv3(x))      # (batch, 128, 7, 7)

        # === FiLM: Early fusion with scalars ===
        gamma, beta = self.film_gen(scalars)
        x = self.film_layer(x, gamma, beta)  # Context-aware features!

        # === ATTENTION on context-aware features ===
        x, attention_info = self.cbam(x)  # (batch, 128, 7, 7)

        # === ADDITIONAL SPATIAL PROCESSING ===
        x = F.relu(self.conv4(x))  # (batch, 128, 4, 4) - spatial reasoning!

        # === FLATTEN VISUAL FEATURES ===
        visual_features = x.view(x.size(0), -1)  # (batch, 2048)

        # === PROCESS SCALARS (late fusion pathway) ===
        scalar_features = self.scalar_network(scalars)  # (batch, 32)

        # === CONCATENATE VISUAL AND SCALAR FEATURES ===
        combined = torch.cat([visual_features, scalar_features], dim=1)  # (batch, 2080)

        # === DECISION LAYERS ===
        x = F.relu(self.fc1(combined))  # (batch, 512)
        x = F.relu(self.fc2(x))         # (batch, 128)

        # === ACTOR-CRITIC HEADS ===
        action_logits = self.actor(x)  # (batch, num_actions)
        value = self.critic(x)         # (batch, 1)

        if return_attention:
            return action_logits, value, attention_info
        else:
            return action_logits, value

    def get_action(self, obs: dict, deterministic: bool = False):
        """
        Sample action from policy.

        Args:
            obs: Single observation (will be batched internally)
            deterministic: If True, take argmax instead of sampling

        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            value: State value estimate
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if isinstance(obs, dict):
                obs_batched = {}
                for k, v in obs.items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() == 0:
                            obs_batched[k] = v.unsqueeze(0)
                        elif v.dim() == 3 and k == 'pov':
                            obs_batched[k] = v.unsqueeze(0)
                        else:
                            obs_batched[k] = v
                    else:
                        obs_batched[k] = torch.tensor([v])
            else:
                obs_batched = obs.unsqueeze(0)

            logits, value = self.forward(obs_batched)

            if deterministic:
                action = logits.argmax(dim=1)
                # For deterministic actions, log_prob is not well-defined
                # Return 0 as placeholder
                log_prob = torch.zeros_like(action, dtype=torch.float32)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, obs: dict, actions: torch.Tensor):
        """
        Evaluate actions for PPO training.

        Args:
            obs: Batch of observations
            actions: (batch,) tensor of actions to evaluate

        Returns:
            log_probs: (batch,) log probabilities of the actions
            values: (batch, 1) state values
            entropy: (batch,) entropy of the action distribution
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy

    def get_action_and_value(self, obs: dict, action: torch.Tensor = None):
        """
        Get action, log probability, entropy, and value for PPO.

        Args:
            obs: Observation dict
            action: Optional action to evaluate (for training). If None, samples action.

        Returns:
            (action, log_prob, entropy, value): Tuple for PPO training
        """
        action_logits, value = self.forward(obs)

        # Create categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Squeeze value from (batch, 1) to (batch,) to match PPO expectations
        value = value.squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: dict) -> torch.Tensor:
        """Get only the value estimate (for GAE computation)."""
        _, value = self.forward(obs)
        # Squeeze value from (batch, 1) to (batch,) to match PPO expectations
        return value.squeeze(-1)


if __name__ == "__main__":
    print("Testing FiLMCBAMPolicyNetwork...")

    network = FiLMCBAMPolicyNetwork(num_actions=23, input_channels=4, num_scalars=9)

    # Count parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {num_params:,} (~{num_params/1e6:.2f}M)")

    # Test with dict observation
    print("\n  Testing with dict observation...")
    obs = {
        'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32),
        'time_left': torch.tensor([0.8, 0.5]),
        'yaw': torch.tensor([45.0, -30.0]),
        'pitch': torch.tensor([0.0, 10.0]),
        'place_table_safe': torch.tensor([1.0, 0.0]),
        'inv_logs': torch.tensor([5.0, 2.0]),
        'inv_planks': torch.tensor([0.0, 4.0]),
        'inv_sticks': torch.tensor([0.0, 0.0]),
        'inv_table': torch.tensor([0.0, 1.0]),
        'inv_axe': torch.tensor([0.0, 0.0])
    }
    logits, value = network(obs)
    print(f"    Input pov shape: {obs['pov'].shape}")
    print(f"    Output logits shape: {logits.shape}")
    print(f"    Output value shape: {value.shape}")
    assert logits.shape == (2, 23), f"Expected (2, 23), got {logits.shape}"
    assert value.shape == (2, 1), f"Expected (2, 1), got {value.shape}"

    # Test with return_attention
    print("\n  Testing with return_attention=True...")
    logits, value, attention_info = network(obs, return_attention=True)
    print(f"    Logits shape: {logits.shape}")
    print(f"    Value shape: {value.shape}")
    print(f"    Attention info keys: {attention_info.keys()}")

    # Test with tensor only
    print("\n  Testing with tensor only...")
    pov_only = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32)
    logits, value = network(pov_only)
    print(f"    Input shape: {pov_only.shape}")
    print(f"    Logits shape: {logits.shape}")
    print(f"    Value shape: {value.shape}")
    assert logits.shape == (2, 23)
    assert value.shape == (2, 1)

    # Test action selection
    print("\n  Testing action selection...")
    single_obs = {
        'pov': torch.randint(0, 256, (4, 84, 84), dtype=torch.float32),
        'time_left': torch.tensor(0.5),
        'yaw': torch.tensor(0.0),
        'pitch': torch.tensor(0.0),
        'place_table_safe': torch.tensor(1.0),
        'inv_logs': torch.tensor(3.0),
        'inv_planks': torch.tensor(0.0),
        'inv_sticks': torch.tensor(0.0),
        'inv_table': torch.tensor(0.0),
        'inv_axe': torch.tensor(0.0)
    }

    action, log_prob, val = network.get_action(single_obs, deterministic=False)
    print(f"    Selected action (stochastic): {action}")
    print(f"    Log prob: {log_prob:.3f}")
    print(f"    Value: {val:.3f}")
    assert 0 <= action < 23

    action_det, _, _ = network.get_action(single_obs, deterministic=True)
    print(f"    Selected action (deterministic): {action_det}")
    assert 0 <= action_det < 23

    # Test evaluate_actions
    print("\n  Testing evaluate_actions...")
    actions = torch.tensor([5, 10])
    log_probs, values, entropy = network.evaluate_actions(obs, actions)
    print(f"    Log probs shape: {log_probs.shape}")
    print(f"    Values shape: {values.shape}")
    print(f"    Entropy shape: {entropy.shape}")
    assert log_probs.shape == (2,)
    assert values.shape == (2, 1)
    assert entropy.shape == (2,)

    print("\n✅ FiLMCBAMPolicyNetwork test passed!")
