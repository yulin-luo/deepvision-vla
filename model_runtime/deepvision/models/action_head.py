"""
Action Head Module for Continuous Action Prediction

This module contains MLP-based architectures for predicting continuous actions
from transformer hidden states, using L1 regression.
"""

import torch
import torch.nn as nn


ACTION_QUERY_CONTRACTS = frozenset({
    "scalar_per_dim_v1",
    "per_step_shared_v1",
    "decoupled_per_dim_v1",
})
CHUNK_EXECUTION_CONTRACTS = frozenset({
    "reject",
    "receding_horizon_first",
})


def action_query_count(action_chunk, action_dim, action_query_contract):
    """Return the number of transformer query positions for an action contract."""
    action_chunk = int(action_chunk)
    action_dim = int(action_dim)
    if action_chunk <= 0:
        raise ValueError(f"action_chunk must be positive, got {action_chunk}")
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    if action_query_contract in ("scalar_per_dim_v1", "decoupled_per_dim_v1"):
        return action_chunk * action_dim
    if action_query_contract == "per_step_shared_v1":
        return action_chunk
    raise ValueError(
        "action_query_contract must be one of "
        f"{sorted(ACTION_QUERY_CONTRACTS)}, got {action_query_contract!r}"
    )


def action_head_contract(action_query_contract):
    if action_query_contract == "scalar_per_dim_v1":
        return "flatten_hxd_hidden_then_mlp_resnet_v1"
    if action_query_contract == "per_step_shared_v1":
        return "shared_per_step_mlp_resnet_to_7d_v1"
    if action_query_contract == "decoupled_per_dim_v1":
        return "decoupled_per_dim_trunk_shared_then_per_dim_linear_v1"
    raise ValueError(
        "action_query_contract must be one of "
        f"{sorted(ACTION_QUERY_CONTRACTS)}, got {action_query_contract!r}"
    )


def select_action_for_execution(predicted_actions, chunk_execution_contract):
    """Select one complete action from a predicted [H,D] action chunk."""
    if predicted_actions.ndim != 2:
        raise ValueError(
            "predicted_actions must have shape [H,D] before execution, got "
            f"{tuple(predicted_actions.shape)}"
        )
    if predicted_actions.shape[0] <= 0 or predicted_actions.shape[1] <= 0:
        raise ValueError(
            "predicted_actions must have non-empty H and D dimensions, got "
            f"{tuple(predicted_actions.shape)}"
        )
    if predicted_actions.shape[0] == 1:
        return predicted_actions[0]
    if chunk_execution_contract == "receding_horizon_first":
        return predicted_actions[0]
    if chunk_execution_contract == "reject":
        raise ValueError(
            "Multi-step action chunks require an explicit execution policy; "
            "use chunk_execution_contract='receding_horizon_first' to execute only "
            "the first complete action before replanning."
        )
    raise ValueError(
        "chunk_execution_contract must be one of "
        f"{sorted(CHUNK_EXECUTION_CONTRACTS)}, got {chunk_execution_contract!r}"
    )


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        """
        Args:
            num_blocks: Number of residual blocks
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        action_chunk=1,
    ):
        """
        Args:
            input_dim: Hidden dimension from transformer
            hidden_dim: Hidden dimension for MLP
            action_dim: Action space dimension (e.g., 7 for robotics: 3D position + 3D rotation + gripper)
            action_chunk: Number of action steps to predict (for action chunking)
        """
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.model = MLPResNet(
            num_blocks=2, 
            input_dim=input_dim * action_dim * action_chunk, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim * action_chunk
        )

    def predict_action(self, actions_hidden_states):
        """
        Predict continuous actions from hidden states.
        
        Args:
            actions_hidden_states: Last hidden states of Transformer corresponding to action tokens
                                   Shape: (batch_size, action_chunk * action_dim, hidden_dim)
        
        Returns:
            Predicted actions of shape (batch_size, action_chunk, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        # Flatten the hidden states: (batch_size, action_chunk * action_dim * hidden_dim)
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, -1)
        # Predict actions: (batch_size, action_chunk * action_dim)
        action = self.model(rearranged_actions_hidden_states)
        # Reshape to: (batch_size, action_chunk, action_dim)
        action = action.reshape(batch_size, self.action_chunk, self.action_dim)
        return action
    
    def forward(self, actions_hidden_states):
        """
        Forward pass (alias for predict_action for compatibility).
        
        Args:
            actions_hidden_states: Shape (batch_size, action_chunk * action_dim, hidden_dim)
        
        Returns:
            Predicted actions of shape (batch_size, action_chunk, action_dim)
        """
        return self.predict_action(actions_hidden_states)


class PerStepSharedActionHead(nn.Module):
    """Apply one shared MLPResNet to each action-step query independently."""

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        action_chunk=1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.action_chunk = int(action_chunk)
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.action_chunk <= 0:
            raise ValueError(f"action_chunk must be positive, got {self.action_chunk}")
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_dim,
        )

    def predict_action(self, actions_hidden_states):
        if actions_hidden_states.ndim != 3:
            raise ValueError(
                "Per-step action hidden states must have shape [B,H,D], got "
                f"{tuple(actions_hidden_states.shape)}"
            )
        if actions_hidden_states.shape[1] != self.action_chunk:
            raise ValueError(
                "Per-step action query count mismatch: "
                f"got {actions_hidden_states.shape[1]}, expected {self.action_chunk}"
            )
        if actions_hidden_states.shape[2] != self.input_dim:
            raise ValueError(
                "Per-step action hidden width mismatch: "
                f"got {actions_hidden_states.shape[2]}, expected {self.input_dim}"
            )
        return self.model(actions_hidden_states)

    def forward(self, actions_hidden_states):
        return self.predict_action(actions_hidden_states)


class _MLPResNetTrunk(nn.Module):
    """The MLPResNet stem WITHOUT its output layer (``fc2``).

    Architecturally byte-identical to :class:`MLPResNet` up to and including
    ``layer_norm2``: ``layer_norm1 -> fc1 -> relu -> mlp_resnet_blocks ->
    layer_norm2``. It produces the ``hidden_dim`` feature that :class:`MLPResNet`
    would feed into ``fc2``. Submodule names match ``MLPResNet`` exactly (minus
    ``fc2``) so a per-step-shared head's ``fc1``/blocks/norms transfer 1:1 into
    ``trunk.*`` and its ``fc2`` rows split into the per-dim final heads.
    """

    def __init__(self, num_blocks, input_dim, hidden_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        return x


class DecoupledPerDimActionHead(nn.Module):
    """Per-dim action head with a shared trunk and a decoupled final layer.

    Token layout is the ``scalar_per_dim`` one: ``N_a = action_chunk *
    action_dim`` action-query tokens laid out chunk-major (token ``c*D + j`` is
    step ``c``, DoF ``j``). Every token is passed through one *shared* trunk MLP
    (``input_dim -> hidden_dim``, :class:`_MLPResNetTrunk`); then the ``j``-th
    DoF is read out by its OWN final linear (``hidden_dim -> 1``). The final head
    for DoF ``j`` is shared across all chunk steps (so an H16 head and an H1 head
    reuse the same seven final linears). Unlike the ``scalar_per_dim`` flatten
    head, tokens are never flattened together: each scalar output depends only on
    its own query token's hidden state (decoupled per-dim readout, YOLO-style).

    This structure makes the trunk = a per-step-shared head's MLPResNet stem and
    each final head = one row of that head's ``fc2``, so a per-step-shared
    pretrain head can be loaded by row-wise remapping (see
    ``pretrain_checkpoint.load_shared_action_head_export_rowwise_decoupled``).
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        action_chunk=1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.action_chunk = int(action_chunk)
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.action_chunk <= 0:
            raise ValueError(f"action_chunk must be positive, got {self.action_chunk}")
        self.trunk = _MLPResNetTrunk(
            num_blocks=2,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        )
        self.final_heads = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.action_dim)]
        )

    def predict_action(self, actions_hidden_states):
        if actions_hidden_states.ndim != 3:
            raise ValueError(
                "Decoupled per-dim action hidden states must have shape "
                f"[B,N_a,D], got {tuple(actions_hidden_states.shape)}"
            )
        batch_size, num_tokens, hidden_width = actions_hidden_states.shape
        expected_tokens = self.action_chunk * self.action_dim
        if num_tokens != expected_tokens:
            raise ValueError(
                "Decoupled per-dim action query count mismatch: "
                f"got {num_tokens}, expected {expected_tokens} "
                f"(action_chunk*action_dim = {self.action_chunk}*{self.action_dim})"
            )
        if hidden_width != self.input_dim:
            raise ValueError(
                "Decoupled per-dim action hidden width mismatch: "
                f"got {hidden_width}, expected {self.input_dim}"
            )
        # [B, N_a, D] -> [B, chunk, dim, D] (chunk-major token order).
        per_dim_hidden = actions_hidden_states.reshape(
            batch_size, self.action_chunk, self.action_dim, self.input_dim
        )
        # Shared trunk applied independently per (step, DoF) token.
        trunk_features = self.trunk(per_dim_hidden)  # [B, chunk, dim, hidden]
        # Decoupled final layer: DoF j read out by its own linear, shared across steps.
        per_dim_scalars = [
            self.final_heads[dim](trunk_features[:, :, dim, :])  # [B, chunk, 1]
            for dim in range(self.action_dim)
        ]
        action = torch.cat(per_dim_scalars, dim=-1)  # [B, chunk, dim]
        return action

    def forward(self, actions_hidden_states):
        return self.predict_action(actions_hidden_states)


def build_action_head(
    *,
    input_dim,
    hidden_dim,
    action_dim,
    action_chunk,
    action_query_contract,
):
    if action_query_contract == "scalar_per_dim_v1":
        return L1RegressionActionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_chunk=action_chunk,
        )
    if action_query_contract == "per_step_shared_v1":
        return PerStepSharedActionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_chunk=action_chunk,
        )
    if action_query_contract == "decoupled_per_dim_v1":
        return DecoupledPerDimActionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_chunk=action_chunk,
        )
    raise ValueError(
        "action_query_contract must be one of "
        f"{sorted(ACTION_QUERY_CONTRACTS)}, got {action_query_contract!r}"
    )
