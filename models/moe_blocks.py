import torch
import torch.nn as nn
import copy
from typing import Optional
from transformers.activations import ACT2FN
from .lora import LoRALinear


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if getattr(config, "lora", False):
            self.Wi = LoRALinear(
                config.hidden_size,
                int(config.intermediate_size) * 2,
                bias=config.mlp_bias,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
        else:
            self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        if getattr(config, "lora", False):
            self.Wo = LoRALinear(
                config.intermediate_size,
                config.hidden_size,
                bias=config.mlp_bias,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
        else:
            self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor, assignment: Optional[torch.Tensor] = None) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class SentenceEnforcedSwitchMoeBlock(nn.Module):
    def __init__(self, config, expert, pretrained=True):
        """
        Sentence-level MoE block.
        Each example in the batch is routed to a single expert based on `assignment`.
        """
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        if not pretrained:
            self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_size)
            assignment: Tensor of shape (batch,) with expert indices in 0...num_experts-1
        
        Returns:
            Tensor of shape (batch, seq_len, hidden_size) with expert outputs.
        """
        # Prepare an output tensor (will have the same ordering as the input)
        outputs = hidden_states.new_empty(hidden_states.size())

        # Loop over each expert and process the examples assigned to it.
        for expert_idx in range(self.num_experts):
            # Create a mask for examples assigned to the current expert.
            mask = (assignment == expert_idx)
            if mask.any():
                # Select the hidden states corresponding to this expert.
                expert_input = hidden_states[mask]
                # Pass through the expert.
                expert_output = self.experts[expert_idx](expert_input)
                # Place the outputs back in the correct positions.
                outputs[mask] = expert_output
        return outputs
