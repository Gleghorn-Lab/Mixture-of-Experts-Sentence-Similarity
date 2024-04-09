import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceTokenTypeMoeBlock(nn.Module):
    def __init__(self, config, expert):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        # Initialize an empty tensor for the final hidden states
        final_hidden_states = torch.zeros_like(hidden_states).to(hidden_states.dtype)
        # Process each expert's input separately
        for expert_id in range(self.num_experts):
            # Mask to select the hidden states corresponding to the current expert
            expert_mask = token_type_ids == expert_id
            # Select the hidden states for the current expert
            expert_hidden_states = hidden_states[expert_mask]
            # Process the hidden states through the expert
            processed_states = self.experts[expert_id](expert_hidden_states).to(hidden_states.dtype)
            # Place the processed states back in the corresponding positions
            final_hidden_states[expert_mask] = processed_states

        return final_hidden_states  # (batch, sequence_length, hidden_dim)


class SentenceSwitchMoeBlock(nn.Module):
    def __init__(self, config, expert):
        """
        Sentence level MoE, single expert chosen
        """
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_output = self.gate(hidden_states) # (batch, sequence_length, n_experts)
        router_logits = router_output.mean(dim=1) # (batch, n_experts)
        router_choice = router_logits.argmax(dim=-1) # (batch)
        final_hidden_states = torch.stack([self.experts[router_choice[i]](hidden_states[i]) for i in range(len(hidden_states))])
        return final_hidden_states, router_logits # (batch, sequence_length, hidden_dim), (batch, num_experts)


class SentenceEnforcedSwitchMoeBlock(nn.Module): ### Test
    def __init__(self, config, expert):
        """
        Sentence level MoE, single expert chosen
        """
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor, router_labels: torch.tensor) -> torch.Tensor:
        # (batch, seq_len, hidden_size), (batch,) -> from 0 to num_experts-1
        sorted_indices = torch.argsort(router_labels) # sort in order of expert idx
        hidden_states = hidden_states[sorted_indices] # apply sort
        router_labels = router_labels[sorted_indices] # apply sort
        expert_idxs = torch.unique(router_labels) # find all experts needed
        bins = torch.bincount(router_labels)
        bins = bins[bins != 0]
        grouped_hidden_states = torch.split(hidden_states, tuple(bins)) # split sorted hidden_states
        expert_outputs = []
        for idx, group in zip(expert_idxs, grouped_hidden_states):
            expert_output = self.experts[idx](group) # sne batched groups to their experts
            expert_outputs.append(expert_output)

        concatenated_outputs = torch.cat(expert_outputs, dim=0)
        final_hidden_states = concatenated_outputs[torch.argsort(sorted_indices)] # put back to original order
        return final_hidden_states  # (batch, sequence_length, hidden_dim)


class SentenceTopKMoeBlock(nn.Module):
    def __init__(self, config, expert):
        """
        Sentence level MoE, topk expert aggregated
        """
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.topk = config.topk
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_output = self.gate(hidden_states)  # (batch, sequence_length, n_experts)
        router_logits = router_output.mean(dim=1)  # (batch, n_experts)
        router_weights = F.softmax(router_logits, dim=-1)  # (batch, n_experts)
        router_weights, selected_experts = torch.topk(router_weights, self.topk, dim=-1)  # (batch, topk)

        # Compute expert outputs
        expert_outputs = torch.stack([self.experts[i](hidden_states) for i in range(self.num_experts)], dim=1)  # (batch, num_experts, sequence_length, hidden_dim)

        # Select top k expert outputs
        selected_expert_outputs = expert_outputs[torch.arange(expert_outputs.size(0)).unsqueeze(1), selected_experts]  # (batch, topk, sequence_length, hidden_dim)

        # Compute weighted sum of selected expert outputs
        router_weights = router_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, topk, 1, 1)
        final_hidden_states = (router_weights * selected_expert_outputs).sum(dim=1)  # (batch, sequence_length, hidden_dim)

        return final_hidden_states, router_logits  # (batch, sequence_length, hidden_dim), (batch, num_experts)


class TokenTopKMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """
    def __init__(self, config, expert):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.topk

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        router_logits = router_logits.reshape(batch_size, sequence_length, -1).mean(dim=1)
        return final_hidden_states, router_logits  # (batch, sequence_length, hidden_dim), (batch, num_experts)