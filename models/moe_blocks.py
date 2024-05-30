import torch
import torch.nn as nn
import torch.nn.functional as F


class NoMoeBlock(nn.Module):
    def __init__(self, config, expert):
        """
        Stand in for no moe, but extended MLP
        """
        super().__init__()
        self.experts = nn.ModuleList([expert(config) for _ in range(1)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.experts[0](hidden_states)


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
    Adapted from Huggingface Mixtral
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


class MHRouter(nn.Module):
    def __init__(self, num_experts, hidden_dim, num_heads):
        super().__init__()
        self.expert_embedding = nn.Parameter(torch.randn(hidden_dim // num_heads, num_experts)) # (h, e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : hidden_states (B * L * n, h)
        return torch.matmul(x, self.expert_embedding) # (B * L * n, e)


class MultiHeadMoeBlock(nn.Module):
    def __init__(self, config, expert):
        super().__init__()
        self.hidden_dim = config.hidden_size # d
        self.num_experts = config.num_experts # e
        self.num_heads = config.num_heads # n
        self.topk = config.topk # k
        self.head_dim = self.hidden_dim // self.num_heads # h
        self.rounded_dim = (self.hidden_dim // self.num_heads) * self.num_heads # r

        self.multi_head_layer = nn.Linear(self.hidden_dim, self.rounded_dim)
        self.router = MHRouter(self.num_experts, self.hidden_dim, self.num_heads)

        config.hidden_size = self.head_dim
        config.intermediate_size = config.intermediate_size // self.num_heads
        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])
        self.merge_layer = nn.Linear(self.rounded_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : hidden_states (B, L, d)
        bs, L, d = x.size()
        # If hidden_dim is not divisible by num_heads r != d
        x = self.multi_head_layer(x) # (B, L, r)
        x = x.reshape(bs * L * self.num_heads, self.head_dim).contiguous() # (B * L * n, h)
        ### Router
        router_logits = self.router(x) # (B * L * n, e)
        router_weights = router_logits.softmax(dim=-1) # (B * L * n, e)
        router_weights, selected_experts = torch.topk(router_weights, self.topk, dim=-1) # (B * L * n, k), (B * L * n, k)
        # Call experts densely, faster than selective loops
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # (B * L * n, e, h)
        # Select top-k expert outputs
        selected_expert_outputs = expert_outputs[torch.arange(expert_outputs.size(0)).unsqueeze(1), selected_experts] # (B * L * n, k, h)
        # Multiply selected expert outputs with router weights elementwise
        weighted_expert_outputs = selected_expert_outputs * router_weights.unsqueeze(-1) # (B * L * n, k, h)
        # Combine topk expert outputs
        x = weighted_expert_outputs.sum(dim=1) # (B * L * n, h)
        # Back to original shape
        x = x.reshape(bs, L, self.rounded_dim) # (B, L, r)
        x = self.merge_layer(x) # (B, L, d)
        return x
