import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from models.losses import clip_loss, MNR_loss
from models.outputs import SentenceSimilarityOutput
from .modeling_modern_bert import (
    ModernBertModel,
    ModernBertConfig,
)


def mean_pooling(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
    return torch.sum(x * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = mean_pooling(hidden_states, attention_mask)
        x = self.layernorm(x)
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        return x


class MoEBertForSentenceSimilarity(PreTrainedModel):
    def __init__(self, config: ModernBertConfig, base_model: ModernBertModel):
        super().__init__(config)
        self.bert = base_model
        self.pooler = Pooler(config)
        self.loss_fct = clip_loss if config.loss_type == "clip" else MNR_loss

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def base_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        assignment: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        else:
            batch_size, seq_len = input_ids.shape[:2]
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        attention_mask, sliding_window_mask = self.bert._update_attention_mask(
            attention_mask, output_attentions=output_attentions
        )

        hidden_states = self.bert.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        for encoder_layer in self.bert.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                assignment=assignment,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.bert.final_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)        

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
    def forward(self, a_docs, b_docs, assignment: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> SentenceSimilarityOutput:
        input_ids_a, attention_mask_a = a_docs['input_ids'], a_docs['attention_mask']
        input_ids_b, attention_mask_b = b_docs['input_ids'], b_docs['attention_mask']

        state_a = self.base_forward(input_ids_a, attention_mask_a, assignment=assignment)
        state_b = self.base_forward(input_ids_b, attention_mask_b, assignment=assignment)

        emb_a = self.pooler(state_a.last_hidden_state, attention_mask_a)
        emb_b = self.pooler(state_b.last_hidden_state, attention_mask_b)
        loss = self.loss_fct(emb_a, emb_b)
        
        return SentenceSimilarityOutput(
            loss=loss,
            logits=torch.cat([emb_a, emb_b], dim=1),
        )

