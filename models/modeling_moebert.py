import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertAttention, BertPreTrainedModel, BertEmbeddings

from models.outputs import *
from models.moe_blocks import *
from models.losses import *


class BertExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_up = nn.Linear(config.hidden_size, config.intermediate_size) # BertIntermediate dense
        self.intermediate_down = nn.Linear(config.intermediate_size, config.hidden_size) # BertOutput dense
        self.new_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.GELU()
    
    def forward(self, hidden_states):
        hidden_states = self.act(self.intermediate_up(hidden_states)) * self.new_linear(hidden_states)
        hidden_states = self.dropout(self.intermediate_down(hidden_states))
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        if config.token_moe:
            self.moe_block = TokenTopKMoeBlock(config, expert=BertExpert)
        else:
            if config.moe_type == 'switch': self.moe_block = SentenceSwitchMoeBlock(config, expert=BertExpert)
            elif config.moe_type == 'topk': self.moe_block = SentenceTopKMoeBlock(config, expert=BertExpert)
            elif config.moe_type == 'tokentype': self.moe_block = SentenceTokenTypeMoeBlock(config, expert=BertExpert)
            else: print(f'Incorrect MOE type {config.moe_type}, try again')
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        token_type_ids=None,
        router_labels=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output, router_logits = self.feed_forward_chunk(attention_output, router_labels, token_type_ids)

        outputs = (layer_output,) + outputs
        
        outputs = outputs + (router_logits,) # router logits last
        return outputs

    def feed_forward_chunk(self, attention_output, router_labels=None, token_type_ids=None): # calls moe_blocks
        router_logits = None
        attention_output_ln = self.LayerNorm(attention_output)
        if token_type_ids is not None and router_labels is None:
            layer_output = self.moe_block(attention_output_ln, token_type_ids=token_type_ids)
        elif router_labels is not None:
            layer_output = self.moe_block(attention_output_ln, router_labels=router_labels)
        else:
            layer_output = self.moe_block(attention_output_ln)
        if isinstance(layer_output, tuple):
            layer_output, router_logits = layer_output
        layer_output = layer_output + attention_output
        return layer_output, router_logits


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        token_type_ids=None,
        router_labels=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_router_logits = ()

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    token_type_ids,
                    router_labels,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    token_type_ids,
                    router_labels,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            router_logits = layer_outputs[-1]
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-2],) # change to -2 because router last
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            all_router_logits = all_router_logits + (router_logits,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return MoEBertOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_logits=all_router_logits
        )


class MoEBertModel(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        router_labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            router_labels=router_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return MoEBertOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            router_logits=encoder_outputs.router_logits
        )


class MoEBertForSentenceSimilarity(nn.Module):
    def __init__(self, config, bert=None):
        super().__init__()
        self.bert = MoEBertModel(config, add_pooling_layer=True) if bert is None else bert
        self.contrastive_loss = clip_loss
        self.temp = nn.Parameter(torch.tensor(0.7))
        self.aux_loss = LoadBalancingLoss(config)
        self.MI = config.MI_loss
        if self.MI:
            self.MI_loss = MILoss(config)
    
    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, r_labels=None, labels=None):
        if random.random() < 0.5:
            outputa = self.bert(input_ids=input_ids_a, attention_mask=attention_mask_a)
            outputb = self.bert(input_ids=input_ids_b, attention_mask=attention_mask_b)
        else:
            outputa = self.bert(input_ids=input_ids_b, attention_mask=attention_mask_b)
            outputb = self.bert(input_ids=input_ids_a, attention_mask=attention_mask_a)

        emba = outputa.pooler_output
        embb = outputb.pooler_output

        c_loss = self.contrastive_loss(emba, embb, self.temp)

        router_logits = tuple((a + b) / 2 for a, b in zip(outputa.router_logits, outputb.router_logits))
        r_loss = self.aux_loss(router_logits)
        if r_labels != None and self.MI:
            r_loss = r_loss + self.MI_loss(router_logits, r_labels)

        logits = (emba, embb)
        loss = c_loss + r_loss
        
        return SentenceSimilarityOutput(
            logits=logits,
            loss=loss
        )
    

class BertForSentenceSimilarity(nn.Module):
    def __init__(self, config=None, bert=None):
        super().__init__()
        self.bert = MoEBertModel(config, add_pooling_layer=True) if bert is None else bert
        self.contrastive_loss = clip_loss
        self.temp = nn.Parameter(torch.tensor(0.7))

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, r_labels=None, labels=None):
        if random.random() < 0.5:
            outputa = self.bert(input_ids=input_ids_a, attention_mask=attention_mask_a)
            outputb = self.bert(input_ids=input_ids_b, attention_mask=attention_mask_b)
        else:
            outputa = self.bert(input_ids=input_ids_b, attention_mask=attention_mask_b)
            outputb = self.bert(input_ids=input_ids_a, attention_mask=attention_mask_a)

        emba = outputa.pooler_output
        embb = outputb.pooler_output

        loss = self.contrastive_loss(emba, embb, self.temp)

        logits = (emba, embb)
        return SentenceSimilarityOutput(
            logits=logits,
            loss=loss
        )