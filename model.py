import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from transformers.models.bert.modeling_bert import BertAttention, apply_chunking_to_forward, BertPreTrainedModel, BertEmbeddings
from transformers.models.bert.modeling_bert import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions


@dataclass
class MoEBertOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    """
    Bert past and cross output with router logits added

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.use_router_loss=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
            modules.
    """
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoEBertOutputWithPoolingAndCrossAttentions(BaseModelOutputWithPoolingAndCrossAttentions):
    """
    Bert pooling and cross output with router logits added
    
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.

        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.use_router_loss=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
            modules.
    """
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


class MiniMoE(nn.Module):
    def __init__(self, model, specific=True, balance=False, c_scale=1.0, r_scale=0.1):
        super().__init__()
        from losses import specified_expert_loss, load_balancing_loss, MNR_loss
        self.bert = model
        self.specific = specific
        self.balance = balance
        self.router_loss = specified_expert_loss if specific else load_balancing_loss
        self.contrastive_loss = MNR_loss
        self.c_scale = c_scale
        self.r_scale = r_scale
    
    def forward(self, batch1, batch2, labels=None):
        outputa = self.bert(**batch1)
        outputb = self.bert(**batch2)

        emba = outputa.pooler_output
        embb = outputb.pooler_output

        c_loss = self.contrastive_loss(emba, embb, scale=self.c_scale)

        router_logits = tuple((a + b) / 2 for a, b in zip(outputa.router_logits, outputb.router_logits))
        if self.specific:
            r_loss = self.router_loss(router_logits, labels) * self.r_scale if labels is not None else 0
        elif self.balance:
            r_loss = self.router_loss(router_logits) * self.r_scale
        return emba, embb, router_logits, c_loss, r_loss


class MiniMoELoadWeights:
    def __init__(self, base_model, tokenizer, domains):
        self.bert_base = base_model # base bert model
        self.tokenizer = tokenizer # bert tokenizer
        self.domains = domains # list of special tokens to take place of CLS
        self.config = self.bert_base.config
        self.model_type = 'Model' # for weight assignment logic, refers to BertModel instead of BertForSequenceClassification, for example
        self.config.num_experts = len(domains) # each domain gets a specific set of experts

    def get_seeded_model(self):
        start_time = time.time()
        model = BertMoEModel(config=self.config)
        model = self.match_weights(model)

        with torch.no_grad():
            model.resize_token_embeddings(len(self.tokenizer) + len(self.domains))
            # Add new tokens to the tokenizer
            added_tokens = {'additional_special_tokens' : self.domains}
            self.tokenizer.add_special_tokens(added_tokens)
            # Seed the embedding with the [CLS] token embedding
            cls_token_embedding = model.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id, :].clone()
            for token in self.domains:
                model.embeddings.word_embeddings.weight[self.tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()

        end_time = time.time()
        print('Model loaded in ', round((end_time - start_time) / 60, 2), 'minutes')
        total, effective, mem = self.count_parameters(model)
        print(f'{total} million total parameters')
        print(f'{effective} million effective parameters')
        print(f'Approximately {mem} GB of memory in fp32\n')
        return model, self.tokenizer

    def check_for_match(self, model): # Test for matching parameters
        all_weights_match = True
        for name, param in self.bert_base.named_parameters(): # for shared parameters
            if name in model.state_dict():
                pre_trained_weight = param.data
                moe_weight = model.state_dict()[name].data
                if not torch.equal(pre_trained_weight, moe_weight):
                    all_weights_match = False
                    break
    
        for i in range(self.config.num_hidden_layers): # for experts
            for j in range(self.config.num_experts):
                moe_encoder_layer = model.bert.encoder.layer[i] if self.model_type != 'Model' else model.encoder.layer[i]
                bert_encoder_layer = self.bert_base.bert.encoder.layer[i] if self.model_type != 'Model' else self.bert_base.encoder.layer[i] 
                if not torch.equal(moe_encoder_layer.moe_block.experts[j].intermediate_up.weight,
                                bert_encoder_layer.intermediate.dense.weight):
                    all_weights_match = False
                if not torch.equal(moe_encoder_layer.moe_block.experts[j].intermediate_down.weight,
                                bert_encoder_layer.output.dense.weight):
                    all_weights_match = False

        if all_weights_match:
            print('All weights match')
        else:
            print('Some weights differ')

    def match_weights(self, model): # Seeds MoBert experts with linear layers of bert
        self.check_for_match(model)
        for name1, param1 in self.bert_base.named_parameters():
            for name2, param2 in model.named_parameters():
                if name1 == name2:
                    model.state_dict()[name2].data.copy_(param1.data)

        for i in range(self.config.num_hidden_layers):
            for j in range(self.config.num_experts):
                moe_encoder_layer = model.bert.encoder.layer[i] if self.model_type != 'Model' else model.encoder.layer[i]
                bert_encoder_layer = self.bert_base.bert.encoder.layer[i] if self.model_type != 'Model' else self.bert_base.encoder.layer[i] 
                moe_encoder_layer.moe_block.experts[j].intermediate_up = copy.deepcopy(bert_encoder_layer.intermediate.dense)
                moe_encoder_layer.moe_block.experts[j].intermediate_down = copy.deepcopy(bert_encoder_layer.output.dense)
        self.check_for_match(model)
        return model

    def count_parameters_in_layer(self, layer):
        """Counts parameters in a regular layer."""
        return sum(p.numel() for p in layer.parameters())

    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        non_effective_params = 0
        j = 0 # only one called at a time
        for i in range(self.config.num_hidden_layers):
            moe_encoder_layer = model.encoder.layer[i] if self.model_type == 'Model' else model.bert.encoder.layer[i]
            non_effective_params += self.count_parameters_in_layer(moe_encoder_layer.moe_block.experts[j].intermediate_up)
            non_effective_params += self.count_parameters_in_layer(moe_encoder_layer.moe_block.experts[j].intermediate_down)
        effective_params = total_params - non_effective_params
        memory_bytes = total_params * 4  # 4 bytes for 32-bit floats
        memory_gig = round(memory_bytes / (1024 ** 3), 2)
        return round(total_params / 1e6, 1), round(effective_params / 1e6, 1), memory_gig


class BertExpert(nn.Module):
    """
    Combined Esm intermediate and output linear layers for MOE
    """
    def __init__(self, config):
        super().__init__()
        self.intermediate_up = nn.Linear(config.hidden_size, config.intermediate_size) # BertIntermediate dense
        self.intermediate_down = nn.Linear(config.intermediate_size, config.hidden_size) # BertOutput dense
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.intermediate_up(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.intermediate_down(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([BertExpert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_output = self.gate(hidden_states) # (batch, sequence_length, n_experts)
        router_logits = router_output.mean(dim=1) # (batch, n_experts)
        router_choice = F.softmax(router_logits, dim=-1).argmax(dim=-1) # (batch)
        final_hidden_states = torch.stack([self.experts[router_choice[i]](hidden_states[i]) for i in range(len(hidden_states))])
        return final_hidden_states, router_logits # (batch, sequence_length, hidden_dim), (batch, num_experts)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
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
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.moe_block = BertMoeBlock(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
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

        layer_output, router_logits = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        return outputs, router_logits

    def feed_forward_chunk(self, attention_output): # calls moe_blocks
        attention_output_ln = self.LayerNorm(attention_output)
        layer_output, router_logits = self.moe_block(attention_output_ln)
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
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

            layer_outputs, router_logits = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            all_router_logits = all_router_logits + (router_logits,)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

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
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            router_logits=all_router_logits
        )


class BertMoEModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

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
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

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

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return MoEBertOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            router_logits=encoder_outputs.router_logits
        )
