import random
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from .modeling_moebert import MoEBertModel
from .modeling_moesm import MoEsmPreTrainedModel, MoEsmModel
from .losses import clip_loss, LoadBalancingLoss, MI_loss, SpecifiedExpertLoss, get_loss_fct 
from .outputs import SentenceSimilarityOutput


class MoEBertForSentenceSimilarity(BertPreTrainedModel):
    def __init__(self, config, bert=None):
        super().__init__(config)
        self.bert = MoEBertModel(config, add_pooling_layer=True) if bert is None else bert
        self.contrastive_loss = clip_loss
        self.temp = nn.Parameter(torch.tensor(0.7))
        self.aux_loss = LoadBalancingLoss(config)
        self.EX = config.expert_loss
        if self.EX:
            if config.MI:
                self.expert_loss = MI_loss(config)
            else:
                self.expert_loss = SpecifiedExpertLoss(config)
    
    def forward(self, a, b,
                att_a=None, att_b=None, r_labels=None, labels=None):
        if random.random() < 0.5:
            outputa = self.bert(input_ids=a, attention_mask=att_a)
            outputb = self.bert(input_ids=b, attention_mask=att_b)
        else:
            outputa = self.bert(input_ids=b, attention_mask=att_b)
            outputb = self.bert(input_ids=a, attention_mask=att_a)

        emba = outputa.pooler_output
        embb = outputb.pooler_output

        c_loss = self.contrastive_loss(emba, embb, self.temp)

        router_logits = tuple((a + b) / 2 for a, b in zip(outputa.router_logits, outputb.router_logits))
        r_loss = self.aux_loss(router_logits)
        if r_labels != None and self.EX:
            r_loss = r_loss + self.expert_loss(router_logits, r_labels)

        logits = (emba, embb)
        loss = c_loss + r_loss
        
        return SentenceSimilarityOutput(
            logits=logits,
            loss=loss
        )


class BertForSentenceSimilarity(BertPreTrainedModel):
    def __init__(self, config=None, bert=None):
        super().__init__(config)
        from transformers import BertModel
        self.bert = BertModel(config, add_pooling_layer=True) if bert is None else bert
        self.contrastive_loss = clip_loss
        self.temp = nn.Parameter(torch.tensor(0.7))

    def forward(self, a, b,
                att_a=None, att_b=None, r_labels=None, labels=None):
        if random.random() < 0.5:
            outputa = self.bert(input_ids=a, attention_mask=att_a)
            outputb = self.bert(input_ids=b, attention_mask=att_b)
        else:
            outputa = self.bert(input_ids=b, attention_mask=att_b)
            outputb = self.bert(input_ids=a, attention_mask=att_a)

        emba = outputa.pooler_output
        embb = outputb.pooler_output

        loss = self.contrastive_loss(emba, embb, self.temp)

        logits = (emba, embb)

        return SentenceSimilarityOutput(
            logits=logits,
            loss=loss
        )


class MoEsmForSentenceSimilarity(MoEsmPreTrainedModel):
    def __init__(self, config, esm=None):
        super().__init__(config)
        self.esm = MoEsmModel(config, add_pooling_layer=True) if esm is None else esm
        self.contrastive_loss = clip_loss
        self.temp = nn.Parameter(torch.tensor(0.7))
        self.aux_loss = LoadBalancingLoss(config)
        self.EX = config.expert_loss
        if self.EX:
            if config.MI:
                self.expert_loss = MI_loss(config)
            else:
                self.expert_loss = SpecifiedExpertLoss(config)
    
    def forward(self, a, b,
                att_a=None, att_b=None, r_labels=None, labels=None):
        if random.random() < 0.5:
            outputa = self.bert(input_ids=a, attention_mask=att_a)
            outputb = self.bert(input_ids=b, attention_mask=att_b)
        else:
            outputa = self.bert(input_ids=b, attention_mask=att_b)
            outputb = self.bert(input_ids=a, attention_mask=att_a)

        emba = outputa.pooler_output
        embb = outputb.pooler_output

        c_loss = self.contrastive_loss(emba, embb, self.temp)

        router_logits = tuple((a + b) / 2 for a, b in zip(outputa.router_logits, outputb.router_logits))
        r_loss = self.aux_loss(router_logits)
        if r_labels != None and self.EX:
            r_loss = r_loss + self.expert_loss(router_logits, r_labels)

        logits = (emba, embb)
        loss = c_loss + r_loss
        
        return SentenceSimilarityOutput(
            logits=logits,
            loss=loss
        )


class EsmForSentenceSimilarity(MoEsmPreTrainedModel):
    def __init__(self, config, esm=None):
        super().__init__(config)
        from transformers import EsmModel
        self.esm = EsmModel(config, add_pooling_layer=True) if esm is None else esm
        self.contrastive_loss = clip_loss
        self.temp = nn.Parameter(torch.tensor(0.7))

    def forward(self, a, b,
                att_a=None, att_b=None, r_labels=None, labels=None):
        if random.random() < 0.5:
            outputa = self.bert(input_ids=a, attention_mask=att_a)
            outputb = self.bert(input_ids=b, attention_mask=att_b)
        else:
            outputa = self.bert(input_ids=b, attention_mask=att_b)
            outputb = self.bert(input_ids=a, attention_mask=att_a)

        emba = outputa.pooler_output
        embb = outputb.pooler_output

        loss = self.contrastive_loss(emba, embb, self.temp)

        logits = (emba, embb)
        return SentenceSimilarityOutput(
            logits=logits,
            loss=loss
        )


class MoEsmForTripletSimilarity(MoEsmPreTrainedModel):
    def __init__(self, config, esm=None):
        super().__init__(config)
        self.esm = MoEsmModel(config, add_pooling_layer=True) if esm is None else esm
        self.contrastive_loss = nn.TripletMarginLoss()
        self.aux_loss = LoadBalancingLoss(config)
        self.EX = config.expert_loss
        if self.EX:
            if config.MI:
                self.expert_loss = MI_loss(config)
            else:
                self.expert_loss = SpecifiedExpertLoss(config)
    
    def embed(self, ids, att=None):
        return self.esm(input_ids=ids, attention_mask=att).pooler_output

    def forward(self, pos, anc, neg,
                att_p=None, att_a=None, att_n=None, r_labels=None):
        outp = self.esm(input_ids=pos, attention_mask=att_p)
        outa = self.esm(input_ids=anc, attention_mask=att_a)
        outn = self.esm(input_ids=neg, attention_mask=att_n)
        
        p = outp.pooler_output
        a = outa.pooler_output
        n = outn.pooler_output

        c_loss = self.contrastive_loss(p, a, n)

        router_logits = tuple((a + b + c) / 2 for a, b, c in
                              zip(outp.router_logits, outa.router_logits, outn.router_logits))

        r_loss = self.aux_loss(router_logits)
        if r_labels != None and self.EX:
            r_loss = r_loss + self.expert_loss(router_logits, r_labels)

        logits = (p, a, n)
        loss = c_loss + r_loss
        
        return SentenceSimilarityOutput(
            logits=logits,
            loss=loss
        )


class EsmForTripletSimilarity(MoEsmPreTrainedModel):
    def __init__(self, config, esm=None):
        super().__init__(config)
        from transformers import EsmModel
        self.esm = EsmModel(config, add_pooling_layer=True) if esm is None else esm
        self.contrastive_loss = nn.TripletMarginLoss()

    def embed(self, ids, att=None):
        return self.esm(input_ids=ids, attention_mask=att).pooler_output
    
    def forward(self, pos, anc, neg,
                att_p=None, att_a=None, att_n=None, r_labels=None):
        outp = self.esm(input_ids=pos, attention_mask=att_p)
        outa = self.esm(input_ids=anc, attention_mask=att_a)
        outn = self.esm(input_ids=neg, attention_mask=att_n)
        
        p = outp.pooler_output
        a = outa.pooler_output
        n = outn.pooler_output

        loss = self.contrastive_loss(p, a, n)
        logits = (p, a, n)
        
        return SentenceSimilarityOutput(
            logits=logits,
            loss=loss
        )


class LinearClassifier(nn.Module):
    def __init__(self, cfg, task_type='binary', num_labels=2):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.num_layers = cfg.num_layers
        self.input_layer = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            self.hidden_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.output_layer = nn.Linear(cfg.hidden_dim, num_labels)
        self.loss_fct = get_loss_fct(task_type)

    def forward(self, embeddings, labels=None):
        embeddings = self.gelu(self.input_layer(embeddings))
        for i in range(self.num_layers):
            embeddings = self.dropout(self.gelu(self.hidden_layers[i](embeddings)))
        logits = self.output_layer(embeddings)
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
