import random
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from models.modeling_moebert import MoEBertModel
from models.losses import clip_loss, LoadBalancingLoss, MI_loss, SpecifiedExpertLoss 
from models.outputs import SentenceSimilarityOutput


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

