import torch
import torch.nn as nn
import numpy as np

from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .model_protein_moe import trans_basic_block, trans_basic_block_Config


class ProteinVec(PreTrainedModel):
    def __init__(self, t5, moe_path):
        vec_model_cpnt = moe_path + '/protein_vec.ckpt'
        vec_model_config = moe_path + '/protein_vec_params.json'
        vec_model_config = trans_basic_block_Config.from_json(vec_model_config)
        super().__init__(vec_model_config)

        self.t5 = t5.eval()
        self.moe = trans_basic_block.load_from_checkpoint(vec_model_cpnt, config=vec_model_config).eval()

        self.contrastive_loss = nn.TripletMarginLoss()
        self.aspect_to_keys_dict = {
            0: ['ENZYME'], # EC
            1: ['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'], # Cofactor
            2: ['MFO'], # MF
            3: ['BPO'], # BP
            4: ['CCO'], # CC
            5: ['PFAM'], # IP
            6: ['GENE3D'] # 3D 
        }
        self.all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])

    def get_mask(self, aspect):
        sampled_keys = self.aspect_to_keys_dict[aspect]
        masks = [self.all_cols[k] in sampled_keys for k in range(len(self.all_cols))]
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None,:]
        return masks
        
    def embed_batch(self, input_ids, attention_mask, aspect):
        ### t5
        masks = self.get_mask(aspect)
        with torch.no_grad():
            embedding = self.t5(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        vecs = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emb = embedding[seq_num][:seq_len-1].unsqueeze(0)
            ### moe
            padding = torch.zeros(seq_emb.shape[0:2]).type(torch.BoolTensor).to(seq_emb)
            out_seq = self.moe.make_matrix(seq_emb, padding)
            vec = self.moe(out_seq, masks)
            vecs.append(vec)
        return torch.cat(vecs, dim=0)

    def forward(self, pos, anc, neg,
                att_p=None, att_a=None, att_n=None, r_labels=None):
        p = self.embed_batch(input_ids=pos, attention_mask=att_p, aspect=r_labels[0].item())
        a = self.embed_batch(input_ids=anc, attention_mask=att_a, aspect=r_labels[0].item())
        n = self.embed_batch(input_ids=neg, attention_mask=att_n, aspect=r_labels[0].item())

        loss = self.contrastive_loss(p, a, n)
        logits = (p, a, n)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss
        )
