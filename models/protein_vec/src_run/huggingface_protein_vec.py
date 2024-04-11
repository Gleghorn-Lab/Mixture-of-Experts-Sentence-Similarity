import torch
import torch.nn as nn
import numpy as np
import os
import inspect
import pytorch_lightning as pl
from dataclasses import asdict
from transformers import PreTrainedModel, T5EncoderModel, PretrainedConfig, T5Config
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm.auto import tqdm

try:
    from .model_protein_moe import trans_basic_block, trans_basic_block_Config
    from .model_protein_vec_single_variable import trans_basic_block_single, trans_basic_block_Config_single
    from .embed_structure_model import trans_basic_block_tmvec, trans_basic_block_Config_tmvec
except:
    from model_protein_moe import trans_basic_block, trans_basic_block_Config
    from model_protein_vec_single_variable import trans_basic_block_single, trans_basic_block_Config_single
    from embed_structure_model import trans_basic_block_tmvec, trans_basic_block_Config_tmvec

class ProteinVecConfig(PretrainedConfig):
    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model",
                     "num_attention_heads": "num_heads",
                     "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        ### T5
        vocab_size=128,
        d_model=1024,
        d_kv=128,
        d_ff=16384,
        num_layers=24,
        num_decoder_layers=None,
        num_heads=32,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=None,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        classifier_dropout=0.0,

        ### Aspect Vecs
        ec_d_model=1024,
        ec_nhead=4,
        ec_num_layers=2,
        ec_dim_feedforward=2048,
        ec_out_dim=512,
        ec_dropout=0.1,
        ec_activation="relu",
        ec_num_variables=10,
        ec_vocab=20,
        ec_lr0=0.0001,
        ec_warmup_steps=500,
        ec_p_bernoulli=0.5,

        gene3d_d_model=1024,
        gene3d_nhead=4,
        gene3d_num_layers=2,
        gene3d_dim_feedforward=2048,
        gene3d_out_dim=512,
        gene3d_dropout=0.1,
        gene3d_activation="relu",
        gene3d_num_variables=10,
        gene3d_vocab=20,
        gene3d_lr0=0.0001,
        gene3d_warmup_steps=500,
        gene3d_p_bernoulli=0.5,

        bp_d_model=1024,
        bp_nhead=4,
        bp_num_layers=4,
        bp_dim_feedforward=2048,
        bp_out_dim=512,
        bp_dropout=0.1,
        bp_activation="relu",
        bp_num_variables=10,
        bp_vocab=20,
        bp_lr0=0.0001,
        bp_warmup_steps=500,
        bp_p_bernoulli=0.5,

        cc_d_model=1024,
        cc_nhead=4,
        cc_num_layers=4,
        cc_dim_feedforward=2048,
        cc_out_dim=512,
        cc_dropout=0.1,
        cc_activation="relu",
        cc_num_variables=10,
        cc_vocab=20,
        cc_lr0=0.0001,
        cc_warmup_steps=500,
        cc_p_bernoulli=0.5,

        mf_d_model=1024,
        mf_nhead=4,
        mf_num_layers=4,
        mf_dim_feedforward=2048,
        mf_out_dim=512,
        mf_dropout=0.1,
        mf_activation="relu",
        mf_num_variables=10,
        mf_vocab=20,
        mf_lr0=0.0001,
        mf_warmup_steps=500,
        mf_p_bernoulli=0.5,

        pfam_d_model=1024,
        pfam_nhead=4,
        pfam_num_layers=2,
        pfam_dim_feedforward=2048,
        pfam_out_dim=512,
        pfam_dropout=0.1,
        pfam_activation="relu",
        pfam_num_variables=10,
        pfam_vocab=20,
        pfam_lr0=0.0001,
        pfam_warmup_steps=500,
        pfam_p_bernoulli=0.5,

        tm_d_model=1024,
        tm_nhead=4,
        tm_num_layers=4,
        tm_dim_feedforward=2048,
        tm_out_dim=512,
        tm_dropout=0.1,
        tm_activation="relu",
        tm_lr0=0.0001,
        tm_warmup_steps=300,

        vec_d_model=512,
        vec_nhead=4,
        vec_num_layers=2,
        vec_dim_feedforward=2048,
        vec_out_dim=512,
        vec_dropout=0.1,
        vec_activation="relu",
        vec_num_variables=10,
        vec_vocab=20,
        vec_lr0=0.0001,
        vec_warmup_steps=500,
        vec_p_bernoulli=0.5,

        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        # EC parameters
        self.ec_d_model = ec_d_model
        self.ec_nhead = ec_nhead
        self.ec_num_layers = ec_num_layers
        self.ec_dim_feedforward = ec_dim_feedforward
        self.ec_out_dim = ec_out_dim
        self.ec_dropout = ec_dropout
        self.ec_activation = ec_activation
        self.ec_num_variables = ec_num_variables
        self.ec_vocab = ec_vocab
        self.ec_lr0 = ec_lr0
        self.ec_warmup_steps = ec_warmup_steps
        self.ec_p_bernoulli = ec_p_bernoulli

        # GENE3D parameters
        self.gene3d_d_model = gene3d_d_model
        self.gene3d_nhead = gene3d_nhead
        self.gene3d_num_layers = gene3d_num_layers
        self.gene3d_dim_feedforward = gene3d_dim_feedforward
        self.gene3d_out_dim = gene3d_out_dim
        self.gene3d_dropout = gene3d_dropout
        self.gene3d_activation = gene3d_activation
        self.gene3d_num_variables = gene3d_num_variables
        self.gene3d_vocab = gene3d_vocab
        self.gene3d_lr0 = gene3d_lr0
        self.gene3d_warmup_steps = gene3d_warmup_steps
        self.gene3d_p_bernoulli = gene3d_p_bernoulli

        # BP parameters
        self.bp_d_model = bp_d_model
        self.bp_nhead = bp_nhead
        self.bp_num_layers = bp_num_layers
        self.bp_dim_feedforward = bp_dim_feedforward
        self.bp_out_dim = bp_out_dim
        self.bp_dropout = bp_dropout
        self.bp_activation = bp_activation
        self.bp_num_variables = bp_num_variables
        self.bp_vocab = bp_vocab
        self.bp_lr0 = bp_lr0
        self.bp_warmup_steps = bp_warmup_steps
        self.bp_p_bernoulli = bp_p_bernoulli

        # CC parameters
        self.cc_d_model = cc_d_model
        self.cc_nhead = cc_nhead
        self.cc_num_layers = cc_num_layers
        self.cc_dim_feedforward = cc_dim_feedforward
        self.cc_out_dim = cc_out_dim
        self.cc_dropout = cc_dropout
        self.cc_activation = cc_activation
        self.cc_num_variables = cc_num_variables
        self.cc_vocab = cc_vocab
        self.cc_lr0 = cc_lr0
        self.cc_warmup_steps = cc_warmup_steps
        self.cc_p_bernoulli = cc_p_bernoulli

        # MF parameters
        self.mf_d_model = mf_d_model
        self.mf_nhead = mf_nhead
        self.mf_num_layers = mf_num_layers
        self.mf_dim_feedforward = mf_dim_feedforward
        self.mf_out_dim = mf_out_dim
        self.mf_dropout = mf_dropout
        self.mf_activation = mf_activation
        self.mf_num_variables = mf_num_variables
        self.mf_vocab = mf_vocab
        self.mf_lr0 = mf_lr0
        self.mf_warmup_steps = mf_warmup_steps
        self.mf_p_bernoulli = mf_p_bernoulli

        # PFAM parameters
        self.pfam_d_model = pfam_d_model
        self.pfam_nhead = pfam_nhead
        self.pfam_num_layers = pfam_num_layers
        self.pfam_dim_feedforward = pfam_dim_feedforward
        self.pfam_out_dim = pfam_out_dim
        self.pfam_dropout = pfam_dropout
        self.pfam_activation = pfam_activation
        self.pfam_num_variables = pfam_num_variables
        self.pfam_vocab = pfam_vocab
        self.pfam_lr0 = pfam_lr0
        self.pfam_warmup_steps = pfam_warmup_steps
        self.pfam_p_bernoulli = pfam_p_bernoulli

        # Vec parameters
        self.vec_d_model = vec_d_model
        self.vec_nhead = vec_nhead
        self.vec_num_layers = vec_num_layers
        self.vec_dim_feedforward = vec_dim_feedforward
        self.vec_out_dim = vec_out_dim
        self.vec_dropout = vec_dropout
        self.vec_activation = vec_activation
        self.vec_num_variables = vec_num_variables
        self.vec_vocab = vec_vocab
        self.vec_lr0 = vec_lr0
        self.vec_warmup_steps = vec_warmup_steps
        self.vec_p_bernoulli = vec_p_bernoulli

        # TM parameters
        self.tm_d_model = tm_d_model
        self.tm_nhead = tm_nhead
        self.tm_num_layers = tm_num_layers
        self.tm_dim_feedforward = tm_dim_feedforward
        self.tm_out_dim = tm_out_dim
        self.tm_dropout = tm_dropout
        self.tm_activation = tm_activation
        self.tm_lr0 = tm_lr0
        self.tm_warmup_steps = tm_warmup_steps

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class BasicConfig(trans_basic_block_Config):
    @classmethod
    def from_huggingface(cls, prefix, proteinvec_config):
        current_attributes = cls.__dataclass_fields__.keys()
        filtered_attributes = {k[len(prefix)+1:]: v for k, v in proteinvec_config.__dict__.items() if k.startswith(prefix)}
        config_dict = {k: v for k, v in filtered_attributes.items() if k in current_attributes}
        config = cls(**config_dict)
        return config


class TmConfig(trans_basic_block_Config_tmvec):
    @classmethod
    def from_huggingface(cls, prefix, proteinvec_config):
        current_attributes = cls.__dataclass_fields__.keys()
        filtered_attributes = {k[len(prefix)+1:]: v for k, v in proteinvec_config.__dict__.items() if k.startswith(prefix)}
        config_dict = {k: v for k, v in filtered_attributes.items() if k in current_attributes}
        config = cls(**config_dict)
        return config
    

class SingleConfig(trans_basic_block_Config_single):
    @classmethod
    def from_huggingface(cls, prefix, proteinvec_config):
        current_attributes = cls.__dataclass_fields__.keys()
        filtered_attributes = {k[len(prefix)+1:]: v for k, v in proteinvec_config.__dict__.items() if k.startswith(prefix)}
        config_dict = {k: v for k, v in filtered_attributes.items() if k in current_attributes}
        config = cls(**config_dict)
        return config


class HF_trans_basic_block(trans_basic_block):
    def __init__(self, config: ProteinVecConfig):
        pl.LightningModule.__init__(self)
        self.config = config

        encoder_config = BasicConfig.from_huggingface(prefix='vec', proteinvec_config=config)
        encoder_args = {k: v for k, v in asdict(encoder_config).items() if k in inspect.signature(nn.TransformerEncoderLayer).parameters}    
        self.dropout = nn.Dropout(encoder_config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_config.num_layers)
        self.mlp_1 = nn.Linear(encoder_config.d_model, encoder_config.out_dim)
        self.mlp_2 = nn.Linear(encoder_config.out_dim, encoder_config.out_dim)
        
        self.trip_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.pdist = nn.PairwiseDistance(p=2)

        self.model_aspect_tmvec = trans_basic_block_tmvec(TmConfig.from_huggingface('tm', config))
        self.model_aspect_pfam = trans_basic_block_single(SingleConfig.from_huggingface('pfam', config))
        self.model_aspect_gene3D = trans_basic_block_single(SingleConfig.from_huggingface('gene3d', config))
        self.model_aspect_ec = trans_basic_block_single(SingleConfig.from_huggingface('ec', config))
        self.model_aspect_mfo = trans_basic_block_single(SingleConfig.from_huggingface('mf', config))
        self.model_aspect_bpo = trans_basic_block_single(SingleConfig.from_huggingface('bp', config))
        self.model_aspect_cco = trans_basic_block_single(SingleConfig.from_huggingface('cc', config))


class ProteinVec(PreTrainedModel):
    def __init__(self, config: ProteinVecConfig):
        super().__init__(config)
        self.config = config

        self.t5 = T5EncoderModel(config=T5Config.from_pretrained('lhallee/prot_t5_enc'))
        self.moe = HF_trans_basic_block(config)

        self.contrastive_loss = nn.TripletMarginLoss()
        self.aspect_to_keys_dict = {
            '[EC]': ['ENZYME'],
            '[MF]': ['MFO'],
            '[BP]': ['BPO'],
            '[CC]': ['CCO'],
            '[IP]': ['PFAM'],
            '[3D]': ['GENE3D'],
            'ALL': ['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'] 
        }
        self.all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])

    def to_eval(self):
        self.t5 = self.t5.eval()
        self.moe = self.moe.eval()

    def load_from_disk(self,
                       aspect_path='models/protein_vec/src_run/protein_vec_models',
                       t5_path='lhallee/prot_t5_enc'):
        state_dict = torch.load(os.path.join(aspect_path, 'protein_vec.ckpt'))['state_dict']
        self.moe.load_state_dict(state_dict)

        self.moe.model_aspect_tmvec = trans_basic_block_tmvec.load_from_checkpoint(
            os.path.join(aspect_path, 'tm_vec_swiss_model_large.ckpt'),
            config=TmConfig.from_huggingface('tm', self.config)
        )
        self.moe.model_aspect_pfam = trans_basic_block_single.load_from_checkpoint(
            os.path.join(aspect_path, 'aspect_vec_pfam.ckpt'),
            config=SingleConfig.from_huggingface('pfam', self.config)
        )
        self.moe.model_aspect_gene3D = trans_basic_block_single.load_from_checkpoint(
            os.path.join(aspect_path, 'aspect_vec_gene3d.ckpt'),
            config=SingleConfig.from_huggingface('gene3d', self.config)
        )
        self.moe.model_aspect_ec = trans_basic_block_single.load_from_checkpoint(
            os.path.join(aspect_path, 'aspect_vec_ec.ckpt'),
            config=SingleConfig.from_huggingface('ec', self.config)
        )
        self.moe.model_aspect_mfo = trans_basic_block_single.load_from_checkpoint(
            os.path.join(aspect_path, 'aspect_vec_go_mfo.ckpt'),
            config=SingleConfig.from_huggingface('mf', self.config)
        )
        self.moe.model_aspect_bpo = trans_basic_block_single.load_from_checkpoint(
            os.path.join(aspect_path, 'aspect_vec_go_bpo.ckpt'),
            config=SingleConfig.from_huggingface('bp', self.config)
        )
        self.moe.model_aspect_cco = trans_basic_block_single.load_from_checkpoint(
            os.path.join(aspect_path, 'aspect_vec_go_cco.ckpt'),
            config=SingleConfig.from_huggingface('cc', self.config)
        )

        self.t5 = T5EncoderModel.from_pretrained(t5_path)

    def get_mask(self, aspect):
        sampled_keys = np.array(self.aspect_to_keys_dict[aspect])
        masks = [self.all_cols[k] in sampled_keys for k in range(len(self.all_cols))]
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None,:]
        return masks

    def featurize_prottrans(self, input_ids, attention_mask):
        with torch.no_grad():
            embedding = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state
        features = [] 
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            features.append(seq_emd)
        prottrans_embedding = torch.tensor(features[0])
        prottrans_embedding = torch.unsqueeze(prottrans_embedding, 0)
        return(prottrans_embedding)

    def embed_vec(self, prottrans_embedding, masks):
        padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(prottrans_embedding)
        out_seq = self.moe.make_matrix(prottrans_embedding, padding)
        vec_embedding = self.moe(out_seq, masks)
        return(vec_embedding)

    def embed(self, input_ids, attention_mask, aspect, progress=False):
        masks = self.get_mask(aspect)
        embed_all_sequences = []
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        if progress:
            for id, mask in tqdm(zip(input_ids, attention_mask), total=len(input_ids)):
                protrans_sequence = self.featurize_prottrans(id.unsqueeze(0), mask.unsqueeze(0))
                embedded_sequence = self.embed_vec(protrans_sequence, masks)
                embed_all_sequences.append(embedded_sequence.detach().cpu()) # if there is enough to need progress you probably need to keep in RAM
        else:
            for id, mask in zip(input_ids, attention_mask):
                protrans_sequence = self.featurize_prottrans(id.unsqueeze(0), mask.unsqueeze(0))
                embedded_sequence = self.embed_vec(protrans_sequence, masks)
                embed_all_sequences.append(embedded_sequence)
        return torch.cat(embed_all_sequences)

    def forward(self, pos, anc, neg,
                att_p=None, att_a=None, att_n=None, r_labels=None):
        p = self.embed(input_ids=pos, attention_mask=att_p, aspect=r_labels[0].item())
        a = self.embed(input_ids=anc, attention_mask=att_a, aspect=r_labels[0].item())
        n = self.embed(input_ids=neg, attention_mask=att_n, aspect=r_labels[0].item())

        loss = self.contrastive_loss(p, a, n)
        logits = (p, a, n)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss
        )
