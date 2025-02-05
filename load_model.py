import time
import copy
import torch
from transformers import BertModel, BertTokenizer
from models.modeling_moebert import MoEBertModel
from models.modeling_moe_bert import MoEBertForSentenceSimilarity, BertForSentenceSimilarity
from utils import add_new_tokens, load_from_weight_path


def load_models(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    base_model = BertModel.from_pretrained(args.model_path,
                                            hidden_dropout_prob = args.hidden_dropout_prob,
                                            attention_probs_dropout_prob = 0.0)
    config = base_model.config
    config.__dict__.update(vars(args))
    if args.MOE:
        loader = MoEBertLoadWeights(args, config, base_model=base_model, tokenizer=tokenizer)
        base_model, tokenizer = loader.get_seeded_model()
        model = MoEBertForSentenceSimilarity(config, base_model)
    else:
        model = BertForSentenceSimilarity(config, base_model)

    if args.domains is not None and args.new_special_tokens:
        model, tokenizer = add_new_tokens(args, model, tokenizer)
    model = load_from_weight_path(args, model)
    print(model)
    return model, tokenizer


class MoEBertLoadWeights:
    def __init__(self, args, model_config, base_model, tokenizer):
        self.bert_base = base_model # base bert model
        self.tokenizer = tokenizer # bert tokenizer
        self.config = model_config
        self.domains = args.domains # list of special tokens to take place of CLS
        self.new_tokens = args.new_special_tokens
        self.single_moe = args.single_moe

    def get_seeded_model(self):
        start_time = time.time()
        model = MoEBertModel(config=self.config)
        model = self.match_weights(model)

        if self.new_tokens:
            with torch.no_grad():
                model.resize_token_embeddings(len(self.tokenizer) + len(self.domains))
                # Add new tokens to the tokenizer
                added_tokens = {'additional_special_tokens' : self.domains}
                print('Adding tokens')
                print(added_tokens)
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

    def check_for_match(self, model):  # Test for matching parameters
        all_weights_match = True
        for name, param in self.bert_base.named_parameters():  # for shared parameters
            if name in model.state_dict():
                pre_trained_weight = param.data
                moe_weight = model.state_dict()[name].data
                if not torch.equal(pre_trained_weight, moe_weight):
                    all_weights_match = False
                    break

        if self.single_moe:
            middle_layer_index = self.config.num_hidden_layers // 2
            for j in range(self.config.num_experts):
                try:
                    moe_encoder_layer = model.bert.encoder.layer[middle_layer_index]
                    bert_encoder_layer = self.bert_base.bert.encoder.layer[middle_layer_index]
                except AttributeError:
                    moe_encoder_layer = model.encoder.layer[middle_layer_index]
                    bert_encoder_layer = self.bert_base.encoder.layer[middle_layer_index]

                if not torch.equal(moe_encoder_layer.moe_block.experts[j].intermediate_up.weight,
                                   bert_encoder_layer.intermediate.dense.weight):
                    all_weights_match = False
                if not torch.equal(moe_encoder_layer.moe_block.experts[j].intermediate_down.weight,
                                   bert_encoder_layer.output.dense.weight):
                    all_weights_match = False
        else:
            for i in range(self.config.num_hidden_layers):  # for experts
                for j in range(self.config.num_experts):
                    try:
                        moe_encoder_layer = model.bert.encoder.layer[i]
                        bert_encoder_layer = self.bert_base.bert.encoder.layer[i]
                    except AttributeError:
                        moe_encoder_layer = model.encoder.layer[i]
                        bert_encoder_layer = self.bert_base.encoder.layer[i]

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

    def match_weights(self, model):  # Seeds MoBert experts with linear layers of bert
        self.check_for_match(model)
        for name1, param1 in self.bert_base.named_parameters():
            for name2, param2 in model.named_parameters():
                if name1 == name2:
                    model.state_dict()[name2].data.copy_(param1.data)

        if self.single_moe:
            middle_layer_index = self.config.num_hidden_layers // 2
            for j in range(self.config.num_experts):
                try:
                    moe_encoder_layer = model.bert.encoder.layer[middle_layer_index]
                    bert_encoder_layer = self.bert_base.bert.encoder.layer[middle_layer_index]
                except AttributeError:
                    moe_encoder_layer = model.encoder.layer[middle_layer_index]
                    bert_encoder_layer = self.bert_base.encoder.layer[middle_layer_index]

                moe_encoder_layer.moe_block.experts[j].intermediate_up = copy.deepcopy(bert_encoder_layer.intermediate.dense)
                moe_encoder_layer.moe_block.experts[j].intermediate_down = copy.deepcopy(bert_encoder_layer.output.dense)
        else:
            for i in range(self.config.num_hidden_layers):
                for j in range(self.config.num_experts):
                    try:
                        moe_encoder_layer = model.bert.encoder.layer[i]
                        bert_encoder_layer = self.bert_base.bert.encoder.layer[i]
                    except AttributeError:
                        moe_encoder_layer = model.encoder.layer[i]
                        bert_encoder_layer = self.bert_base.encoder.layer[i]

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
        if self.single_moe:
            middle_layer_index = self.config.num_hidden_layers // 2
            for j in range(self.config.num_experts - self.config.topk):
                try:
                    moe_encoder_layer = model.bert.encoder.layer[middle_layer_index]
                except AttributeError:
                    moe_encoder_layer = model.encoder.layer[middle_layer_index]

                non_effective_params += self.count_parameters_in_layer(moe_encoder_layer.moe_block.experts[j])
        else:
            for j in range(self.config.num_experts - self.config.topk):
                for i in range(self.config.num_hidden_layers):
                    try:
                        moe_encoder_layer = model.bert.encoder.layer[i]
                    except AttributeError:
                        moe_encoder_layer = model.encoder.layer[i]

                    non_effective_params += self.count_parameters_in_layer(moe_encoder_layer.moe_block.experts[j])

        effective_params = total_params - non_effective_params
        memory_bytes = total_params * 4  # 4 bytes for 32-bit floats
        memory_gig = round(memory_bytes / (1024 ** 3), 2)
        return round(total_params / 1e6, 1), round(effective_params / 1e6, 1), memory_gig
