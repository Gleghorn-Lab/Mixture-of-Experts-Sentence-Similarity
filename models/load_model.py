import time
import copy
import torch
from transformers import (
    EsmModel,
    EsmTokenizer,
    BertModel,
    BertTokenizer
)
from .modeling_moebert import MoEBertModel
from .configuration_moesm import MoEsmConfig
from .modeling_moesm import (
    MoEsmModel,
    MoEsmForMaskedLM,
    MoEsmForSequenceClassification,
    MoEsmForTokenClassification
)
from .model_zoo import (
    EsmForSentenceSimilarity,
    EsmForTripletSimilarity,
    MoEsmForSentenceSimilarity,
    MoEBertForSentenceSimilarity,
    BertForSentenceSimilarity,
    MoEsmForTripletSimilarity
)


def load_models(args):
    if args.ESM:
        tokenizer = EsmTokenizer.from_pretrained(args.model_path)
        base_model = EsmModel.from_pretrained(args.model_path,
                                               hidden_dropout_prob = args.hidden_dropout_prob,
                                               attention_probs_dropout_prob = 0.0)
        if args.MOE:
            loader = MoEsmLoadWeights(args)
            model, tokenizer = loader.get_seeded_model(tokenizer=tokenizer)
        elif args.model_type.lower() == 'sentencesimilarity':
            model = EsmForSentenceSimilarity(base_model.config, base_model)
        else:
            model = EsmForTripletSimilarity(base_model.config, base_model)
    
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        base_model = BertModel.from_pretrained(args.model_path,
                                               hidden_dropout_prob = args.hidden_dropout_prob,
                                               attention_probs_dropout_prob = 0.0)

        domains = args.domains
        if domains is not None and args.new_special_tokens:
            with torch.no_grad():
                base_model.resize_token_embeddings(len(tokenizer) + len(domains))
                # Add new tokens to the tokenizer
                added_tokens = {'additional_special_tokens' : domains}
                tokenizer.add_special_tokens(added_tokens)
                # Seed the embedding with the [CLS] token embedding
                try:  
                    cls_token_embedding = base_model.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
                    for token in domains:
                        base_model.embeddings.word_embeddings.weight[tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()
                except AttributeError:
                    cls_token_embedding = base_model.esm.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
                    for token in domains:
                        base_model.esm.embeddings.word_embeddings.weight[tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()

        config = base_model.config
        config.__dict__.update(vars(args))
        if args.MOE:
            loader = MoEBertLoadWeights(args, config, base_model=base_model, tokenizer=tokenizer)
            base_model, tokenizer = loader.get_seeded_model()
            model = MoEBertForSentenceSimilarity(config, base_model)
        else:
            model = BertForSentenceSimilarity(config, base_model)
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


class MoEsmLoadWeights:
    """
    For seeding MoEsm models with pretrained weights from ESM, or loading MoEsm weights from huggingface hub
    """
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.model_type = args.model_type
        self.num_experts = args.num_experts
        self.topk = args.topk
        self.num_labels = args.num_labels
        self.esm_base = None
        self.config = None
        self.single_moe = args.single_moe

    def get_seeded_model(self, tokenizer): # seed new MoEsm with Esm
        start_time = time.time()
        if self.model_type.lower() == 'model':
            from transformers import EsmModel as EsmModelTransformers
            self.esm_base = EsmModelTransformers.from_pretrained(self.model_path)
            model = MoEsmModel(config=self.config)

        elif self.model_type.lower() == 'maskedlm':
            from transformers import EsmForMaskedLM as EsmModelTransformers
            self.esm_base = EsmModelTransformers.from_pretrained(self.model_path)
            self.config = self.get_config(self.esm_base)
            model = MoEsmForMaskedLM(config=self.config)

        elif self.model_type.lower() == 'sequenceclassification':
            from transformers import EsmForSequenceClassification as EsmModelTransformers
            self.esm_base = EsmModelTransformers.from_pretrained(self.model_path, num_labels=self.num_labels)
            self.config = self.get_config(self.esm_base)
            model = MoEsmForSequenceClassification(config=self.config)

        elif self.model_type.lower() == 'tokenclassification':
            from transformers import EsmForTokenClassification as EsmModelTransformers
            self.esm_base = EsmModelTransformers.from_pretrained(self.model_path, num_labels=self.num_labels)
            self.config = self.get_config(self.esm_base)
            model = MoEsmForTokenClassification(config=self.config)
        
        elif self.model_type.lower() == 'sentencesimilarity':
            from transformers import EsmModel as EsmModelTransformers
            self.esm_base = EsmModelTransformers.from_pretrained(self.model_path)
            self.config = self.get_config(self.esm_base)
            model = MoEsmForSentenceSimilarity(config=self.config)

        elif self.model_type.lower() == 'triplet':
            from transformers import EsmModel as EsmModelTransformers
            self.esm_base = EsmModelTransformers.from_pretrained(self.model_path)
            self.config = self.get_config(self.esm_base)
            model = MoEsmForTripletSimilarity(config=self.config)

        else: print(f'You entered {self.model_type}\nValid options are:\nModel , MaskedLM , SequenceClassification , TokenClassification , Triplet , SentenceSimilarity')
        
        model = self.match_weights(model)

        if self.args.domains is not None and self.args.new_special_tokens:
            with torch.no_grad():
                model.resize_token_embeddings(len(tokenizer) + len(self.args.domains))
                # Add new tokens to the tokenizer
                added_tokens = {'additional_special_tokens' : self.args.domains}
                print('Adding tokens')
                print(added_tokens)
                tokenizer.add_special_tokens(added_tokens)
                # Seed the embedding with the [CLS] token embedding
                try:  
                    cls_token_embedding = model.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
                    for token in self.args.domains:
                        model.embeddings.word_embeddings.weight[tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()
                except AttributeError:
                    cls_token_embedding = model.esm.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
                    for token in self.args.domains:
                        model.esm.embeddings.word_embeddings.weight[tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()

        end_time = time.time()
        print('Model loaded in ', round((end_time - start_time) / 60, 2), 'minutes')
        total, effective, mem = self.count_parameters(model)
        print(f'{total} million total parameters')
        print(f'{effective} million effective parameters')
        print(f'Approximately {mem} GB of memory in fp32\n')
        return model, tokenizer

    def get_pretrained_model(self): # Load Pretrained MoEsm
        start_time = time.time()
        if self.model_type.lower() == 'model':
            model = MoEsmModel.from_pretrained(self.model_path, use_router_loss=self.use_router_loss)

        elif self.model_type.lower() == 'maskedlm':
            model = MoEsmForMaskedLM.from_pretrained(self.model_path, use_router_loss=self.use_router_loss)

        elif self.model_type.lower() == 'sequenceclassification':
            model = MoEsmForSequenceClassification.from_pretrained(self.model_path, num_labels=self.num_labels, use_router_loss=self.use_router_loss)

        elif self.model_type.lower() == 'tokenclassification':
            model = MoEsmForTokenClassification.from_pretrained(self.model_path, num_labels=self.num_labels, use_router_loss=self.use_router_loss)

        elif self.model_type.lower() == 'sentencesimilarity':
            model = MoEsmForSentenceSimilarity.from_pretrained(self.model_path)

        elif self.model_type.lower() == 'triplet':
            model = MoEsmForTripletSimilarity.from_pretrained(self.model_path)

        else: print(f'You entered {self.model_type}\nValid options are:\nModel , MaskedLM , SequenceClassification , TokenClassification , Triplet , SentenceSimilarity')

        self.config = self.get_config(model)
        end_time = time.time()
        print('Model loaded in ', round((end_time - start_time) / 60, 2), 'minutes')
        total, effective, mem = self.count_parameters(model)
        print(f'{total} million total parameters')
        print(f'{effective} million effective parameters')
        print(f'Approximately {mem} GB of memory in fp32')
        return model

    def get_config(self, model):
        esm_config = model.config
        if isinstance(esm_config, MoEsmConfig):
            config = esm_config
            return config
        else:
            config = MoEsmConfig(
                vocab_size=esm_config.vocab_size,
                mask_token_id=esm_config.mask_token_id,
                pad_token_id=esm_config.pad_token_id,
                hidden_size=esm_config.hidden_size,
                num_hidden_layers=esm_config.num_hidden_layers,
                num_attention_heads=esm_config.num_attention_heads,
                intermediate_size=esm_config.intermediate_size,
                max_position_embeddings=esm_config.max_position_embeddings,
                initializer_range=esm_config.initializer_range,
                layer_norm_eps=esm_config.layer_norm_eps,
                position_embedding_type=esm_config.position_embedding_type,
                use_cache=esm_config.use_cache,
                emb_layer_norm_before=esm_config.emb_layer_norm_before,
                token_dropout=esm_config.token_dropout,
                is_folding_model=esm_config.is_folding_model,
                esmfold_config=esm_config.esmfold_config,
                vocab_list=esm_config.vocab_list,
                num_experts=self.args.num_experts,
                topk=self.args.topk,
                hidden_dropout_prob=self.args.hidden_dropout_prob,
                attention_probs_dropout_prob=self.args.attention_probs_dropout_prob,
                moe_type=self.args.moe_type,
                token_moe=self.args.token_moe,
                num_tasks=self.args.num_tasks,
                contact_head=self.args.contact_head,
                domains=self.args.domains,
                wBAL=self.args.wBAL,
                wEX=self.args.wEX,
                expert_loss=self.args.expert_loss,
                single_moe=self.args.single_moe
            )
        return config

    def check_for_match(self, model): # Test for matching parameters
        all_weights_match = True
        for name, param in self.esm_base.named_parameters(): # for shared parameters
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
                    moe_encoder_layer = model.esm.encoder.layer[middle_layer_index]
                except AttributeError:
                    moe_encoder_layer = model.encoder.layer[middle_layer_index]
                try:
                    esm_encoder_layer = self.esm_base.esm.encoder.layer[middle_layer_index]
                except AttributeError:
                    esm_encoder_layer = self.esm_base.encoder.layer[middle_layer_index]
                if not torch.equal(moe_encoder_layer.moe_block.experts[j].intermediate_up.weight,
                                   esm_encoder_layer.intermediate.dense.weight):
                    all_weights_match = False
                if not torch.equal(moe_encoder_layer.moe_block.experts[j].intermediate_down.weight,
                                   esm_encoder_layer.output.dense.weight):
                    all_weights_match = False
        else:
            for i in range(self.config.num_hidden_layers): # for experts
                for j in range(self.config.num_experts):
                    try:
                        moe_encoder_layer = model.esm.encoder.layer[i]
                    except AttributeError:
                        moe_encoder_layer = model.encoder.layer[i]
                    try:
                        esm_encoder_layer = self.esm_base.esm.encoder.layer[i]
                    except AttributeError:
                        esm_encoder_layer = self.esm_base.encoder.layer[i]
                    if not torch.equal(moe_encoder_layer.moe_block.experts[j].intermediate_up.weight,
                                       esm_encoder_layer.intermediate.dense.weight):
                        all_weights_match = False
                    if not torch.equal(moe_encoder_layer.moe_block.experts[j].intermediate_down.weight,
                                       esm_encoder_layer.output.dense.weight):
                        all_weights_match = False

        if all_weights_match:
            print('All weights match')
        else:
            print('Some weights differ')

    def match_weights(self, model): # Seeds MoEsm experts with linear layers of Esm
        self.check_for_match(model)
        for name1, param1 in self.esm_base.named_parameters():
            for name2, param2 in model.named_parameters():
                if name1 == name2:
                    model.state_dict()[name2].data.copy_(param1.data)

        if self.single_moe:
            middle_layer_index = self.config.num_hidden_layers // 2
            for j in range(self.config.num_experts):
                try:
                    moe_encoder_layer = model.esm.encoder.layer[middle_layer_index]
                except AttributeError:
                    moe_encoder_layer = model.encoder.layer[middle_layer_index]
                try:
                    esm_encoder_layer = self.esm_base.esm.encoder.layer[middle_layer_index]
                except AttributeError:
                    esm_encoder_layer = self.esm_base.encoder.layer[middle_layer_index]
                moe_encoder_layer.moe_block.experts[j].intermediate_up = copy.deepcopy(esm_encoder_layer.intermediate.dense)
                moe_encoder_layer.moe_block.experts[j].intermediate_down = copy.deepcopy(esm_encoder_layer.output.dense)
        else:
            for i in range(self.config.num_hidden_layers):
                for j in range(self.config.num_experts):
                    try:
                        moe_encoder_layer = model.esm.encoder.layer[i]
                    except AttributeError:
                        moe_encoder_layer = model.encoder.layer[i]
                    try:
                        esm_encoder_layer = self.esm_base.esm.encoder.layer[i]
                    except AttributeError:
                        esm_encoder_layer = self.esm_base.encoder.layer[i]
                    moe_encoder_layer.moe_block.experts[j].intermediate_up = copy.deepcopy(esm_encoder_layer.intermediate.dense)
                    moe_encoder_layer.moe_block.experts[j].intermediate_down = copy.deepcopy(esm_encoder_layer.output.dense)
        
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
                    moe_encoder_layer = model.esm.encoder.layer[middle_layer_index]
                except AttributeError:
                    moe_encoder_layer = model.encoder.layer[middle_layer_index]
                non_effective_params += self.count_parameters_in_layer(moe_encoder_layer.moe_block.experts[j])
        else:
            for i in range(self.config.num_hidden_layers):
                for j in range(self.config.num_experts - self.config.topk):
                    try:
                        moe_encoder_layer = model.esm.encoder.layer[i]
                    except AttributeError:
                        moe_encoder_layer = model.encoder.layer[i]
                    non_effective_params += self.count_parameters_in_layer(moe_encoder_layer.moe_block.experts[j])

        effective_params = total_params - non_effective_params
        memory_bytes = total_params * 4  # 4 bytes for 32-bit floats
        memory_gig = round(memory_bytes / (1024 ** 3), 2)
        return round(total_params / 1e6, 1), round(effective_params / 1e6, 1), memory_gig