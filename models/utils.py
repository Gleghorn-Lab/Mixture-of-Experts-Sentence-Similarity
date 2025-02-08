import copy
import torch
from torchinfo import summary
from typing import Any, List, Tuple
from transformers import AutoTokenizer
from .modern_bert_config import ModernBertConfig
from .modeling_modern_bert import ModernBertModel
from .moe_blocks import SentenceEnforcedSwitchMoeBlock
from .modeling_moe_bert import MoEBertForSentenceSimilarity


def convert_to_moe_bert(config: ModernBertConfig, model: ModernBertModel) -> ModernBertModel:
    """
    Seeds all experts with the original weights from the pretrained model.
    mlp (Expert) at ModernBertModel.layers[i].mlp
    """
    for layer in model.layers:
        Expert = copy.deepcopy(layer.mlp)
        layer.mlp = SentenceEnforcedSwitchMoeBlock(config, Expert, pretrained=True)
    return model


def add_new_tokens(model: ModernBertModel, tokenizer: Any, domains: List[str]) -> Tuple[ModernBertModel, Any]:
    """
    Adds args.domains as new tokens, seeds with CLS
    embeddings live in ModernBertModel.embeddings.tok_embeddings
    """
    with torch.no_grad():
        model.resize_token_embeddings(len(tokenizer) + len(domains))
        # Add new tokens to the tokenizer
        added_tokens = {'additional_special_tokens' : domains}
        tokenizer.add_special_tokens(added_tokens)
        # Seed the embedding with the [CLS] token embedding
        try: 
            cls_token_embedding = model.embeddings.tok_embeddings.weight[tokenizer.cls_token_id, :].clone()
            for token in domains:
                model.embeddings.tok_embeddings.weight[tokenizer.convert_tokens_to_ids(token), :] = cls_token_embedding.clone()
        except AttributeError as e:
            print(e)
            cls_token_embedding = model.bert.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
            for token in domains:

                model.bert.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids(token), :] = cls_token_embedding.clone()
    return model, tokenizer


def prepare_model(
        pretrained_path: str,
        domains: List[str],
        lora: bool = False,
        moe: bool = True,
        clip_loss: bool = True,
    ) -> Tuple[MoEBertForSentenceSimilarity, Any]:
    """
    Loads a pretrained model and adds new tokens
    """
    config = ModernBertConfig.from_pretrained(pretrained_path)
    config.lora = lora
    config.loss_type = 'clip' if clip_loss else 'mnr_loss'
    if moe:
        config.num_experts = len(domains)
    else:
        config.num_experts = 1
    model = ModernBertModel.from_pretrained(pretrained_path, config=config)
    print('Pre MOE number of parameters:', sum(p.numel() for p in model.parameters()))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model, tokenizer = add_new_tokens(model, tokenizer, domains)
    config.vocab_size = len(tokenizer)
    model = convert_to_moe_bert(config, model) if moe else model
    model = MoEBertForSentenceSimilarity(config, model)
    print('Post MOE number of parameters:', sum(p.numel() for p in model.parameters()))
    return model, tokenizer


if __name__ == "__main__":
    # py -m models.utils
    model_path = 'answerdotai/ModernBERT-base'
    domains = ['[COPD]', '[CVD]', '[CANCER]']
    lora = False
    moe = True
    model, tokenizer = prepare_model(model_path, domains, lora, moe)
    summary(model)
    print(model)
    print(tokenizer.additional_special_tokens)