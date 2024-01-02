import torch
from trainer import *
from torch.utils.data import DataLoader as TorchLoader
from transformers import BertModel, BertTokenizer
from model import MiniMoELoadWeights, MiniMoE


class config:
    model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    data_paths = ['lhallee/abstract_domain_cvd', 'lhallee/abstract_domain_copd']
    epochs = 100
    domains = ['[CVD]', '[COPD]']
    batch_size = 2
    lr = 1e-4
    validate_interval = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    base_model = BertModel.from_pretrained(config.model_path)
    base_model.config.attention_probs_dropout_prob = 0.05
    base_model.config.hidden_dropout_prob = 0.05
    #base_model.max_position_embeddings = 2048 # this would be a good addition
    loader = MiniMoELoadWeights(base_model=base_model, tokenizer=tokenizer, domains=config.domains)
    model, tokenizer = loader.get_seeded_model()
    mini = MiniMoE(model)

    train_dataset, valid_dataset, test_dataset = get_datasets(config.data_paths, tokenizer, config.domains)
    train_loader = TorchLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = TorchLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = TorchLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(mini.parameters(), lr=config.lr)
    train(config, mini, optimizer, train_loader, valid_loader)
