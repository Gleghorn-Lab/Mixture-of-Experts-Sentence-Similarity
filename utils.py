import yaml
from transformers import BertModel, BertTokenizer, EsmModel, EsmTokenizer

from models.load_model import MoEBertLoadWeights, MoEsmLoadWeights
from models.modeling_moesm import MoEsmForSentenceSimilarity, EsmForSentenceSimilarity
from models.modeling_moebert import BertForSentenceSimilarity, MoEBertForSentenceSimilarity


def get_yaml(yaml_file):
    if yaml_file == None:
        return None
    else:
        with open(yaml_file, 'r') as file:
            args = yaml.safe_load(file)
        return args


def log_metrics(log_path, metrics, details=None, header=None):
    def log_nested_dict(d, parent_key=''):
        filtered_results = {}
        for k, v in d.items():
            new_key = f'{parent_key}_{k}' if parent_key else k
            if isinstance(v, dict):
                filtered_results.update(log_nested_dict(v, new_key))
            elif 'runtime' not in k and 'second' not in k:
                filtered_results[new_key] = round(v, 5) if isinstance(v, (float, int)) else v
        return filtered_results

    filtered_results = log_nested_dict(metrics)

    with open(log_path, 'a') as f:
        f.write('\n')
        if header is not None:
            f.write(header + '\n')
        if details is not None:
            for k, v in details.items():
                f.write(f'{k}: {v}\n')
        for k, v in filtered_results.items():
            f.write(f'{k}: {v}\n')
        f.write('\n')


def load_model(args):
    if args['ESM']:
        tokenizer = EsmTokenizer.from_pretrained(args['model_path'])
        base_model = EsmModel.from_pretrained(args['model_path'],
                                               hidden_dropout_prob = args['hidden_dropout_prob'],
                                               attention_probs_dropout_prob = 0.0)
        if args['MOE']:
            loader = MoEsmLoadWeights(args)
            model, tokenizer = loader.get_seeded_model(tokenizer=tokenizer)
        else:
            model = EsmForSentenceSimilarity(base_model.config, base_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(args['model_path'])
        base_model = BertModel.from_pretrained(args['model_path'],
                                               hidden_dropout_prob = args['hidden_dropout_prob'],
                                               attention_probs_dropout_prob = 0.0)
        config = base_model.config
        for key, value in args.items():
            setattr(config, key, value)
        if args['MOE']:
            loader = MoEBertLoadWeights(args, config, base_model=base_model, tokenizer=tokenizer)
            base_model, tokenizer = loader.get_seeded_model()
            model = MoEBertForSentenceSimilarity(config, base_model)
        else:
            model = BertForSentenceSimilarity(config, base_model)
    print(model)
    return model, tokenizer


