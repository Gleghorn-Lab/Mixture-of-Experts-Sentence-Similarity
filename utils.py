import yaml
import torch


def get_yaml(yaml_file):
    if yaml_file == None:
        return None
    else:
        with open(yaml_file, 'r') as file:
            args = yaml.safe_load(file)
        return args


def load_from_weight_path(args, model):
    if args.weight_path != None:
        if args.huggingface_username in args.weight_path:
            model = model.from_pretrained(args.weight_path, token=args.token, config=model.config)
        else:
            try:
                model.load_state_dict(torch.load(args.weight_path)) # for torch
            except:
                from safetensors.torch import load_model
                load_model(model, args.weight_path) # for safetensors
        print(f'Loaded from {args.weight_path}')
    return model


def add_new_tokens(args, model, tokenizer):
    """
    Adds args.domains as new tokens, seeds with CLS
    """
    domains = args.domains
    with torch.no_grad():
        model.resize_token_embeddings(len(tokenizer) + len(domains))
        # Add new tokens to the tokenizer
        added_tokens = {'additional_special_tokens' : domains}
        tokenizer.add_special_tokens(added_tokens)
        # Seed the embedding with the [CLS] token embedding
        try:  
            cls_token_embedding = model.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
            for token in domains:
                model.embeddings.word_embeddings.weight[tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()
        except AttributeError:
            cls_token_embedding = model.esm.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
            for token in domains:
                model.esm.embeddings.word_embeddings.weight[tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()
    return model, tokenizer


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


def print_tensor_flow(model, input_tensor):
    names, weights, inputs, outputs = [], [], [], []
    def hook_fn(module, input, output):
        if hasattr(module, 'weight'):
            names.append(module.__class__.__name__)
            weights.append(tuple(module.weight.shape))
            inputs.append(tuple(input[0].shape))
            outputs.append(tuple(output.shape))
        elif 'pool' in module.__class__.__name__.lower():
            names.append(module.__class__.__name__)
            weights.append('None')
            inputs.append(tuple(input[0].shape))
            outputs.append(tuple(output.shape))
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module):
            handles.append(module.register_forward_hook(hook_fn))
    with torch.no_grad():
        _ = model(input_tensor)
    for handle in handles:
        handle.remove()
    print('Tensor flow')
    print('{:<25} {:<25} {:<25} {:<25}'.format('Layer', 'Weight Size', 'Input size', 'Output size'))
    print('-' * 100)
    for name, weight, input, output in zip(names, weights, inputs, outputs):
        print('{:<25} {:<25} {:<25} {:<25}'.format(str(name), str(weight), str(input), str(output)))


def calc_memory(model, inputs, device, name='Layer'):
    model.to(device)
    try:
        inputs = inputs.to(device)
        memory_before = torch.cuda.memory_allocated(device)
        _ = model(inputs)        
    except:
        inputs = (input.to(device) for input in inputs)
        memory_before = torch.cuda.memory_allocated(device)
        _ = model(*inputs)

    memory_after = torch.cuda.memory_allocated(device)
    vram_usage = memory_after - memory_before
    print(f"Process {name} VRAM usage: {vram_usage / 1024 / 1024:.2f} MB")
    model.cpu()
    del model
    torch.cuda.empty_cache()


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = total_params - frozen_params
    
    bfloat_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
    float_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.float16)
    half_precision_params = bfloat_params + float_params

    print(f"Total parameters: {total_params / 1e6:,}")
    print(f"Frozen parameters: {frozen_params / 1e6:,}")
    print(f"Trainable parameters: {trainable_params / 1e6:,}")
    print(f"Half precision parameters (bfloat16 or float16): {half_precision_params / 1e6:,}")
