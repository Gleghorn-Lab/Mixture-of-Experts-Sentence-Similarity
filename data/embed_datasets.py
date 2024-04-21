import torch
import numpy as np
import sqlite3
from tqdm.auto import tqdm


def prepare_embed_standard_model(model, tokenizer, max_length, full, pooling, device):
    def embed_standard_model(seqs):
        embeddings = []
        with torch.no_grad():
            for seq in tqdm(seqs, desc='Embedding batch'):
                ids = tokenizer(seq,
                        add_special_tokens=True,
                        padding=False,
                        return_token_type_ids=False,
                        return_tensors='pt').input_ids.to(device)
                output = model(ids)
                try:
                    emb = output.last_hidden_state.float()
                except:
                    emb = output.hidden_states[-1].float()
                if full:
                    if emb.size(1) < max_length:
                        padding_needed = max_length - emb.size(1)
                        emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed, 0, 0), value=0)
                    else:
                        emb = emb[:, :max_length, :]
                else:
                    if pooling == 'cls':
                        emb = emb[:, 0, :]
                    elif pooling == 'mean':
                        emb = torch.mean(emb, dim=1, keepdim=False)
                    else:
                        emb = torch.max(emb, dim=1, keepdim=False)[0]
                embeddings.append(emb.detach().cpu().numpy())
        return embeddings
    return embed_standard_model


def prepare_embed_double_model(model, tokenizer, device):
    def embed_double_model(seqs):
        embeddings = []
        with torch.no_grad():
            for seq in tqdm(seqs, desc='Embedding batch'):
                toks = tokenizer(seq,
                                add_special_tokens=True,
                                padding=False,
                                return_token_type_ids=False,
                                return_tensors='pt')
                ids = toks.input_ids[:, 1:].to(device) # remove cls token
                mask = toks.attention_mask[:, 1:].to(device)
                base_ids = model.tokenizer_base(seq,
                                add_special_tokens=True,
                                padding=False,
                                return_token_type_ids=False,
                                return_tensors='pt').input_ids.to(device)
                emb = model.embed(base_ids, ids, mask).float().detach().cpu().numpy()
                embeddings.append(emb)
        return embeddings
    return embed_double_model


def prepare_embed_moe_model(model, tokenizer, device, domain, expert, full, max_length, add_token=False):
    def embed_moe_model(seqs):
        embeddings = []
        with torch.no_grad():
            for seq in tqdm(seqs, desc='Embedding batch'):
                ids = tokenizer(seq,
                                add_special_tokens=True,
                                padding=False,
                                return_token_type_ids=False,
                                return_tensors='pt').input_ids.to(device)
                if add_token:
                    ids[0][0] = tokenizer(domain, add_special_tokens=False).input_ids[0]
                if full:
                    emb = model.embed_matrix(ids, r_labels=expert).float().detach().cpu().numpy()
                    if emb.size(1) < max_length:
                        padding_needed = max_length - emb.size(1)
                        emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed, 0, 0), value=0)
                    else:
                        emb = emb[:, :max_length, :]
                else:
                    emb = model.embed(ids, r_labels=expert).float().detach().cpu().numpy()
                embeddings.append(emb)
        return emb
    return embed_moe_model


def prepare_embed_protein_vec_dataset(model):
    def embed_protein_vec_dataset(seqs, aspect_token):
        with torch.no_grad():
            embeds = model.embed(seqs, aspect_token)
        return embeds.tolist()
    return embed_protein_vec_dataset


def embed_data(cfg,
               seqs,
               model,
               tokenizer,
               expert=None,
               domain=None):
    
    model_type = cfg.model_type.lower()
    sql = cfg.sql
    db_file = cfg.db_path
    full = cfg.full
    max_length = cfg.max_length
    device = cfg.device
    add_token = cfg.new_special_tokens
    pooling = cfg.pooling

    model.eval()
    embeddings = []
    batch_size = 1000

    if model_type == 'triplet':
        embed_seqs = prepare_embed_moe_model(model, tokenizer, device, domain, expert, full, max_length, add_token)
    elif model_type == 'proteinvec':
        embed_seqs = prepare_embed_protein_vec_dataset(model)
    elif model_type == 'double':
        embed_seqs = prepare_embed_double_model(model, tokenizer, device)
    else:
        embed_seqs = prepare_embed_standard_model(model, tokenizer, max_length, full, pooling, device)

    for i in tqdm(range(0, len(seqs), batch_size), desc='Batches'):
        batch_seqs = seqs[i:i + batch_size]
        embs = embed_seqs(batch_seqs)
        if sql:
            with sqlite3.connect(db_file) as conn:
                c = conn.cursor()
                c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")
                for seq, emb in zip(batch_seqs, embs):
                    emb_data = np.array(emb).tobytes()
                    c.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, emb_data))
                conn.commit()
        else:
            embeddings.extend(embs)
    if embeddings:
        return embeddings
