import torch
import numpy as np
import sqlite3
from tqdm.auto import tqdm


def embed_standard_plm(cfg, model, tokenizer, seqs, full=False, pooling='mean', max_length=512):
    model.eval()
    input_embeddings = []
    with torch.no_grad():
        for sample in tqdm(seqs, desc='Embedding'):
            ids = tokenizer(sample,
                            add_special_tokens=True,
                            padding=False,
                            return_token_type_ids=False,
                            return_tensors='pt').input_ids.to(cfg.device)
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
                input_embeddings.append(emb.detach().cpu().numpy())
            else:
                if pooling == 'cls':
                    emb = emb[:, 0, :]
                elif pooling == 'mean':
                    emb = torch.mean(emb, dim=1, keepdim=False)
                else:
                    emb = torch.max(emb, dim=1, keepdim=False)[0]
                input_embeddings.append(emb.detach().cpu().numpy())
    return input_embeddings


def embed_moe_dataset(args, model, tokenizer, seqs, expert, domain):
    model.eval()
    input_embeddings = []
    add_token = args.new_special_tokens
    with torch.no_grad():
        for sample in tqdm(seqs, desc='Embedding'):
            ids = tokenizer(sample,
                            add_special_tokens=True,
                            padding=False,
                            return_token_type_ids=False,
                            return_tensors='pt').input_ids.to(args.device)
            if add_token:
                ids[0][0] = tokenizer(domain, add_special_tokens=False).input_ids[0]
            if args.full:
                emb = model.embed_matrix(ids, r_labels=expert).float().detach().cpu().numpy()
                if emb.size(1) < args.max_length:
                    padding_needed = args.max_length - emb.size(1)
                    emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed, 0, 0), value=0)
                else:
                    emb = emb[:, :args.max_length, :]
            else:
                emb = model.embed(ids, r_labels=expert).float().detach().cpu().numpy()
            input_embeddings.append(emb)
    return input_embeddings


def embed_protein_vec_dataset(args, model, tokenizer, seqs, aspect_token):
    model.eval()
    toks = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(toks['input_ids']).to(args.device)
    attention_mask = torch.tensor(toks['attention_mask']).to(args.device)
    embeds = model.embed(input_ids, attention_mask, aspect_token, progress=True).detach().cpu().numpy()
    return embeds.tolist()


def embed_dataset_and_save(cfg, model, tokenizer, seqs):
    model.eval()
    db_file = cfg.db_path
    batch_size = 1000
    full = cfg.full
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")
        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), batch_size), desc='Batches'):
                batch_seqs = seqs[i:i + batch_size]
                embeddings = []
                for sample in tqdm(batch_seqs, desc='Embedding'):  # Process embeddings in batches
                    ids = tokenizer(sample,
                                    add_special_tokens=True,
                                    padding=False,
                                    return_token_type_ids=False,
                                    return_tensors='pt').input_ids.to(cfg.device)
                    if full:
                        embedding = model.embed_matrix(ids).detach().cpu().numpy()
                    else:
                        embedding = model(ids).detach().cpu().numpy()
                    embeddings.append(embedding) # add if cfg.full for matrix

                for seq, emb in zip(batch_seqs, embeddings):
                    emb_data = np.array(emb).tobytes()
                    c.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, emb_data))
                conn.commit()


def save_embeddings_to_disk(emb_dict, db_file="embeddings.db"):
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")

        for seq, emb in emb_dict.items():
            emb_data = np.array(emb).tobytes()  # Serialize using NumPy
            c.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, emb_data))
        conn.commit()


def load_embedding(seq, db_file="embeddings.db"):
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
        row = result.fetchone()
        if row is not None:
            emb_data = row[0]
            return np.frombuffer(emb_data, dtype=np.float32).reshape(-1)  # Deserialize
        else:
            return None
