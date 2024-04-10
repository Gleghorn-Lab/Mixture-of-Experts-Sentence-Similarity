import torch
import gc
import sqlite3
import os

from transformers import T5EncoderModel, T5Tokenizer

from data.load_data import get_fine_tune_data, get_seqs
from models.protein_vec.src_run.model_protein_moe import trans_basic_block, trans_basic_block_Config
from models.protein_vec.src_run.utils_search import *

from tqdm.auto import tqdm
from datasets import load_dataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Load model
    orig_cwd = os.getcwd()

    vec_model_cpnt = 'protein_vec_models/protein_vec.ckpt'
    vec_model_config = 'protein_vec_models/protein_vec_params.json'

    tokenizer = T5Tokenizer.from_pretrained("lhallee/prot_t5_enc", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("lhallee/prot_t5_enc").half().to(device).eval()

    os.chdir('models/protein_vec/src_run/')
    vec_model_config = trans_basic_block_Config.from_json(vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(vec_model_cpnt, config=vec_model_config).to(device).eval()

    os.chdir(orig_cwd)
    gc.collect()

    ### Embedding

    def embed_seqs(model, model_deep, tokenizer, seqs, device):
        sampled_keys = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None,:]

        embed_all_sequences = []
        for seq in tqdm(seqs, desc='Embedding'): 
            protrans_sequence = featurize_prottrans([seq], model, tokenizer, device)
            embedded_sequence = embed_vec(protrans_sequence, model_deep, masks, device)
            embed_all_sequences.append(embedded_sequence)
        return np.concatenate(embed_all_sequences)

    ### Get seqs

    dataset_paths = [
        'lhallee/EC_reg',
        'lhallee/CC_reg',
        'lhallee/MF_reg',
        'lhallee/BP_reg',
        'lhallee/dl_binary_reg',
        'lhallee/dl_ten_reg',
        'lhallee/MetalIonBinding_reg'
    ]

    class args:
        trim=True
        max_length=1024

    datasets, all_seqs, train_sets, valid_sets, test_sets, num_labels, task_types = [], [], [], [], [], [], []


    for i, data_path in enumerate(dataset_paths):
        train_set, valid_set, test_set, num_label, task_type = get_fine_tune_data(args, data_path)

        train_seqs, train_labels = get_seqs(train_set)
        valid_seqs, valid_labels = get_seqs(valid_set)
        test_seqs, test_labels = get_seqs(test_set)

        train_seqs, train_labels = train_seqs, train_labels
        valid_seqs, valid_labels = valid_seqs, valid_labels
        test_seqs, test_labels = test_seqs, test_labels

        train_sets.append((train_seqs, train_labels))
        valid_sets.append((valid_seqs, valid_labels))
        test_sets.append((test_seqs, test_labels))

        all_seqs.extend(train_seqs + valid_seqs + test_seqs)

    all_seqs = list(set(all_seqs))


    ### Embed
    embeds = embed_seqs(model, model_deep, tokenizer, all_seqs, device)

    ### Save with sql
    db_file = 'prot_vec_local.db'

    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")
        for seq, emb in tqdm(zip(all_seqs, embeds), total=len(all_seqs), desc='Saving to disk'):
            emb_data = np.array(emb).tobytes()
            c.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, emb_data))
        conn.commit()


if __name__ == '__main__':
    main()