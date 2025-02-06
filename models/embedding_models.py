import torch
import os
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional, List, Callable, Union
from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def mean_pooling(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
    return torch.sum(x * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SimpleTextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def build_collator(tokenizer) -> Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]:
    def _collate_fn(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function for batching texts."""
        return tokenizer(texts, return_tensors="pt", padding='longest', pad_to_multiple_of=8)
    return _collate_fn


class EmbeddingMixin:
    def embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device

    def _read_texts_from_db(self, db_path: str) -> set[str]:
        """Read strings from SQLite database."""
        import sqlite3
        texts = []
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT text FROM embeddings")
            while True:
                row = c.fetchone()
                if row is None:
                    break
                texts.append(row[0])
        return set(texts)

    def embed_dataset(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 2,
        embed_dtype: torch.dtype = torch.float32,
        cls_pooling: bool = False,
        num_workers: int = 0,
        sql: bool = False,
        save: bool = False,
        sql_db_path: str = 'embeddings.db',
        save_path: str = 'embeddings.pth',
    ) -> Optional[dict[str, torch.Tensor]]:
        #texts = list(set([text[:max_len] for text in texts])) # trim beforehand
        texts = sorted(texts, key=len, reverse=True)
        collate_fn = build_collator(tokenizer)
        device = self.device

        if sql:
            import sqlite3
            conn = sqlite3.connect(sql_db_path)
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (text text PRIMARY KEY, embedding blob)')
            already_embedded = self._read_texts_from_db(sql_db_path)
            to_embed = [text for text in texts if text not in already_embedded]

            print(f"Found {len(already_embedded)} already embedded texts in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new texts")
            if len(to_embed) > 0:
                dataset = SimpleTextDataset(to_embed)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
                with torch.no_grad():
                    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                        seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                        embeddings = self.embed(input_ids, attention_mask, cls_pooling).float().cpu() # sql requires float32
                        for seq, emb in zip(seqs, embeddings):
                            c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", 
                                    (seq, emb.cpu().numpy().tobytes()))
                        
                        if (i + 1) % 100 == 0:
                            conn.commit()
            
                conn.commit()
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = torch.load(save_path, map_location='cpu', weights_only=True)
            to_embed = [text for text in texts if text not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded texts in {save_path}")
            print(f"Embedding {len(to_embed)} new texts")
        else:
            to_embed = texts
            print(f"Embedding {len(to_embed)} new texts")

        if len(to_embed) > 0:
            dataset = SimpleTextDataset(to_embed)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    embeddings = self.embed(input_ids, attention_mask, cls_pooling).to(embed_dtype).cpu()
                    for seq, emb in zip(seqs, embeddings):
                        embeddings_dict[seq] = emb.view(1, -1)

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict


class BaseEmbedder(nn.Module, EmbeddingMixin):
    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_pooling: bool = False) -> torch.Tensor:
        return self.embed(input_ids, attention_mask, cls_pooling)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_pooling: bool = False) -> torch.Tensor:
        if cls_pooling:
            return self.model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        else:
            last_hidden_state = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
            return mean_pooling(last_hidden_state, attention_mask)


class SentenceTransformerEmbedder(nn.Module, EmbeddingMixin):
    # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    def __init__(self, model_path: str):

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_pooling: bool = False) -> torch.Tensor:
        return self.embed(input_ids, attention_mask, cls_pooling)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_pooling: bool = False) -> torch.Tensor:
        assert not cls_pooling, "MiniEmbedder does not support cls_pooling"

        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)[0]
        sentence_embeddings = mean_pooling(last_hidden_state, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    

class LlamaEmbedder(nn.Module, EmbeddingMixin):
    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_pooling: bool = False) -> torch.Tensor:
        return self.embed(input_ids, attention_mask, cls_pooling)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_pooling: bool = False) -> torch.Tensor:
        # cls is eos for llama
        if cls_pooling:
            eos_tokens = (input_ids == self.tokenizer.eos_token_id)
            if not torch.any(eos_tokens):
                raise ValueError("No eos token found in input")
            cls_token_index = torch.argmax(eos_tokens.long(), dim=1)
            return self.model(input_ids, attention_mask=attention_mask).last_hidden_state[torch.arange(input_ids.size(0)), cls_token_index, :]
        else:
            last_hidden_state = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
            return mean_pooling(last_hidden_state, attention_mask)


class TfidfEmbedder(nn.Module, EmbeddingMixin):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(max_features=4096)
        self.is_fitted = False
        # Dummy parameters to make device property work
        self.model = nn.Parameter(torch.zeros(1))
        
    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer on a corpus of texts"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
    def forward(self, texts: List[str], *args, **kwargs) -> torch.Tensor:
        return self.embed(texts)

    def embed_dataset(self, texts: List[str]) -> torch.Tensor:
        """
        Embed texts using TF-IDF. Note: This embedder takes raw texts instead of input_ids
        """
        if not self.is_fitted:
            raise RuntimeError("TfidfEmbedder must be fitted before embedding")
            
        if isinstance(texts, torch.Tensor):
            raise ValueError("TfidfEmbedder expects raw texts, not input_ids")

        embeddings_dict = {}
        # Convert sparse matrix to dense tensor
        embeddings = self.vectorizer.transform(texts).toarray()
        print(embeddings.shape)
        for text, embedding in zip(texts, embeddings):
            embeddings_dict[text] = torch.from_numpy(embedding).float().view(1, -1)
        return embeddings_dict


relevant_paths = {
    'ModernBERT-base': 'answerdotai/ModernBERT-base',
    'ModernBERT-large': 'answerdotai/ModernBERT-large',
    'BERT-base': 'google-bert/bert-base-uncased',
    'BERT-large': 'google-bert/bert-large-uncased',
    'Mini': 'sentence-transformers/all-MiniLM-L6-v2',
    'MPNet': 'sentence-transformers/all-mpnet-base-v2',
    'E5-base': 'intfloat/e5-base-v2',
    'E5-large': 'intfloat/e5-large-v2',
    'RoBERTa-base': 'FacebookAI/roberta-base',
    'RoBERTa-large': 'FacebookAI/roberta-large',
    'Llama-3.2-1B': 'meta-llama/Llama-3.2-1B',
    'SciBERT': 'allenai/scibert_scivocab_uncased',
    'PubmedBERT': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    'BioBERT': 'dmis-lab/biobert-v1.1',
    'TF-IDF': None,
}


model_to_class_dict = {
    'Mini': SentenceTransformerEmbedder,
    'MPNet': SentenceTransformerEmbedder,
    'E5-base': SentenceTransformerEmbedder,
    'E5-large': SentenceTransformerEmbedder,
    'ModernBERT-base': BaseEmbedder,
    'ModernBERT-large': BaseEmbedder,
    'BERT-base': BaseEmbedder,
    'BERT-large': BaseEmbedder,
    'RoBERTa-base': BaseEmbedder,
    'RoBERTa-large': BaseEmbedder,
    'DeBERTa-v3-base': BaseEmbedder,
    'SciBERT': BaseEmbedder,
    'PubmedBERT': BaseEmbedder,
    'BioBERT': BaseEmbedder,
    'Llama-3.2-1B': LlamaEmbedder,
    'TF-IDF': TfidfEmbedder,
}
