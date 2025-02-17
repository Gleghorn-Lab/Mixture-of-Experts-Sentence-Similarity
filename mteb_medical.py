import mteb
import numpy as np
import torch
from mteb.encoder_interface import PromptType
from models.embedding_models import BaseEmbedder


class BaseEncoder(BaseEmbedder):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.
        
        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.
            
        Returns:
            The encoded sentences.
        """
        tokenized = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding='longest',
            pad_to_multiple_of=8,
            truncation=True,
            max_length=512,
            add_special_tokens=True)
        input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        input_ids, attention_mask = input_ids.to(self._device), attention_mask.to(self._device)

        with torch.no_grad():
            embeddings = self.embed(input_ids, attention_mask)
        return embeddings.cpu().numpy()



if __name__ == "__main__":
    # py -m benchmarks.mteb_medical
    from mteb import MTEB, Benchmark, get_tasks

    model_path = 'answerdotai/ModernBERT-base'
    model = BaseEncoder(model_path)

    medical_benchmark = Benchmark(
        name="MTEB(Medical, v1)",
        tasks=get_tasks(
            tasks=[
                "CUREv1",
                "NFCorpus",
                "TRECCOVID",
                "TRECCOVID-PL",
                "SciFact",
                "SciFact-PL",
                "MedicalQARetrieval",
                "PublicHealthQA",
                "MedrxivClusteringP2P.v2",
                "MedrxivClusteringS2S.v2",
                "CmedqaRetrieval",
                "CMedQAv2-reranking",
            ],
            languages=["eng"]
        ),
        description="A curated set of MTEB tasks designed to evaluate systems in the context of medical information retrieval.",
        reference="",
        citation=None,
    )
    evaluation = MTEB(tasks=medical_benchmark)
    results = evaluation.run(model, verbosity=2)
    for result in results:
        print(result.task_name)
        print(result.scores)
