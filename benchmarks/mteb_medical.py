import mteb
from mteb.encoder_interface import PromptType
import numpy as np


class CustomModel:
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
        pass

model = CustomModel()
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = MTEB(tasks=tasks)
evaluation.run(model)