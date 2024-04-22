import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Union, Any
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
### For custom trainer
import numpy as np
from typing import Optional, List
from torch.utils.data import DataLoader
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, has_length, denumpify_detensorize
from transformers.trainer_pt_utils import EvalLoopContainer, IterableDatasetShard, find_batch_size
from transformers.utils import logging

logger = logging.get_logger(__name__)


class TopkTallyCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        try:
            tally = model.aux_loss.tally.detach().cpu().numpy()
            model.aux_loss.reset_tally()
        except:
            tally = model.expert_loss.tally.detach().cpu().numpy()
            model.expert_loss.reset_tally()
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(tally.shape[0]), tally)
        plt.xlabel('Expert Index')
        plt.ylabel('Tally of Topk Chosen Results')
        plt.title(f'Topk Tally at Global Step {state.global_step}')
        plt.savefig(f'topk_tally_{state.global_step}.png')
        plt.close()


class DoubleTrainer(Trainer):
    def __init__(self, *args, **kwargs):


        self.loss = kwargs.pop('loss', None)
        self.temp = torch.tensor(kwargs.pop("temp", 1.0))
        super().__init__(*args, **kwargs)
        self.accumulated_a = []
        self.accumulated_b = []

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            outputs = model(**inputs)
            logits = outputs.logits
            a, b = logits[0], logits[1]

            self.accumulated_a.append(a)
            self.accumulated_b.append(b)

            if len(self.accumulated_a) == self.args.gradient_accumulation_steps:
                emb_a = torch.cat(self.accumulated_a, dim=0)
                emb_b = torch.cat(self.accumulated_b, dim=0)
                loss = self.loss(emb_a, emb_b, self.temp)
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                self.accelerator.backward(loss)
                self.accumulated_a = []
                self.accumulated_b = []
                return loss.detach()

        return torch.tensor(float('nan')) # returning this will make the logger only log the real losses

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        ### Reset accumulated logits
        self.accumulated_a = []
        self.accumulated_b = []
        args = self.args
        gradient_accumulation_steps = args.gradient_accumulation_steps

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            if self.is_fsdp_enabled:
                self.model = model
            if model is not self.model:
                self.model_wrapped = model
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        batch_size = self.args.eval_batch_size
        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)
        if args.past_index >= 0:
            self._past = None
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        observed_num_examples = 0
        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size
            ### Primarily updated here
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            a, b = logits[0], logits[1]
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None
            self.accumulated_a.append(a)
            self.accumulated_b.append(b)
            if len(self.accumulated_a) == self.args.gradient_accumulation_steps:
                emb_a = torch.cat(self.accumulated_a, dim=0)
                emb_b = torch.cat(self.accumulated_b, dim=0)
                loss = self.loss(emb_a, emb_b, self.temp)
                losses = self.gather_function((loss.repeat(batch_size * gradient_accumulation_steps)))
                all_losses.add(losses)
                self.accumulated_a = []
                self.accumulated_b = []
            ###
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                all_inputs.add(inputs_decode)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                all_preds.add(logits)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self.gather_function((labels))
                all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


def HF_trainer(model,
               train_dataset,
               valid_dataset,
               compute_metrics=None,
               data_collator=None,
               patience=1,
               EX=False,
               double=False,
               *args, **kwargs):
    training_args = TrainingArguments(load_best_model_at_end=True, *args, **kwargs)

    if EX:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience), TopkTallyCallback()]
    else:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    if double:
        from models.losses import clip_loss
        trainer = DoubleTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=callbacks,
            loss=clip_loss
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=callbacks
        )
    return trainer
