import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState


class PlotMIGateCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        mi_task_gate = model.MI_loss.MI_task_gate.detach().cpu()
        mi_task_gate = mi_task_gate / mi_task_gate.sum(axis=-1, keepdims=True) 
        plt.imshow(mi_task_gate, cmap='viridis')
        plt.colorbar()
        plt.savefig(f"MI_task_gate_plot_{state.global_step}.png")
        plt.close()


def HF_trainer(model,
               train_dataset,
               valid_dataset,
               compute_metrics=None,
               data_collator=None,
               patience=1,
               MI=False,
               *args, **kwargs):
    training_args = TrainingArguments(load_best_model_at_end=True, *args, **kwargs)

    if MI:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience), PlotMIGateCallback()]
    else:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

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