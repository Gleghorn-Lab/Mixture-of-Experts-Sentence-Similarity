import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState


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


def HF_trainer(model,
               train_dataset,
               valid_dataset,
               compute_metrics=None,
               data_collator=None,
               patience=1,
               EX=False,
               token=None,
               **kwargs):
    training_args = TrainingArguments(load_best_model_at_end=True, hub_token=token, **kwargs)

    if EX:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience), TopkTallyCallback()]
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
