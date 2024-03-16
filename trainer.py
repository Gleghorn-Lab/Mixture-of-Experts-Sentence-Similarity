from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


def HF_trainer(model,
               train_dataset,
               valid_dataset,
               compute_metrics=None,
               data_collator=None,
               patience=3,
               *args, **kwargs):
    training_args = TrainingArguments(load_best_model_at_end=True, *args, **kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )
    return trainer