import argparse
import os

os.environ['TRANSFORMERS_CACHE'] = '.trans_cache'
os.environ['HF_DATASETS_CACHE'] = '.datasets_cache'
os.environ['HF_HOME'] = '.hf_home'

import wandb
import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    EvalPrediction, DataCollatorWithPadding
from evaluate import load
import sys


def parse_args():
    args = sys.argv[1:]
    nr_seeds = int(args[0])
    nr_train_samples = int(args[1])
    nr_val_samples = int(args[2])
    nr_test_samples = int(args[3])

    print("Command-line arguments:")
    print(f"Number of seeds: {nr_seeds}")
    print(f"Number of training samples: {nr_train_samples}")
    print(f"Number of validation samples: {nr_val_samples}")
    print(f"Number of test samples: {nr_test_samples}")

    return argparse.Namespace(nr_seeds=nr_seeds, nr_train_samples=nr_train_samples, nr_val_samples=nr_val_samples,
                              nr_test_samples=nr_test_samples)


def get_processed_splits(dataset, tokenizer, nr_train_samples, nr_val_samples, nr_test_samples):
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples['text'])
        return result

    train_split = dataset['train'].map(preprocess_function, batched=True)
    val_split = dataset['validation'].map(preprocess_function, batched=True)
    test_split = dataset['test'].map(preprocess_function, batched=True)

    if nr_train_samples != -1:
        train_split = train_split.select(range(nr_train_samples))
    if nr_val_samples != -1:
        val_split = val_split.select(range(nr_val_samples))
    if nr_test_samples != -1:
        test_split = test_split.select(range(nr_test_samples))

    return train_split, val_split, test_split


def get_compute_metrics():
    accuracy_metric = load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)['accuracy']
        return {"accuracy": accuracy}

    return compute_metrics


def fine_tune(dataset, model_name: str, args):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset, val_dataset, test_split = get_processed_splits(dataset, tokenizer, args.nr_train_samples,
                                                                  args.nr_val_samples, args.nr_test_samples)

    best_trainer = None
    best_accuracy = 0
    accuracies = []
    train_times = []
    for i in range(args.nr_seeds):
        print(f"Fine-tuning {model_name} with seed {i}...")

        wandb.init(
            project="anlp_ex1",
            config={
                "dataset": "SST2",
            },
            name=f"{model_name}_seed_{i}"
        )

        training_args = TrainingArguments(seed=i, output_dir=f"results/{model_name}/seed_{i}",
                                          overwrite_output_dir=True)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=get_compute_metrics(),
            data_collator=data_collator,
        )

        train_res = trainer.train()
        train_times.append(train_res.metrics['train_runtime'])

        model.eval()
        metrics = trainer.evaluate(eval_dataset=val_dataset)
        wandb.log(metrics)
        wandb.finish()
        acc = metrics['eval_accuracy']
        accuracies.append(acc)

        if acc > best_accuracy:
            best_accuracy = acc
            best_trainer = trainer

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return best_trainer, test_split, avg_accuracy, std_accuracy, train_times


def predict(trainer, test_split):
    training_args = TrainingArguments(seed=trainer.args.seed,
                                      output_dir=os.path.join(trainer.args.output_dir, 'test_results'),
                                      overwrite_output_dir=True, per_device_eval_batch_size=1)

    trainer.args = training_args
    trainer.model.eval()
    predictions = trainer.predict(test_split)
    trainer.data_collator = None  # avoid padding for predictions
    preds = predictions.predictions
    preds = np.argmax(preds, axis=1)

    with open('predictions.txt', 'w') as f:
        for input_sentence, label in zip(test_split['text'], preds):
            f.write(f'{input_sentence}###{label}\n')

    return predictions.metrics["test_runtime"]


def main():
    args = parse_args()

    model_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
    dataset = load_dataset("SetFit/sst2")
    best_trainer = None
    best_test_split = None
    best_mean_acc = 0
    all_train_times = []
    open('res.txt', 'w').close()
    for model_name in model_names:
        trainer, test_split, mean_acc, std_acc, train_times = fine_tune(dataset, model_name, args)
        all_train_times.extend(train_times)

        with open('res.txt', 'a') as f:
            f.write(f'{model_name},{mean_acc} +- {std_acc}\n')

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_trainer = trainer
            best_test_split = test_split

    with open('res.txt', 'a') as f:
        f.write('----\n')
        for time in all_train_times:
            f.write(f'train time, {time}\n')

    pred_time = predict(best_trainer, best_test_split)

    with open('res.txt', 'a') as f:
        f.write(f'predict time, {pred_time}\n')


if __name__ == '__main__':
    main()
