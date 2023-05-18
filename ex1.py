import argparse
import os

os.environ['TRANSFORMERS_CACHE'] = '/sci/nosnap/tomhope/noystl/anlp_ex1/.trans_cache'
os.environ['HF_DATASETS_CACHE'] = '/sci/nosnap/tomhope/noystl/anlp_ex1/.datasets_cache'
os.environ['HF_HOME'] = '/sci/nosnap/tomhope/noystl/anlp_ex1/.hf_home'

import wandb
import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    EvalPrediction, DataCollatorWithPadding
from evaluate import load

def parse_args():
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr_seeds", type=int, help="Number of seeds to be used for each model.", default=3)
    parser.add_argument("--nr_train_samples", type=int,
                        help="Number of samples to be used during training or -1 if all training samples should be used.",
                        default=-1)
    parser.add_argument("--nr_val_samples", type=int,
                        help="Number of samples to be used during validation or -1 if all validation samples should be used.",
                        default=-1)
    parser.add_argument("--nr_test_samples", type=int,
                        help="Number of samples for which the model will predict a sentiment or -1 if a sentiment should be predicted for all test samples.",
                        default=-1)
    args = parser.parse_args()

    # Print out the command-line arguments
    print("Command-line arguments:")
    print(f"Number of seeds: {args.nr_seeds}")
    print(f"Number of training samples: {args.nr_train_samples}")
    print(f"Number of validation samples: {args.nr_val_samples}")
    print(f"Number of test samples: {args.nr_test_samples}")

    return args


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
