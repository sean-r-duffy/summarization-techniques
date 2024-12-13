import csv
import os

import pandas as pd
from transformers import PegasusTokenizerFast, BigBirdPegasusForConditionalGeneration, BigBirdPegasusConfig, pipeline
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import torch
from rouge import Rouge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
MAX_LENGTH = 4096
EPOCHS = 2
MIN_CHARS = MAX_LENGTH * 1
MAX_CHARS = MAX_LENGTH * 6
TRAIN_N = 10000
VAL_N = 100

model_config = BigBirdPegasusConfig(
    encoder_layers = 1,
    encoder_ffn_dim = 1024,
    encoder_attention_heads = 2,
    decoder_layers = 1,
    decoder_ffn_dim = 1024,
    decoder_attention_heads = 2,
    max_position_embeddings=4096
)

tokenizer = PegasusTokenizerFast.from_pretrained("google/pegasus-large")

def load(path, train_size=TRAIN_N, val_size=VAL_N, input_col='article'):
    dataset = load_dataset(path)
    train_data, val_data = dataset['train'], dataset['validation']

    def get_length(example):
        example['article_length'] = len(example[input_col])
        return example

    train_data = train_data.map(get_length)
    train_data = train_data.filter(lambda x: MIN_CHARS < x['article_length'] < MAX_CHARS)
    train_data = train_data.shuffle(seed=41).select(range(train_size))
    val_data = val_data.shuffle(seed=41).select(range(val_size))

    return train_data, val_data


def preprocess_function(examples, input_col='article', target_col='abstract'):
    inputs = examples[input_col]
    targets = examples[target_col]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


class AbstractiveSummarizer:
    def __init__(self, name, model_path=None):
        if model_path is None:
            self.model = BigBirdPegasusForConditionalGeneration(config=model_config).to(device)
        else:
            self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(model_path).to(device)
        self.train_data = None
        self.val_data = None
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=tokenizer, device=device)
        self.name = name

    def train(self, dataset, checkpoint_dir, input_col='article', target_col='abstract'):
        train_data, val_data = load(dataset)
        self.train_data = train_data.map(lambda x: preprocess_function(x, input_col=input_col, target_col=target_col), batched=True)
        self.val_data = val_data.map(lambda x: preprocess_function(x, input_col=input_col, target_col=target_col), batched=True)
        training_args = Seq2SeqTrainingArguments(
            run_name=self.name,
            output_dir=checkpoint_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            eval_strategy="epoch",
            save_total_limit=3,
            report_to='none'
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            processing_class=tokenizer
        )

        trainer.train()

    def generate(self, text, max_length=500, min_length=30, do_sample=True):
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample, truncation=True)
        return summary[0]['summary_text']

    def evaluate(self, dataset, dataset_name, n=100, text_col='article', target_col='abstract'):
        data = load_dataset(dataset)
        data = data['test'].shuffle(seed=41).select(range(n))
        rouge = Rouge()

        os.makedirs(f'results/{self.name}', exist_ok=True)
        with open(f'results/{self.name}/{dataset_name}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["abstract", "generated", "rouge1-r", "rouge1-p", "rouge1-f",
                             "rouge2-r", "rouge2-p", "rouge2-f", "rougel-r", "rougel-p", "rougel-f"])

            for example in data:
                article = example[text_col]
                abstract = example[target_col]
                generated = self.generate(article)

                scores = rouge.get_scores(generated, abstract, avg=True)

                writer.writerow([abstract, generated, scores['rouge-1']['r'], scores['rouge-1']['p'], scores['rouge-1']['f'],
                                 scores['rouge-2']['r'], scores['rouge-2']['p'], scores['rouge-2']['f'],
                                 scores['rouge-l']['r'], scores['rouge-l']['p'], scores['rouge-l']['f']])

