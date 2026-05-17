"""
Scaffold to fine-tune a seq2seq denoising model (mT5/mBART/VietT5) on generated pairs.

Quick usage (example):
python scripts/finetune_mt5.py --train_file data/denoise_pairs.jsonl --model_name google/mt5-small --output_dir outputs/denoiser

This is a minimal, ready-to-edit scaffold and assumes a GPU environment and the packages
in requirements-denoiser.txt are installed.
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def tokenize_fn(examples, tokenizer, max_input_len=128, max_target_len=128):
    inputs = examples['noisy']
    targets = examples['clean']
    model_inputs = tokenizer(inputs, max_length=max_input_len, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_len, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def main(train_file: str, model_name: str, output_dir: str, epochs: int, batch_size: int):
    ds = load_dataset('json', data_files={'train': train_file})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenized = ds['train'].map(lambda ex: tokenize_fn(ex, tokenizer), batched=True, remove_columns=ds['train'].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        predict_with_generate=True,
        logging_steps=100,
        save_total_limit=3,
        num_train_epochs=epochs,
        fp16=True,
        save_strategy='epoch',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_file', required=True)
    p.add_argument('--model_name', default='google/mt5-small')
    p.add_argument('--output_dir', default='outputs/denoiser')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=8)
    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args.train_file, args.model_name, args.output_dir, args.epochs, args.batch_size)
