"""train.py
Fine-tune mBART on local parallel data using HuggingFace Trainer.
"""
import argparse
import os
from datasets import load_metric
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import prepare_dataset
import numpy as np
import evaluate

def preprocess_function(examples, tokenizer, max_length=128):
    inputs = examples['en']
    targets = examples['vi']
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/mbart-en-vi')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    datasets = prepare_dataset(args.data_dir)
    if args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    else:
        from model import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer()

    tokenized = datasets.map(lambda ex: preprocess_function(ex, tokenizer, max_length=args.max_length), batched=True, remove_columns=['en','vi'])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    bleu = evaluate.load('sacrebleu')

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        return {'bleu': result['score']}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_strategy='steps',
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized.get('validation', None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print('Saved model to', args.output_dir)

if __name__ == '__main__':
    main()
