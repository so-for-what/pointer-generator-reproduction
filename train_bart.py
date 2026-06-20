"""
Fine-tune facebook/bart-base on CNN/Daily Mail for summarization.
"""
import os
import sys
import json
import torch
import gc
import numpy as np
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration, BartTokenizerFast,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
    TrainerCallback
)
from rouge_score import rouge_scorer

# Config
MODEL_NAME = "facebook/bart-base"
MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 142
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 3e-5
EPOCHS = 3
USE_SUBSET = 50000  # 50000 for quick results, 0 for full dataset

device = torch.device('cuda')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)

# Load dataset
print("Loading CNN/Daily Mail...")
dataset = load_dataset('cnn_dailymail', '3.0.0')
if USE_SUBSET > 0:
    dataset['train'] = dataset['train'].select(range(min(USE_SUBSET, len(dataset['train']))))
    dataset['validation'] = dataset['validation'].select(range(min(1000, len(dataset['validation']))))

print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

def preprocess(examples):
    inputs = [doc for doc in examples['article']]
    targets = [high for high in examples['highlights']]
    
    model_inputs = tokenizer(
        inputs, max_length=MAX_SOURCE_LEN, truncation=True, padding=False
    )
    
    labels = tokenizer(
        text_target=targets, max_length=MAX_TARGET_LEN, truncation=True, padding=False
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print("Tokenizing...")
tokenized = dataset.map(
    preprocess, batched=True,
    remove_columns=dataset['train'].column_names,
)

# Load model
print("Loading BART-base...")
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# ROUGE evaluation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Replace -100 with pad_token_id in both predictions and labels
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Clip any out-of-range values
    max_id = tokenizer.vocab_size
    predictions = np.clip(predictions, 0, max_id - 1)
    labels = np.clip(labels, 0, max_id - 1)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    scores = {t: {'p': [], 'r': [], 'f': []} for t in ['rouge1', 'rouge2', 'rougeL']}
    for pred, ref in zip(decoded_preds, decoded_labels):
        result = scorer.score(ref, pred)
        for t in scores:
            scores[t]['p'].append(result[t].precision)
            scores[t]['r'].append(result[t].recall)
            scores[t]['f'].append(result[t].fmeasure)
    
    return {
        'rouge1': np.mean(scores['rouge1']['f']),
        'rouge2': np.mean(scores['rouge2']['f']),
        'rougeL': np.mean(scores['rougeL']['f']),
    }

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir='./bart-summarization',
    eval_strategy='steps',
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=4,
    fp16=True,
    dataloader_num_workers=0,
    warmup_steps=500,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='rougeL',
    greater_is_better=True,
    report_to='none',
    remove_unused_columns=False,
)

# Custom callback to print GPU memory
class MemoryCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 100 == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU mem: {mem:.2f} GB", flush=True)
            torch.cuda.reset_peak_memory_stats()

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[MemoryCallback()],
)

print("Starting training...")
trainer.train()

# Final evaluation
print("\n=== Final Evaluation ===")
results = trainer.evaluate(tokenized['test'].select(range(500)))
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

# Save model
trainer.save_model('./bart-summarization/final')
