# you have to run split first to access the correct files here
# this is just an example to give a guide for a starting point, HIGHLY recommend make your own version
# question why things work the way they do

import argparse, os
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a classifier on Adobe comments.")
    p.add_argument("--csv", required=True, help="Train CSV path")
    p.add_argument("--val", required=True, help="Validation CSV path")
    p.add_argument("--test", required=True, help="Test CSV path")
    p.add_argument("--text-col", default="cleaned_comment_for_sentiment", help="Text column name")
    p.add_argument("--label-col", default="sentiment_label", help="Label column name (sentiment_label or engagement_label)")
    p.add_argument("--model", default="distilroberta-base", help="HF model name (e.g., distilroberta-base, roberta-base, bert-base-uncased)")
    p.add_argument("--outdir", default="models/run", help="Where to save model + tokenizer")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()

def load_df(path, text_col, label_col):
    df = pd.read_csv(path).dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    return df

def build_label_maps(series):
    classes = sorted(series.unique().tolist())
    label2id = {c:i for i,c in enumerate(classes)}
    id2label = {i:c for c,i in label2id.items()}
    return label2id, id2label

def df_to_hfds(df, text_col, label_col, label2id):
    tmp = df[[text_col, label_col]].copy()
    tmp["label"] = tmp[label_col].map(label2id)
    return Dataset.from_pandas(tmp[[text_col, "label"]])

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    log_dir = os.path.join("logs", os.path.basename(args.outdir.strip("/")))
    os.makedirs(log_dir, exist_ok=True)

    # Load data
    train_df = load_df(args.csv, args.text_col, args.label_col)
    val_df   = load_df(args.val, args.text_col, args.label_col)
    test_df  = load_df(args.test, args.text_col, args.label_col)

    # Label maps from TRAIN (avoid unseen labels leakage)
    label2id, id2label = build_label_maps(train_df[args.label_col])
    num_labels = len(label2id)
    if num_labels < 2:
        raise ValueError(f"Need >=2 classes in {args.label_col}, got: {list(label2id.keys())}")

    # Convert to HF datasets
    train_ds = df_to_hfds(train_df, args.text_col, args.label_col, label2id)
    val_ds   = df_to_hfds(val_df,   args.text_col, args.label_col, label2id)
    test_ds  = df_to_hfds(test_df,  args.text_col, args.label_col, label2id)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    def tok_fn(batch):
        return tok(batch[args.text_col], truncation=True, padding=True, max_length=args.max_length)
    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds   = val_ds.map(tok_fn, batched=True)
    test_ds  = test_ds.map(tok_fn, batched=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Metrics
    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    # Training args
    targs = TrainingArguments(
        output_dir=args.outdir,
        learning_rate=args.lr,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=log_dir,
        report_to=[],  # disable WandB, etc.
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Evaluate on test
    test_metrics = trainer.evaluate(test_ds)
    print("\n=== TEST METRICS ===")
    for k, v in test_metrics.items():
        if k.startswith("eval_"):
            print(f"{k}: {v}")

    # Detailed class report
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=1)
    print("\nClass report on TEST:\n")
    print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(num_labels)]))

    # Save
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)
    print(f"\n✅ Saved to: {args.outdir}")
    print(f"Labels: {label2id}")
    print(f"Text: {args.text_col} | Label: {args.label_col} | Model: {args.model}")

if __name__ == "__main__":
    main()
