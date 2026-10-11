"""CMPSC 448 midterm: CNN and LSTM text authorship classification.

Run from the project root:
    python src/train_models.py

Reads data/llm_responses.csv; writes outputs to results/.
Splits on prompt_id so that the three responses to each prompt stay together.
"""
from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "llm_responses.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
LABELS = ["ChatGPT", "Qwen", "Gemini"]
MODES = ["input", "output", "combined"]
MAX_VOCAB = 15000
MAX_LENGTH = 300
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 16
EPOCHS = 12
LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training on:", DEVICE, flush=True)


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)


def get_text(row, mode: str) -> str:
    if mode == "input":
        return row.LLM_input
    if mode == "output":
        return row.LLM_output
    return row.LLM_input + " <SEP> " + row.LLM_output


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    needed = ["prompt_id", "category", "LLM_name", "LLM_input", "LLM_output"]
    if any(c not in df.columns for c in needed):
        raise ValueError(f"Missing columns: {set(needed) - set(df.columns)}")
    if df[needed].isna().any().any():
        raise ValueError("Dataset has missing values in required columns")
    if len(df) != 300 or df.prompt_id.nunique() != 100:
        print("Note: dataset differs from expected 300 responses / 100 prompts")
    if df.duplicated(["prompt_id", "LLM_name"]).any():
        raise ValueError("Duplicate prompt_id and LLM_name combination")
    if set(df.LLM_name.unique()) != set(LABELS):
        raise ValueError(f"Expected model labels {LABELS}; found {df.LLM_name.unique().tolist()}")
    for pid, group in df.groupby("prompt_id"):
        if set(group.LLM_name) != set(LABELS):
            raise ValueError(f"Missing model for prompt_id {pid}")
        if group.LLM_input.nunique() != 1:
            raise ValueError(f"Different prompts among models for prompt_id {pid}")
    return df


def make_splits(df: pd.DataFrame):
    ids = sorted(df.prompt_id.unique())
    train_ids, holdout_ids = train_test_split(ids, test_size=0.30, random_state=SEED)
    val_ids, test_ids = train_test_split(holdout_ids, test_size=0.50, random_state=SEED)
    pieces = [df[df.prompt_id.isin(part)].copy() for part in (train_ids, val_ids, test_ids)]
    pd.DataFrame({"prompt_id": ids, "split": ["train" if x in train_ids else "val" if x in val_ids else "test" for x in ids]}).to_csv(RESULTS / "prompt_splits.csv", index=False)
    return pieces


def build_vocab(train: pd.DataFrame):
    # Train-only data prevents vocabulary leakage from validation and testing.
    counts = Counter()
    for text in pd.concat([train.LLM_input, train.LLM_output]):
        counts.update(tokens(text))
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
    for token, _ in counts.most_common(MAX_VOCAB - len(vocab)):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


class ResponsesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict, mode: str):
        self.labels = torch.tensor([LABELS.index(label) for label in df.LLM_name], dtype=torch.long)
        encoded = []
        for row in df.itertuples(index=False):
            seq = [vocab.get(t, 1) for t in tokens(get_text(row, mode))[:MAX_LENGTH]]
            seq += [0] * (MAX_LENGTH - len(seq))
            encoded.append(seq)
        self.inputs = torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.inputs[i], self.labels[i]


class CNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.conv = nn.Conv1d(EMBEDDING_DIM, 64, kernel_size=3)
        self.classifier = nn.Linear(64, 3)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        features = torch.relu(self.conv(embedded)).amax(dim=2)
        return self.classifier(features)


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_DIM, 3)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.classifier(h[-1])


def evaluate(model, loader):
    model.eval()
    true, predicted = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE)).argmax(dim=1).cpu().tolist()
            true.extend(y.tolist())
            predicted.extend(out)
    return {
        "accuracy": accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
        "truth": true,
        "predictions": predicted,
    }


def run_experiment(name, model_class, mode, train, val, test, vocab):
    print(f"\n=== {name}: {mode} ===", flush=True)
    train_loader = DataLoader(ResponsesDataset(train, vocab, mode), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ResponsesDataset(val, vocab, mode), batch_size=BATCH_SIZE)
    test_loader = DataLoader(ResponsesDataset(test, vocab, mode), batch_size=BATCH_SIZE)
    torch.manual_seed(SEED)
    model = model_class(len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_epoch, best_weights = -1.0, None, None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        val_scores = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:02d}/{EPOCHS}: loss={avg_loss:.4f} val_f1={val_scores['macro_f1']:.3f}", flush=True)
        history.append({"epoch": epoch, "training_loss": avg_loss, "validation_f1": val_scores["macro_f1"]})
        if val_scores["macro_f1"] > best_f1:
            best_f1 = val_scores["macro_f1"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    score = evaluate(model, test_loader)
    print(f"TEST: accuracy={score['accuracy']:.3f}, macro F1={score['macro_f1']:.3f}", flush=True)
    prefix = f"{name}_{mode}"
    pd.DataFrame(history).to_csv(RESULTS / f"{prefix}_history.csv", index=False)
    (RESULTS / f"{prefix}_classification_report.txt").write_text(
        classification_report(score["truth"], score["predictions"], labels=range(3), target_names=LABELS, zero_division=0)
    )
    matrix = confusion_matrix(score["truth"], score["predictions"], labels=range(3))
    disp = ConfusionMatrixDisplay(matrix, display_labels=LABELS)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name}: {mode}")
    plt.tight_layout()
    plt.savefig(RESULTS / f"{prefix}_confusion.png", dpi=150)
    plt.close()
    # Predictions make it possible to audit the mistakes by prompt.
    pd.DataFrame({
        "prompt_id": test.prompt_id.to_numpy(),
        "actual": [LABELS[x] for x in score["truth"]],
        "predicted": [LABELS[x] for x in score["predictions"]],
    }).to_csv(RESULTS / f"{prefix}_predictions.csv", index=False)
    return {"model": name, "input_type": mode, "best_epoch": best_epoch,
            "validation_macro_f1": best_f1, "test_accuracy": score["accuracy"],
            "test_macro_f1": score["macro_f1"]}


def main():
    df = load_data()
    print("Rows:", len(df), "| Prompts:", df.prompt_id.nunique(), flush=True)
    print(df.LLM_name.value_counts().to_string(), flush=True)
    train, val, test = make_splits(df)
    print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}", flush=True)
    vocab = build_vocab(train)
    print("Training vocabulary size:", len(vocab), flush=True)
    (RESULTS / "experiment_settings.json").write_text(json.dumps({
        "seed": SEED, "max_length": MAX_LENGTH, "max_vocab": MAX_VOCAB,
        "vocabulary_size": len(vocab), "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM, "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS, "learning_rate": LR,
        "split": "70/15/15 by prompt_id", "device": str(DEVICE),
    }, indent=2))
    rows = []
    for model_name, cls in [("CNN", CNN), ("LSTM", LSTM)]:
        for mode in MODES:
            rows.append(run_experiment(model_name, cls, mode, train, val, test, vocab))
            pd.DataFrame(rows).to_csv(RESULTS / "experiment_results.csv", index=False)
    print("\nDONE! Results saved in:", RESULTS, flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()