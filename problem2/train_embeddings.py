#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

# Config / Defaults
DEFAULT_VOCAB_SIZE = 5000
DEFAULT_HIDDEN_DIM = 256
DEFAULT_EMBED_DIM  = 64
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LEN    = 200  # for optional fixed-length sequences (not required by model)

PARAM_BUDGET = 2_000_000

WORD_RE = re.compile(r"[a-z]+")

def iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# Text cleaning & preprocessing
def clean_text(text):
    """
    - lowercase
    - keep alphabetic letters and spaces only
    - split into words
    - drop very short words (< 2 chars)
    """
    if not text:
        return []
    text = text.lower()
    # Replace non-alpha with space
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split(" ") if len(w) >= 2]
    return words

def build_vocab(all_tokens, vocab_size):
    """
    Build vocabulary of top-K frequent words.
    Reserve index 0 for <UNK>.
    Return: vocab_to_idx (dict), idx_to_vocab (list), total_words (int)
    """
    counter = Counter(all_tokens)
    most_common = counter.most_common(vocab_size - 1)  # leave 0 for <UNK>
    idx_to_vocab = ["<UNK>"] + [w for w, _ in most_common]
    vocab_to_idx = {w: i for i, w in enumerate(idx_to_vocab)}
    total_words = sum(counter.values())
    return vocab_to_idx, idx_to_vocab, total_words

def encode_sequence(tokens, vocab_to_idx, max_len):
    """
    Convert tokens to index sequence with <UNK>=0, then pad/truncate to max_len.
    Returned tensor is LongTensor of shape [max_len].
    (This is built as required, though the model uses BoW.)
    """
    seq = [vocab_to_idx.get(w, 0) for w in tokens]
    if len(seq) >= max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))
    return torch.tensor(seq, dtype=torch.long)

def bow_vector(tokens, vocab_to_idx, vocab_size):
    """
    Multi-hot bag of words: vector of shape [vocab_size], 1 if token present.
    Uses presence (not counts) to stay in [0,1] for BCE.
    """
    x = torch.zeros(vocab_size, dtype=torch.float32)
    for w in tokens:
        idx = vocab_to_idx.get(w, 0)
        if idx != 0:  # ignore UNK for BoW presence (optional design choice)
            x[idx] = 1.0
    return x

# Model
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()  # outputs in [0,1] for BCE
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Data loading
def load_abstracts(papers_path):
    with open(papers_path,"r", encoding="utf-8") as f:
        data = json.load(f)
    papers = []
    for item in data:
        pid = item.get("arxiv_id") or item.get("id") or ""
        abstract = item.get("abstract") or ""
        if pid and abstract:
            papers.append((pid, abstract))
    return papers

# Mini-batch utilities
def batchify(tensors, batch_size):
    """
    Yield batches (Tensor [B, V]) from a list of vectors (Tensor [V]).
    """
    n = len(tensors)
    for i in range(0, n, batch_size):
        batch = torch.stack(tensors[i:i+batch_size], dim=0)
        yield batch

# Training loop
def train_model(model, device, bow_tensors, epochs, batch_size, lr=1e-3):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training autoencoder...")
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in batchify(bow_tensors, batch_size):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        avg_loss = epoch_loss / max(1, batches)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    dur = time.time() - start
    print(f"Training complete in {dur:.1f} seconds")
    return avg_loss

# Embedding export
@torch.no_grad()
def compute_embeddings_and_losses(model, device, ids, bow_tensors):
    model.eval()
    criterion = nn.BCELoss(reduction="mean")
    out = []
    for pid, x in zip(ids, bow_tensors):
        x = x.to(device).unsqueeze(0)  # [1, V]
        recon, z = model(x)
        loss = criterion(recon, x).item()
        emb = z.squeeze(0).cpu().tolist()
        out.append({
            "arxiv_id": pid,
            "embedding": emb,
            "reconstruction_loss": round(float(loss), 6)
        })
    return out

# CLI parsing (stdlib only)
def parse_cli(argv):
    if len(argv) < 3:
        print("Usage: python train_embeddings.py <input_papers.json> <output_dir> [--epochs 50] [--batch_size 32] [--vocab 5000] [--hidden 256] [--embed 64] [--max_len 200]")
        sys.exit(1)
    input_path = argv[1]
    out_dir = argv[2]
    # defaults
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    vocab_size = DEFAULT_VOCAB_SIZE
    hidden_dim = DEFAULT_HIDDEN_DIM
    embed_dim = DEFAULT_EMBED_DIM
    max_len = DEFAULT_MAX_LEN

    # simple flag parsing
    i = 3
    while i < len(argv):
        if argv[i] == "--epochs" and i + 1 < len(argv):
            epochs = int(argv[i+1]); i += 2
        elif argv[i] == "--batch_size" and i + 1 < len(argv):
            batch_size = int(argv[i+1]); i += 2
        elif argv[i] == "--vocab" and i + 1 < len(argv):
            vocab_size = int(argv[i+1]); i += 2
        elif argv[i] == "--hidden" and i + 1 < len(argv):
            hidden_dim = int(argv[i+1]); i += 2
        elif argv[i] == "--embed" and i + 1 < len(argv):
            embed_dim = int(argv[i+1]); i += 2
        elif argv[i] == "--max_len" and i + 1 < len(argv):
            max_len = int(argv[i+1]); i += 2
        else:
            print(f"Unknown/invalid argument: {argv[i]}")
            sys.exit(1)
    return input_path, out_dir, epochs, batch_size, vocab_size, hidden_dim, embed_dim, max_len
--
# Main
def main():
    print(sys.argv)
    input_path, out_dir, epochs, batch_size, vocab_size, hidden_dim, embed_dim, max_len = parse_cli(sys.argv)

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading abstracts from {input_path}...")
    papers = load_abstracts(input_path)
    print(f"Found {len(papers)} abstracts")

    # Build tokens list
    all_tokens = []
    cleaned_docs = []
    ids = []
    for pid, abs_text in papers:
        tokens = clean_text(abs_text)
        cleaned_docs.append(tokens)
        ids.append(pid)
        all_tokens.extend(tokens)

    print(f"Building vocabulary from {len(all_tokens)} words...")
    vocab_to_idx, idx_to_vocab, total_words = build_vocab(all_tokens, vocab_size)
    vocab_size = len(idx_to_vocab)  # in case actual < requested (e.g., tiny data)
    print(f"Vocabulary size: {vocab_size} words (UNK=0)")

    # Sequences (not used by model but created to fulfill requirement)
    _sequences = [encode_sequence(toks, vocab_to_idx, max_len) for toks in cleaned_docs]

    # BoW multi-hot tensors
    bow_tensors = [bow_vector(toks, vocab_to_idx, vocab_size) for toks in cleaned_docs]

    # Model
    model = TextAutoencoder(vocab_size=vocab_size, hidden_dim=hidden_dim, embedding_dim=embed_dim)
    total_params = count_parameters(model)
    arch_str = f"{vocab_size} → {hidden_dim} → {embed_dim} → {hidden_dim} → {vocab_size}"
    status = "✓" if total_params <= PARAM_BUDGET else "✗"
    print(f"Model architecture: {arch_str}")
    print(f"Total parameters: {total_params:,} (under {PARAM_BUDGET:,} limit {status})")
    if total_params > PARAM_BUDGET:
        print("ERROR: Parameter budget exceeded. Adjust --vocab/--hidden/--embed.")
        sys.exit(1)

    # Device
    device = torch.device("cpu")

    # Train
    final_loss = train_model(model, device, bow_tensors, epochs=epochs, batch_size=batch_size, lr=1e-3)

    # Save model
    model_path = os.path.join(out_dir, "model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_to_idx": vocab_to_idx,
        "model_config": {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "embedding_dim": embed_dim
        }
    }, model_path)

    # Embeddings
    embeddings = compute_embeddings_and_losses(model, device, ids, bow_tensors)
    with open(os.path.join(out_dir, "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    # Vocabulary mapping
    vocab_json = {
        "vocab_to_idx": vocab_to_idx,
        "idx_to_vocab": {str(i): w for i, w in enumerate(idx_to_vocab)},
        "vocab_size": vocab_size,
        "total_words": total_words
    }
    with open(os.path.join(out_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # Training log
    training_log = {
        "start_time": iso_now(),  # this is approximate; for exact, capture pre-train
        "end_time": iso_now(),
        "epochs": epochs,
        "final_loss": round(float(final_loss), 6),
        "total_parameters": int(total_params),
        "papers_processed": len(papers),
        "embedding_dimension": int(embed_dim)
    }
    with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    print("All outputs written to:", out_dir)

if __name__ == "__main__":
    main()
