# train_test.py — IMDB Sentiment Classification with Routing Transformer
# Adapted from the standard Transformer's IMDB train_test to use the
# Routing Transformer (sparse multihead attention: local + routing).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm

from model import build_routing_transformer

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# CONFIG
# =====================================================
MAX_LEN = 256          # Shorter than 512 to save memory with routing attention
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4

# Routing Transformer hyperparameters
D_MODEL = 256          # Must be even (split between local & routing heads)
N_LAYERS = 4
N_HEADS = 4            # Must be even (half local, half routing)
D_FF = 1024
DROPOUT = 0.1
NUM_CLUSTERS = 16      # Number of clusters for routing attention
WINDOW_SIZE = 32       # Window size for local attention

# =====================================================
# LOAD IMDB DATASET
# =====================================================
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE)

print(f"Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")

# =====================================================
# BUILD ROUTING TRANSFORMER
# =====================================================
# We only need the encoder for classification, but build_routing_transformer
# creates both encoder + decoder. We'll use only the encoder side.
routing_model = build_routing_transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=2,            # Not used (we use a custom classifier head)
    src_seq_len=MAX_LEN,
    tgt_seq_len=MAX_LEN,
    d_model=D_MODEL,
    N=N_LAYERS,
    h=N_HEADS,
    dropout=DROPOUT,
    d_ff=D_FF,
    num_clusters=NUM_CLUSTERS,
    window_size=WINDOW_SIZE,
).to(device)

# =====================================================
# CLASSIFIER WITH CLS POOLING
# =====================================================
class RoutingClassifier(nn.Module):
    """
    Wraps the Routing Transformer encoder for binary sentiment classification.
    Uses [CLS]-style pooling (first token) → dropout → linear → 2 classes.
    """
    def __init__(self, routing_transformer, d_model):
        super().__init__()
        self.encoder = routing_transformer.encoder
        self.src_embed = routing_transformer.src_embed
        self.src_pos = routing_transformer.src_pos
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_mask):
        # --- Embeddings ---
        x = self.src_embed(input_ids)                       # (B, L, D)
        x = self.src_pos(x)                                 # + positional encoding

        # --- Attention mask  (B,1,1,L) for broadcast ---
        mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # --- Encoder (sparse multihead: local + routing) ---
        x = self.encoder(x, mask)                           # (B, L, D)

        # --- CLS token pooling (first token) ---
        cls_repr = x[:, 0, :]                               # (B, D)
        cls_repr = self.dropout(cls_repr)
        return self.classifier(cls_repr)                    # (B, 2)


classifier_model = RoutingClassifier(routing_model, D_MODEL).to(device)

total_params = sum(p.numel() for p in classifier_model.parameters())
trainable_params = sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)
print(f"\nModel Parameters: {total_params:,} total, {trainable_params:,} trainable")

# =====================================================
# TRAINING SETUP
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier_model.parameters(), lr=LR, weight_decay=0.01)

# Simple linear warmup + cosine decay scheduler
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =====================================================
# TRAIN LOOP
# =====================================================
print(f"\n{'='*60}")
print(f"  Training Routing Transformer on IMDB")
print(f"  Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LR}")
print(f"  d_model: {D_MODEL} | layers: {N_LAYERS} | heads: {N_HEADS}")
print(f"  clusters: {NUM_CLUSTERS} | window: {WINDOW_SIZE}")
print(f"  Device: {device}")
print(f"{'='*60}\n")

for epoch in range(EPOCHS):
    classifier_model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = classifier_model(input_ids, mask)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/total:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1} — Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

# =====================================================
# EVALUATION
# =====================================================
print(f"\n{'='*60}")
print("  Evaluating on IMDB Test Set")
print(f"{'='*60}\n")

classifier_model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    loop = tqdm(test_loader, desc="Testing")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = classifier_model(input_ids, mask)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
roc_auc = roc_auc_score(all_labels, all_probs)
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\n========== IMDB RESULTS (Routing Transformer) ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# =====================================================
# GPU MEMORY SUMMARY
# =====================================================
if torch.cuda.is_available():
    print(f"\n========== GPU MEMORY ==========")
    print(f"Allocated : {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"Reserved  : {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    print(f"Peak      : {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
