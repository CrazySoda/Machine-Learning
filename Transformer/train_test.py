#normal attention

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os

from datasets import load_dataset
from transformers import BertModel, BertTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm

from model_updated import build_transformer, multihead_attentionblock
from gpu_profiler import GPUProfiler

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# CONFIG
# =====================================================
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 30
LR = 2e-5
D_MODEL = 768
N_LAYERS = 12
N_HEADS = 12
D_FF = 3072

CHECKPOINT_DIR = "./checkpoints_normal"
CHECKPOINT_INTERVAL_SEC = 3600   # save every 1 hour
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =====================================================
# PROFILER
# =====================================================
import model_updated as _model_module

profiler = GPUProfiler(logfile="gpu_profile_normal.log", reset=True, tensorboard_logdir="./tb_logs_normal")
_model_module.profiler = profiler

# =====================================================
# LOAD IMDB DATASET
# =====================================================
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(dataset["test"],  batch_size=BATCH_SIZE)

# =====================================================
# BUILD TRANSFORMER
# =====================================================
transformer_model = build_transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=2,
    src_seq_len=MAX_LEN,
    tgt_seq_len=MAX_LEN,
    d_model=D_MODEL,
    N=N_LAYERS,
    h=N_HEADS,
    d_ff=D_FF,
    attention_type="standard",
).to(device)


# =====================================================
# CLASSIFIER WITH CLS POOLING
# =====================================================
class Classifier(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(D_MODEL, 2)
        self.src_embed_layernorm = nn.LayerNorm(D_MODEL)
        self.src_embed_dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        x = self.transformer.src_embed(input_ids)
        x = x + self.transformer.src_pos.pe[:, : x.shape[1], :]
        x = self.src_embed_layernorm(x)
        x = self.src_embed_dropout(x)
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        x = self.transformer.encoder(x, mask)
        cls_representation = x[:, 0, :]
        cls_representation = self.dropout(cls_representation)
        return self.classifier(cls_representation)


classifier_model = Classifier(transformer_model).to(device)

# =====================================================
# CHECKPOINT HELPERS
# =====================================================
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
METRICS_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "metrics_checkpoint.json")


def save_checkpoint(model, optimizer, epoch, batch_idx, epoch_metrics, last_checkpoint_time):
    tmp_path = CHECKPOINT_PATH + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "last_checkpoint_time": last_checkpoint_time,
        },
        tmp_path,
    )
    os.replace(tmp_path, CHECKPOINT_PATH)   # atomic replace to avoid corrupt files

    with open(METRICS_CHECKPOINT_PATH, "w") as f:
        json.dump(epoch_metrics, f, indent=2)

    print(f"\n  [Checkpoint saved at epoch {epoch+1}, batch {batch_idx}]")


def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, 0, None   # start_epoch, start_batch, last_checkpoint_time

    print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch_metrics_loaded = {}
    if os.path.exists(METRICS_CHECKPOINT_PATH):
        with open(METRICS_CHECKPOINT_PATH) as f:
            epoch_metrics_loaded = json.load(f)

    print(f"  Resumed from epoch {ckpt['epoch']+1}, batch {ckpt['batch_idx']}")
    return ckpt["epoch"], ckpt["batch_idx"], ckpt["last_checkpoint_time"], epoch_metrics_loaded

# =====================================================
# LOAD BERT & TRANSFER WEIGHTS  (only if no checkpoint)
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier_model.parameters(), lr=LR)

resume_result = load_checkpoint(classifier_model, optimizer)

if len(resume_result) == 4:
    start_epoch, resume_batch, last_checkpoint_time, epoch_metrics = resume_result
    # If we finished mid-epoch, advance to next epoch to re-run it cleanly
    # (mid-epoch resume requires DataLoader state saving which is complex;
    #  simplest safe approach is to redo the interrupted epoch)
    if resume_batch > 0:
        print(f"  Note: interrupted mid-epoch — re-running epoch {start_epoch+1} from scratch.")
    else:
        start_epoch = start_epoch  # completed epoch, start from next
else:
    start_epoch = 0
    last_checkpoint_time = None
    epoch_metrics = {
        "training_time_per_epoch_sec": [],
        "peak_memory_per_epoch_MB": [],
        "loss_per_epoch": [],
        "accuracy_per_epoch": [],
        "layer_profiles_per_epoch": [],
    }

    # Transfer BERT weights only on fresh start
    print("Loading pretrained BERT...")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)

    print("Transferring weights...")

    classifier_model.transformer.src_embed.embedding.weight.data.copy_(
        bert.embeddings.word_embeddings.weight.data
    )
    classifier_model.transformer.src_pos.pe[:, :MAX_LEN, :].data.copy_(
        bert.embeddings.position_embeddings.weight.data.unsqueeze(0)
    )
    classifier_model.src_embed_layernorm.weight.data.copy_(bert.embeddings.LayerNorm.weight.data)
    classifier_model.src_embed_layernorm.bias.data.copy_(bert.embeddings.LayerNorm.bias.data)

    for i, layer in enumerate(bert.encoder.layer):
        our_layer = classifier_model.transformer.encoder.layers[i]

        our_layer.self_attention_block.w_q.weight.data.copy_(layer.attention.self.query.weight.data)
        our_layer.self_attention_block.w_q.bias.data.copy_(layer.attention.self.query.bias.data)
        our_layer.self_attention_block.w_k.weight.data.copy_(layer.attention.self.key.weight.data)
        our_layer.self_attention_block.w_k.bias.data.copy_(layer.attention.self.key.bias.data)
        our_layer.self_attention_block.w_v.weight.data.copy_(layer.attention.self.value.weight.data)
        our_layer.self_attention_block.w_v.bias.data.copy_(layer.attention.self.value.bias.data)
        our_layer.self_attention_block.w_o.weight.data.copy_(layer.attention.output.dense.weight.data)
        our_layer.self_attention_block.w_o.bias.data.copy_(layer.attention.output.dense.bias.data)
        our_layer.self_attention_block.dropout.p = layer.attention.output.dropout.p

        our_layer.feed_forward_block.linear_1.weight.data.copy_(layer.intermediate.dense.weight.data)
        our_layer.feed_forward_block.linear_1.bias.data.copy_(layer.intermediate.dense.bias.data)
        our_layer.feed_forward_block.linear_2.weight.data.copy_(layer.output.dense.weight.data)
        our_layer.feed_forward_block.linear_2.bias.data.copy_(layer.output.dense.bias.data)

        our_layer.residual_connections[0].norm.alpha.data.copy_(layer.attention.output.LayerNorm.weight.data)
        our_layer.residual_connections[0].norm.bias.data.copy_(layer.attention.output.LayerNorm.bias.data)
        our_layer.residual_connections[1].norm.alpha.data.copy_(layer.output.LayerNorm.weight.data)
        our_layer.residual_connections[1].norm.bias.data.copy_(layer.output.LayerNorm.bias.data)

    print("Weight transfer complete.")

    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialise checkpoint timer if not loaded from checkpoint
if last_checkpoint_time is None:
    last_checkpoint_time = time.time()

# =====================================================
# HELPER: evaluate on test set
# =====================================================
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, mask)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels, all_probs


# =====================================================
# TRAIN LOOP
# =====================================================
for epoch in range(start_epoch, EPOCHS):
    classifier_model.train()
    total_loss = 0.0

    profiler.start_epoch_collection()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    epoch_start = time.time()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(loop):
        input_ids = batch["input_ids"].to(device)
        mask      = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = classifier_model(input_ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        # ---- Hourly checkpoint ----
        now = time.time()
        if now - last_checkpoint_time >= CHECKPOINT_INTERVAL_SEC:
            save_checkpoint(
                classifier_model, optimizer,
                epoch, batch_idx + 1,
                epoch_metrics, now,
            )
            last_checkpoint_time = now

    epoch_time = time.time() - epoch_start

    peak_mem_MB = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem_MB = torch.cuda.max_memory_allocated() / 1024 ** 2

    avg_loss = total_loss / len(train_loader)
    layer_summary = profiler.end_epoch_collection()

    # ---- Per-epoch accuracy on test set ----
    epoch_acc, _, _, _ = evaluate(classifier_model, test_loader)
    classifier_model.train()

    # Store metrics
    epoch_metrics["training_time_per_epoch_sec"].append(round(epoch_time, 3))
    epoch_metrics["peak_memory_per_epoch_MB"].append(round(peak_mem_MB, 2))
    epoch_metrics["loss_per_epoch"].append(round(avg_loss, 6))
    epoch_metrics["accuracy_per_epoch"].append(round(epoch_acc, 6))
    epoch_metrics["layer_profiles_per_epoch"].append(layer_summary)

    # ---- End-of-epoch checkpoint (always save after completing an epoch) ----
    last_checkpoint_time = time.time()
    save_checkpoint(
        classifier_model, optimizer,
        epoch + 1, 0,          # batch_idx=0 signals epoch boundary
        epoch_metrics, last_checkpoint_time,
    )

    print(f"Epoch {epoch+1}  Loss: {avg_loss:.4f}  Acc: {epoch_acc:.4f}  "
          f"Time: {epoch_time:.1f}s  Peak VRAM: {peak_mem_MB:.1f} MB")
    print("  Per-layer summary:")
    for name, stats in layer_summary.items():
        print(f"    {name:<25} mean_time={stats['mean_time_ms']:.2f} ms  "
              f"mean_mem_delta={stats['mean_mem_delta_MB']:.2f} MB  "
              f"calls={stats['calls']}")

# =====================================================
# SAVE TRAINING METRICS
# =====================================================
GPUProfiler.save_metrics(epoch_metrics, "normal_attention_training_metrics.json")

# =====================================================
# FINAL EVALUATION
# =====================================================
final_acc, all_preds, all_labels, all_probs = evaluate(classifier_model, test_loader)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary"
)
roc_auc     = roc_auc_score(all_labels, all_probs)
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\n========== IMDB RESULTS (Normal Attention) ==========")
print(f"Accuracy  : {final_acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

eval_metrics = {
    "accuracy":         round(final_acc, 6),
    "precision":        round(precision, 6),
    "recall":           round(recall, 6),
    "f1":               round(f1, 6),
    "roc_auc":          round(roc_auc, 6),
    "confusion_matrix": conf_matrix.tolist(),
}
GPUProfiler.save_metrics(eval_metrics, "normal_attention_eval_metrics.json")