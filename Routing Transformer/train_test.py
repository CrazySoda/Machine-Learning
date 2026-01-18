import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from model import build_routing_transformer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
import math

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SRC_LANG = "en"
TGT_LANG = "de"

SRC_SEQ_LEN = 60
TGT_SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 40

D_MODEL = 256
N = 4
H = 4 # Must be even for our implementation (half local, half routing)
D_FF = 1024
DROPOUT = 0.25
PAD_IDX = 0

NUM_CLUSTERS = 8
WINDOW_SIZE = 10

# ===============================
# LOAD DATASET
# ===============================
print("Downloading dataset...")
dataset = load_dataset("Helsinki-NLP/opus_books", "de-en")
train_data = list(dataset["train"])

train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=42)
print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

# ===============================
# TOKENIZER (BPE)
# ===============================
def train_tokenizer(sentences):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=12000,
        min_frequency=2,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )
    tokenizer.train_from_iterator(sentences, trainer)
    return tokenizer

print("Training tokenizers...")
src_tokenizer = train_tokenizer(s["translation"][SRC_LANG] for s in train_data)
tgt_tokenizer = train_tokenizer(s["translation"][TGT_LANG] for s in train_data)

src_tokenizer.save("src_tokenizer.json")
tgt_tokenizer.save("tgt_tokenizer.json")

SRC_VOCAB_SIZE = src_tokenizer.get_vocab_size()
TGT_VOCAB_SIZE = tgt_tokenizer.get_vocab_size()
print("Vocab sizes:", SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)

# ===============================
# DATASET
# ===============================
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def encode(self, tokenizer, text, max_len):
        ids = tokenizer.encode(text).ids
        ids = [1] + ids + [2]
        ids = ids[:max_len]
        ids += [PAD_IDX] * (max_len - len(ids))
        return torch.tensor(ids)

    def __getitem__(self, idx):
        item = self.data[idx]["translation"]
        return (
            self.encode(src_tokenizer, item[SRC_LANG], SRC_SEQ_LEN),
            self.encode(tgt_tokenizer, item[TGT_LANG], TGT_SEQ_LEN),
        )

    def __len__(self):
        return len(self.data)

train_loader = DataLoader(TranslationDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TranslationDataset(val_data), batch_size=BATCH_SIZE)

# ===============================
# MASKS
# ===============================
def create_src_mask(src):
    return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt):
    seq_len = tgt.size(1)
    padding = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
    nopeak = torch.tril(torch.ones((1, seq_len, seq_len), device=tgt.device)).bool()
    return padding & nopeak

# ===============================
# MODEL
# ===============================
model = build_routing_transformer(
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
    SRC_SEQ_LEN, TGT_SEQ_LEN,
    d_model=D_MODEL, N=N, h=H,
    dropout=DROPOUT, d_ff=D_FF,
    num_clusters=NUM_CLUSTERS,
    window_size=WINDOW_SIZE
).to(DEVICE)

criterion = nn.NLLLoss(ignore_index=PAD_IDX)

# ===============================
# OPTIMIZER + WARMUP
# ===============================
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup=4000):
        self.optimizer = optimizer
        self.step_num = 0
        self.d_model = d_model
        self.warmup = warmup

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup ** -1.5
        )
        for p in self.optimizer.param_groups:
            p["lr"] = lr
        self.optimizer.step()

scheduler = WarmupScheduler(optimizer, D_MODEL)

# ===============================
# BLEU
# ===============================
smooth_fn = SmoothingFunction().method1

def evaluate(loader):
    model.eval()
    bleu = []

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_mask = create_src_mask(src)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_mask = create_tgt_mask(tgt_in)

            enc = model.encode(src, src_mask)
            dec = model.decode(enc, src_mask, tgt_in, tgt_mask)
            out = model.project(dec)

            pred = out.argmax(-1)

            for p, t in zip(pred, tgt_out):
                p = [x for x in p.tolist() if x not in [0,1,2]]
                t = [x for x in t.tolist() if x not in [0,1,2]]
                if len(t) > 0:
                    bleu.append(sentence_bleu([t], p, smoothing_function=smooth_fn))

    return sum(bleu) / len(bleu)

# ===============================
# TRAIN
# ===============================
for epoch in range(EPOCHS):
    model.train()
    total = 0

    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = create_src_mask(src)
        tgt_mask = create_tgt_mask(tgt_in)

        enc = model.encode(src, src_mask)
        dec = model.decode(enc, src_mask, tgt_in, tgt_mask)
        out = model.project(dec)

        loss = criterion(out.reshape(-1, out.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        scheduler.step()

        total += loss.item()

    print(f"Epoch {epoch+1} Loss: {total/len(train_loader):.3f}")
    print("Validation BLEU:", evaluate(val_loader))
