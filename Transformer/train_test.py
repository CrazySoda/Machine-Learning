import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from model import build_transformer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SRC_LANG = "en"
TGT_LANG = "de"

SRC_SEQ_LEN = 50
TGT_SEQ_LEN = 50
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

D_MODEL = 512
N = 6
H = 8
D_FF = 2048
DROPOUT = 0.1
PAD_IDX = 0

# ===============================
# LOAD DATASET
# ===============================
print("Downloading dataset...")
dataset = load_dataset("Helsinki-NLP/opus_books", "de-en")
train_data = list(dataset["train"])

# Split 95% train / 5% validation
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=42)
print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

# ===============================
# TOKENIZER TRAINING
# ===============================
def train_tokenizer(sentences):
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
        min_frequency=2
    )
    tokenizer.train_from_iterator(sentences, trainer)
    return tokenizer

print("Training tokenizers...")
src_tokenizer = train_tokenizer(
    s["translation"][SRC_LANG] for s in train_data
)
tgt_tokenizer = train_tokenizer(
    s["translation"][TGT_LANG] for s in train_data
)

# Save tokenizers
src_tokenizer.save("src_tokenizer.json")
tgt_tokenizer.save("tgt_tokenizer.json")
print("Tokenizers saved!")

SRC_VOCAB_SIZE = src_tokenizer.get_vocab_size()
TGT_VOCAB_SIZE = tgt_tokenizer.get_vocab_size()

# ===============================
# DATASET CLASS
# ===============================
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def encode(self, tokenizer, text, max_len):
        tokens = tokenizer.encode(text).ids
        tokens = [1] + tokens + [2]  # <sos> <eos>
        tokens = tokens[:max_len]
        tokens += [PAD_IDX] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]["translation"]
        src = self.encode(src_tokenizer, item[SRC_LANG], SRC_SEQ_LEN)
        tgt = self.encode(tgt_tokenizer, item[TGT_LANG], TGT_SEQ_LEN)
        return src, tgt

    def __len__(self):
        return len(self.data)

train_loader = DataLoader(TranslationDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TranslationDataset(val_data), batch_size=BATCH_SIZE)

# ===============================
# MASK FUNCTIONS
# ===============================
def create_src_mask(src):
    return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt):
    seq_len = tgt.size(1)
    padding_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
    nopeak_mask = torch.tril(torch.ones((1, seq_len, seq_len), device=tgt.device)).bool()
    return padding_mask & nopeak_mask

# ===============================
# MODEL
# ===============================
model = build_transformer(
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    SRC_SEQ_LEN,
    TGT_SEQ_LEN,
    d_model=D_MODEL,
    N=N,
    h=H,
    dropout=DROPOUT,
    d_ff=D_FF
).to(DEVICE)

criterion = nn.NLLLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===============================
# BLEU helper
# ===============================
smooth_fn = SmoothingFunction().method1

def evaluate(loader):
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    bleu_scores = []

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_mask = create_src_mask(src)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = create_tgt_mask(tgt_input)

            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, src_mask, tgt_input, tgt_mask)
            out = model.project(dec_out)

            predicted_tokens = out.argmax(dim=-1)

            # Token-level accuracy
            mask = (tgt_output != PAD_IDX)
            correct_tokens += (predicted_tokens == tgt_output).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            # BLEU per sentence
            for pred_seq, true_seq in zip(predicted_tokens, tgt_output):
                pred_tokens = [t for t in pred_seq.tolist() if t not in [PAD_IDX, 1, 2]]
                true_tokens = [t for t in true_seq.tolist() if t not in [PAD_IDX, 1, 2]]
                if len(true_tokens) > 0:
                    bleu_scores.append(sentence_bleu([true_tokens], pred_tokens, smoothing_function=smooth_fn))

    token_acc = correct_tokens / total_tokens
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return token_acc, avg_bleu

# ===============================
# TRAIN + VALIDATE LOOP
# ===============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_src_mask(src)
        tgt_mask = create_tgt_mask(tgt_input)

        enc_out = model.encode(src, src_mask)
        dec_out = model.decode(enc_out, src_mask, tgt_input, tgt_mask)
        out = model.project(dec_out)

        loss = criterion(out.reshape(-1, out.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_train_loss:.4f}")

    # Evaluate on validation set after each epoch
    val_token_acc, val_bleu = evaluate(val_loader)
    print(f"Validation Token Accuracy: {val_token_acc*100:.2f}%, Validation BLEU: {val_bleu*100:.2f}%\n")

# ===============================
# SAVE MODEL
# ===============================
torch.save(model.state_dict(), "transformer_en_de.pth")
print("Training complete. Model saved.")
