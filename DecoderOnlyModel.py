import json
from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import time


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# BOS = Beginning of Sequence → marks the start of a Modbus message.
# SEP = Separator → marks the boundary between the query and the response.
# EOS = End of Sequence → marks the end of the full sequence.
# PAD = Padding token → used to pad shorter sequences so they fit in a batch.
# VOCAB_SIZE = 260 → bytes (0–255) + 4 special tokens (256–259)

# Constants
BOS, SEP, EOS, PAD = 256, 257, 258, 259
VOCAB_SIZE = 260

# Load and preprocess dataset
#hexadecimal string into a list of byte values 
def hex_to_bytes(hex_str):
    # 2-character group is treated as a hex byte and converted to decimal
    return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

def preprocess(example):
    q = hex_to_bytes(example["query"])
    r = hex_to_bytes(example["response"])
    return {
        "input_ids": [BOS] + q + [SEP] + r + [EOS]
    }

with open("modbus_dataset_test.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

with open("modbus_dataset.jsonl", "r") as f:
    test_data = [json.loads(line) for line in f]

#turns a list of dictionaries into a Dataset object 
train_dataset = Dataset.from_list([preprocess(d) for d in train_data])
test_dataset = Dataset.from_list([preprocess(d) for d in test_data])

# Model
class DecoderOnlyTransformer(nn.Module):
    # vocab_size = how many distinct tokens we can embed
    #d_model =  size of each embedding vector. 
    #n_heads: number of attention heads. Helps the model focus on different parts of the sequence simultaneously.
    # n_layers: number of transformer layers stacked
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        #Converts each token ID into a dense vector of dimension d_model.
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(512, d_model)
        # Even though we call it "encoder layer", apply causal masks later to make it autoregressive like a decoder.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
            ) for _ in range(n_layers)
        ]) # dim_feedforward = Size of the hidden layer inside the feed-forward network of the Transformer.
        self.norm = nn.LayerNorm(d_model)
        #Maps the output of the model (of shape [batch, seq_len, d_model]) to logits over the vocabulary.
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size() # B is the batch size, T is the sequence length (number of tokens in each input)

        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(x) + self.pos_emb(positions)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device) # Ensures the model can only attend to current and past tokens, not future ones

        for layer in self.layers:
            x = layer(x, src_mask=tgt_mask)  # Applies each encoder layer sequentially

        x = self.norm(x) 
        return self.output(x) # Projects the final hidden states back to vocabulary space: for each token position, the model predicts a probability distribution over the 260 tokens (0–259)


# Collate
def collate(batch):
    #Ensures all sequences are of shape [max_len], Required because Transformers need all sequences in a batch to be the same length
    max_len = max(len(x["input_ids"]) for x in batch)
    padded = [x["input_ids"] + [PAD] * (max_len - len(x["input_ids"])) for x in batch]
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "labels": torch.tensor(padded, dtype=torch.long)
    }

# Sample hyperparameters
def sample_hparams():
    return {
        "d_model": random.choice([128, 256]),
        "n_heads": random.choice([2, 4]),
        "n_layers": random.choice([2, 4]),
        "lr": random.choice([1e-4, 5e-4]),
        "batch_size": random.choice([8, 16]),
        "epochs": 3
    }

# Generate
def generate(model, input_seq, max_len=32):
    model.eval()
    with torch.no_grad():
        #Copies the input sequence so the original is not modified
        seq = input_seq[:]
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long).to(next(model.parameters()).device)
            logits = model(x)
            next_token = logits[0, -1].argmax().item()
            if next_token == EOS:
                break
            seq.append(next_token)
        return seq

# Evaluate
def evaluate(model, dataset, device, print_limit=5):
    correct, total, shown = 0, 0, 0
    for item in dataset:
        tokens = item["input_ids"]
        sep_idx = tokens.index(SEP)
        query = tokens[:sep_idx+1]
        true = tokens[sep_idx+1:-1]

        pred = generate(model, query)
        pred = pred[pred.index(SEP)+1:]
        if EOS in pred:
            pred = pred[:pred.index(EOS)]

        matches = sum(1 for i in range(min(len(true), len(pred))) if true[i] == pred[i])
        correct += matches
        total += len(true)

        if shown < print_limit:
            def tohex(x): return ''.join(f"{b:02x}" for b in x)
            print("\nQuery:             ", tohex(query[1:-1]))
            print("True Response:     ", tohex(true))
            print("Predicted Response:", tohex(pred))
            shown += 1
    acc = 100 * correct / total
    print(f"\n✅ Byte-Level Accuracy: {acc:.2f}%")
    return acc

# Training
def train_transformer(model, dataloader, hparams, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for epoch in range(hparams["epochs"]):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)[:, :-1, :]
            targets = y[:, 1:]

            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Run one trial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hparams = sample_hparams()
model = DecoderOnlyTransformer(VOCAB_SIZE, hparams["d_model"], hparams["n_heads"], hparams["n_layers"])
train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=collate)



train_transformer(model, train_loader, hparams, device)
evaluate(model, test_dataset, device)



def generate_with_timing(model, input_seq, max_len=32):
    model.eval()
    with torch.no_grad():
        seq = input_seq[:]

        start_time = time.time()  # ⏱️ Start timing

        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long).to(next(model.parameters()).device)
            logits = model(x)
            next_token = logits[0, -1].argmax().item()
            if next_token == EOS:
                break
            seq.append(next_token)

        end_time = time.time()  # ⏱️ End timing
        duration = end_time - start_time

        return seq, duration


total_time = 0
num_samples = 0
accurate_tokens = 0
total_tokens = 0

for item in test_dataset:
    tokens = item["input_ids"]
    sep_idx = tokens.index(SEP)
    query = tokens[:sep_idx+1]
    true_response = tokens[sep_idx+1:-1]  # excludes EOS

    pred, inference_time = generate_with_timing(model, query)
    pred = pred[pred.index(SEP)+1:]
    if EOS in pred:
        pred = pred[:pred.index(EOS)]

    # Accuracy calculation
    matches = sum(1 for i in range(min(len(true_response), len(pred))) if true_response[i] == pred[i])
    accurate_tokens += matches
    total_tokens += len(true_response)

    total_time += inference_time
    num_samples += 1

    if num_samples <= 5:
        def tohex(x): return ''.join(f"{b:02x}" for b in x)
        print("\n🧪 Sample", num_samples)
        print("Query:              ", tohex(query[1:-1]))
        print("True Response:      ", tohex(true_response))
        print("Predicted Response: ", tohex(pred))
        print(f"Inference Time:     {inference_time:.6f} seconds")

# 🧠 Summary
avg_time = total_time / num_samples
byte_accuracy = 100 * accurate_tokens / total_tokens

print(f"\n✅ Average Inference Time per Sample: {avg_time:.6f} seconds")
print(f"✅ Byte-Level Accuracy: {byte_accuracy:.2f}%")


