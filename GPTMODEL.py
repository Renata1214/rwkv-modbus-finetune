from datasets import Dataset
import torch
import json
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import DataLoader
from torch.optim import AdamW  # ✅ Use PyTorch's version
import random
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Load data
with open("modbus_dataset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Byte tokenizer: hex string -> list of integers (0–255)
def hex_to_bytes(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

# Format: [BOS] + query_bytes + [SEP] + response_bytes + [EOS]
BOS, SEP, EOS = 256, 257, 258  # extra tokens

def preprocess(example):
    q_bytes = hex_to_bytes(example["query"])
    r_bytes = hex_to_bytes(example["response"])
    return {
        "input_ids": [BOS] + q_bytes + [SEP] + r_bytes + [EOS]
    }

# Tokenize
tokenized_data = [preprocess(entry) for entry in data]

# Build Dataset
dataset = Dataset.from_list(tokenized_data)

# Define GPT config (tiny for start)
vocab_size = 259  # 0–255 bytes + BOS + SEP + EOS
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=128,
    n_embd=256,
    n_layer=4,
    n_head=4
)

model = GPT2LMHeadModel(config)


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [0] * (max_len - len(x["input_ids"])) for x in batch]
    attention_mask = [[1]*len(x["input_ids"]) + [0]*(max_len - len(x["input_ids"])) for x in batch]
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-4)

for epoch in range(3):  # You can increase this
    model.train()
    for batch in train_loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")



def generate_response(query_hex, model, max_length=64):
    model.eval()
    with torch.no_grad():
        query_bytes = hex_to_bytes(query_hex)
        input_ids = torch.tensor([[BOS] + query_bytes + [SEP]], dtype=torch.long).to(device)
        output = model.generate(
            input_ids=input_ids,
            max_length=len(input_ids[0]) + max_length,
            eos_token_id=EOS,
            pad_token_id=0,
        )
        generated = output[0].tolist()
        resp_bytes = generated[generated.index(SEP)+1:]
        resp_bytes = resp_bytes[:resp_bytes.index(EOS)] if EOS in resp_bytes else resp_bytes
        return ''.join(f"{b:02x}" for b in resp_bytes)


input_file = "modbus_dataset.jsonl"
output_file = "modbus_predictions.jsonl"

results = []

# Open input file and predict
with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in tqdm(fin, desc="Generating responses"):
        data = json.loads(line)
        query = data["query"]
        true_response = data["response"]

        try:
            pred_response = generate_response(query, model)
        except Exception as e:
            pred_response = f"ERROR: {str(e)}"

        output = {
            "query": query,
            "true_response": true_response,
            "predicted_response": pred_response
        }

        fout.write(json.dumps(output) + "\n")

output_file = "modbus_predictions.jsonl"

total_percentage = 0
count = 0

with open(output_file, "r") as f:
    for line in f:
        entry = json.loads(line)
        true_bytes = hex_to_bytes(entry["true_response"])
        pred_bytes = hex_to_bytes(entry["predicted_response"]) if "predicted_response" in entry else []

        # Minimum length to avoid division by zero
        if not true_bytes:
            continue

        # Compare up to length of true_bytes
        min_len = min(len(true_bytes), len(pred_bytes))
        matches = sum(1 for i in range(min_len) if true_bytes[i] == pred_bytes[i])
        percentage = (matches / len(true_bytes)) * 100
        total_percentage += percentage
        count += 1

average_correctness = total_percentage / count if count > 0 else 0
print(f"Average Byte-Level Correctness: {average_correctness:.2f}%")



#Trying with Hyperparameters

# Define hyperparameter search space
search_space = {
    "n_embd": [128, 256, 512],
    "n_layer": [2, 4, 6, 8],
    "n_head": [2, 4, 8],
    "learning_rate": [5e-5, 1e-4, 5e-4],
    "batch_size": [4, 8, 16, 32],
    "epochs": [2, 3, 5, 7]
}

# Perform random search to generate trial combinations
num_trials = 5
random_trials = []

for _ in range(num_trials):
    trial = {
        "n_embd": random.choice(search_space["n_embd"]),
        "n_layer": random.choice(search_space["n_layer"]),
        "n_head": random.choice(search_space["n_head"]),
        "learning_rate": random.choice(search_space["learning_rate"]),
        "batch_size": random.choice(search_space["batch_size"]),
        "epochs": random.choice(search_space["epochs"])
    }
    random_trials.append(trial)

# Convert to DataFrame and display or save
df = pd.DataFrame(random_trials)
print(df)

# Optionally, save to CSV
df.to_csv("random_search_trials.csv", index=False)

def bytes_to_hex(byte_list):
    return ''.join(f'{b:02x}' for b in byte_list)

def evaluate_byte_accuracy(model, dataset, device, print_limit=5):
    model.eval()
    correct_total = 0
    total_total = 0
    printed = 0  # Track how many samples we've printed

    with torch.no_grad():
        for item in dataset:
            input_ids = item["input_ids"]
            sep_idx = input_ids.index(SEP)
            query = input_ids[:sep_idx + 1]
            true_response = input_ids[sep_idx + 1:-1]

            input_tensor = torch.tensor([query], dtype=torch.long).to(device)
            output = model.generate(
                input_ids=input_tensor,
                max_length=len(query) + len(true_response) + 5,
                eos_token_id=EOS,
                pad_token_id=0
            )
            generated = output[0].tolist()

            # Extract predicted response
            try:
                pred_response = generated[generated.index(SEP) + 1:]
                if EOS in pred_response:
                    pred_response = pred_response[:pred_response.index(EOS)]
            except ValueError:
                pred_response = []

            # Accuracy calculation
            compare_len = min(len(true_response), len(pred_response))
            correct = sum(1 for i in range(compare_len) if true_response[i] == pred_response[i])
            correct_total += correct
            total_total += len(true_response)

            # Print details (limit to first N)
            if printed < print_limit:
                print(f"\n🧪 Sample {printed+1}")
                print("Query:             ", bytes_to_hex(query[1:-1]))  # skip BOS and SEP
                print("True Response:     ", bytes_to_hex(true_response))
                print("Predicted Response:", bytes_to_hex(pred_response))
                printed += 1

    accuracy = (correct_total / total_total) * 100 if total_total > 0 else 0
    print(f"\n✅ Byte-Level Accuracy: {accuracy:.2f}%")
    return accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

for idx, trial in df.iterrows():
    print(f"\n🔁 Trial {idx+1} - Config: {trial.to_dict()}")

    # Build model
    config = GPT2Config(
        vocab_size=259,
        n_positions=128,
        n_embd=int(trial["n_embd"]),
        n_layer=int(trial["n_layer"]),
        n_head=int(trial["n_head"])
    )

    model = GPT2LMHeadModel(config).to(device)

    # DataLoader
    loader = DataLoader(dataset, batch_size=int(trial["batch_size"]), shuffle=True, collate_fn=collate)




    print(f"Example item from dataset:\n{dataset[0]}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=trial["learning_rate"])

    # Training loop
    for epoch in range(int(trial["epochs"])):
        model.train()
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Evaluation
    accuracy = evaluate_byte_accuracy(model, dataset, device)
    print(f"🧠 Accuracy for Trial {idx+1}: {accuracy:.2f}%")

    results.append({
        **trial.to_dict(),
        "final_loss": loss.item(),
        "byte_accuracy": accuracy
    })

results_df = pd.DataFrame(results)
results_df.to_csv("trial_results.csv", index=False)
print("✅ Best trial based on accuracy:")
print(results_df.sort_values("byte_accuracy", ascending=False).iloc[0])


# Load all trials
results_df = pd.read_csv("trial_results.csv")

# Get best config by byte-level accuracy
best_trial = results_df.sort_values("byte_accuracy", ascending=False).iloc[0]

# Extract and cast
config_dict = {
    "n_embd": int(best_trial["n_embd"]),
    "n_layer": int(best_trial["n_layer"]),
    "n_head": int(best_trial["n_head"])
}
batch_size = int(best_trial["batch_size"])
learning_rate = float(best_trial["learning_rate"])
epochs = int(best_trial["epochs"])

vocab_size = 259  # 0–255 + BOS + SEP + EOS
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=128,
    n_embd=config_dict["n_embd"],
    n_layer=config_dict["n_layer"],
    n_head=config_dict["n_head"]
)

model = GPT2LMHeadModel(config).to(device)


# Load and preprocess
with open("modbus_dataset_test.jsonl", "r") as f:
    test_data = [json.loads(line) for line in f]

test_dataset = Dataset.from_list([preprocess(x) for x in test_data])

accuracy = evaluate_byte_accuracy(model, test_dataset, device, print_limit=10)
print(f"\n🎯 Accuracy on Test Set: {accuracy:.2f}%")
