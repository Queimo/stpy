# Import necessary libraries
from transformers import AutoTokenizer, EsmModel
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
import pandas as pd
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 1. Load the model and tokenizer
model_path = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = EsmModel.from_pretrained(model_path)

# 2. Load and preprocess the dataset
df = pd.read_csv("./data_exploration/data/ProtSpatial/protein_data raw.csv", index_col=0)
df["Plate"] = df["Plate"].str.replace("Plate_", "").astype(int)
X = df["variant"].tolist()[:100]
y = df["Fitness"].tolist()[:100]

# 3. Split data into training and validation sets
splitter = GroupShuffleSplit(test_size=0.20, n_splits=2, random_state=712)
train_idx, val_idx = next(splitter.split(X, y, X))
X_train, X_val = [X[i] for i in train_idx], [X[i] for i in val_idx]
y_train, y_val = [y[i] for i in train_idx], [y[i] for i in val_idx]

# 4. Create Hugging Face datasets
train_data = {"variant": X_train, "Fitness": y_train}
val_data = {"variant": X_val, "Fitness": y_val}
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

# 5. Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["variant"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 6. Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "Fitness"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "Fitness"])

# 7. Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # 8. Check DataLoader and Model Integration
# for batch in train_loader:
#     input_ids = batch["input_ids"]
#     attention_mask = batch["attention_mask"]
#     labels = batch["Fitness"]

#     # Forward pass (example)
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     print(f"Output shape: {outputs.last_hidden_state.shape}")  # Check output dimensions
#     break

# 9. (Optional) Save tokenized datasets for future use
train_dataset.save_to_disk("./train_dataset")
val_dataset.save_to_disk("./val_dataset")

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 1
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to(device)
# 10. Model Fine-Tuning Placeholder
# You can now use the prepared train_loader and val_loader to train your model

model_name = model_path.split("/")[-1]
batch_size = 8

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.distributed.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 1)

args = TrainingArguments(
    f"{model_name}-finetuned-localization",
    eval_strategy= "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    push_to_hub=False,
    report_to="none",  # Disable logging to external services like WandB
    no_cuda=False,     # Ensure CUDA is used if available
    local_rank=-1, 
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset.to(device),
    eval_dataset=val_dataset.to(device),
    tokenizer=tokenizer,
)

trainer.train()
