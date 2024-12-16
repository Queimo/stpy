# Import necessary libraries
from transformers import AutoTokenizer, EsmModel, AutoConfig
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
import pandas as pd
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# 1. Load the model and tokenizer
# model_name = "esm2_t33_650M_UR50D"
model_name = "esm2_t6_8M_UR50D"
model_path = "facebook/" + model_name
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Load and preprocess the dataset
df = pd.read_csv("./data_exploration/data/ProtSpatial/protein_data raw.csv", index_col=0)
df["Plate"] = df["Plate"].str.replace("Plate_", "").astype(int)
X = df["variant"].tolist()
y = df["Fitness"].tolist()

# 4. Create Hugging Face datasets
train_data = {"variant": X, "labels": y}
train_dataset = Dataset.from_dict(train_data)

# 5. Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["variant"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)

# 6. Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 7. Create DataLoaders
batch_size = 8

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 1
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, problem_type="regression").to(device)
# 10. Model Fine-Tuning Placeholder
# You can now use the prepared train_loader and val_loader to train your model

#freeze base model
for param in model.base_model.parameters():
    param.requires_grad = False

# print number of trainable parameters with transformers library
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import get_scheduler
from transformers import DataCollatorWithPadding

# Model, datasets, and tokenizer are assumed to be pre-defined
data_collator = DataCollatorWithPadding(tokenizer)

# Prepare data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
# Load the fine-tuned model

emb_list = []


# open csv file
import csv

with open(f"./data_exploration/data/ProtSpatial/{model_name}_embeddings.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["variant", "embedding"])
    # for batch in train_loader:


    for variant in df["variant"].unique():
        batch = tokenizer(variant, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        # Move inputs to the appropriate device

        # Perform inference
        with torch.no_grad():
            hss = model(**batch, output_hidden_states=True).hidden_states
            for hs in hss:
                # emb_dict[variant] = str(list(torch.mean(hs, dim=1).squeeze().cpu().numpy()))
                writer.writerow([variant, str(list(torch.mean(hs, dim=1).squeeze().cpu().numpy()))])      
                     


