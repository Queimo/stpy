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
# model_path = "facebook/esm2_t33_650M_UR50D"
model_name = "esm2_t6_8M_UR50D"
model_path = "facebook/" + model_name
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Load and preprocess the dataset
df = pd.read_csv("./data_exploration/data/ProtSpatial/protein_data raw.csv", index_col=0)
df["Plate"] = df["Plate"].str.replace("Plate_", "").astype(int)
X = df["variant"].tolist()
y = df["Fitness"].tolist()

# 3. Split data into training and validation sets
splitter = GroupShuffleSplit(test_size=0.20, n_splits=2, random_state=712)
train_idx, val_idx = next(splitter.split(X, y, X))
X_train, X_val = [X[i] for i in train_idx], [X[i] for i in val_idx]
y_train, y_val = [y[i] for i in train_idx], [y[i] for i in val_idx]

# 4. Create Hugging Face datasets
train_data = {"variant": X_train, "labels": y_train}

val_data = {"variant": X_val, "labels": y_val}
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

# 5. Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["variant"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 6. Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

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
from transformers import AdamW, get_scheduler
from transformers import DataCollatorWithPadding

# Model, datasets, and tokenizer are assumed to be pre-defined
data_collator = DataCollatorWithPadding(tokenizer)

# Define the LightningModule
class ESMRegressionModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-2, weight_decay=0.01):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# Prepare data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Instantiate the LightningModule
model_module = ESMRegressionModule(model)

# Define the Lightning Trainer
trainer = pl.Trainer(
    max_epochs=3,
    log_every_n_steps=10,
    gradient_clip_val=1.0,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.WandbLogger(project="esm-finetuning", entity="queimo"),
    ],
)

# Train the model
trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


#load the fine-tuned model
fine_tuned_model_path = f"./{model_name}-finetuned-localization"

model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path).to(device)

# 11. Inference on validation set
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt

# Load the fine-tuned model
fine_tuned_model_path = f"./{model_name}-finetuned-localization"
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path).to(device)

# too much memory
# res = model(val_dataset["input_ids"], attention_mask=val_dataset["attention_mask"], labels=val_dataset["labels"])


def scores(model, val_dataset):
    y_pred = []
    batch_size = 16  # Set your desired batch size

    for i in range(0, len(val_dataset), batch_size):
        # Select a batch of examples
        input_ids = val_dataset["input_ids"][i:i + batch_size].to(device)
        attention_mask = val_dataset["attention_mask"][i:i + batch_size].to(device)
        labels = val_dataset["labels"][i:i + batch_size].to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Collect predictions
        y_pred.extend(outputs.logits.squeeze().detach().cpu().numpy())

    # Now `y_pred` contains predictions for all examples in the validation set

    y_pred = np.array(y_pred)
    y_true = val_dataset["labels"].numpy()
    # Calculate the mean squared error and R^2 score
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

mse, r2 = scores(model, val_dataset)
print(f"Vali Mean Squared Error: {mse}")
print(f"Vali R^2 Score: {r2}")

mse, r2 = scores(model, train_dataset)
print(f"Train Mean Squared Error: {mse}")
print(f"Train R^2 Score: {r2}")
