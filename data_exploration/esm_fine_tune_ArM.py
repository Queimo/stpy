# Import necessary libraries
from transformers import AutoTokenizer, EsmModel, AutoConfig
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np  

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# 1. Load the model and tokenizer
# model_path = "facebook/esm2_t33_650M_UR50D"
model_name = "esm2_t6_8M_UR50D"
model_path = "facebook/" + model_name
tokenizer = AutoTokenizer.from_pretrained(model_path)

import pickle
def load_dataset(path, prefilter=True):
    data_dict = pickle.load(open(path, "rb"))
    df = data_dict["df"]
    if prefilter:
        df = df[df["big_OD"]]
    else:
        pass
    
    y = df["norm_TSNAK"].values
    variants = df["seq"].values
    sasa_colums = ['s111', 's112', 's118', 's119', 's121', 'sasa_sum']
    X_sasa = df[sasa_colums].values
    X_ddg = df[["ddg_mean", "ddg_std"]].values
    X_rosetta = np.concatenate((X_sasa, X_ddg), axis=1)    
    
    return variants, X_rosetta, y


# Load data
path = r".\data_exploration\data\ArM\data_set_dict.pkl"
X, X_rosetta, y = load_dataset(path, prefilter=True)

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
from transformers import get_scheduler
from transformers import DataCollatorWithPadding

# Model, datasets, and tokenizer are assumed to be pre-defined
data_collator = DataCollatorWithPadding(tokenizer)

# Define the LightningModule
class ESMRegressionModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-1, weight_decay=0.01):
        super().__init__()
        self.model = model.base_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.linear = torch.nn.Linear(320, 1)
        self.loss = torch.nn.MSELoss()

    def forward(self, **batch):
        b = {k: v for k, v in batch.items() if k != "labels"}
        h = self.model(**b)
        y = self.linear(torch.sum(h[0], dim=1)).squeeze()
        return y

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = self.loss(outputs, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = self.loss(outputs, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# Prepare data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Instantiate the LightningModule
model_module = ESMRegressionModule(model)
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="esm-regression")
# Define the Lightning Trainer
trainer = pl.Trainer(
    max_epochs=1,
    log_every_n_steps=10,
    logger=wandb_logger,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
)

# Train the model
trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


#load the fine-tuned model
fine_tuned_model_path = f"./{model_name}-finetuned-localization"

# 11. Inference on validation set
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt

# Load the fine-tuned model
fine_tuned_model_path = f"./{model_name}-finetuned-localization"
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path).to(device)

def scores(model_module, val_dataloader):
    model_module.eval()  # Ensure the model is in evaluation mode
    y_pred = []
    y_true = []

    # Iterate through the validation dataloader
    for batch in val_dataloader:
        # Move inputs to the appropriate device
        labels = batch["labels"].to(model_module.device)

        # Perform inference
        with torch.no_grad():
            outputs = model_module.model(**batch)

        # Collect predictions and true labels
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Calculate the mean squared error and R^2 score
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, r2


mse, r2 = scores(model_module, val_loader)
print(f"Vali Mean Squared Error: {mse}")
print(f"Vali R^2 Score: {r2}")

mse, r2 = scores(model_module, train_loader)
print(f"Train Mean Squared Error: {mse}")
print(f"Train R^2 Score: {r2}")
