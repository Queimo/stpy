import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb  # Import wandb for logging
from lightning.pytorch.loggers import WandbLogger

# Load dataset
def load_dataset(path):
    data_dict = pickle.load(open(path, "rb"))
    df = data_dict["df"]
    X = data_dict["X"]
    y = df["norm_TSNAK"].values
    torch.manual_seed(2)
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    return X, y

# Define a simple neural network class for PyTorch Lightning
class SimpleNN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims):
        super(SimpleNN, self).__init__()
        layers = []
        in_dim = input_dim
        
        # Add hidden layers dynamically based on hidden_dims
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))  # Linear layer
            layers.append(nn.ReLU())                 # ReLU activation
            in_dim = h_dim  # Set the current layer output as the next input
        
        # Output layer
        layers.append(nn.Linear(in_dim, 1))  # Output layer with 1 unit (regression)

        # Combine all layers into a single sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Use Adam optimizer with learning rate of 0.001
        return optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        X_train, y_train = batch
        y_pred = self(X_train)
        loss = nn.MSELoss()(y_pred, y_train)
        # Log training loss to wandb
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_val, y_val = batch
        y_pred = self(X_val)
        val_loss = nn.MSELoss()(y_pred, y_val)
        # Log validation loss to wandb
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

# Function to setup the model
def setup_nn(input_dim, hidden_dims):
    return SimpleNN(input_dim, hidden_dims)

# Main block
if __name__ == "__main__":
    # Initialize WandB
    # wandb.init(project="protein_fitness_regression", entity="queimo")

    wandb_logger = WandbLogger(project="protein_fitness_regression", entity="queimo")
    # Load data
    path = r".\data_exploration\data_set_dict.pkl"
    X, y = load_dataset(path)

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

    # Setup the model
    model = setup_nn(input_dim=X_train.shape[1], hidden_dims=[1000, 50])

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=1000, 
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='min')],
        logger=wandb_logger
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    trainer.validate(model, val_loader)
