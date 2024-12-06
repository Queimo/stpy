import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch
import matplotlib.pyplot as plt

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

# Main block
if __name__ == "__main__":
    # Load data
    path = r".\data_exploration\data_set_dict.pkl"
    X, y = load_dataset(path)

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define RandomForestRegressor
    rf_model = RandomForestRegressor(
        n_estimators=100,  # Number of trees
        max_depth=None,    # Allow trees to grow fully
        random_state=42,   # For reproducibility
        n_jobs=-1          # Use all available cores
    )

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    print(f"Training R^2: {train_r2:.4f}")
    print(f"Validation R^2: {val_r2:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Validation MSE: {val_mse:.4f}")
    
    # Feature Importance
    feature_importances = rf_model.feature_importances_
    top_idx = np.argsort(feature_importances)[::-1][:20]  # Top 20 features

    # Print and plot top 20 features
    print("Top feature importances:")
    for idx in top_idx:
        print(f"Feature {idx}: Importance {feature_importances[idx]:.8f}")
        plt.bar(idx, feature_importances[idx], color="red")
    
    # Finalize plot
    plt.title("Top Feature Importances")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.show()
