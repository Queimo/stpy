import pickle
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
import torch

# Load dataset
def load_dataset(path, prefilter=True):
    data_dict = pickle.load(open(path, "rb"))
    df = data_dict["df"]
    if prefilter:
        X = data_dict["X"][df["big_OD"]]
        y = df["norm_TSNAK"][df["big_OD"]].values
        groups = df["variant"][df["big_OD"]].values
    else:
        X = data_dict["X"]
        y = df["norm_TSNAK"].values
        groups = df["variant"].values
        
    return X, y, groups

# Main block
if __name__ == "__main__":
    # Load data
    path = r".\data_exploration\data_set_dict.pkl"
    X, y, groups = load_dataset(path, prefilter=True)

    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    # Use the indices to split the data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Split data into train and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Standardize y
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    # Define RidgeCV with cross-validation
    ridge_cv = RidgeCV(
        alphas=np.linspace(270, 280, 5),  # Range of alpha values
        cv=5,  # 5-fold cross-validation
        scoring="r2",  # Use R^2 as the scoring metric
          # Store the cross-validation values
    )

    # Fit the model
    ridge_cv.fit(X_train, y_train)

    # Print results
    print(f"Best alpha: {ridge_cv.alpha_}")
    print(f"Training R^2: {ridge_cv.score(X_train, y_train):.4f}")
    print(f"Validation R^2: {ridge_cv.score(X_val, y_val):.4f}")
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    
    topk = 20
    # Print top 20 coefficients
    print("Top coefficients:")
    top_idx = np.argsort(np.abs(ridge_cv.coef_))[::-1][:topk]
    for ii, idx in enumerate(top_idx):
        print(f"Feature {idx}: {ridge_cv.coef_[idx]:.8f}")
        plt.bar(ii, abs(ridge_cv.coef_[idx]))
        
        # idx is tick label

    # replace x ticks with feature names
    plt.xticks(range(topk), top_idx)
    rotate = plt.xticks(rotation=90)
    small = plt.xticks(fontsize=6)
    
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
        
    plt.show()
        