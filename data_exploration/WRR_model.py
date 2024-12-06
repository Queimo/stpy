import pickle
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score

# Load dataset
def load_dataset(path, prefilter=True):
    data_dict = pickle.load(open(path, "rb"))
    df = data_dict["df"]
    if prefilter:
        X = data_dict["X"][df["big_OD"]]
        y = df["norm_TSNAK"][df["big_OD"]].values
        groups = df["variant"][df["big_OD"]].values
        od = df["OD"][df["big_OD"]].values
    else:
        X = data_dict["X"]
        y = df["norm_TSNAK"].values
        groups = df["variant"].values
        od = df["OD"].values
        
    return X, y, groups, od

# Main block
if __name__ == "__main__":
    # Load data
    path = r".\data_exploration\data_set_dict.pkl"
    X, y, groups, od = load_dataset(path, prefilter=True)

    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    # Use the indices to split the data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    od_train, od_val = od[train_idx], od[val_idx]
    
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

    # Assume we have X_train, y_train, and s_inv (weights)
    # X_train: (n_samples, n_features)
    # y_train: (n_samples,)
    # s_inv: (n_samples,)

    # Ridge penalty (regularization strength)

    class WeightedRidgeRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, alpha=1.0, a=0.5, b=0.5):
            """
            Parameters:
            - alpha: Regularization strength.
            - a: Scaling parameter for the sigmoid function.
            - b: Offset parameter for the sigmoid function.
            """
            self.alpha = alpha
            self.a = a
            self.b = b

        def fit(self, X, y, sample_weights=None):
            """
            Fit the Weighted Ridge Regression model.
            
            Parameters:
            - X: Training data of shape (n_samples, n_features).
            - y: Target values of shape (n_samples,).
            - sample_weights: Additional weights for computing W. Shape (n_samples,).
            """
            # Validate input arrays
            X, y = check_X_y(X, y)

            # Use provided sample_weights or initialize as ones
            if sample_weights is None:
                sample_weights = np.ones(X.shape[0])

            # Compute weights using the sigmoid function
            s_inv = 1 / (1 + np.exp(-self.a * (sample_weights - self.b)))

            # Construct diagonal weight matrix
            W = np.diag(s_inv)

            # Weighted Ridge regression formula: beta = (X^T W X + alpha I)^-1 X^T W y
            XTWX = X.T @ W @ X
            ridge_matrix = XTWX + self.alpha * np.eye(X.shape[1])
            XTWy = X.T @ W @ y

            # Solve for coefficients
            self.coef_ = np.linalg.solve(ridge_matrix, XTWy)

            # Save attributes for later use
            self.X_ = X
            self.y_ = y
            self.sample_weights_ = sample_weights

            return self

        def predict(self, X):
            """
            Predict using the Weighted Ridge Regression model.
            
            Parameters:
            - X: Input data of shape (n_samples, n_features).
            
            Returns:
            - Predictions of shape (n_samples,).
            """
            # Check if the model is fitted
            check_is_fitted(self, ["coef_"])

            # Validate input array
            X = check_array(X)

            # Predict using the linear model
            return X @ self.coef_


    # Define your regressor
    regressor = WeightedRidgeRegressor(alpha=270.0)

    # Define the parameter grid for cross-validation
    param_grid = {
        "a": [10.0],  # Possible values for 'a'
        "b": np.linspace(0, 0.1, 100)  # Possible values for 'b'
    }

    # Custom scoring metric (e.g., R^2)
    scorer = make_scorer(r2_score)

    # Perform GridSearchCV
    cv = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring=scorer, cv=5)
    cv.fit(X_train, y_train, sample_weights=od_train)

    # Print the best parameters and R^2 score
    print(f"Best parameters: {cv.best_params_}")
    print(f"Best cross-validation R^2: {cv.best_score_:.4f}")

    # Make predictions using the best model
    best_model = cv.best_estimator_
    y_val_pred = best_model.predict(X_val)
    print(f"Validation R^2: {r2_score(y_val, y_val_pred):.4f}")


    import matplotlib.pyplot as plt
    fig = plt.figure()
    
    beta = best_model.coef_
    
    topk = 20
    # Print top 20 coefficients
    print("Top coefficients:")
    top_idx = np.argsort(np.abs(beta))[::-1][:topk]
    for ii, idx in enumerate(top_idx):
        print(f"Feature {idx}: {beta[idx]:.8f}")
        plt.bar(ii, abs(beta[idx]))
        
        # idx is tick label

    # replace x ticks with feature names
    plt.xticks(range(topk), top_idx)
    rotate = plt.xticks(rotation=90)
    small = plt.xticks(fontsize=6)
    
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
        
    plt.show()
        