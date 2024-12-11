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
    path = r".\data_exploration\data\ArM\data_set_dict.pkl"
    X, y, groups, od = load_dataset(path, prefilter=True)

    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 0)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    # Use the indices to split the data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    od_train, od_val = od[train_idx], od[val_idx]

    #from validation set drop certain outliers which have OD < 0.018
    X_val = X_val[od_val > 0.018]
    y_val = y_val[od_val > 0.018]
    od_val = od_val[od_val > 0.018]
    
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

        def fit(self, X, y, od=None):
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
            if od is None:
                od = np.ones(X.shape[0])

            # Compute weights using the sigmoid function
            # s_inv = 1 / (1 + np.exp(-self.a * (od - self.b)))
            
            # discret
            s_inv = np.where(od < self.b, 0, 1)

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
            self.sample_weights_ = od

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
    regressor = WeightedRidgeRegressor(alpha=80.0)

    # Perform GridSearchCV
    param_grid = {
        "alpha": [80],  # Range of alpha values
        "a": [1000.0],  # Fixed 'a' value
        "b": np.linspace(0.0, 0.21, 20)  # Range of 'b' values  
    }

    scorer = make_scorer(r2_score)
    cv = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring=scorer, cv=15, return_train_score=True, n_jobs=-1)
    cv.fit(X_train, y_train, od=od_train)

    import matplotlib.pyplot as plt
    # Plot the contour plot

    from sklearn_evaluation import plot

    plot.grid_search(cv.cv_results_, change='b', kind='line')
    
    rotate = plt.xticks(rotation=90)

    # Extract results
    # results = cv.cv_results_
    alpha_values = param_grid["alpha"]
    b_values = param_grid["b"]
    # mean_cv_r2 = results["mean_test_score"]  # Cross-validation R^2 for each 'b'
    # std_cv_r2 = results["std_test_score"]  # Standard deviation of CV R^2
    validation_r2 = []

    # Evaluate validation R^2 for each model
    for b in b_values:
        model = WeightedRidgeRegressor(alpha=80, a=10.0, b=b)
        model.fit(X_train, y_train, od=od_train)
        y_val_pred = model.predict(X_val)
        validation_r2.append(r2_score(y_val, y_val_pred))
    
    # calc validation_r2 for best model
    model = cv.best_estimator_
    model.fit(X_train, y_train, od=od_train)
    y_val_pred = model.predict(X_val)
    
    validation_r2_f = r2_score(y_val, y_val_pred)
    
    print(f"Validation R^2: {validation_r2_f:.4f}")
    print(f"Best parameters: {cv.best_params_}")
    

    # Plotting Cross-Validation R^2 and Validation R^2
    plt.figure(figsize=(12, 6))
    
    mean_cv_r2 = cv.cv_results_["mean_test_score"]
    std_cv_r2 = cv.cv_results_["std_test_score"]

    # # Plot CV R^2
    plt.plot(b_values, mean_cv_r2, label="CV R^2", color="blue")
    plt.fill_between(b_values, mean_cv_r2 - std_cv_r2, mean_cv_r2 + std_cv_r2, color="blue", alpha=0.2)

    # # Plot Validation R^2
    plt.plot(b_values, validation_r2, label="Validation R^2", color="orange")
    
    # # Add vertical line for best 'b' value
    best_b = cv.best_params_["b"]
    plt.axvline(best_b, color="red", linestyle="--", label=f"Best b: {best_b:.4f}")
    
    od_train_plot = od_train[od_train < 0.2]
    zero_axis = np.zeros(len(od_train_plot))
    plt.scatter(od_train_plot, zero_axis, color="black", label="OD < 0.2")
    

    # Add labels, legend, and grid
    plt.xlabel("Parameter b")
    plt.ylabel("R^2 Score")
    plt.title("R^2 Scores for Weighted Ridge Regression over cutoff Parameter b: (OD - b)")
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()
    
    #validation parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Validation Parity Plot")
    plt.grid(True)
    plt.show()

    #train parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, model.predict(X_train), alpha=0.5)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Train Parity Plot")
    plt.grid(True)
    plt.show()
    
    # Print the best parameters and cross-validation R^2
    print(f"Best parameters: {cv.best_params_}")
    print(f"Best cross-validation R^2: {cv.best_score_:.4f}")
    # print(f"Validation R^2: {validation_r2[np.argmax(mean_cv_r2)]:.4f}")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    
    beta = cv.best_estimator_.coef_
    
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
        