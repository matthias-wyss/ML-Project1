import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocess import preprocess
from helpers import create_csv_submission


def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio.

    If the ratio is 0.8, 80% of the dataset will be used for training,
    and the remaining 20% will be for testing. If the ratio multiplied 
    by the number of samples is not an integer, np.floor is used to 
    determine the split index.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0, 1].
        seed: integer for random seed.

    Returns:
        x_tr: numpy array containing the training data.
        x_te: numpy array containing the testing data.
        y_tr: numpy array containing the training labels.
        y_te: numpy array containing the testing labels.

    Example:
        >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
        (array([2, 3, 4, 10, 1, 6, 0, 7, 12, 9]), array([8, 11, 5]), array([2, 3, 4, 10, 1, 6, 0, 7, 12, 9]), array([8, 11, 5]))
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Get the total number of samples
    N = len(x)

    # Generate random permutation of indices
    indices = np.random.permutation(N)

    # Determine the split index
    split_idx = int(np.floor(ratio * N))

    # Split the indices into training and testing sets
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Create training and testing data and labels
    x_tr, x_te = x[train_indices], x[test_indices]
    y_tr, y_te = y[train_indices], y[test_indices]

    return x_tr, x_te, y_tr, y_te


def initialize_weights(x, w):
    """
    Initialize weights for the model.

    Args:
        x: Input features.
        w: Existing weights.

    Returns:
        A numpy array of initialized weights.
    """
    if w is None:
        return np.zeros(x.shape[1])  # Initialize weights to zero
    return w


def predict(w, x, method):
    """
    Generate predictions based on the model weights and input features.

    Args:
        w: Weights of the model.
        x: Input features.
        method: The method used for prediction.

    Returns:
        Predicted labels based on the specified method.
    """
    if method in ["logistic_regression", "reg_logistic_regression"]:
        # Compute predicted probabilities for logistic regression
        y_pred_prob = sigmoid(x @ w)
        return (y_pred_prob >= 0.5).astype(int)  # Binary classification
    else:
        # Compute raw predictions for other methods
        raw_predictions = x @ w
        return (raw_predictions >= 0).astype(int)


def calculate_accuracy(y_pred, y_true):
    """
    Calculate the accuracy of the predictions.

    Args:
        y_pred: Predicted labels.
        y_true: True labels.

    Returns:
        Accuracy as a float.
    """
    return np.mean(y_pred == y_true)


def calculate_f1_score(y_pred, y_true):
    """
    Calculate the F1 score of the predictions.

    Args:
        y_pred: Predicted labels.
        y_true: True labels.

    Returns:
        F1 score as a float.
    """
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1-Score
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0


def show_parameters(method, lambda_=None, gamma=None, max_iters=None):
    """
    Print model name and its parameters.

    Args:
        method: str, the name of the method.
        lambda_: float, regularization parameter.
        gamma: float, learning rate.
        max_iters: integer, maximum number of iterations.
    """
    if method in ["mean_squared_error_gd", "mean_squared_error_sgd"]:
        print(f"Method: {method}, Gamma: {gamma}, Max Iters: {max_iters}")
    elif method == "least_squares":
        print(f"Method: {method}")
    elif method == "ridge_regression":
        print(f"Method: {method}, Lambda: {lambda_}")
    elif method in ["logistic_regression", "reg_logistic_regression"]:
        print(f"Method: {method}, Lambda: {lambda_}, Gamma: {gamma}, Max Iters: {max_iters}")
    else:
        print(f"Method: {method} (parameters unknown or unspecified)")


def count_unique_labels(y_pred):
    """
    Count and print occurrences of each unique label in predictions.

    Args:
        y_pred: numpy.ndarray, predicted labels.

    Returns:
        None
    """
    unique_labels, counts = np.unique(y_pred, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        print(f"Number of predicted labels {label}: {count}")


import numpy as np
import csv  # Ensure to import csv for the create_csv_submission function

def predict_with_method(
    x_train,
    y_train,
    x_test,
    test_ids,
    method,
    lambda_=0.1,
    initial_w=None,
    max_iters=1000,
    gamma=0.01,
    replace_nan_by=None,
    column_nan_threshold=None,
    row_nan_threshold=None,
    continuous_threshold=None,
    normalization_method=None,
    outliers=None,
    z_score_threshold=None,
    max_false_percentage=None,
    balance_method=None,
    target_minority_ratio=None,
    noise_ratio=None,
    add_bias=None,
    pca_components=None,
    preprocess_verbose=False
):
    """
    Predict labels for the test dataset using a specified regression method.

    Args:
        x_train: numpy.ndarray, training input features.
        y_train: numpy.ndarray, training target labels.
        x_test: numpy.ndarray, testing input features.
        test_ids: numpy.ndarray, IDs for test samples.
        method: str, the regression method to use.
        lambda_: float, regularization parameter (default is 0.1).
        initial_w: numpy.ndarray, initial weights for gradient descent.
        max_iters: int, maximum iterations for gradient descent (default is 1000).
        gamma: float, learning rate for gradient descent (default is 0.01).
        replace_nan_by: int, number to replace NaNs by.
        column_nan_threshold: threshold for removing features with too many NaN values.
        row_nan_threshold: threshold for removing samples with too many NaN values.
        continuous_threshold: threshold for determining continuous features.
        normalization_method: method to normalize features.
        outliers: strategy for handling outliers.
        z_score_threshold: threshold for identifying outliers based on Z-scores.
        max_false_percentage: maximum allowable percentage of false samples.
        balance_method: method for balancing the dataset.
        target_minority_ratio: ratio for balancing the target classes.
        add_bias: boolean, whether to add a bias term to the model.
        pca_components: int, number of components for PCA.
        preprocess_verbose: boolean, if True, print preprocessing steps.

    Returns:
        numpy.ndarray, a 2D array of IDs and predicted labels.
    """
    # Initialize weights if not provided
    initial_w = initialize_weights(x_train, initial_w)

    # Preprocess training and test data
    if preprocess_verbose:
        print("Preprocessing data...")
    
    preprocessed_x_train, preprocessed_x_test, preprocessed_y_train = preprocess(
        x_train, x_test, y_train,
        replace_nan_by=replace_nan_by,
        column_nan_threshold=column_nan_threshold,
        row_nan_threshold=row_nan_threshold,
        continuous_threshold=continuous_threshold,
        normalization_method=normalization_method,
        outliers=outliers,
        z_score_threshold=z_score_threshold,
        max_false_percentage=max_false_percentage,
        balance_method=balance_method,
        target_minority_ratio=target_minority_ratio,
        noise_ratio=noise_ratio,
        add_bias=add_bias,
        pca_components=pca_components,
        verbose=preprocess_verbose
    )

    # Train the model using the specified method
    if method == "mean_squared_error_gd":
        w, _ = mean_squared_error_gd(preprocessed_y_train, preprocessed_x_train, initial_w, max_iters, gamma)
    elif method == "mean_squared_error_sgd":
        w, _ = mean_squared_error_sgd(preprocessed_y_train, preprocessed_x_train, initial_w, max_iters, gamma)
    elif method == "least_squares":
        w, _ = least_squares(preprocessed_y_train, preprocessed_x_train)
    elif method == "ridge_regression":
        w, _ = ridge_regression(preprocessed_y_train, preprocessed_x_train, lambda_)
    elif method == "logistic_regression":
        w, _ = logistic_regression(preprocessed_y_train, preprocessed_x_train, initial_w, max_iters, gamma)
    elif method == "reg_logistic_regression":
        w, _ = reg_logistic_regression(preprocessed_y_train, preprocessed_x_train, lambda_, initial_w, max_iters, gamma)
    else:
        raise ValueError("Invalid method specified.")

    # Make predictions on the test set using the helper function
    y_pred = predict(w, preprocessed_x_test, method)

    # Map 0 to -1 for binary classification
    y_pred = np.where(y_pred == 0, -1, y_pred)

    # Count occurrences of each unique label and print them
    count_unique_labels(y_pred)

    # Use the create_csv_submission function to save predictions
    create_csv_submission(test_ids, y_pred, "predictions.csv")

    return np.column_stack((test_ids, y_pred))



def test_with_method(
    x_tr,
    y_tr,
    x_te,
    y_te,
    method,
    lambda_=0.1,
    initial_w=None,
    max_iters=1000,
    gamma=0.01,
    verbose=False,
):
    """
    Train a model using the specified method and evaluate it on the test data.

    Args:
        x_tr: numpy.ndarray, training features.
        y_tr: numpy.ndarray, training labels.
        x_te: numpy.ndarray, testing features.
        y_te: numpy.ndarray, testing labels.
        method: str, the training method to use.
        lambda_: float, regularization parameter.
        initial_w: numpy.ndarray, initial weights.
        max_iters: int, maximum iterations for gradient descent.
        gamma: float, learning rate.
        verbose: bool, if True print additional information.

    Returns:
        accuracy: float, accuracy.
        f1: float, F1-score.
    """
    # Initialize weights
    initial_w = initialize_weights(x_tr, initial_w)

    # Train the model using the specified method
    if method == "mean_squared_error_gd":
        w, _ = mean_squared_error_gd(y_tr, x_tr, initial_w, max_iters, gamma)
    elif method == "mean_squared_error_sgd":
        w, _ = mean_squared_error_sgd(y_tr, x_tr, initial_w, max_iters, gamma)
    elif method == "least_squares":
        w, _ = least_squares(y_tr, x_tr)
    elif method == "ridge_regression":
        w, _ = ridge_regression(y_tr, x_tr, lambda_)
    elif method == "logistic_regression":
        w, _ = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
    elif method == "reg_logistic_regression":
        w, _ = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
    else:
        raise ValueError("Invalid method specified.")

    # Make predictions on the test set
    y_pred = predict(w, x_te, method)

    # Calculate accuracy and F1 score
    accuracy = calculate_accuracy(y_pred, y_te)
    f1_score = calculate_f1_score(y_pred, y_te)

    # Print the results if verbose is set to True
    if verbose:
        print(f"Method: {method}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1_score:.4f}\n")
        
    return accuracy, f1_score




def k_fold_split(x, y, k, seed=1):
    """
    Split data into K folds for cross-validation.
    
    Args:
        x: numpy array of shape (N, D), where N is the number of samples and D is the number of features.
        y: numpy array of shape (N,), representing the labels corresponding to the samples.
        k: integer, the number of folds for cross-validation.
        seed: integer, random seed for reproducibility of the split.
    
    Returns:
        folds_x: list of numpy arrays containing the K folds for features.
        folds_y: list of numpy arrays containing the K folds for labels.
    """
    np.random.seed(seed)  # Set random seed for reproducibility
    N = len(x)  # Get the number of samples
    indices = np.random.permutation(N)  # Randomly permute indices for shuffling the data
    
    # Split indices into K approximately equal parts
    folds_x = np.array_split(x[indices], k)  # Split features into K folds
    folds_y = np.array_split(y[indices], k)  # Split labels into K folds
    
    return folds_x, folds_y  # Return the feature and label folds



def cross_validate(x, y, k, method, replace_nan_by, column_nan_threshold, row_nan_threshold, continuous_threshold, 
    normalization_method, outliers, z_score_threshold, max_false_percentage, balance_method, target_minority_ratio, noise_ratio, 
    add_bias, pca_components, lambda_=0.1, initial_w=None, max_iters=1000, gamma=0.01, preprocess_verbose=False, 
    cross_validation_verbose=False):
    """
    Perform k-fold cross-validation.
    
    Args:
        x: numpy array of shape (N, D), where N is the number of samples and D is the number of features.
        y: numpy array of shape (N,), representing the labels for the samples.
        k: integer, the number of folds for cross-validation.
        method: str, the training method to use (e.g., 'least_squares', 'ridge_regression', etc.).
        replace_nan_by: int, number to replace NaNs by.
        column_nan_threshold: threshold for removing features with too many NaN values.
        row_nan_threshold: threshold for removing samples with too many NaN values.
        continuous_threshold: threshold for determining continuous features.
        normalization_method: method to normalize features.
        outliers: strategy for handling outliers.
        z_score_threshold: threshold for identifying outliers based on Z-scores.
        max_false_percentage: maximum allowable percentage of false samples.
        balance_method: method for balancing the dataset.
        target_minority_ratio: ratio for balancing the target classes.
        add_bias: boolean, whether to add a bias term to the model.
        pca_components: int, number of components for PCA.
        lambda_: float, regularization parameter for methods like Ridge and Logistic Regression.
        initial_w: numpy array, initial weights for gradient methods.
        max_iters: integer, maximum number of iterations for gradient methods.
        gamma: float, learning rate for gradient methods.
        preprocess_verbose: boolean, if True, print preprocessing steps.
        cross_validation_verbose: boolean, if True, print cross-validation steps.
    
    Returns:
        avg_accuracy: float, average accuracy across all folds.
        avg_f1: float, average F1-score across all folds.
    """
    # Split data into K folds for cross-validation
    folds_x, folds_y = k_fold_split(x, y, k)
    
    accuracies = []  # List to store accuracies for each fold
    f1_scores = []   # List to store F1 scores for each fold
    
    for i in range(k):
        # Use the i-th fold as the test set and the remaining as the training set
        x_te = folds_x[i]
        y_te = folds_y[i]
        
        # Convert labels from -1 to 0 for compatibility with the model
        y_te = np.where(y_te == -1, 0, y_te)

        # Combine the remaining folds to create the training set
        x_tr = np.concatenate([folds_x[j] for j in range(k) if j != i])
        y_tr = np.concatenate([folds_y[j] for j in range(k) if j != i])
        
        if preprocess_verbose:
            print(f'Preprocessing for fold {i+1}:\n')

        # Preprocess training and test data separately (fit on train, apply to test)
        preprocessed_x_train, preprocessed_x_test, preprocessed_y_train = preprocess(
            x_tr, x_te, y_tr, 
            replace_nan_by=replace_nan_by,
            column_nan_threshold=column_nan_threshold, 
            row_nan_threshold=row_nan_threshold, 
            continuous_threshold=continuous_threshold, 
            normalization_method=normalization_method, 
            outliers=outliers, 
            z_score_threshold=z_score_threshold, 
            max_false_percentage=max_false_percentage, 
            balance_method=balance_method, 
            target_minority_ratio=target_minority_ratio, 
            noise_ratio=noise_ratio, 
            add_bias=add_bias, 
            pca_components=pca_components,
            verbose=preprocess_verbose
        )
        
        if preprocess_verbose:
            print()
            
        if cross_validation_verbose:
            print(f'Results for fold {i+1}:')

        # Test the model with the specified training method and calculate accuracy and F1 score
        accuracy, f1 = test_with_method(
            preprocessed_x_train, preprocessed_y_train, preprocessed_x_test, y_te, method, 
            lambda_=lambda_, 
            initial_w=initial_w, 
            max_iters=max_iters, 
            gamma=gamma, 
            verbose=cross_validation_verbose
        )

        # Append the results for this fold
        accuracies.append(accuracy)
        f1_scores.append(f1)

    # Calculate average accuracy and F1 score across all folds
    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    
    # Print the model parameters and their values
    show_parameters(method=method, lambda_=lambda_, gamma=gamma, max_iters=max_iters)
    
    # Print the average accuracy and F1 score
    print(f"Average accuracy: {avg_accuracy:.4f}, Average F1-Score: {avg_f1:.4f}\n")
    
    return avg_accuracy, avg_f1  # Return the average accuracy and F1 score
