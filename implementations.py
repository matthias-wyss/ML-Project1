import numpy as np

# 1. Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform linear regression using gradient descent.

    Parameters:
    y : array_like
        The target values.
    tx : array_like
        The feature matrix.
    initial_w : array_like
        Initial weights.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Step size (learning rate).

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    w = initial_w  # Initialize weights
    for n in range(max_iters):
        # Compute the gradient
        gradient = -tx.T @ (y - tx @ w) / len(y)
        # Update weights
        w = w - gamma * gradient
    # Calculate the loss
    loss = np.sum((y - tx @ w) ** 2) / (2 * len(y))
    return w, loss


# 2. Linear regression using stochastic gradient descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Perform linear regression using stochastic gradient descent.

    Parameters:
    y : array_like
        The target values.
    tx : array_like
        The feature matrix.
    initial_w : array_like
        Initial weights.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Step size (learning rate).

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    w = initial_w  # Initialize weights
    for n in range(max_iters):
        for i in range(len(y)):
            # Compute the gradient for each sample
            gradient = -(y[i] - tx[i] @ w) * tx[i] / len(y)
            # Update weights
            w = w - gamma * gradient
    # Calculate the loss
    loss = np.sum((y - tx @ w) ** 2) / (2 * len(y))
    return w, loss


# 3. Least squares regression using normal equations
def least_squares(y, tx):
    """
    Perform least squares regression using normal equations.

    Parameters:
    y : array_like
        The target values.
    tx : array_like
        The feature matrix.

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    # Calculate optimal weights using the normal equation
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    # Calculate the loss
    loss = np.sum((y - tx @ w) ** 2) / (2 * len(y))
    return w, loss


# 4. Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """
    Perform ridge regression using normal equations.

    Parameters:
    y : array_like
        The target values.
    tx : array_like
        The feature matrix.
    lambda_ : float
        The regularization parameter.

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    # Regularization term
    lambda_prime = 2 * len(y) * lambda_
    # Calculate optimal weights using the regularized normal equation
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(tx.shape[1]), tx.T @ y)
    # Calculate the loss
    loss = np.sum((y - tx @ w) ** 2) / (2 * len(y))
    return w, loss


# 5. Logistic regression using gradient descent
def sigmoid(t):
    """
    Compute the sigmoid function.

    Parameters:
    t : array_like
        Input values.

    Returns:
    array_like
        Sigmoid function values.
    """
    return 1 / (1 + np.exp(-t))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform logistic regression using gradient descent.

    Parameters:
    y : array_like
        The target values (binary).
    tx : array_like
        The feature matrix.
    initial_w : array_like
        Initial weights.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Step size (learning rate).

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    w = initial_w  # Initialize weights
    for n in range(max_iters):
        # Compute the gradient
        gradient = tx.T @ (sigmoid(tx @ w) - y) / len(y)
        # Update weights
        w -= gamma * gradient

    # Calculate the loss using cross-entropy
    loss = -np.mean(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))
    return w, loss




# 6. Regularized logistic regression using gradient descent
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression using gradient descent.

    Parameters:
    y : array_like
        The target values (binary).
    tx : array_like
        The feature matrix.
    lambda_ : float
        Regularization parameter.
    initial_w : array_like
        Initial weights.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Step size (learning rate).

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    w = initial_w  # Initialize weights
    for n in range(max_iters):
        # Compute the gradient with regularization term
        gradient = tx.T @ (sigmoid(tx @ w) - y) / len(y) + 2 * lambda_ * w
        # Update weights
        w -= gamma * gradient

    # Calculate the loss using cross-entropy with regularization
    loss = -np.mean(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))
    return w, loss


