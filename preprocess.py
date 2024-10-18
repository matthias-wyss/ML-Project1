import numpy as np

def preprocess(
    x_train,
    x_test,
    y_train,
    replace_nan_by,
    column_nan_threshold,
    row_nan_threshold,
    continuous_threshold,
    normalization_method,
    outliers,
    z_score_threshold,
    max_false_percentage,
    balance_method,
    target_minority_ratio,
    add_bias,
    pca_components=None,
    verbose=False,
):
    """
    Preprocess the input data by converting labels, removing single-value columns, 
    handling NaNs, removing outliers, balancing the data, normalizing features, and 
    adding a bias term.

    Args:
        x_train (numpy.ndarray): Training feature matrix.
        x_test (numpy.ndarray): Testing feature matrix.
        y_train (numpy.ndarray): Training labels.
        replace_nan_by (int): Number to replace NaNs by.
        column_nan_threshold (float): Threshold to remove columns with too many NaNs.
        row_nan_threshold (float): Threshold to remove rows with too many NaNs.
        continuous_threshold (float): Threshold to determine if features are continuous.
        normalization_method (str): Method for normalization or standardization.
        outliers (str): Method for handling outliers ('remove' or 'clip').
        z_score_threshold (float): Z-score threshold for outlier removal.
        max_false_percentage (float): Maximum allowable percentage of false positives.
        balance_method (str): Method for balancing the data ('downsampling' or 'upsampling').
        target_minority_ratio (float): Desired ratio of the minority class.
        add_bias (bool): Whether to add a bias term to the feature matrix.
        pca_components (int): number of components for PCA.
        verbose (bool): Whether to print verbose output.

    Returns:
        x_train_preprocessed (numpy.ndarray): Preprocessed training feature matrix.
        x_test_preprocessed (numpy.ndarray): Preprocessed testing feature matrix.
        y_train_preprocessed (numpy.ndarray): Preprocessed training labels.
    """
    # Convert y_train labels from -1 and 1 to 0 and 1
    y_train = convert_labels(y_train, verbose=verbose)

    # Remove columns with a single unique value from both x_train and x_test
    x_train, x_test = remove_single_value_columns(x_train, x_test, verbose=verbose)
    
    if replace_nan_by is not None:
        # Replace all NaN by replace_nan_by if it is defined
        x_train, x_test = replace_nan(x_train, x_test, replacement_value=replace_nan_by, verbose=verbose)
    else:
        # Remove columns with a high percentage of NaN values
        x_train, x_test = remove_columns_with_high_nan(x_train, x_test, threshold=column_nan_threshold, verbose=verbose)

        # Remove rows with a high percentage of NaN values
        x_train, y_train = remove_rows_with_high_nan(x_train, y_train, threshold=row_nan_threshold, verbose=verbose)

        # Fill remaining NaNs in numerical features
        x_train = fill_missing_values(x_train, continuous_threshold, verbose=verbose)
        x_test = fill_missing_values(x_test, continuous_threshold, verbose=verbose)

    # Normalize or standardize x_train and x_test with the given method
    x_train, x_test = normalize_or_standardize(x_train, x_test, method=normalization_method, verbose=verbose)

    # Handle outliers
    if outliers == "remove":
        x_train, y_train = remove_outliers(
            x_train,
            y_train,
            threshold=z_score_threshold,
            max_false_percentage=max_false_percentage,
            verbose=verbose,
        )
    elif outliers == "clip":
        x_train = clip_outliers(x_train, threshold=z_score_threshold, verbose=verbose)

    # Check for imbalance and balance the data if needed
    if balance_method is not None:
        x_train, y_train = check_and_balance_data(
            x_train,
            y_train,
            method=balance_method,
            target_minority_ratio=target_minority_ratio,
            verbose=verbose,
        )

    # Add a bias term to x_train and x_test if specified
    if add_bias:
        x_train, x_test = add_bias_term(x_train, x_test, verbose=verbose)
        
    # Perform PCA if specified
    if pca_components is not None and pca_components > 0:
        # Apply PCA to training data
        x_train, components, train_mean = perform_pca(x_train, n_components=pca_components, verbose=verbose)  
        
        # Center x_test using the mean from x_train
        x_test_centered = x_test - train_mean  
        # Apply PCA to testing data using the same components
        x_test = np.dot(x_test_centered, components)


    return x_train, x_test, y_train



def perform_pca(x, n_components, verbose=False):
    """
    Perform PCA on the given dataset.

    Args:
        x (numpy.ndarray): The input data (samples x features).
        n_components (int): Number of principal components to keep.

    Returns:
        x_pca (numpy.ndarray): Transformed data in the PCA space.
        components (numpy.ndarray): The principal components (eigenvectors).
        mean (numpy.ndarray): Mean vector used for centering.
    """
    # Center the data
    x_mean = np.mean(x, axis=0)
    x_centered = x - x_mean

    # Compute the covariance matrix
    cov_matrix = np.cov(x_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    top_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Transform the data
    x_pca = np.dot(x_centered, top_eigenvectors)
    
    if verbose:
        print(f'PCA performs to reach {n_components} components.')

    return x_pca, top_eigenvectors, x_mean



def check_and_balance_data(
    x_train, y_train, method="downsampling", target_minority_ratio=0.5, verbose=False
):
    """
    Check for class imbalance and balance the data according to the specified method.

    Args:
        x_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        method (str): Method for balancing ('downsampling' or 'upsampling').
        target_minority_ratio (float): Desired ratio of the minority class.
        verbose (bool): Whether to print verbose output.

    Returns:
        x_train_balanced (numpy.ndarray): Balanced training feature matrix.
        y_train_balanced (numpy.ndarray): Balanced training labels.
    """
    # Identify unique classes and their counts
    unique_classes, class_counts = np.unique(y_train, return_counts=True)

    # Ensure the dataset is binary
    if len(unique_classes) != 2:
        raise ValueError("This function is designed for binary datasets only.")

    # Determine major and minor classes based on counts
    if class_counts[0] > class_counts[1]:
        major_class, minor_class = unique_classes[0], unique_classes[1]
        major_count, minor_count = class_counts[0], class_counts[1]
    else:
        major_class, minor_class = unique_classes[1], unique_classes[0]
        major_count, minor_count = class_counts[1], class_counts[0]

    # Calculate target sizes for balancing
    total_count = major_count + minor_count
    target_minor_size = int(target_minority_ratio * total_count)
    target_major_size = total_count - target_minor_size

    # Get indices of major and minor classes
    major_indices = np.where(y_train == major_class)[0]
    minor_indices = np.where(y_train == minor_class)[0]

    # Balance the dataset using the specified method
    if method == "downsampling":
        # Downsample the majority class
        if target_major_size < major_count:
            major_indices = np.random.choice(major_indices, size=target_major_size, replace=False)

        if verbose:
            print(
                f"Majority class downsampled. New target sizes: {target_major_size} for majority class, {target_minor_size} for minority class."
            )

    elif method == "upsampling":
        # Upsample the minority class
        if target_minor_size > minor_count:
            extra_indices = np.random.choice(
                minor_indices, size=target_minor_size - minor_count, replace=True
            )
            minor_indices = np.concatenate([minor_indices, extra_indices])

        if verbose:
            print(
                f"Minority class upsampled. New target sizes: {target_major_size} for majority class, {target_minor_size} for minority class."
            )

    elif method is None:
        return x_train, y_train

    else:
        raise ValueError(f"Method {method} not implemented. Use 'downsampling' or 'upsampling'.")

    # Combine indices of major and minor classes and shuffle
    balanced_indices = np.concatenate([major_indices, minor_indices])
    np.random.shuffle(balanced_indices)

    # Extract the balanced data
    x_train_balanced = x_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]

    return x_train_balanced, y_train_balanced


def convert_labels(y_train, verbose=False):
    """
    Convert the labels from -1 and 1 to 0 and 1.

    Args:
        y_train (numpy.ndarray): Training labels containing values -1 and 1.

    Returns:
        y_train_converted (numpy.ndarray): Converted labels containing values 0 and 1.
    """
    # Map -1 to 0 and keep 1 as is
    y_train_converted = np.where(y_train == -1, 0, y_train)
    if verbose:
        print("Labels converted from -1 to 0.")

    return y_train_converted


def convert_labels_back(y_train, verbose=False):
    """
    Convert the labels from 0 and 1 to -1 and 1.

    Args:
        y_train (numpy.ndarray): Training labels containing values 0 and 1.

    Returns:
        y_train_converted (numpy.ndarray): Converted labels containing values -1 and 1.
    """
    # Map 0 to -1 and keep 1 as is
    y_train_converted = np.where(y_train == 0, -1, y_train)
    if verbose:
        print("Labels converted from 0 to -1.")

    return y_train_converted


def remove_single_value_columns(x_train, x_test, verbose=False):
    """
    Remove columns from x_train and x_test where all values are the same (nunique == 1).

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.

    Returns:
        x_train_cleaned (numpy.ndarray): Updated x_train with single-value columns removed.
        x_test_cleaned (numpy.ndarray): Updated x_test with single-value columns removed.
    """
    # Shape before
    num_feature_before = x_train.shape[1]

    # Identify columns in x_train with a single unique value
    single_value_columns = [
        col
        for col in range(x_train.shape[1])
        if len(np.unique(x_train[:, col][~np.isnan(x_train[:, col])])) == 1
    ]

    # Filter the columns in x_train and x_test
    x_train_cleaned = np.delete(x_train, single_value_columns, axis=1)
    x_test_cleaned = np.delete(x_test, single_value_columns, axis=1)

    # Shape after
    num_feature_after = x_train_cleaned.shape[1]
    
    if verbose:
        # Print the updated shape of the datasets
        print(f"{num_feature_before - num_feature_after} features where all values are the same removed.")
    
    return x_train_cleaned, x_test_cleaned



def replace_nan(x_train, x_test, replacement_value=-1, verbose=False):
    """
    Replace all NaN values in x_train and x_test with the specified replacement value.

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.
        replacement_value (float or int): Value to replace NaN with (default is -1).
        verbose (bool): If True, print out a confirmation message (default is False).

    Returns:
        x_train_cleaned (numpy.ndarray): Updated x_train with NaN replaced by the replacement_value.
        x_test_cleaned (numpy.ndarray): Updated x_test with NaN replaced by the replacement_value.
    """
    # Replace NaN values with the specified replacement_value in both x_train and x_test
    x_train_cleaned = np.where(np.isnan(x_train), replacement_value, x_train)
    x_test_cleaned = np.where(np.isnan(x_test), replacement_value, x_test)

    if verbose:
        print(f"Replaced all NaN values with {replacement_value}.")

    return x_train_cleaned, x_test_cleaned



def remove_columns_with_high_nan(x_train, x_test, threshold=0.5, verbose=False):
    """
    Remove columns with a percentage of NaN values exceeding the specified threshold from x_train and x_test.

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.
        threshold (float): Percentage threshold for NaN values in columns.

    Returns:
        x_train_cleaned (numpy.ndarray): Updated x_train with columns removed.
        x_test_cleaned (numpy.ndarray): Updated x_test with columns removed.
    """
    # Calculate the percentage of NaN values in each column of x_train
    nan_percentage = np.mean(np.isnan(x_train), axis=0)

    # Identify columns where NaN percentage exceeds the threshold
    columns_to_remove = np.where(nan_percentage > threshold)[0]

    # Remove columns from both x_train and x_test
    x_train_cleaned = np.delete(x_train, columns_to_remove, axis=1)
    x_test_cleaned = np.delete(x_test, columns_to_remove, axis=1)

    if verbose:
        print(f"Removed {len(columns_to_remove)} columns with NaN percentage higher than {threshold*100}%.")

    return x_train_cleaned, x_test_cleaned


def remove_rows_with_high_nan(x_train, y_train, threshold=0.5, verbose=False):
    """
    Remove rows from x_train where the percentage of NaN values exceeds the specified threshold.

    Args:
        x_train (numpy.ndarray): Training dataset.
        y_train (numpy.ndarray): Training labels.
        threshold (float): Percentage threshold for NaN values in rows.

    Returns:
        x_train_cleaned (numpy.ndarray): Updated x_train with rows removed.
        y_train_cleaned (numpy.ndarray): Updated y_train with corresponding rows removed.
    """
    # Calculate the percentage of NaN values in each row
    nan_percentage_per_row = np.mean(np.isnan(x_train), axis=1)

    # Identify rows where NaN percentage exceeds the threshold
    rows_to_keep = np.where(nan_percentage_per_row <= threshold)[0]

    # Keep only the valid rows in x_train and y_train
    x_train_cleaned = x_train[rows_to_keep]
    y_train_cleaned = y_train[rows_to_keep]

    if verbose:
        print(f"Removed {x_train.shape[0] - len(rows_to_keep)} rows with NaN percentage higher than {threshold*100}%.")

    return x_train_cleaned, y_train_cleaned


def fill_missing_values(x_data, continuous_threshold=10, verbose=False):
    """
    Fill missing values in the dataset based on the nature of the features (continuous or categorical).

    Args:
        x_data (numpy.ndarray): Input dataset with missing values.
        continuous_threshold (int): Threshold to determine if a feature is continuous (number of unique values).
        verbose (bool): Whether to print verbose output.

    Returns:
        x_filled (numpy.ndarray): Dataset with missing values filled.
    """
    # Create a copy of the original data to avoid modifying it directly
    x_filled = x_data.copy()
    
    # Initialize counters for filled columns
    continuous_count = 0
    categorical_count = 0

    # Iterate through each column in the dataset
    for col in range(x_data.shape[1]):
        col_data = x_data[:, col]
        # Check for NaNs in the column
        if np.any(np.isnan(col_data)):
            # Find unique values in the column, ignoring NaNs
            unique_values = np.unique(col_data[~np.isnan(col_data)])

            # Check if the column is continuous based on the number of unique values
            if len(unique_values) > continuous_threshold:
                # Fill NaNs with the mean value of the column
                mean_value = np.nanmean(col_data)
                x_filled[:, col] = np.where(np.isnan(col_data), mean_value, col_data)
                continuous_count += 1
                
            else:
                # For categorical columns, fill NaNs with the mode value
                mode_value = np.nan if len(unique_values) == 0 else unique_values[0]
                x_filled[:, col] = np.where(np.isnan(col_data), mode_value, col_data)
                categorical_count += 1

    # Print the total number of filled columns of each type if verbose is True
    if verbose:
        print(f"Total filled continuous columns: {continuous_count}")
        print(f"Total filled categorical columns: {categorical_count}")

    return x_filled




def normalize_or_standardize(x_train, x_test, method="standardize", verbose=False):
    """
    Normalize or standardize the dataset.

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.
        method (str): Method for scaling, either 'normalize' (min-max) or 'standardize' (z-score).
        verbose (bool): Whether to print verbose output.

    Returns:
        x_train_scaled (numpy.ndarray): Scaled training dataset.
        x_test_scaled (numpy.ndarray): Scaled testing dataset.
    """
    if method == "normalize":
        # Min-max normalization to scale features between 0 and 1
        min_vals = np.nanmin(x_train, axis=0)
        max_vals = np.nanmax(x_train, axis=0)
        x_train_scaled = (x_train - min_vals) / (max_vals - min_vals)
        x_test_scaled = (x_test - min_vals) / (max_vals - min_vals)

        if verbose:
            print("Data normalized using min-max scaling.")
    elif method == "standardize":
        # Z-score standardization (zero mean, unit variance)
        means = np.nanmean(x_train, axis=0)
        stds = np.nanstd(x_train, axis=0)
        x_train_scaled = (x_train - means) / stds
        x_test_scaled = (x_test - means) / stds

        if verbose:
            print("Data standardized using z-score scaling.")
    else:
        raise ValueError(f"Normalization method '{method}' is not recognized.")

    return x_train_scaled, x_test_scaled


def remove_outliers(x_train, y_train, threshold=3, max_false_percentage=0.3, verbose=False):
    """
    Remove outliers from the dataset based on z-score and a false percentage threshold.

    Args:
        x_train (numpy.ndarray): Training dataset.
        y_train (numpy.ndarray): Training labels.
        threshold (float): Z-score threshold to identify outliers.
        max_false_percentage (float): Maximum percentage of non-outlier values allowed in a row.
        verbose (bool): Whether to print verbose output.

    Returns:
        x_train_cleaned (numpy.ndarray): Training dataset with outliers removed.
        y_train_cleaned (numpy.ndarray): Training labels with outliers removed.
    """
    # Compute the z-scores of x_train
    z_scores = np.abs((x_train - np.nanmean(x_train, axis=0)) / np.nanstd(x_train, axis=0))

    # Create a mask of non-outliers
    outlier_mask = z_scores < threshold

    # Calculate the percentage of non-outlier values in each row of the mask
    false_percentages = 1 - np.mean(outlier_mask, axis=1)

    # Check if the false percentage is less than the maximum allowed
    non_outlier_rows = false_percentages < max_false_percentage

    # Remove rows with outliers based on the non-outlier mask
    x_train_cleaned = x_train[non_outlier_rows]
    y_train_cleaned = y_train[non_outlier_rows]

    if verbose:
        num_removed = np.sum(~non_outlier_rows)
        print(f"Removed {num_removed} outliers (z-score > {threshold}).")

    return x_train_cleaned, y_train_cleaned


def clip_outliers(x_train, threshold=3, verbose=False):
    """
    Clip outliers in the dataset to the z-score threshold.

    Args:
        x_train (numpy.ndarray): Training dataset.
        threshold (float): Z-score threshold to identify outliers.
        verbose (bool): Whether to print verbose output.

    Returns:
        x_train_clipped (numpy.ndarray): Training dataset with outliers clipped.
    """
    # Compute the z-scores of x_train
    z_scores = np.abs((x_train - np.nanmean(x_train, axis=0)) / np.nanstd(x_train, axis=0))

    # Clip outliers to the threshold
    x_train_clipped = np.where(z_scores > threshold, np.sign(x_train) * threshold, x_train)

    if verbose:
        print(f"Clipped outliers (z-score > {threshold}) to the threshold.")

    return x_train_clipped


def add_bias_term(x_train, x_test, verbose=False):
    """
    Add a bias term (column of ones) to x_train and x_test.

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.
        verbose (bool): Whether to print verbose output.

    Returns:
        x_train_with_bias (numpy.ndarray): Training dataset with bias term added.
        x_test_with_bias (numpy.ndarray): Testing dataset with bias term added.
    """
    # Add a column of ones to the start of x_train and x_test
    x_train_with_bias = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    x_test_with_bias = np.hstack([np.ones((x_test.shape[0], 1)), x_test])

    if verbose:
        print("Bias term added to feature matrices.")

    return x_train_with_bias, x_test_with_bias
