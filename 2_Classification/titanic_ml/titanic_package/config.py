# config.py

# Configuration settings for the Titanic model package

# Path to the Titanic dataset
DATA_PATH = "C:/Users/srini/Desktop/tianic-test/data/train.csv"  # Update with the actual path or set a relative path if in the same directory

# Model parameters (can add more as needed)
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    # Add other model parameters if needed
}

# Data processing options (like test size, random state)
DATA_SPLIT = {
    "test_size": 0.2,
    "random_state": 42
}
