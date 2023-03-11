import numpy as np

def load_data():
    # Replace the following lines with your code to load your data
    # X is your feature matrix, y is your target vector
    X = np.random.rand(100, 10)  # Small dataset
    y = np.random.randint(2, size=100)
    X_large = np.random.rand(1000, 40)  # Large dataset
    y_large = np.random.randint(2, size=1000)

    return X, y, X_large, y_large
