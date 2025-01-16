import numpy as np
import matplotlib.pyplot as plt
from model import LSTMModel


def load_data(filepath: str) -> np.ndarray:
    """
    Loads data from a file.
    
    Args:
        filepath (str): The path to the data file.
        
    Returns:
        np.ndarray: The loaded data.
    """
    return np.loadtxt(filepath)


def plot_results(test_data: np.ndarray, predictions: np.ndarray) -> None:
    """
    Visualizes the test results.
    
    Args:
        test_data (np.ndarray): Actual test data.
        predictions (np.ndarray): Model predictions.
    """
    plt.plot(range(len(test_data)), test_data, label="Test Data")
    plt.plot(range(len(predictions)), predictions, label="Predictions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load training and testing data
    train_data = load_data("data/train.txt")
    test_data = load_data("data/test.txt")

    # Create the model
    model = LSTMModel(input_dim=1, learning_rate=0.1)

    # Train the model
    model.train(train_data, iterations=1000)

    # Test the model
    predictions = [model.forward_pass(x) for x in test_data]

    # Visualize results
    plot_results(test_data, predictions)
