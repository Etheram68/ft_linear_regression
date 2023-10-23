import argparse
from pathlib import Path
from ai.LinearRegression import LinearRegression


def main():
    """
    Main function to train a linear regression prediction model using the provided dataset.

    Command-line arguments:
    - --file (str): Path to the dataset file (default="data/data.csv").
    - --learning-rate (float): Learning rate for gradient descent (default=0.001).
    - --num_iterations (int): Number of iterations for gradient descent (default=2000).

    Trains a linear regression model using the specified dataset and hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Execute this programe for train new prediction model")
    parser.add_argument("--file", type=str, default="data/data.csv", help="Path to dataset file (default=data.csv)")
    parser.add_argument("--learning-rate", type=float, default="0.001", help="Set learning rate (default=0.01)")
    parser.add_argument("--num_iterations", type=int, default="2000", help="Set number iterations gradient Descent (default=2000)")

    args = parser.parse_args()
    file = Path(args.file)

    if file.is_file():
        lr = LinearRegression(file=file)
    else:
        print(f"File {file} doesn't exist")
        exit(1)

    lr.fit(learning_rate=args.learning_rate, num_iterations=args.num_iterations)

if __name__ == "__main__":
    main()
