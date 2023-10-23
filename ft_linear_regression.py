import argparse
import pandas as pd
from pathlib import Path


class LinearRegression:
    def __init__(self, file):
        self.file = file
        self.theta0 = 0
        self.theta1 = 0
        self.mean_km = 0
        self.std_km = 0


    def _read_file(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file)
        except pd.errors.EmptyDataError:
            print(f"File {self.file} is empty")
            exit()
        return df


    def _gradient_descent(self, X, Y, learning_rate, num_iterations, convergence_threshold=0.001) -> tuple:
        n = float(len(X))
        theta0 = 0
        theta1 = 0

        for _ in range(num_iterations):
            Y_pred = [theta0 + theta1 * x for x in X]
            D_theta0 = (-2/n) * sum(Y[i] - Y_pred[i] for i in range(len(Y)))
            D_theta1 = (-2/n) * sum(X[i] * (Y[i] - Y_pred[i]) for i in range(len(X)))

            new_theta0 = theta0 - learning_rate * D_theta0
            new_theta1 = theta1 - learning_rate * D_theta1

            diff = abs(new_theta0 - theta0) + abs(new_theta1 - theta1)
            if diff < convergence_threshold:
                break

            theta0 = new_theta0
            theta1 = new_theta1
        return theta0, theta1


    def _save_result(self) -> None:
        try:
            with open("thetas", "w") as file:
                file.write(f"{self.theta0},{self.theta1},{self.mean_km},{self.std_km}")
        except IOError as e:
            print(f"Error: {e}")
            exit(1)


    def fit(self, learning_rate, num_iterations) -> None:
        df = self._read_file()

        # Normilize mileage
        self.mean_km = df['km'].mean()
        self.std_km = df['km'].std()
        df['km'] = (df['km'] - self.mean_km) / self.std_km

        self.theta0, self.theta1 = self._gradient_descent(df['km'], df['price'], learning_rate=learning_rate, num_iterations=num_iterations)
        self._save_result()


def main():
    parser = argparse.ArgumentParser(description="Execute this programe for train new prediction model")
    parser.add_argument("--file", type=str, default="data.csv", help="Path to dataset file (default=data.csv)")
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
