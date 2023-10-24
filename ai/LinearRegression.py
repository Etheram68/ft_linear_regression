import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, file):
        """
        Initialize a LinearRegression object.

        Args:
        file (str): The file path to the dataset.
        """
        self.file = file
        self.theta0 = 0
        self.theta1 = 0
        self.mean_km = 0
        self.std_km = 0
        self.cost_history = []


    def read_file(self) -> pd.DataFrame:
        """
        Read the dataset from the specified file.

        Returns:
        pd.DataFrame: The dataset as a Pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.file)
        except pd.errors.EmptyDataError:
            print(f"File {self.file} is empty")
            exit()
        return df


    def normalize_features(self, df) -> pd.DataFrame:
        """
        Normalize the 'km' feature in the dataset.

        Args:
        df (pd.DataFrame): The dataset as a Pandas DataFrame.

        Returns:
        pd.DataFrame: The dataset with normalized 'km' feature.
        """
        self.mean_km = df['km'].mean()
        self.std_km = df['km'].std()
        df['km'] = (df['km'] - self.mean_km) / self.std_km
        return df

    def gradient_descent(self, X, Y, learning_rate, num_iterations) -> tuple:
        """
        Perform gradient descent to find the linear regression parameters.

        Args:
        X (pd.Series): The feature (normalized 'km') values.
        Y (pd.Series): The target (price) values.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations to run gradient descent.

        Returns:
        tuple: A tuple containing the optimized theta0 and theta1 values.
        """
        n = float(len(X))
        theta0 = 0
        theta1 = 0
        self.cost_history = np.zeros(num_iterations)

        for _ in range(num_iterations):
            Y_pred = theta0 + theta1 * X
            D_theta0 = (-2/n) * sum(Y - Y_pred)
            D_theta1 = (-2/n) * sum(X * (Y - Y_pred))

            theta0 = theta0 - learning_rate * D_theta0
            theta1 = theta1 - learning_rate * D_theta1

            self.cost_history[_] = sum((Y_pred - Y) ** 2) / (2 * n)
        return theta0, theta1


    def save_result(self) -> None:
        """
        Save the results (theta0, theta1, mean_km, std_km) to a file.
        """
        try:
            with open("thetas", "w") as file:
                file.write(f"{self.theta0},{self.theta1},{self.mean_km},{self.std_km}")
        except IOError as e:
            print(f"Error: {e}")
            exit(1)


    def display_result(self, cpy_km, df, num_iterations) -> None:
        """
        Display the results of linear regression and the cost function graph.

        Args:
        cpy_km (pd.Series): A copy of the 'km' feature values.
        df (pd.DataFrame): The dataset as a Pandas DataFrame.
        num_iterations (int): The number of iterations used in gradient descent.
        """
        plt.figure(1)
        plt.title("Result of linear regression")
        plt.scatter(cpy_km, df['price'])
        plt.plot(cpy_km, self.theta0 + self.theta1 * df['km'], color='red')
        plt.xlabel('km')
        plt.ylabel('price')
        plt.legend(['Data Points', 'Linear Regression'])
        plt.figure(2)
        plt.title("Cost function, gradient descent")
        plt.plot(range(num_iterations), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()


    def determination_coef(self, y, pred) -> float:
        """
        Calculate the determination coefficient (R-squared).

        Args:
        y (pd.Series): The actual target values.
        pred (pd.Series): The predicted target values.

        Returns:
        float: The determination coefficient (R-squared) value.
        """
        u = ((y - pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)

    def fit(self, learning_rate, num_iterations, graph) -> None:
        """
        Fit the linear regression model to the dataset.

        Args:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations to run gradient descent.
        graph (bool): Whether to display the results graph.

        """
        df = self.read_file()

        try:
            cpy_km = df['km'].copy()
            df = self.normalize_features(df)

            self.theta0, self.theta1 = self.gradient_descent(df['km'], df['price'], learning_rate=learning_rate, num_iterations=num_iterations)
            determination_coef = self.determination_coef(df['price'], self.theta0 + self.theta1 * df['km'])
            print(f"Determination coefficient: {determination_coef}")
            if graph:
                self.display_result(cpy_km, df, num_iterations)
        except KeyError as e:
            print(f"Error: The fit function lookings for column name {e}")
            exit(1)
        self.save_result()
