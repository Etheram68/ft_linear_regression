import pandas as pd


class LinearRegression:
    def __init__(self, file):
        """
        Initialize the LinearRegression model with a dataset file path and initial parameter values.

        Parameters:
        - file (str): Path to the dataset file.

        Attributes:
        - file (str): Path to the dataset file.
        - theta0 (float): Intercept parameter.
        - theta1 (float): Slope parameter.
        - mean_km (float): Mean value of the 'km' feature for normalization.
        - std_km (float): Standard deviation of the 'km' feature for normalization.
        """
        self.file = file
        self.theta0 = 0
        self.theta1 = 0
        self.mean_km = 0
        self.std_km = 0


    def _read_file(self) -> pd.DataFrame:
        """
        Read the dataset from the specified file into a pandas DataFrame.

        Returns:
        - df (pd.DataFrame): The dataset as a DataFrame.

        Raises:
        - pd.errors.EmptyDataError: If the file is empty.
        """
        try:
            df = pd.read_csv(self.file)
        except pd.errors.EmptyDataError:
            print(f"File {self.file} is empty")
            exit()
        return df


    def _gradient_descent(self, X, Y, learning_rate, num_iterations, convergence_threshold=0.01) -> tuple:
        """
        Perform gradient descent to optimize model parameters.

        Parameters:
        - X (list): Feature data.
        - Y (list): Target data.
        - learning_rate (float): Step size for gradient descent.
        - num_iterations (int): Maximum number of iterations.
        - convergence_threshold (float, optional): Threshold for convergence.

        Returns:
        - (theta0, theta1) (tuple): Optimized parameters (intercept and slope).

        Description:
        This function iteratively updates the model parameters using gradient descent
        until convergence or the maximum number of iterations is reached.
        """
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
        """
        Save the model's parameters (theta0, theta1, mean_km, and std_km) to a file called "thetas".

        Raises:
        - IOError: If an error occurs while writing to the file.
        """
        try:
            with open("thetas", "w") as file:
                file.write(f"{self.theta0},{self.theta1},{self.mean_km},{self.std_km}")
        except IOError as e:
            print(f"Error: {e}")
            exit(1)


    def fit(self, learning_rate, num_iterations) -> None:
        """
        Train the linear regression model using gradient descent.

        Parameters:
        - learning_rate (float): Step size for gradient descent.
        - num_iterations (int): Maximum number of iterations.

        Description:
        This function reads the dataset, normalizes the 'km' feature, and then uses gradient descent
        to optimize the model's parameters. It saves the parameters to a file.
        """
        df = self._read_file()

        # Normilize mileage
        try:
            self.mean_km = df['km'].mean()
            self.std_km = df['km'].std()
            df['km'] = (df['km'] - self.mean_km) / self.std_km

            self.theta0, self.theta1 = self._gradient_descent(df['km'], df['price'], learning_rate=learning_rate, num_iterations=num_iterations)
        except KeyError as e:
            print(f"Error: The fit function lookings for column name {e}")
            exit(1)
        self._save_result()
