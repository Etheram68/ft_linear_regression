import argparse
from pathlib import Path

def load_fit_result(file: Path) -> tuple:
    try:
        with open(file, "r") as f:
            data = f.readline().split(",")
            theta0 = float(data[0])
            theta1 = float(data[1])
            mean_km = float(data[2])
            std_km = float(data[3])
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    return theta0, theta1, mean_km, std_km

def mileage_normalized(mileage: int, mean_km: float, std_km: float) -> float:
    return (mileage - mean_km) / std_km

def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + (theta1 * mileage)

def main() -> None:
    """
    Main function to estimate the price of a car based on its mileage using a trained model.

    Command-line arguments:
    - mileage (int): Mileage of the car for price estimation.
    - --file (str): Path to the training file containing model parameters (default="thetas").

    Prints the estimated price of the car.
    """
    parser = argparse.ArgumentParser(description="Execute this programe for train new prediction model")
    parser.add_argument('mileage', type=int, help="Mileage of car for estimate price")
    parser.add_argument("--file", type=str, default="thetas", help="Path to training file (default=thetas)")

    args = parser.parse_args()
    file = Path(args.file)

    if file.is_file():
        theta0, theta1, mean_km, std_km = load_fit_result(file)
    else:
        print(f"File {file} doesn't exist")
        exit(1)

    predicted_price = estimate_price(mileage_normalized(args.mileage, mean_km, std_km), theta0, theta1)

    if predicted_price < 0:
        predicted_price = 0

    print(f"Price estimated for car with {args.mileage} km: {predicted_price:.2f}€")


if __name__ == "__main__":
    main()
