import numpy as np
import matplotlib.pyplot as plt

def calculate_error(true_path, predicted_path):
    """
    Calculate the Euclidean distance error between the true path and a predicted path.

    :param true_path: 2D array of shape (2, N) representing the true path [x, y].
    :param predicted_path: 2D array of shape (2, N) representing the predicted path [x, y].
    :return: 1D array of errors for each time step.
    """
    return np.sqrt(np.sum((true_path - predicted_path) ** 2, axis=0))

def plot_error(true_path, predicted_ekf, robot_odometry, save_path=None):
    """
    Plot the error between the True Path vs Predicted EKF and True Path vs Robot Odometry.

    :param true_path: 2D array of shape (2, N) representing the true path [x, y].
    :param predicted_ekf: 2D array of shape (2, N) representing the EKF predicted path [x, y].
    :param robot_odometry: 2D array of shape (2, N) representing the robot odometry path [x, y].
    :param save_path: Optional path to save the plot as an image.
    """
    # Calculate errors
    error_ekf = calculate_error(true_path, predicted_ekf)
    error_odometry = calculate_error(true_path, robot_odometry)

    # Plot errors
    plt.figure(figsize=(10, 6))
    plt.plot(error_ekf, label="True Path vs Predicted EKF", color="red", linewidth=2)
    plt.plot(error_odometry, label="True Path vs Robot Odometry", color="blue", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Error [m]")
    plt.title("Error Comparison")
    plt.legend()
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Error plot saved at: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage (replace with actual data)
    true_path = np.array([[0, 1, 2], [0, 1, 2]])  # Replace with hxTrue[0:2, :]
    predicted_ekf = np.array([[0, 1.1, 2.1], [0, 0.9, 1.8]])  # Replace with hxEst[0:2, :]
    robot_odometry = np.array([[0, 1.2, 2.2], [0, 1.0, 2.0]])  # Replace with hxDR[0:2, :]

    plot_error(true_path, predicted_ekf, robot_odometry)