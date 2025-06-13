"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)
Adapted by David Antunes (@davidamantunes)
"""

import math
import time  # Add this at the top with other imports
import os

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plots

import matplotlib.pyplot as plt
import numpy as np
from utils.angle import angle_mod
from utils.plot import plot_covariance_ellipse, plot_arrow
from slam_performance import PerformanceTracker
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis
from error_plot import plot_error  # Import the error plotting function

# Process noise covariance matrix, variance of x and y is 0.5m, yaw is 30deg
scale = 1
Cx = np.diag([0.3*scale, 0.3*scale, np.deg2rad(30.0)]) ** 2

#  Simulation parameter
Q_sim = np.diag([0.3, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([0.5, np.deg2rad(5.0)]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 110.0  # simulation time [s]
MAX_RANGE = 10.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True


def ekf_slam(xEst, PEst, u, z):
    # Start the stopwatch
    start_time = time.time()
    
    # Predict
    xEst, PEst = predict(xEst, PEst, u)
    initP = np.eye(2)

    # Update
    for iz in range(len(z[:, 0])):  # for each observation (line of z)
        min_id = search_correspond_landmark_id(xEst, PEst, z[iz, 0:2]) # 1st iteration, it will be 0

        nLM = calc_n_lm(xEst) # 1st iteration, it will be 0
        if min_id == nLM: # 1st iteration, both are 0, i.e. add new landmark
            print("New LM")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :]))) # Extend state vector
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP)))) # Extend covariance matrix
            # Update the state and covariance with the augmented versions
            xEst = xAug
            PEst = PAug
        lm = get_landmark_position_from_state(xEst, min_id) # Get the position of the landmark from the state vector
        
        # Calculate 3 crutial parameters
        # y: innovation (z_sensor - z_predicted) [z_predicted = h(u_predicted, 0 noise)], S: covariance of the innovation (H @ PEst @ H.T + Q), H: Jacobian of the measurement model (h function)
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

        K = (PEst @ H.T) @ np.linalg.inv(S) # Kalman gain
        # Update state and covariance
        xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])
    
    # Stop the stopwatch and calculate elapsed time
    elapsed_time = time.time() - start_time
    
    return xEst, PEst, elapsed_time


def predict(xEst, PEst, u):
    """
    Performs the prediction step of EKF SLAM

    :param xEst: nx1 state vector
    :param PEst: nxn covariance matrix
    :param u:    2x1 control vector
    :returns:    predicted state vector, predicted covariance
    """
    G, Fx = jacob_motion(xEst, u)
    xEst[0:STATE_SIZE] = motion_model(xEst[0:STATE_SIZE], u)
    # As Fx is an identity matrix, multipliying it with Cx will not change anything
    PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx 
    return xEst, PEst


def calc_input(time):
    v = 1.0  # [m/s]
    # yaw_rate = 0.1  # [rad/s]
    yaw_rate = 2 * np.pi / SIM_TIME  # Adjust yaw rate to complete a full circle in SIM_TIME
    u = np.array([[v, yaw_rate]]).T
    return u

def calc_square_input(time):
    """
    Generate input for a square trajectory (10x10 meters).
    The robot alternates between moving straight and turning 90 degrees.
    """
    v = 0.5  # [m/s], constant forward velocity
    side_length = 15.0  # Length of each side of the square
    turn_duration = np.pi / 2 / v  # Time to complete a 90-degree turn
    straight_duration = side_length / v  # Time to complete a straight segment

    # Total time for one side + one turn
    segment_duration = straight_duration + turn_duration

    # Determine the phase of the motion
    phase_time = time % segment_duration

    if phase_time < straight_duration:
        # Move straight
        yaw_rate = 0.0  # No turning
    else:
        # Turn 90 degrees
        yaw_rate = np.pi / 2 / turn_duration  # Constant yaw rate for 90-degree turn

    u = np.array([[v, yaw_rate]]).T
    return u


def observation(xTrue, xd, u, Aruco):
    """
    :param xTrue: the true pose of the system
    :param xd:    the current noisy estimate of the system
    :param u:     the current control input
    :param Aruco:  the true position of the landmarks

    :returns:     Computes the true position, observations, dead reckoning (noisy) position,
                  and noisy control function
    """
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = np.zeros((0, 3))

    for i in range(len(Aruco[:, 0])): # Test all beacons, only add the ones we can see (within MAX_RANGE)

        # Calculate relative coordinates:
        dx = Aruco[i, 0] - xTrue[0, 0]
        dy = Aruco[i, 1] - xTrue[1, 0]
        # Calculate distance to landmark (Euclidean distance (√(dx² + dy²))):
        d = math.hypot(dx, dy)
        # Calculate relative bearing (angle):
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:  # Only process landmarks within sensing range
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise (distance measurement)
            angle_n = angle + np.random.randn() * Q_sim[1, 1] ** 0.5  # add noise (angle measurement)
            zi = np.array([dn, angle_n, i]) # [distance, angle, landmark ID]
            z = np.vstack((z, zi)) # Add this observation to the observations matrix

    # add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = (F @ x) + (B @ u) # Predict the next state ( "@" is matrix multiplication)
    return x


def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_motion(x, u):
    """
    Calculates the jacobian of motion model.

    :param x: The state, including the estimated position of the system
    :param u: The control function
    :returns: G:  Jacobian
              Fx: STATE_SIZE x (STATE_SIZE + 2 * num_landmarks) matrix where the left side is an identity matrix
    """
    # np.eye(STATE_SIZE) = 3x3 identity, 3x8 zeros for All landmarks with 2 values (each with x, y)
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_lm(x)))))
    # print("Fx =\n", Fx) DEBUG

    # u[0, 0] é a velocidade linear e x[2, 0] é o ângulo
    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)
    # print("jF =\n", jF) DEBUG

    # Reason why we do Fx.T @ jF @ Fx and not just jF is to create a bigger matrix with jF in the top-left and zeros elsewhere.
    G = np.eye(len(x)) + Fx.T @ jF @ Fx
    # print("G =\n", G) DEBUG

    return G, Fx,


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def get_landmark_position_from_state(x, ind):
    # Extracts the position (x,y coordinates) of a specific landmark from the state vector.
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm


def search_correspond_landmark_id(xAug, PAug, zi):
    """
    Landmark association with Mahalanobis distance (measure of the distance between a point and a distribution)

    If this landmark is at least M_DIST_TH units away from all known landmarks,
    it is a NEW landmark.

    :param xAug: The estimated state
    :param PAug: The estimated covariance
    :param zi:   the read measurements of specific landmark
    :returns:    landmark id
    """

    nLM = calc_n_lm(xAug) # Number of landmarks in the state vector

    min_dist = []

    for i in range(nLM): # Goes through all landmarks in the state vector
        lm = get_landmark_position_from_state(xAug, i)
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)
        min_dist.append(y.T @ np.linalg.inv(S) @ y) # Squared Mahalanobis distance
 
    min_dist.append(M_DIST_TH)  # new landmark
    # A small Mahalanobis distance indicates a good match between a measurement and a landmark
    # This means that if we choose the landmark with the smallest Mahalanobis distance, it is likely to be the correct one.
    # The index of the minimum distance corresponds to the landmark ID
    min_id = min_dist.index(min(min_dist)) 

    return min_id


def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2] # Calculate relative position vector (x_delta, y_delta)= (x_lm - x_est, y_lm - y_est)
    q = (delta.T @ delta)[0, 0] # Calculate squared distance
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0] # Calculate expected bearing angle
    # zp is h(u predicted, 0 noise)
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]]) # Construct predicted measurement, i.e. [distance, angle], based on the relative position from the robot to the landmark to the extimated position of the robot
    
    y = (z - zp).T # Calculate innovation, i.e. the difference between the measurement (from the sensor) and the predicted measurement (done above)
    y[1] = pi_2_pi(y[1])
    
    H = jacob_h(q, delta, xEst, LMid + 1) # Calculate measurement Jacobian
    S = H @ PEst @ H.T + Cx[0:2, 0:2] # Project state uncertainty + measurement noise

    return y, S, H


def jacob_h(q, delta, x, i):
    """
    Calculates the jacobian of the measurement function

    :param q:     the range from the system pose to the landmark
    :param delta: the difference between a landmark position and the estimated system position
    :param x:     the state, including the estimated system position
    :param i:     landmark id + 1
    :returns:     the jacobian H
    """
    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def pi_2_pi(angle):
    return angle_mod(angle)

def choose_trajectory():
    """
    Prompt the user to choose a trajectory type.
    """
    print("Choose a trajectory:")
    print("1 - Circle")
    print("2 - Square")
    
    while True:
        choice = input("Enter 1 or 2: ")
        if choice == "1":
            print("You chose a circular trajectory.")
            return calc_input  # Circular trajectory function
        elif choice == "2":
            print("You chose a square trajectory.")
            return calc_square_input  # Square trajectory function
        else:
            print("Invalid choice. Please enter 1 or 2.")


def mahalanobis_distance(point, mean, covariance):
    """
    Calculates the Mahalanobis distance between a point and a distribution.
    """
    diff = point - mean
    inv_cov = np.linalg.inv(covariance)
    return np.sqrt(diff.T @ inv_cov @ diff)

def match_landmarks_mahalanobis(true_landmarks, estimated_landmarks, estimated_covariances):
    """
    Matches true landmarks to estimated landmarks using Mahalanobis distance.
    true_landmarks: Nx2 array of true landmark positions
    estimated_landmarks: Nx2 array of estimated landmark positions
    estimated_covariances: List of Nx2x2 covariance matrices for estimated landmarks
    Returns the matched true and estimated landmarks.
    """
    n = true_landmarks.shape[0]
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = mahalanobis_distance(true_landmarks[i], estimated_landmarks[j], estimated_covariances[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return true_landmarks[row_ind], estimated_landmarks[col_ind]

def align_landmarks_no_scale(true_landmarks, estimated_landmarks):
    """
    Aligns estimated landmarks to true landmarks using rotation and translation (no scaling).
    Returns the aligned landmarks, rotation matrix, and translation vector.
    """
    mu_true = np.mean(true_landmarks, axis=0)
    mu_est = np.mean(estimated_landmarks, axis=0)
    X = estimated_landmarks - mu_est
    Y = true_landmarks - mu_true

    # Optimal rotation (Kabsch algorithm)
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and translation
    X_aligned = (X @ R) + mu_true
    t = mu_true - mu_est @ R
    return X_aligned, R, t


def main():
    print(__file__ + " start!!")

    sim_time = 0.0
    
    # Initialize the performance tracker
    tracker = PerformanceTracker()


    # Aruco positions [x, y]
    Aruco = np.array([[5, 0],
                     [10, 0],
                     [15, 2],
                     [17, -2],
                     [16, 10],
                     [16, 16],
                     [13, 16],
                     [5, 16],
                     [0, 16],
                     [2, 14],
                     [2, 10]])

    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)
    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    
    trajectory_function = choose_trajectory()

    while SIM_TIME >= sim_time:
        sim_time += DT
        
        # Use the chosen trajectory function to calculate input
        u = trajectory_function(sim_time)

        # xTrue is the true state of the robot, and xDR is the dead reckoning state
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, Aruco)
        
        xEst, PEst, elapsed_time = ekf_slam(xEst, PEst, ud, z)
        
        # Calculate total landmarks detected
        total_landmarks = calc_n_lm(xEst)
        
        # Add data to the tracker
        if (total_landmarks < 12):
            tracker.add_data_point(total_landmarks, elapsed_time)
        
        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(Aruco[:, 0], Aruco[:, 1], "*k")  # Plot landmarks
            plt.plot(xEst[0], xEst[1], ".r")  # Plot robot position

            # Plot covariance ellipse for the robot
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst[0:2, 0:2], chi2=3.0, color="-r")

            # Plot robot orientation as an arrow
            plot_arrow(xEst[0, 0], xEst[1, 0], xEst[2, 0], arrow_length=1.0)

            # Plot landmarks with covariance ellipses
            for i in range(calc_n_lm(xEst)):
                lm_x = xEst[STATE_SIZE + i * 2, 0]
                lm_y = xEst[STATE_SIZE + i * 2 + 1, 0]
                lm_cov = PEst[STATE_SIZE + i * 2:STATE_SIZE + i * 2 + 2,
                              STATE_SIZE + i * 2:STATE_SIZE + i * 2 + 2]
                plt.plot(lm_x, lm_y, "xg")  # Plot landmark position
                plot_covariance_ellipse(lm_x, lm_y, lm_cov, chi2=3.0, color="-g")

            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")  # True path
            plt.plot(hxDR[0, :], hxDR[1, :], "-k")  # Dead reckoning path
            plt.plot(hxEst[0, :], hxEst[1, :], "-r")  # EKF SLAM path
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    # After simulation completes, use the tracker to save data and create visualizations
    tracker.save_data()
    tracker.plot_performance()
    tracker.plot_mean_times_by_landmarks()  # Add this line
    tracker.print_stats()
    
    # Still save the trajectory plot as before
    plt.figure(figsize=(10, 8))
    plt.plot(Aruco[:, 0], Aruco[:, 1], "*k", label="Landmarks")
    plt.plot(hxTrue[0, :], hxTrue[1, :], "-b", label="True Path")
    plt.plot(hxDR[0, :], hxDR[1, :], "-k", label="Robot Odometry")
    plt.plot(hxEst[0, :], hxEst[1, :], "-r", label="EKF SLAM")

    # Plot landmarks with covariance ellipses
    for i in range(calc_n_lm(xEst)):
        lm_x = xEst[STATE_SIZE + i * 2, 0]
        lm_y = xEst[STATE_SIZE + i * 2 + 1, 0]
        lm_cov = PEst[STATE_SIZE + i * 2:STATE_SIZE + i * 2 + 2,
                      STATE_SIZE + i * 2:STATE_SIZE + i * 2 + 2]
        plt.plot(lm_x, lm_y, "xg")  # Plot landmark position
        plot_covariance_ellipse(lm_x, lm_y, lm_cov, chi2=3.0, color="-g")  # Add covariance ellipse

    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    filepath = os.path.join("performance", "final_plot.png")  # Save in the 'performance' folder
    plt.savefig(filepath)
    print(f"Final plot saved as '{filepath}'.")

    # Final plot with covariance ellipses
    plt.figure(figsize=(10, 8))

    # Extrai os marcos estimados do vetor de estado final
    nLM = int((len(xEst) - STATE_SIZE) / LM_SIZE)
    est_lms = np.array([[xEst[STATE_SIZE + i * 2, 0], xEst[STATE_SIZE + i * 2 + 1, 0]] for i in range(nLM)])
    est_covs = [PEst[STATE_SIZE + i * 2:STATE_SIZE + i * 2 + 2, STATE_SIZE + i * 2:STATE_SIZE + i * 2 + 2] for i in range(nLM)]

    # Correspondência ótima (matching) entre LM reais e estimados
    Aruco_matched, est_lms_matched = match_landmarks_mahalanobis(Aruco, est_lms, est_covs)

    # Alinhamento (rotação + translação, sem escala)
    est_lms_aligned, R_align, t_align = align_landmarks_no_scale(Aruco_matched, est_lms_matched)

    # Landmarks reais
    plt.scatter(Aruco_matched[:, 0], Aruco_matched[:, 1], c='blue', marker='*', s=120, label='Real Landmarks')

    # Landmarks estimados alinhados
    plt.scatter(est_lms_aligned[:, 0], est_lms_aligned[:, 1], c='red', marker='x', s=80, label='Predicted Landmarks')

    # Liga landmarks reais aos alinhados
    for i in range(Aruco_matched.shape[0]):
        plt.plot([Aruco_matched[i, 0], est_lms_aligned[i, 0]], [Aruco_matched[i, 1], est_lms_aligned[i, 1]], ':r', alpha=0.5)

    # Adiciona elipses de covariância para os landmarks estimados
    for i, cov in enumerate(est_covs):
        aligned_lm = est_lms_aligned[i]
        plot_covariance_ellipse(aligned_lm[0], aligned_lm[1], cov, chi2=3.0, color="-g")

    # Trajetórias
    plt.plot(hxDR[0, :], hxDR[1, :], '-k', linewidth=3, label='Robot Odometry')
    plt.plot(hxEst[0, :], hxEst[1, :], '-r', linewidth=2, label='Predicted EKF')
    plt.plot(hxTrue[0, :], hxTrue[1, :], ':c', linewidth=2, label='True Path')

    # Configurações do gráfico
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    # Salva o plot final
    plot_dir = "performance"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "plot_FINAL.png"), dpi=300)
    print(f"Final plot saved as '{os.path.join(plot_dir, 'plot_FINAL.png')}'.")

    # After simulation completes, plot the error
    error_plot_dir = "error"  # Change directory to 'error'
    os.makedirs(error_plot_dir, exist_ok=True)  # Ensure the folder exists
    error_plot_path = os.path.join(error_plot_dir, "error_plot.png")
    plot_error(hxTrue[0:2, :], hxEst[0:2, :], hxDR[0:2, :], save_path=error_plot_path)
    print(f"Error plot saved as '{error_plot_path}'.")

if __name__ == '__main__':
    main()
