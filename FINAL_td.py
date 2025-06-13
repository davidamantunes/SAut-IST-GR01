import math
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plots

import matplotlib.pyplot as plt
import numpy as np
from utils.angle import angle_mod
from utils.plot import plot_covariance_ellipse, plot_arrow
from utils.get_u_from_pose import extract_velocity_vector
import subprocess
import re
from utils.get_aruco import process_image_at_time, get_topic_time_range
from scipy.optimize import linear_sum_assignment

# Process noise covariance matrix, variance of x and y is 0.5m, yaw is 30deg
Cx = np.diag([0.01, 0.01, np.deg2rad(-50.0)]) ** 2 # Process noise covariance
R = np.diag([3000, np.deg2rad(1700)]) ** 2       # Observation noise covariance
angulo_extra = 0

#  Simulation parameter
R_sim = np.zeros((2, 2))

DT = 0.1  # time tick [s]
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True

# Global variables to store the extracted velocity data
times_global = None
u_global = None

# Define variables for the rosbag information
rosbag_file = "oito.bag"
camera_topic = "/usb_cam/image_raw"
rosbag_start_time = None
rosbag_end_time = None


def ekf_slam(xEst, PEst, u, z, id_to_lm_idx):
    # Predict
    xEst, PEst = predict(xEst, PEst, u)
    initP = np.eye(2)

    # Update
    for iz in range(len(z[:, 0])):  # for each observation (line of z)
        obs_id = int(z[iz, 2])
        if obs_id in id_to_lm_idx:
            min_id = id_to_lm_idx[obs_id]
        else:
            min_id = calc_n_lm(xEst)  # New landmark

        nLM = calc_n_lm(xEst)
        if min_id == nLM:
            print(f"New LM for ID {obs_id}")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            xEst = xAug
            PEst = PAug
            id_to_lm_idx[obs_id] = nLM  # Map this ID to the new landmark index

        lm = get_landmark_position_from_state(xEst, id_to_lm_idx[obs_id])
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2], id_to_lm_idx[obs_id])

        K = (PEst @ H.T) @ np.linalg.inv(S)
        xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst, id_to_lm_idx


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
    v = 1.0  # [m/s], constant forward velocity
    side_length = 10.0  # Length of each side of the square
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

def load_velocity_data(bag_file=None):
    """
    Load velocity data from the ROS bag file
    """
    global times_global, u_global, rosbag_file
    if bag_file is None:
        bag_file = rosbag_file
    print(f"Loading velocity data from bag file: {bag_file}")
    times, u = extract_velocity_vector(bag_file)
    if u is None:
        print(f"Failed to extract velocity data from bag file: {bag_file}")
        return False
    
    # Print some statistics about the loaded data
    times_global = times
    u_global = u
    duration = times[-1] - times[0]
    avg_interval = duration / (len(times) - 1) if len(times) > 1 else 0
    
    print(f"Successfully loaded {u.shape[1]} velocity samples")
    print(f"Time range: 0 to {duration:.2f}s, Avg interval: {avg_interval:.4f}s")
    print(f"Linear velocity range: {np.min(u[0]):.2f} to {np.max(u[0]):.2f} m/s")
    print(f"Angular velocity range: {np.min(u[1]):.2f} to {np.max(u[1]):.2f} rad/s")
    return True

def real_input(time):
    """
    Generate the real input for a trajectory using actual data from the ROS bag,
    with proper time scaling to match simulation time.
    """
    global times_global, u_global
    
    # Initialize data if not done yet
    if times_global is None or u_global is None:
        success = load_velocity_data()
        if not success:
            raise RuntimeError("Failed to load velocity data from bag file. Check the file path and make sure the bag file exists.")
    
    # Scale simulation time to bag file time range
    if len(times_global) > 1:
        max_data_time = times_global[-1]
        # Scale the current simulation time to fit within the available data time range
        # This ensures we use the full dataset even if simulation time exceeds data time
        scaled_time = (time % max_data_time) if max_data_time > 0 else 0
    else:
        scaled_time = 0
    
    # Find the closest time index
    closest_idx = np.argmin(np.abs(times_global - scaled_time))
    
    # Get the corresponding velocity values
    v = u_global[0, closest_idx]
    yaw_rate = u_global[1, closest_idx]
    
    # Print debug info occasionally
    if time < 0.2 or abs(time - int(time)) < 0.1:  # Print at start and whole seconds
        print(f"Time: {time:.1f}s, Scaled: {scaled_time:.1f}s, Using sample {closest_idx}/{len(times_global)}, v={v:.2f}, yaw_rate={yaw_rate:.2f}")
    
    # Return as control vector
    return np.array([[v, yaw_rate]]).T


def get_aruco_observations(current_time):
    """
    Get ArUco marker observations from the rosbag at the specified time
    
    Args:
        current_time: Current simulation time in seconds
        
    Returns:
        z: numpy array of observations in format [distance, angle, landmark_id]
    """
    global rosbag_start_time, rosbag_end_time
    
    # Initialize the rosbag time range if not done yet
    if rosbag_start_time is None or rosbag_end_time is None:
        rosbag_start_time, rosbag_end_time = get_topic_time_range(rosbag_file, camera_topic)
        if rosbag_start_time is None:
            print("Error: Could not get time range from rosbag")
            return np.zeros((0, 3))
        print(f"Rosbag time range: {rosbag_end_time - rosbag_start_time:.3f} seconds")
    
    # Scale the simulation time to fit within the rosbag time range
    rosbag_duration = rosbag_end_time - rosbag_start_time
    if rosbag_duration <= 0:
        return np.zeros((0, 3))
    
    # Calculate the relative time in the rosbag
    relative_time = (current_time % rosbag_duration)
    target_time = rosbag_start_time + relative_time
    
    # Get detections from the rosbag at the specified time
    detections = []
    try:
        # Call the process_image_at_time function directly
        results = process_image_at_time(rosbag_file, camera_topic, target_time, return_results=True)
        if results:
            detections = results
    except Exception as e:
        print(f"Error getting ArUco observations: {e}")
        return np.zeros((0, 3))
    
    # Process the detections into the format expected by EKF SLAM
    z = np.zeros((0, 3))
    for detection in detections:
        if 'id' in detection and 'distance' in detection and 'angle_rad' in detection:
            marker_id = detection['id']
            distance = detection['distance']
            angle = -detection['angle_rad'] + angulo_extra #ALTERAR
            
            # Add to observation matrix
            zi = np.array([distance, angle, marker_id])
            z = np.vstack((z, zi))
            
            #print(f"Added observation: ID={marker_id}, Distance={distance:.2f}m, Angle={angle:.2f}rad")
    
    return z


# Modify the observation function to use real ArUco detections
def observation(xTrue, xd, u, Aruco, current_time):
    """
    Get observations from ArUco markers in the rosbag
    
    :param xTrue: the true pose of the system
    :param xd: the current noisy estimate of the system
    :param u: the current control input
    :param Aruco: the true position of the landmarks
    :param current_time: current simulation time
    
    :returns: Computes the true position, observations, dead reckoning position, and noisy control
    """
    # Update the true position using the motion model
    xTrue = motion_model(xTrue, u)
    
    # Get real observations from the ArUco markers in the rosbag
    z = get_aruco_observations(current_time)
    
    # If no ArUco markers detected, return empty observation matrix
    #if len(z) == 0:
    #    print(f"No ArUco markers detected at time {current_time:.1f}s")
    
    # Add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T
    
    # Update dead reckoning
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
                   [0.0, 0.0,  DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)
    # print("jF =\n", jF) DEBUG

    # Reason why we do Fx.T @ jF @ Fx and not
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
    min_id = min_dist.index(min(min(min_dist))) 

    return min_id


def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_h(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + R  # <-- aqui usas a nova matriz R
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
    print("3 - Real data")
    
    while True:
        choice = input("Enter 1, 2 or 3: ")
        if choice == "1":
            print("You chose a circular trajectory.")
            return calc_input  # Circular trajectory function
        elif choice == "2":
            print("You chose a square trajectory.")
            return calc_square_input  # Square trajectory function
        elif choice == "3":
            print("You chose a real data trajectory.")
            return real_input  # Real trajectory function
        else:
            print("Invalid choice. Please enter 1 or 2.")


def mahalanobis_distance(x, y, cov):
    """Calcula a distância de Mahalanobis entre x e y com matriz de covariância cov."""
    diff = x - y
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(cov.shape[0])  # fallback se a matriz não for invertível
    return np.sqrt(diff.T @ inv_cov @ diff)

def match_landmarks_mahalanobis(true_landmarks, estimated_landmarks, estimated_covariances):
    """
    Faz correspondência ótima entre marcos reais e estimados usando distância de Mahalanobis.
    true_landmarks: Nx2
    estimated_landmarks: Nx2
    estimated_covariances: lista de Nx2x2 matrizes de covariância
    Retorna os arrays ordenados para alinhamento.
    """
    n = true_landmarks.shape[0]
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = mahalanobis_distance(true_landmarks[i], estimated_landmarks[j], estimated_covariances[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return true_landmarks[row_ind], estimated_landmarks[col_ind]


def align_landmarks_with_scale(true_landmarks, estimated_landmarks):
    """
    Alinha os marcos estimados aos reais usando rotação, translação e escala (Procrustes).
    Retorna os marcos alinhados, o fator de escala, a matriz de rotação e o vetor de translação.
    """
    mu_true = np.mean(true_landmarks, axis=0)
    mu_est = np.mean(estimated_landmarks, axis=0)
    X = estimated_landmarks - mu_est
    Y = true_landmarks - mu_true

    # Escala ótima
    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)
    s = norm_Y / norm_X if norm_X > 0 else 1.0

    # Rotação ótima
    U, _, Vt = np.linalg.svd((X * s).T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Aplica escala, rotação e translação
    X_aligned = (X * s) @ R + mu_true
    t = mu_true - (mu_est * s) @ R
    return X_aligned, s, R, t

def align_landmarks_no_scale(true_landmarks, estimated_landmarks):
    """
    Alinha os marcos estimados aos reais usando apenas rotação e translação (sem escala).
    Retorna os marcos alinhados, a matriz de rotação e o vetor de translação.
    """
    mu_true = np.mean(true_landmarks, axis=0)
    mu_est = np.mean(estimated_landmarks, axis=0)
    X = estimated_landmarks - mu_est
    Y = true_landmarks - mu_true

    # Rotação ótima (Kabsch)
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Aplica rotação e translação (sem escala)
    X_aligned = (X @ R) + mu_true
    t = mu_true - mu_est @ R
    return X_aligned, R, t


def main():
    print(__file__ + " start!!")

    load_velocity_data()
    global SIM_TIME, times_global
    if times_global is not None and len(times_global) > 0:
        SIM_TIME = times_global[-1]
        print(f"SIM_TIME ajustado para {SIM_TIME:.2f} segundos (duração dos dados reais)")
    else:
        SIM_TIME = 1  # fallback padrão TEMPO DE SIMULAÇÂO

    time = 0.0

    Aruco = np.array([[1, 0], #Rosbag volta_4
                     [1.5, 1],
                     [0, 3.5],
                     [-2, 3],
                     [-1.5, 1.5],
                     [-1, 5],
                     [2, 3],
                     [-1, -2.5],
                     [-2, 0],])

    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)
    xDR = np.zeros((STATE_SIZE, 1))

    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    # Defina hxLine ANTES do loop principal!
    hxLine = np.array([
        [0, 0],
        [3, 0],
        [3.5, 0.5]
    ])

    # --- Adicione estas listas para armazenar os erros e landmarks ---
    error_hxline_ekf = []
    error_hxline_odom = []
    n_landmarks_seen = []
    time_list = []

    trajectory_function = choose_trajectory()

    id_to_lm_idx = {}  # Novo: mapeamento de ID para índice de landmark

    while SIM_TIME >= time:
        time += DT

        u = trajectory_function(time)
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, Aruco, time)

        xEst, PEst, id_to_lm_idx = ekf_slam(xEst, PEst, ud, z, id_to_lm_idx)

        x_state = xEst[0:STATE_SIZE]
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        # --- Calcule e armazene os erros e número de landmarks ---
        err_ekf = np.linalg.norm(xTrue[0:2, 0] - xEst[0:2, 0])
        err_odom = np.linalg.norm(xTrue[0:2, 0] - xDR[0:2, 0])
        #Erro em relação ao ponto mais próximo de hxLine
        ekf_pos = xEst[0:2, 0]
        dists_ekf = np.linalg.norm(hxLine - ekf_pos.reshape(1, 2), axis=1)
        err_hxline_ekf = np.min(dists_ekf)

        # Erro Odometria em relação à hxLine
        odom_pos = xDR[0:2, 0]
        dists_odom = np.linalg.norm(hxLine - odom_pos.reshape(1, 2), axis=1)
        err_hxline_odom = np.min(dists_odom)

        error_hxline_ekf.append(err_hxline_ekf)
        error_hxline_odom.append(err_hxline_odom)
        n_landmarks_seen.append(z.shape[0])
        time_list.append(time)

        if show_animation:
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

            #plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")  # True path
            plt.plot(hxDR[0, :], hxDR[1, :], "-k")  # Dead reckoning path
            plt.plot(hxEst[0, :], hxEst[1, :], "-r")  # EKF SLAM path
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)
            
    # Create directory for saving plots if it doesn't exist
    import os
    plot_dir = "ekf_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory '{plot_dir}' for saving plots")

    # Save separate plots for each trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(Aruco[:, 0], Aruco[:, 1], "*k", label="Landmarks")
    plt.plot(hxTrue[0, :], hxTrue[1, :], "-b", linewidth=2, label="True Path")
    plt.legend()
    plt.title("True Robot Path")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "true_path_plot.png"), dpi=300)
    print(f"True path plot saved in '{plot_dir}/true_path_plot.png'")

    plt.figure(figsize=(10, 8))
    plt.plot(Aruco[:, 0], Aruco[:, 1], "*k", label="Landmarks")
    plt.plot(hxDR[0, :], hxDR[1, :], "-k", linewidth=2, label="Dead Reckoning")
    plt.legend()
    plt.title("Dead Reckoning Path")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "dead_reckoning_plot.png"), dpi=300)
    print(f"Dead reckoning plot saved in '{plot_dir}/dead_reckoning_plot.png'")

    plt.figure(figsize=(10, 8))
    plt.plot(Aruco[:, 0], Aruco[:, 1], "*k", label="Landmarks")
    plt.plot(hxEst[0, :], hxEst[1, :], "-r", linewidth=2, label="EKF SLAM")
    plt.legend()
    plt.title("EKF SLAM Path")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "ekf_slam_plot.png"), dpi=300)
    print(f"EKF SLAM plot saved in '{plot_dir}/ekf_slam_plot.png'")

    # Also keep the combined plot
    plt.figure(figsize=(10, 8))
    plt.plot(Aruco[:, 0], Aruco[:, 1], "*k", label="Landmarks")
    plt.plot(hxTrue[0, :], hxTrue[1, :], "-b", label="True Path")
    plt.plot(hxDR[0, :], hxDR[1, :], "-k", label="Dead Reckoning")
    plt.plot(hxEst[0, :], hxEst[1, :], "-r", label="EKF SLAM")

    # --- Acrescenta isto para o robô ---
    robot_x = xEst[0, 0]
    robot_y = xEst[1, 0]
    robot_yaw = xEst[2, 0]
    robot_cov = PEst[0:2, 0:2]

    # Elipse de covariância do robô (vermelho)
    plot_covariance_ellipse(robot_x, robot_y, robot_cov, chi2=3.0, color="-r")
    # Cruz vermelha na posição do robô
    plt.plot(robot_x, robot_y, "xr", markersize=10, label="Robot")
    # Seta da direção do robô (vermelho)
    plot_arrow(robot_x, robot_y, robot_yaw, arrow_length=1.0, head_width=0.15, fc="k", ec="k")

    # Landmarks estimadas e suas elipses
    nLM = int((len(xEst) - 3) / 2)
    for i in range(nLM):
        lm_x = xEst[3 + i * 2, 0]
        lm_y = xEst[3 + i * 2 + 1, 0]
        lm_cov = PEst[3 + i * 2:3 + i * 2 + 2, 3 + i * 2:3 + i * 2 + 2]
        plt.plot(lm_x, lm_y, "xg", label="Estimated LM" if i == 0 else None)
        plot_covariance_ellipse(lm_x, lm_y, lm_cov, chi2=3.0, color="-g")

    plt.legend()
    plt.title("Combined Robot Paths")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "final_plot.png"), dpi=300)
    print(f"Combined plot saved in '{plot_dir}/final_plot.png'")

    # --- Plot automático dos landmarks reais vs previstos (alinhados) ---
    def align_landmarks(true_landmarks, estimated_landmarks):
        """
        Alinha os marcos estimados aos reais usando o método de Procrustes (Kabsch).
        true_landmarks: Nx2 array com as posições reais
        estimated_landmarks: Nx2 array com as posições estimadas
        Retorna os marcos estimados alinhados.
        """
        mu_true = np.mean(true_landmarks, axis=0)
        mu_est = np.mean(estimated_landmarks, axis=0)
        X = estimated_landmarks - mu_est
        Y = true_landmarks - mu_true

        U, _, Vt = np.linalg.svd(X.T @ Y)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        X_aligned = (X @ R) + mu_true
        return X_aligned

    # Extrai os marcos estimados do vetor de estado final
    nLM = int((len(xEst) - 3) / 2)
    est_lms = np.array([[xEst[3 + i*2, 0], xEst[3 + i*2 + 1, 0]] for i in range(nLM)])

    # Alinha os marcos estimados aos reais
    est_lms_aligned = align_landmarks(Aruco, est_lms)

    ## PLOT
    plt.figure(figsize=(10, 8))
    # Extrai os marcos estimados do vetor de estado final
    nLM = int((len(xEst) - 3) / 2)
    est_lms = np.array([[xEst[3 + i*2, 0], xEst[3 + i*2 + 1, 0]] for i in range(nLM)])
    est_covs = [PEst[3 + i*2:3 + i*2 + 2, 3 + i*2:3 + i*2 + 2] for i in range(nLM)]

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

    plt.plot(hxLine[:, 0], hxLine[:, 1], ':c', linewidth=2, label='True path')
    plt.plot(hxEst[0, :], hxEst[1, :], '-g', linewidth=2, label='Robot Odometry')
    plt.plot(hxTrue[0, :], hxTrue[1, :], '-r', linewidth=2, label='Predicted EKF')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_FINAL.png"), dpi=300)

    # --- Adicione este bloco para o gráfico de erro e landmarks ---
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    l1, = ax1.plot(time_list, error_hxline_ekf, label="EKF Error", color='red')
    l2, = ax1.plot(time_list, error_hxline_odom, label="Odometry Error", color='blue')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Distance [m]")

    ax2 = ax1.twinx()
    l3, = ax2.plot(time_list, n_landmarks_seen, label="Landmarks", color='purple', linestyle='dashed')
    ax2.set_ylabel("Number of Landmarks Seen")

    #plt.title("Distância à Linha de Referência (hxLine) e Landmarks Observados")

    lines = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "erro_landmarks_plot.png"), dpi=300)
    print(f"Erro e landmarks plot salvo em '{plot_dir}/erro_landmarks_plot.png'")

    # --- Estatísticas relevantes ---
    print("\n--- Estatísticas Relevantes ---")

    # Comprimento do percurso EKF
    ekf_path = hxEst[:2, :].T  # shape (N, 2)
    ekf_dist = np.linalg.norm(np.diff(ekf_path, axis=0), axis=1)
    ekf_length = np.sum(ekf_dist)
    print(f"Comprimento do percurso EKF: {ekf_length:.2f} m")

    # Comprimento do percurso Verdadeiro
    true_path = hxTrue[:2, :].T
    true_dist = np.linalg.norm(np.diff(true_path, axis=0), axis=1)
    true_length = np.sum(true_dist)
    print(f"Comprimento do percurso Verdadeiro: {true_length:.2f} m")

    # Erro médio/final entre landmarks reais e estimados (após matching e alinhamento)
    lm_errors = np.linalg.norm(Aruco_matched - est_lms_aligned, axis=1)
    print(f"Erro médio entre LM reais e estimados (após alinhamento): {np.mean(lm_errors):.3f} m")
    print(f"Erro máximo entre LM reais e estimados: {np.max(lm_errors):.3f} m")
    print(f"Erro mínimo entre LM reais e estimados: {np.min(lm_errors):.3f} m")
    # ...existing code...
    print(f"Erro final (último passo) EKF vs hxLine: {error_hxline_ekf[-1]:.3f} m")
    print(f"Erro final (último passo) Odometria vs hxLine: {error_hxline_odom[-1]:.3f} m")
    print(f"Erro médio EKF vs hxLine: {np.mean(error_hxline_ekf):.3f} m")
    print(f"Erro médio Odometria vs hxLine: {np.mean(error_hxline_odom):.3f} m")
    print(f"Número total de landmarks estimados: {nLM}")
    print(f"Número total de passos: {len(time_list)}")
    print("--- Fim das Estatísticas ---\n")
# ...existing code...

if __name__ == '__main__':
    main()