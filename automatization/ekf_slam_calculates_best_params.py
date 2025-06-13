import math
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plots

import matplotlib.pyplot as plt
import numpy as np
from utils.angle import angle_mod
from utils.plot import plot_covariance_ellipse, plot_arrow
import subprocess
import re
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
from itertools import product

# Process noise covariance matrix, variance of x and y is 0.5m, yaw is 30deg
#Cx = np.diag([0.1, 0.05, np.deg2rad(30.0)]) ** 2 #Cx = np.diag([0.3, 0.1, np.deg2rad(30.0)]) ** 2

# Define variables for the rosbag information
rosbag_file = "retangulo"  # Use absolute path where the file is located

#angulo_extra = -0.12

#  Simulation parameter
R_sim = np.zeros((2, 2))

DT = 0.1  # time tick [s]
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True

# Global variables to store the extracted velocity data
times_global = None
u_global = None




# --- NOVO: Carregar dados ArUco de ficheiro txt ---
ARUCO_DATA_FILE = f"rosbag_data_{os.path.splitext(os.path.basename(rosbag_file))[0]}/aruco_observations.txt"  # Caminho para o ficheiro txt



aruco_df = None
aruco_time_min = None
aruco_time_max = None

def load_aruco_data():
    global aruco_df, aruco_time_min, aruco_time_max
    if aruco_df is None:
        aruco_df = pd.read_csv(ARUCO_DATA_FILE, sep=r'\s+', engine='python')
        aruco_time_min = aruco_df['t_norm'].min()
        aruco_time_max = aruco_df['t_norm'].max()
        print(f"ArUco data loaded: {len(aruco_df)} rows, t_norm range {aruco_time_min:.3f} to {aruco_time_max:.3f}")

def get_aruco_time_range():
    load_aruco_data()
    return aruco_time_min, aruco_time_max

def process_image_at_time_txt(sim_time, time_window=0.05):
    """
    Devolve as deteções ArUco mais próximas do tempo sim_time (em segundos).
    time_window: tolerância em segundos para considerar deteções próximas.
    """
    load_aruco_data()
    # Seleciona linhas próximas do tempo sim_time
    mask = (np.abs(aruco_df['t_norm'] - sim_time) <= time_window)
    detections = []
    for _, row in aruco_df[mask].iterrows():
        detections.append({
            'id': int(row['id']),
            'distance': float(row['distance']),
            'angle_rad': float(row['angle_rad'])
        })
    return detections

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

def load_velocity_data(pose_file="rosbag_data/robot_poses.txt"):
    """
    Load velocity data from a txt file with pose information.
    """
    global times_global, u_global
    print(f"Loading velocity data from pose file: {pose_file}")
    times, u = extract_velocity_vector_from_txt(pose_file)
    if u is None or len(u) == 0:
        print(f"Failed to extract velocity data from pose file: {pose_file}")
        return False

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
    #if time < 0.2 or abs(time - int(time)) < 0.1:  # Print at start and whole seconds
    #    print(f"Time: {time:.1f}s, Scaled: {scaled_time:.1f}s, Using sample {closest_idx}/{len(times_global)}, v={v:.2f}, yaw_rate={yaw_rate:.2f}")
    
    # Return as control vector
    return np.array([[v, yaw_rate]]).T


# --- NOVO: Obter as deteções ArUco do ficheiro txt ---
def get_aruco_observations(current_time):
    """
    Get ArUco marker observations from the txt file at the specified time
    """
    # Usar o range do ficheiro txt
    global aruco_time_min, aruco_time_max
    if aruco_time_min is None or aruco_time_max is None:
        aruco_time_min, aruco_time_max = get_aruco_time_range()
        print(f"ArUco txt time range: {aruco_time_min:.3f} to {aruco_time_max:.3f} seconds")

    aruco_duration = aruco_time_max - aruco_time_min
    if aruco_duration <= 0:
        return np.zeros((0, 3))

    # Escala o tempo de simulação para o range do ficheiro
    relative_time = (current_time % aruco_duration)
    target_time = aruco_time_min + relative_time

    # Vai buscar deteções próximas desse tempo
    detections = process_image_at_time_txt(target_time, time_window=0.05)

    z = np.zeros((0, 3))
    for detection in detections:
        marker_id = detection['id']
        distance = detection['distance']
        angle = -detection['angle_rad']  + angulo_extra
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


def save_config_txt(config_path= f"ekf_plots_{os.path.splitext(os.path.basename(rosbag_file))[0]}/ekf_config.txt"):
    """
    Guarda os parâmetros Cx, rosbag_file e angulo_extra num ficheiro txt.
    """
    # Extrai os valores da diagonal de Cx (em metros e graus)
    cx_diag = np.sqrt(np.diag(Cx))
    cx_x = cx_diag[0]
    cx_y = cx_diag[1]
    cx_yaw_deg = np.rad2deg(cx_diag[2])
    with open(config_path, "w") as f:
        f.write("# EKF SLAM config file\n")
        f.write(f"Cx_diag: {cx_x:.5f} {cx_y:.5f} {cx_yaw_deg:.2f}  # [m] [m] [deg]\n")
        f.write(f"rosbag_file: {rosbag_file}\n")
        f.write(f"angulo_extra: {angulo_extra}\n")
    print(f"Configuração guardada em '{config_path}'")


def main():
    print(__file__ + " start!!")
    #save_config_txt()  # <-- Adiciona isto logo no início

    load_velocity_data()
    global SIM_TIME, times_global
    if times_global is not None and len(times_global) > 0:
        SIM_TIME = times_global[-1]
        print(f"SIM_TIME ajustado para {SIM_TIME:.2f} segundos (duração dos dados reais)")
    else:
        SIM_TIME = 1  # fallback padrão TEMPO DE SIMULAÇÂO

    time = 0.0


    

    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)
    xDR = np.zeros((STATE_SIZE, 1))

    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

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
    plot_dir = f"ekf_plots_{os.path.splitext(os.path.basename(rosbag_file))[0]}"
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

    # Plot comparativo
    plt.figure(figsize=(8, 6))
    min_landmarks = min(Aruco.shape[0], est_lms.shape[0], est_lms_aligned.shape[0])
    plt.plot(Aruco[:min_landmarks,0], Aruco[:min_landmarks,1], '*k', label='Landmarks reais')
    plt.plot(est_lms[:min_landmarks,0], est_lms[:min_landmarks,1], 'xg', label='Estimados (originais)')
    plt.plot(est_lms_aligned[:min_landmarks,0], est_lms_aligned[:min_landmarks,1], 'or', label='Estimados (alinhados)')
    for i in range(min_landmarks):
        plt.text(Aruco[i,0], Aruco[i,1], f'R{i}', color='k')
        plt.text(est_lms[i,0], est_lms[i,1], f'E{i}', color='g')
        plt.text(est_lms_aligned[i,0], est_lms_aligned[i,1], f'A{i}', color='r')
    plt.legend()
    plt.title('Comparação de marcos reais vs previstos (alinhados)')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "landmarks_alignment.png"), dpi=300)
    print(f"Plot de alinhamento dos landmarks guardado em '{plot_dir}/landmarks_alignment.png'")

    # --- Extrai as covariâncias dos marcos estimados ---
    est_lms_covs = [PEst[3 + i*2:3 + i*2 + 2, 3 + i*2:3 + i*2 + 2] for i in range(nLM)]

    # --- Correspondência ótima usando distância de Mahalanobis ---
    Aruco_matched, est_lms_matched = match_landmarks_mahalanobis(Aruco, est_lms, est_lms_covs)
    est_lms_aligned, scale_factor, R_align, t_align = align_landmarks_with_scale(Aruco_matched, est_lms_matched)

    # --- Alinhar o percurso EKF com a mesma transformação ---
    hxEst_aligned = hxEst.copy()
    # Usar a média dos landmarks estimados usados no alinhamento!
    mu_est = np.mean(est_lms_matched, axis=0)
    hxEst_xy = hxEst[:2, :].T  # shape (N,2)
    # Aplica: centragem -> escala -> rotação -> translação
    hxEst_aligned_xy = (hxEst_xy - mu_est) * scale_factor
    hxEst_aligned_xy = hxEst_aligned_xy @ R_align
    hxEst_aligned_xy = hxEst_aligned_xy + t_align
    hxEst_aligned[:2, :] = hxEst_aligned_xy.T

    # Translada o percurso EKF alinhado para começar em (0,0)
    start_ekf = hxEst_aligned[:2, 0:1]  # shape (2,1)
    hxEst_aligned[:2, :] = hxEst_aligned[:2, :] - start_ekf


    # Plot comparativo final com fator de escala
    plt.figure(figsize=(8, 6))
    min_landmarks = min(Aruco.shape[0], est_lms_aligned.shape[0])
    for i in range(min_landmarks):
        plt.plot([Aruco[i, 0], est_lms_aligned[i, 0]], [Aruco[i, 1], est_lms_aligned[i, 1]], ':r', alpha=0.5)
    plt.plot(Aruco[:,0], Aruco[:,1], '*k', label='Landmarks reais')
    plt.plot(est_lms[:,0], est_lms[:,1], 'xg', label='Estimados (originais)')
    plt.plot(est_lms_aligned[:,0], est_lms_aligned[:,1], 'or', label='Estimados (alinhados)')
    for i in range(Aruco.shape[0]):
        plt.text(Aruco[i,0], Aruco[i,1], f'R{i}', color='k')
        plt.text(est_lms[i,0], est_lms[i,1], f'E{i}', color='g')
        plt.text(est_lms_aligned[i,0], est_lms_aligned[i,1], f'A{i}', color='r')
    plt.legend()
    plt.title(f'Comparação final de marcos reais vs previstos (alinhados)\nFator de escala: {scale_factor:.3f}')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "final_landmarks_alignment.png"), dpi=300)
    print(f"Plot de comparação final guardado em '{plot_dir}/final_landmarks_alignment.png'")

    # --- PLOT FINALÍSSIMO: Landmarks reais, ajustados, percurso real e EKF ---
    plt.figure(figsize=(10, 8))
    # Landmarks reais
    plt.scatter(Aruco[:, 0], Aruco[:, 1], c='blue', marker='*', s=120, label='Landmarks reais')
    # Landmarks ajustados/alinhados
    plt.scatter(est_lms_aligned[:, 0], est_lms_aligned[:, 1], c='red', marker='x', s=80, label='Landmarks ajustados')
    # Percurso real do robô
    plt.plot(hxTrue[0, :], hxTrue[1, :], '-g', linewidth=2, label='Percurso real')
    # Percurso EKF ALINHADO
    plt.plot(hxEst_aligned[0, :], hxEst_aligned[1, :], '--k', linewidth=2, label='Percurso EKF (alinhado)')
    # Liga landmarks reais aos ajustados
    for i in range(Aruco.shape[0]):
        plt.plot([Aruco[i, 0], est_lms_aligned[i, 0]], [Aruco[i, 1], est_lms_aligned[i, 1]], ':r', alpha=0.5)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Plot Finalíssimo: Landmarks e Percursos')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_finalissimo.png"), dpi=300)
    print(f"Plot finalíssimo guardado em '{plot_dir}/plot_finalissimo.png'")

    # --- PLOT FINALÍSSIMO 2: Landmarks reais, ajustados, percurso real e EKF ORIGINAL ---
    plt.figure(figsize=(10, 8))
    # Landmarks reais
    plt.scatter(Aruco[:, 0], Aruco[:, 1], c='blue', marker='*', s=120, label='Landmarks reais')
    # Landmarks ajustados/alinhados
    plt.scatter(est_lms_aligned[:, 0], est_lms_aligned[:, 1], c='red', marker='x', s=80, label='Landmarks ajustados')
    # Percurso real do robô
    plt.plot(hxTrue[0, :], hxTrue[1, :], '-g', linewidth=2, label='Percurso real')
    # Percurso EKF ORIGINAL (não alinhado)
    plt.plot(hxEst[0, :], hxEst[1, :], '--k', linewidth=2, label='Percurso EKF (original)')
    # Liga landmarks reais aos ajustados
    for i in range(Aruco.shape[0]):
        plt.plot([Aruco[i, 0], est_lms_aligned[i, 0]], [Aruco[i, 1], est_lms_aligned[i, 1]], ':r', alpha=0.5)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Plot Finalíssimo 2: Landmarks e Percursos (EKF original)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_finalissimo2.png"), dpi=300)
    print(f"Plot finalíssimo 2 guardado em '{plot_dir}/plot_finalissimo2.png'")

    # --- PLOT FINALÍSSIMO 2.1: EKF original com rotação extra de 5 graus à esquerda ---
    plt.figure(figsize=(10, 8))
    # Landmarks reais
    plt.scatter(Aruco[:, 0], Aruco[:, 1], c='blue', marker='*', s=120, label='Landmarks reais')
    # Landmarks ajustados/alinhados
    plt.scatter(est_lms_aligned[:, 0], est_lms_aligned[:, 1], c='red', marker='x', s=80, label='Landmarks ajustados')
    # Percurso real do robô
    plt.plot(hxTrue[0, :], hxTrue[1, :], '-g', linewidth=2, label='Percurso real')
    # Percurso EKF ORIGINAL (não alinhado) com rotação extra de 5 graus
    theta = np.deg2rad(-5)  # 5 graus em radianos
    R_extra = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
    hxEst_rotated = hxEst.copy()
    hxEst_rotated_xy = hxEst[:2, :].T @ R_extra
    plt.plot(hxEst_rotated_xy[:, 0], hxEst_rotated_xy[:, 1], '--m', linewidth=2, label='EKF (original +5º)')
    # Liga landmarks reais aos ajustados
    for i in range(Aruco.shape[0]):
        plt.plot([Aruco[i, 0], est_lms_aligned[i, 0]], [Aruco[i, 1], est_lms_aligned[i, 1]], ':r', alpha=0.5)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Plot Finalíssimo 2.1: Landmarks e Percursos (EKF original +5º)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_finalissimo2_1.png"), dpi=300)
    print(f"Plot finalíssimo 2.1 guardado em '{plot_dir}/plot_finalissimo2_1.png'")

    # --- Guardar as posições EKF num ficheiro .txt ---
    ekf_positions_path = os.path.join(plot_dir, "ekf_positions.txt")
    with open(ekf_positions_path, "w") as f:
        f.write("x_ekf\ty_ekf\n")
        for x, y in zip(hxEst[0, :], hxEst[1, :]):
            f.write(f"{x:.6f}\t{y:.6f}\n")
    print(f"Posições EKF guardadas em '{ekf_positions_path}'")

    # --- Guardar as posições EKF num ficheiro .txt ---
    ekf_positions_path2 = os.path.join(plot_dir, "ekf_positions2.txt")
    with open(ekf_positions_path2, "w") as f:
        f.write("x_ekf\ty_ekf\n")
        for x, y in zip(hxEst[0, :], hxEst[1, :]):
            f.write(f"{x:.6f}\t{y:.6f}\n")
    print(f"Posições EKF guardadas em '{ekf_positions_path2}'")

    # --- Guardar as posições dos landmarks estimados num ficheiro .txt ---
    ekf_landmarks_path = os.path.join(plot_dir, "ekf_landmarks.txt")
    with open(ekf_landmarks_path, "w") as f:
        f.write("lm_idx\tx_lm\ty_lm\n")
        for i in range(nLM):
            x_lm = xEst[3 + i * 2, 0]
            y_lm = xEst[3 + i * 2 + 1, 0]
            f.write(f"{i}\t{x_lm:.6f}\t{y_lm:.6f}\n")
    print(f"Landmarks EKF guardados em '{ekf_landmarks_path}'")

    # --- Guardar a trajetória real num ficheiro .txt ---
    true_positions_path = os.path.join(plot_dir, "true_positions.txt")
    with open(true_positions_path, "w") as f:
        f.write("x_true\ty_true\n")
        for x, y in zip(hxTrue[0, :], hxTrue[1, :]):
            f.write(f"{x:.6f}\t{y:.6f}\n")
    print(f"Trajetória real guardada em '{true_positions_path}'")

def extract_velocity_vector_from_txt(pose_file):
    """
    Lê poses do ficheiro txt e calcula os vetores de velocidade linear e angular.
    Espera colunas: timestamp, t_norm, x, y, yaw
    Retorna: times (np.array), u (np.array shape 2xN: [v; yaw_rate])
    """
    df = pd.read_csv(pose_file, sep=r'\s+', engine='python')
    x = df['x'].values
    y = df['y'].values
    yaw = df['yaw'].values
    times = df['timestamp'].values

    dt = np.diff(times)
    dx = np.diff(x)
    dy = np.diff(y)
    dyaw = np.diff(yaw)

    dt [dt == 0] = 1e-6

    # Corrige saltos de ângulo
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi

    v = np.sqrt(dx**2 + dy**2) / dt
    yaw_rate = dyaw / dt

    # Para manter o mesmo formato que o original (2, N)
    u = np.vstack([v, yaw_rate])
    # O tempo corresponde ao instante entre duas amostras, pode-se usar o tempo inicial ou médio
    times_mid = times[:-1]  # ou (times[:-1] + times[1:]) / 2

    return times_mid, u

# --- NOVO: Roda simulação para parâmetros dados ---
def run_simulation_for_params(cx_x, cx_y, cx_yaw_deg, angulo_extra_val, show_animation=False):
    """
    Roda a simulação principal com os parâmetros dados e retorna o erro entre hxTrue e hxEst_aligned.
    """
    global Cx, angulo_extra

    # Atualiza os parâmetros
    Cx = np.diag([cx_x, cx_y, np.deg2rad(cx_yaw_deg)]) ** 2
    angulo_extra = angulo_extra_val

    # --- Copia essencial do main() para obter hxTrue e hxEst_aligned ---
    # (sem gráficos, sem guardar ficheiros)
    load_velocity_data()
    global SIM_TIME, times_global
    if times_global is not None and len(times_global) > 0:
        SIM_TIME = times_global[-1]
    else:
        SIM_TIME = 1

    time = 0.0

    # Aruco positions [x, y]
    '''Aruco = np.array([[1.3, 0.44],
                     [2.36, 0.32],
                     [3.86, -0.66],
                     [4.00, 0.88]])'''
    
    '''Aruco = np.array([[1, 0],
                     [1, 3],
                     [-2.6, 1.2],
                     [-5.72, 1.2],
                     [-5.72, 1.2-3.05],
                     [-1, 1.2-3.05],])'''

    Aruco = np.array([[1, 0], #Rosbag volta_4
                     [1, 2],
                     [1, 1],
                     [1, 3.8],
                     [0.5, 4],
                     [-1, 3.8],
                     [-1, 2],
                     [-1.5, 1.5],
                     [1, 2],
                     [-2, 0],
                     [-1, -1]])

    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)
    xDR = np.zeros((STATE_SIZE, 1))

    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    trajectory_function = real_input  # Força sempre real_input

    id_to_lm_idx = {}

    while SIM_TIME >= time:
        time += DT
        u = trajectory_function(time)
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, Aruco, time)
        xEst, PEst, id_to_lm_idx = ekf_slam(xEst, PEst, ud, z, id_to_lm_idx)
        x_state = xEst[0:STATE_SIZE]
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

    # --- Alinhamento igual ao main ---
    nLM = int((len(xEst) - 3) / 2)
    est_lms = np.array([[xEst[3 + i*2, 0], xEst[3 + i*2 + 1, 0]] for i in range(nLM)])
    est_lms_covs = [PEst[3 + i*2:3 + i*2 + 2, 3 + i*2:3 + i*2 + 2] for i in range(nLM)]
    Aruco_matched, est_lms_matched = match_landmarks_mahalanobis(Aruco, est_lms, est_lms_covs)
    est_lms_aligned, scale_factor, R_align, t_align = align_landmarks_with_scale(Aruco_matched, est_lms_matched)
    hxEst_aligned = hxEst.copy()
    mu_est = np.mean(est_lms_matched, axis=0)
    hxEst_xy = hxEst[:2, :].T
    hxEst_aligned_xy = (hxEst_xy - mu_est) * scale_factor
    hxEst_aligned_xy = hxEst_aligned_xy @ R_align
    hxEst_aligned_xy = hxEst_aligned_xy + t_align
    hxEst_aligned[:2, :] = hxEst_aligned_xy.T
    start_ekf = hxEst_aligned[:2, 0:1]
    hxEst_aligned[:2, :] = hxEst_aligned[:2, :] - start_ekf

    # --- Calcula o erro RMSE entre hxTrue e hxEst_aligned ---
    min_len = min(hxTrue.shape[1], hxEst_aligned.shape[1])
    rmse = np.sqrt(np.mean((hxTrue[0, :min_len] - hxEst_aligned[0, :min_len])**2 +
                           (hxTrue[1, :min_len] - hxEst_aligned[1, :min_len])**2))

    # --- Expanded statistical analysis ---
    min_len = min(hxTrue.shape[1], hxEst_aligned.shape[1])

    # Calculate path length (for relative error)
    path_length = 0
    for i in range(1, min_len):
        dx = hxTrue[0, i] - hxTrue[0, i-1]
        dy = hxTrue[1, i] - hxTrue[1, i-1]
        path_length += np.sqrt(dx**2 + dy**2)

    # Calculate squared errors at each point
    squared_errors = (hxTrue[0, :min_len] - hxEst_aligned[0, :min_len])**2 + \
                     (hxTrue[1, :min_len] - hxEst_aligned[1, :min_len])**2
                     
    # Basic statistics
    rmse = np.sqrt(np.mean(squared_errors))
    max_error = np.sqrt(np.max(squared_errors))
    mean_error = np.mean(np.sqrt(squared_errors))
    median_error = np.median(np.sqrt(squared_errors))
    std_dev = np.std(np.sqrt(squared_errors))
    relative_error = (rmse / path_length) * 100 if path_length > 0 else float('inf')

    # Calculate Normalized Estimation Error Squared (NEES) - filter consistency test
    # (This requires access to the state covariance at each timestep)
    # nees_values = []
    # for i in range(min_len):
    #     error = hxTrue[:2, i] - hxEst_aligned[:2, i]
    #     P_i = PEst_history[i][:2, :2]  # Assuming you store covariance history
    #     nees = error.T @ np.linalg.inv(P_i) @ error
    #     nees_values.append(nees)
    # avg_nees = np.mean(nees_values)

    # Print statistics summary
    print(f"RMSE: {rmse:.4f} m")
    print(f"Path Length: {path_length:.4f} m")
    print(f"Relative Error: {relative_error:.2f}%")
    print(f"Max Error: {max_error:.4f} m")
    print(f"Mean Error: {mean_error:.4f} m")
    print(f"Median Error: {median_error:.4f} m")
    print(f"Standard Deviation: {std_dev:.4f} m")
    # print(f"NEES (Consistency): {avg_nees:.4f}")

    # Optional: Create error histogram for visualization
    # plt.figure(figsize=(8, 4))
    # plt.hist(np.sqrt(squared_errors), bins=20)
    # plt.title('Distribution of Position Errors')
    # plt.xlabel('Error (m)')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(plot_dir, "error_histogram.png"))

    return rmse

def grid_search_params(): ##AJUSTAR 
    cx_x_vals = [0.15]
    cx_y_vals = [0.15]
    cx_yaw_deg_vals = [1]
    angulo_extra_vals = [-0.07]

    results = []
    for cx_x, cx_y, cx_yaw_deg, angulo_extra_val in product(cx_x_vals, cx_y_vals, cx_yaw_deg_vals, angulo_extra_vals):
        print(f"Testando: cx_x={cx_x}, cx_y={cx_y}, cx_yaw_deg={cx_yaw_deg}, angulo_extra={angulo_extra_val}")
        try:
            rmse = run_simulation_for_params(cx_x, cx_y, cx_yaw_deg, angulo_extra_val)
            print(f"RMSE = {rmse:.4f}")
            results.append((rmse, cx_x, cx_y, cx_yaw_deg, angulo_extra_val))
        except Exception as e:
            print(f"Erro na simulação: {e}")
            results.append((np.inf, cx_x, cx_y, cx_yaw_deg, angulo_extra_val))
    # Ordena por RMSE crescente
    results.sort()
    print("\nTop 5 melhores combinações:")
    for res in results[:5]:
        print(f"RMSE={res[0]:.4f} | cx_x={res[1]}, cx_y={res[2]}, cx_yaw_deg={res[3]}, angulo_extra={res[4]}")

    # Guardar o melhor resultado num ficheiro txt
    best = results[0]
    # Ensure the directory exists before writing the file
    plot_dir = f"ekf_plots_{os.path.splitext(os.path.basename(rosbag_file))[0]}"
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(plot_dir, "best_grid_search_result.txt"), "w") as f:
        f.write("# Melhor resultado do grid search\n")
        f.write(f"RMSE: {best[0]:.6f}\n")
        f.write(f"cx_x: {best[1]}\n")
        f.write(f"cx_y: {best[2]}\n")
        f.write(f"cx_yaw_deg: {best[3]}\n")
        f.write(f"angulo_extra: {best[4]}\n")
        f.write(f"rosbag_file: {rosbag_file}\n")
    print(f"\nMelhor resultado guardado em '{os.path.join(plot_dir, 'best_grid_search_result.txt')}'")

    return results

if __name__ == "__main__":
    grid_search_params()