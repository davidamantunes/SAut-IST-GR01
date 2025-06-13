import os
import numpy as np
from utils.get_aruco import process_image_at_time, get_topic_time_range
from utils.get_u_from_pose import extract_velocity_vector

# Configurações
rosbag_file = "oito.bag"
camera_topic = "/usb_cam/image_raw"
output_dir = f"ekf_plots_{os.path.splitext(os.path.basename(rosbag_file))[0]}"
os.makedirs(output_dir, exist_ok=True)

# --- Extrair observações ArUco ---
print("A extrair observações ArUco...")
start_time, end_time = get_topic_time_range(rosbag_file, camera_topic)
if start_time is None or end_time is None:
    print("Erro ao obter o intervalo de tempo do tópico.")
    exit(1)

DT = 0.1  # Intervalo de amostragem em segundos
times = np.arange(start_time, end_time, DT)
aruco_txt_path = os.path.join(output_dir, "aruco_observations.txt")
with open(aruco_txt_path, "w") as f:
    f.write("timestamp\tt_norm\tid\tdistance\tangle_rad\n")
    for t in times:
        detections = process_image_at_time(rosbag_file, camera_topic, t, return_results=True)
        if detections:
            for d in detections:
                t_norm = d['timestamp'] - start_time
                f.write(f"{d['timestamp']:.3f}\t{t_norm:.3f}\t{d['id']}\t{d['distance']:.4f}\t{d['angle_rad']:.4f}\n")
print(f"Observações ArUco guardadas em '{aruco_txt_path}'")

# --- Extrair posições do robô (velocidades integradas) ---
print("A extrair posições do robô (odometria integrada)...")
vel_times, vel_data = extract_velocity_vector(rosbag_file)
if vel_data is None:
    print("Erro ao extrair velocidades da rosbag.")
    exit(1)

poses = []
x, y, yaw = 0.0, 0.0, 0.0
poses.append((vel_times[0], x, y, yaw))
for i in range(1, len(vel_times)):
    dt = vel_times[i] - vel_times[i-1]
    v = vel_data[0, i-1]
    w = vel_data[1, i-1]
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    yaw += w * dt
    poses.append((vel_times[i], x, y, yaw))

robot_pose_txt_path = os.path.join(output_dir, "robot_poses.txt")
with open(robot_pose_txt_path, "w") as f:
    f.write("timestamp\tt_norm\tx\ty\tyaw\n")
    for p in poses:
        t_norm = p[0] - start_time
        f.write(f"{p[0]:.3f}\t{t_norm:.3f}\t{p[1]:.4f}\t{p[2]:.4f}\t{p[3]:.4f}\n")
print(f"Posições do robô guardadas em '{robot_pose_txt_path}'")