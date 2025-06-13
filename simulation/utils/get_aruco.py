import cv2
import numpy as np
import sys
import rosbag
from cv_bridge import CvBridge

# Calibração da câmara (matriz e distorção)
camera_matrix = np.array([
    [1448.54497, 0.0, 971.243241],
    [0.0, 1456.51014, 534.162503],
    [0.0, 0.0, 1.0]
])
dist_coeffs = np.array([[-0.45757027, 0.31496389, -0.00193966, -0.00158797, -0.12176998]])

# Tamanho real do marcador ArUco em metros
marker_length = 0.039  # 16 cm

# Dicionário ArUco e parâmetros de deteção
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters_create()

def get_images_from_rosbag(bag_file, topic_name, time_interval=0.1, max_images=1000):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file)
    # Encontrar o tempo mínimo e máximo do tópico
    start_time = None
    end_time = None
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if start_time is None:
            start_time = t.to_sec()
        end_time = t.to_sec()
    if start_time is None or end_time is None:
        print("Não foi possível encontrar mensagens no tópico.")
        bag.close()
        return
    # Gerar os tempos alvo
    target_times = [start_time + i * time_interval for i in range(int((end_time - start_time) / time_interval) + 1)]
    # Ler todas as mensagens e guardar por timestamp
    messages = []
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        messages.append((t.to_sec(), msg))
    # Para cada tempo alvo, encontrar a mensagem mais próxima
    used = set()
    count = 0
    for target in target_times:
        closest = min(messages, key=lambda x: abs(x[0] - target))
        if abs(closest[0] - target) <= time_interval and closest[0] not in used:
            used.add(closest[0])
            frame = bridge.imgmsg_to_cv2(closest[1], desired_encoding="bgr8")
            yield frame, closest[0]
            count += 1
            if count >= max_images:
                break
    bag.close()

def process_image_at_time(bag_file, topic_name, target_time, return_results=False):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file)
    messages = []
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        messages.append((t.to_sec(), msg))
    bag.close()
    if not messages:
        print("Não foi possível encontrar mensagens no tópico.")
        return [] if return_results else None

    # Encontrar a mensagem mais próxima do tempo alvo
    closest = min(messages, key=lambda x: abs(x[0] - target_time))
    frame = bridge.imgmsg_to_cv2(closest[1], desired_encoding="bgr8")
    msg_time = closest[0]

    # Obter o tempo inicial da bag para o tópico
    start_time, _ = get_topic_time_range(bag_file, topic_name)
    if start_time is not None:
        tempo_relativo = msg_time - start_time
        if not return_results:
            print(f"Tempo relativo ao início da rosbag: {tempo_relativo:.3f} segundos")
    else:
        tempo_relativo = None

    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
    corners, ids, _ = cv2.aruco.detectMarkers(frame_undistorted, aruco_dict, parameters=parameters)
    
    # List to store detection results
    results = []
    
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            distance = np.linalg.norm(tvec)
            c = corners[i][0]
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())
            fov_horizontal_deg = 70
            dx = cx - (w / 2)
            angle_deg = (dx / w) * fov_horizontal_deg
            angle_rad = np.deg2rad(angle_deg)
            angle_rad = np.arctan2(np.sin(angle_rad), np.cos(angle_rad))
            
            # Create detection result dictionary
            detection = {
                'id': int(ids[i][0]),
                'distance': float(distance),
                'angle_rad': float(angle_rad),
                'position': tvec.tolist(),
                'rotation': rvec.tolist(),
                'corners': c.tolist(),
                'timestamp': float(msg_time)
            }
            
            results.append(detection)
            
            if not return_results:
                result_message = f"{msg_time:.3f}, ID: {ids[i][0]}, Distância: {distance:.2f} m, Ângulo: {angle_rad:.2f} rad\n"
                print(result_message)
                text = f"ID: {ids[i][0]} Dist: {distance:.2f}m Âng: {angle_rad:.2f} rad"
                cv2.putText(frame_undistorted, text,
                            org=(int(c[0][0]), int(c[0][1]) - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 255, 0),
                            thickness=2,
                            lineType=cv2.LINE_AA)
    else:
        if not return_results:
            result_message = f"{msg_time:.3f}\n"
            print(result_message)
    
    if return_results:
        return results
    
    # Display code is commented out
    """
    cv2.imshow("Deteção ArUco", frame_undistorted)
    cv2.waitKey(10000)  # Fecha após 3 segundos
    cv2.destroyAllWindows()
    """

def get_topic_time_range(bag_file, topic_name):
    bag = rosbag.Bag(bag_file)
    start_time = None
    end_time = None
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if start_time is None:
            start_time = t.to_sec()
        end_time = t.to_sec()
    bag.close()
    if start_time is None or end_time is None:
        print("Não foi possível encontrar mensagens no tópico.")
        return None, None
    return start_time, end_time

if __name__ == "__main__":  # Fixed: added double underscores
    if len(sys.argv) < 4:
        print("Uso: python aruco_test_td.py <bag_file> <topic> <tempo_relativo>")
        sys.exit(1)
    bag_file = sys.argv[1]
    topic = sys.argv[2]
    tempo_relativo = float(sys.argv[3])

    # Obter o tempo inicial da rosbag para o tópico
    start_time, end_time = get_topic_time_range(bag_file, topic)
    if start_time is None:
        print("Não foi possível obter o tempo inicial da rosbag.")
        sys.exit(1)

    print(f"Tempo disponível no tópico '{topic}': {end_time - start_time:.3f} segundos (de {start_time:.3f} a {end_time:.3f})")

    # Calcular o timestamp absoluto
    target_time = start_time + tempo_relativo

    process_image_at_time(bag_file, topic, target_time)