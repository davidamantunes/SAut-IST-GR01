import rosbag
import numpy as np

def extract_velocity_vector(bag_file, topic_name="/pose", skip_count=0):
    """
    Calculates linear and angular velocity from pose position and orientation changes.
    
    Parameters:
    ----------
    bag_file : str
        Path to the ROS bag file.
    topic_name : str
        The pose topic to extract from.
    skip_count : int
        Number of messages to skip between samples (0 for all samples).
        
    Returns:
    -------
    times : numpy.ndarray
        Array of timestamps
    u : numpy.ndarray
        2D array with linear velocity (u[0]) and angular velocity (u[1])
    """
    try:
        import math
        bag = rosbag.Bag(bag_file)
        
        # Lists to store data
        times = []
        linear_velocities = []
        angular_velocities = []
        
        skip = 0
        first_time = None
        prev_pos_x = None
        prev_pos_y = None
        prev_time = None
        prev_quat_z = None
        prev_quat_w = None
        
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            # Skip logic
            if skip_count > 0 and skip < skip_count:
                skip += 1
                continue
                
            # Reset skip counter
            skip = 0
            
            current_time = t.to_sec()
            
            if first_time is None:
                first_time = current_time
            
            # Extract position values
            pos_x = msg.pose.pose.position.x
            pos_y = msg.pose.pose.position.y
            
            # Extract orientation values (we only need z and w for 2D rotation)
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            
            # Calculate time since first message
            time_diff = current_time - first_time
            
            # Calculate linear velocity if we have previous position data
            linear_vel = 0.0
            if prev_pos_x is not None and prev_pos_y is not None and prev_time is not None:
                # Calculate distance moved
                dx = pos_x - prev_pos_x
                dy = pos_y - prev_pos_y
                distance = (dx**2 + dy**2)**0.5  # Euclidean distance
                
                # Calculate time elapsed
                elapsed_time = current_time - prev_time
                
                # Calculate velocity (distance/time)
                if elapsed_time > 0:  # Avoid division by zero
                    linear_vel = distance / elapsed_time
            
            # Calculate angular velocity if we have previous orientation data
            angular_vel = 0.0
            if prev_quat_z is not None and prev_quat_w is not None and prev_time is not None:
                # Calculate yaw angles (simplified for 2D rotation around z-axis)
                yaw = 2 * math.atan2(qz, qw)
                prev_yaw = 2 * math.atan2(prev_quat_z, prev_quat_w)
                
                # Calculate change in yaw
                dyaw = yaw - prev_yaw
                
                # Normalize to [-pi, pi]
                if dyaw > math.pi:
                    dyaw -= 2 * math.pi
                elif dyaw < -math.pi:
                    dyaw += 2 * math.pi
                
                # Time between measurements
                elapsed_time = current_time - prev_time
                
                # Calculate angular velocity
                if elapsed_time > 0:  # Avoid division by zero
                    angular_vel = dyaw / elapsed_time
            
            # Store data if not the first point
            if prev_time is not None:
                times.append(time_diff)
                linear_velocities.append(linear_vel)
                angular_velocities.append(angular_vel)
            
            # Update previous values for next iteration
            prev_pos_x = pos_x
            prev_pos_y = pos_y
            prev_quat_z = qz
            prev_quat_w = qw
            prev_time = current_time
                
        bag.close()
        
        # Convert to numpy arrays
        times = np.array(times)
        u = np.vstack((linear_velocities, angular_velocities))
        
        print(f"Calculated velocities from position data: {u.shape[1]} samples")
        return times, u
        
    except Exception as e:
        print(f"Error calculating velocity vector: {e}")
        return None, None

if __name__ == "__main__":
    # Path to your ROS bag file
    bag_file_path = "2025-05-16-07-49-57.bag"  # Make sure this points to your bag file
    
    # Extract velocity vector
    times, u = extract_velocity_vector(bag_file_path, skip_count=0)
    
    if u is not None:
        print(f"Extracted {u.shape[1]} velocity samples")
        print("\nControl vector u shape:", u.shape)
        print("\nFirst 5 samples of u:")
        for i in range(min(5, u.shape[1])):
            print(f"Time {times[i]:.2f}s: u = [{u[0,i]:.4f}, {u[1,i]:.4f}]")