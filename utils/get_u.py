import rosbag
import numpy as np

def extract_velocity_vector(bag_file, topic_name="/pose", skip_count=0):
    """
    Extracts linear and angular velocity from pose messages into a 2D control vector u.
    
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
        bag = rosbag.Bag(bag_file)
        
        # Lists to store data
        times = []
        linear_velocities = []
        angular_velocities = []
        
        skip = 0
        first_time = None
        
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            # Remove the skipping logic to get every sample
            # Or keep it with default skip_count=0
            if skip_count > 0 and skip < skip_count:
                skip += 1
                continue
                
            # Reset skip counter
            skip = 0
            
            if first_time is None:
                first_time = t.to_sec()
            
            # Store timestamp
            times.append(t.to_sec() - first_time)
            
            # Extract velocity values (using x for linear, z for angular)
            linear_vel = msg.twist.twist.linear.x
            angular_vel = msg.twist.twist.angular.z
            
            # Store velocities
            linear_velocities.append(linear_vel)
            angular_velocities.append(angular_vel)
                
        bag.close()
        
        # Convert to numpy arrays
        times = np.array(times)
        u = np.vstack((linear_velocities, angular_velocities))
        
        return times, u
        
    except Exception as e:
        print(f"Error extracting velocity vector: {e}")
        return None, None

if __name__ == "__main__":
    # Path to your ROS bag file
    bag_file_path = "rosbag2.bag"  # Make sure this points to your bag file
    
    # Extract velocity vector
    times, u = extract_velocity_vector(bag_file_path, skip_count=0)
    
    if u is not None:
        print(f"Extracted {u.shape[1]} velocity samples")
        print("\nControl vector u shape:", u.shape)
        print("\nFirst 5 samples of u:")
        for i in range(min(5, u.shape[1])):
            print(f"Time {times[i]:.2f}s: u = [{u[0,i]:.4f}, {u[1,i]:.4f}]")