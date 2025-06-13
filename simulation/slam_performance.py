import numpy as np
import matplotlib.pyplot as plt
import time
import os  # Add os module for directory operations

class PerformanceTracker:
    """
    Class to track and visualize performance metrics for EKF SLAM algorithm.
    Tracks iteration time and number of landmarks detected.
    """
    def __init__(self):
        self.iteration_times = []
        self.total_landmarks_list = []
        self.iteration_numbers = []
        self.iteration_count = 0
        # Create performance directory if it doesn't exist
        self.performance_dir = "performance"
        os.makedirs(self.performance_dir, exist_ok=True)
        
    def add_data_point(self, total_landmarks, elapsed_time):
        """Add a new data point from a SLAM iteration"""
        self.iteration_count += 1
        self.total_landmarks_list.append(total_landmarks)
        self.iteration_times.append(elapsed_time)
        self.iteration_numbers.append(self.iteration_count)
        
        # Print current data point
        print(f"Iteration {self.iteration_count}:")
        print(f"  Total unique landmarks detected: {total_landmarks}")
        print(f"  Iteration time: {elapsed_time:.6f} seconds")
        
    def save_data(self, filename='performance_data.csv'):
        """Save collected data to a CSV file"""
        # Create performance matrix
        performance_matrix = np.array([self.total_landmarks_list, self.iteration_times])
        
        # Save to CSV in performance directory
        filepath = os.path.join(self.performance_dir, filename)
        np.savetxt(filepath, performance_matrix.T, delimiter=',',
                  header='total_landmarks,elapsed_time', comments='')
        print(f"Performance data saved to {filepath}")
        
    def plot_performance(self, save_path="landmarks_vs_time.png"):
        """Create and save a dual-axis plot of landmarks and execution time"""
        # Create plot
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Plot landmarks on left y-axis
        ax1.plot(self.iteration_numbers, self.total_landmarks_list, 'g-', label='Landmarks')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Landmarks', color='g')
        
        # Plot time on right y-axis
        ax2.plot(self.iteration_numbers, self.iteration_times, 'b-', label='Execution Time')
        ax2.set_ylabel('Time (seconds)', color='b')
        
        plt.title('Landmarks Detected and Execution Time')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        # Save to performance directory
        filepath = os.path.join(self.performance_dir, save_path)
        plt.savefig(filepath)
        print(f"Combined plot saved as '{filepath}'")
        
    def print_stats(self):
        """Print summary statistics of the collected data"""
        if not self.iteration_times:
            print("No data available for statistics")
            return
        
        avg_time = sum(self.iteration_times)/len(self.iteration_times)
        max_time = max(self.iteration_times)
        min_time = min(self.iteration_times)
        
        print("\nPerformance Statistics:")
        print(f"  Total iterations: {self.iteration_count}")
        print(f"  Final landmark count: {self.total_landmarks_list[-1]}")
        print(f"  Average iteration time: {avg_time:.6f} seconds")
        print(f"  Max iteration time: {max_time:.6f} seconds")
        print(f"  Min iteration time: {min_time:.6f} seconds")
        
        # Calculate correlation between landmarks and iteration time
        if len(set(self.total_landmarks_list)) > 1:  # Only if landmarks vary
            correlation = np.corrcoef(self.total_landmarks_list, self.iteration_times)[0, 1]
            print(f"  Correlation between landmarks and time: {correlation:.4f}")
    
    def plot_mean_times_by_landmarks(self, save_path="mean_times_by_landmarks.png"):
        """
        Create a violin plot showing execution time distribution for each unique landmark count.
        Shows the relationship between number of landmarks and computational cost.
        """
        if not self.total_landmarks_list:
            print("No data available for plotting")
            return
            
        # Get unique landmark counts
        unique_landmarks = sorted(set(self.total_landmarks_list))
        mean_times = []
        time_distributions = []
        
        # Collect execution times for each unique landmark count
        for lm_count in unique_landmarks:
            # Find all execution times for this landmark count
            indices = [i for i, lm in enumerate(self.total_landmarks_list) if lm == lm_count]
            times = [self.iteration_times[i] for i in indices]
            
            # Store all times for violin plot
            time_distributions.append(times)
            
            # Calculate mean for annotation
            mean_times.append(np.mean(times))
        
        # Create the violin plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create violin plot
        violin_parts = ax.violinplot(time_distributions, positions=unique_landmarks, 
                                    showmeans=True, showmedians=True, widths=0.8)
        
        # Customize violin plot colors
        for pc in violin_parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        violin_parts['cmeans'].set_edgecolor('blue')  # Change mean to blue
        violin_parts['cmedians'].set_edgecolor('orange')  # Keep median as black
        
        # Add mean values as text annotations
        for i, lm in enumerate(unique_landmarks):
            ax.annotate(f"{mean_times[i]:.6f}s", 
                         (lm, mean_times[i] + 0.0002), 
                         textcoords="offset points",
                         xytext=(0, 10), 
                         ha='center',
                         fontsize=8)
        
        # Add a polynomial fit to show the trend
        if len(unique_landmarks) > 2:  # Need at least 3 points for a meaningful fit
            z = np.polyfit(unique_landmarks, mean_times, 2)
            p = np.poly1d(z)
            x_line = np.linspace(min(unique_landmarks), max(unique_landmarks), 100)
            ax.plot(x_line, p(x_line), 'b--', linewidth=2, 
                    label=f'Quadratic fit: {z[0]:.6f}x² + {z[1]:.6f}x + {z[2]:.6f}')
        
        # Set labels and title
        ax.set_xlabel('Number of Landmarks')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Execution Time Distribution by Number of Landmarks')
        ax.set_xticks(unique_landmarks)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        # Save to performance directory
        filepath = os.path.join(self.performance_dir, save_path)
        plt.savefig(filepath)
        print(f"Violin plot of execution times saved as '{filepath}'")
        
        # Print a table of values
        print("\nExecution Time Statistics by Landmark Count:")
        print("Landmarks | Mean Time (s) | Median Time (s) | Std Dev (s) | Count")
        print("-" * 65)
        for i, lm in enumerate(unique_landmarks):
            times = time_distributions[i]
            count = len(times)
            mean = np.mean(times)
            median = np.median(times)
            std_dev = np.std(times)
            print(f"{lm:9d} | {mean:12.6f} | {median:14.6f} | {std_dev:11.6f} | {count:5d}")

# If this file is run directly, it can load data from a CSV and analyze it
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Load data from file
        data = np.loadtxt(sys.argv[1], delimiter=',', skiprows=1)
        
        tracker = PerformanceTracker()
        tracker.total_landmarks_list = data[:, 0].tolist()
        tracker.iteration_times = data[:, 1].tolist()
        tracker.iteration_numbers = list(range(1, len(data) + 1))
        tracker.iteration_count = len(data)
        
        # Generate plots and stats
        tracker.plot_performance()
        tracker.print_stats()
    else:
        print("Usage: python slam_performance.py performance_data.csv")