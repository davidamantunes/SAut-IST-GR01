import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file (fix the path)
data = pd.read_csv('performance_data.csv')

# Group by total_landmarks and calculate mean elapsed_time for each group
grouped_data = data.groupby('total_landmarks')['elapsed_time'].mean().reset_index()

# Print the mean values for each landmark count
print("Mean elapsed times by landmark count:")
for _, row in grouped_data.iterrows():
    print(f"Landmarks: {int(row['total_landmarks'])}, Mean elapsed time: {row['elapsed_time']:.8f} seconds")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(grouped_data['total_landmarks'], grouped_data['elapsed_time'], marker='o', linestyle='-', markersize=8)

# Add text labels with mean values near each point
for i, row in grouped_data.iterrows():
    plt.annotate(f"{row['elapsed_time']:.8f}s", 
                 (row['total_landmarks'], row['elapsed_time']),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

# Add labels and title
plt.xlabel('Total Landmarks')
plt.ylabel('Mean Elapsed Time (seconds)')
plt.title('Performance Analysis: Mean Elapsed Time vs Total Landmarks')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Improve appearance
plt.tight_layout()
plt.show()