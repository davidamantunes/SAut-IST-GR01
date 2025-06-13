# EKF SLAM - Pioneer 3DX Robot - SAut-IST-GR01

## Description

This project **implements Simultaneous Localization and Mapping (SLAM)** using the **Extended Kalman Filter (EKF)** with a Pioneer 3-DX robot.  
The algorithm enables the robot to **localize itself while building a map of its environment**, improving its positional accuracy over pure odometry estimates.

This work was **developed as a project for LEEC (IST)** to apply EKF-SLAM, both in simulation and with a real-world robot, employing odometry and camera data with ArUco markers.

## Implementation

- **Algorithm**: Extended Kalman Filter (EKF) for SLAM
- **Platform**: Pioneer 3-DX robot
- **Sensors**: Odometry + Camera (with ArUco marker detection)
- **Environment**: Ubuntu + ROS + Python
- **Simulation**: Implemented and evaluated against ground-truth trajectory
- **Real-world experiments**: Validated with data collected by the robot in a physical setting

## Features

- Localization while mapping the environment
- Correction of odometric drift using EKF
- Utilization of landmarks (ArUco markers) for improved accuracy
- Implementation in both simulated and real-world scenarios
- Quantitative comparison between odometric estimates and EKF-corrected trajectory

## Repository Structure

- `automatization/`: Scripts for automating tasks like parameter tuning and data extraction
- `camera/`: Camera calibration files and ROS launch configurations
- `simulation/`: EKF-SLAM simulation scripts and utilities
- `utils/`: Helper scripts for plotting, angle calculations, and data processing
- `README.md`: This document
- Additional files: configuration files, algorithm parameters, and related resources

## Results

- Odometric estimates typically drift over time, causing significant inaccuracies in robot localization.
- The EKF algorithm efficiently corrects these inaccuracies by fusing odometric data with camera detections of ArUco markers.
- The algorithm performs well under a range of scenarios and trajectory complexities, demonstrating robustness and adaptability.

## Improvement Suggestions

- Implement adaptive noise models to account for changing sensor conditions
- Integrate Loop Closure mechanisms to further reduce drift
- Improve robustness in more cluttered or challenging environments

## References

- [R. Ventura, “Derivation of the Kalman Filter and Extended Kalman Filter”, 2018]
- [T. Bailey and H. Durrant-Whyte, “Simultaneous Localization and Mapping (SLAM): Part II”, IEEE Transactions on Robotics, 2006]
- [M. I. Ribeiro, “Kalman and Extended Kalman Filters: Concept, Derivation and Properties”, 2004]

## Repository

[https://github.com/davidamantunes/SAut-IST-GR01.git](https://github.com/davidamantunes/SAut-IST-GR01.git)