import matplotlib.pyplot as plt
import numpy as np
import os

def plot_joint_trajectory(time_stamps, target_qs, real_qs, prefix=""):
    target_qs = np.array(target_qs)
    real_qs = np.array(real_qs)
    num_joints = target_qs.shape[1]
    
    plt.figure(figsize=(12, 8))
    for i in range(num_joints):
        plt.subplot(num_joints, 1, i + 1)
        plt.plot(time_stamps, np.degrees(target_qs[:, i]), label='Target', linewidth=2)
        plt.plot(time_stamps, np.degrees(real_qs[:, i]), label='Real', linestyle='--', linewidth=2)
        plt.ylabel(f'Joint {i+1} (deg)')
        plt.grid(True)
        if i == 0:
            plt.title('Joint Trajectories')
        if i == num_joints - 1:
            plt.xlabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    save_path = f"logs/{prefix}_joint_trajectory.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVED] Joint trajectory plot to {save_path}")

def plot_tcp_trajectory(time_stamps, tcp_poses, prefix=""):
    tcp_poses = np.array(tcp_poses)
    labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
    
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(time_stamps, tcp_poses[:, i], linewidth=2)
        plt.ylabel(labels[i])
        plt.grid(True)
        if i == 0:
            plt.title('TCP Pose Over Time')
        if i == 5:
            plt.xlabel('Time (s)')
    plt.tight_layout()
    save_path = f"logs/{prefix}_tcp_trajectory.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVED] TCP trajectory plot to {save_path}")

def plot_ft_sensor(time_stamps, ft_data, prefix="", wrench=None, selection_vector=None):
    ft_data = np.array(ft_data)
    labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(time_stamps, ft_data[:, i], linewidth=2)
        if wrench and selection_vector:
            if selection_vector[i]:
                plt.axhline(wrench[i], color='red', linestyle='--', linewidth=1)
        plt.ylabel(labels[i])
        plt.grid(True)
        if i == 0:
            plt.title('F/T Sensor Data Over Time')
        if i == 5:
            plt.xlabel('Time (s)')
    plt.tight_layout()
    save_path = f"logs/{prefix}_ft_sensor.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVED] F/T sensor plot to {save_path}")
