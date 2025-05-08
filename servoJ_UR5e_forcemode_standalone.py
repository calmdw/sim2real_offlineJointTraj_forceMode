import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import yaml
import os
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

def getTrajectoryPoint(trajectory, timestep):
    return trajectory[timestep]

def load_config(config_filename):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, config_filename)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default="joint_positions_10April_0245.npy", help='Path to joint_positions.npy file')
    parser.add_argument('--traj_length', type=int, default=375, help='Number of steps to run from trajectory')
    parser.add_argument('--traj_index', type=int, default=5, help='Index to use from joint_positions array')
    parser.add_argument('--rtde_frequency', type=int, default=10, help='Index to use from joint_positions array')
    parser.add_argument('--velocity', type=float, default=0.05)
    parser.add_argument('--acceleration', type=float, default=0.05)
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # === Load joint data ===
    joint_positions = np.load(args.traj_file)
    joint_positions = joint_positions[:args.traj_length] if args.traj_length else joint_positions
    joint_positions_rel = joint_positions - joint_positions[0]
    trajectory_rel = joint_positions_rel[:, args.traj_index, :]
    trajectory_real = trajectory_rel + np.array(config['home_position'])

    # Show loaded info
    print(f"[INFO] Trajectory loaded from {args.traj_file}")
    print(f"[INFO] Shape: {joint_positions.shape}, Using index {args.traj_index}")
    print(f"[INFO] Total steps: {trajectory_real.shape[0]}")
    print(f"[INFO] Home position: {np.degrees(config['home_position'])}")

    # Connect to robot
    rtde_c = RTDEControl("192.168.5.1", args.rtde_frequency)
    rtde_r = RTDEReceive("192.168.5.1")
    
    # Move to home
    rtde_c.moveJ(config['home_position'], 0.5, 0.3)
    target_tcp = rtde_r.getTargetTCPPose()
    print("tcp pose at home \n", target_tcp)
    print("[INFO] Sample trajectory_real.degrees \n", np.degrees(trajectory_real[:10]))
    input("[INPUT] At home. Press Enter to zero F/T sensor.")

    #TODO
    rtde_c.zeroFtSensor()
    print("[INFO] Zeroing F/T sensor...")
    time.sleep(0.2)
    print("[INFO] Zeroed F/T sensor at home position: ", rtde_r.getActualTCPForce())
    input("[INPUT] Confirm F/T sensor reset. Press Enter to move.")
    interrupted = False

    # === Init containers ===
    time_stamps, target_qs, real_qs, tcp_poses, ft_data = [], [], [], [], []

    # Start control loop
    step = 0
    try:
        while step < len(trajectory_real):
            t_start = rtde_c.initPeriod()
            #TODO
            target_q = getTrajectoryPoint(trajectory_real, step)
            actual_ft = rtde_r.getActualTCPForce()

            rtde_c.forceMode(rtde_r.getTargetTCPPose(), config['selection_vector'], config['wrench'],
                             config['force_type'], config['limits'])
            rtde_c.servoJ(target_q, args.velocity, args.acceleration, config['dt'], config['lookahead_time'], config['gain'])

            # Log
            time_stamps.append(step * 1 / args.rtde_frequency)
            target_qs.append(target_q.copy())
            real_qs.append(rtde_r.getTargetQ())
            tcp_poses.append(rtde_r.getTargetTCPPose())
            ft_data.append(actual_ft)
            print(f"[STEP {step}] Target: {np.round(target_q, 3)}, Real: {rtde_r.getTargetQ()}")

            step += 1
            rtde_c.waitPeriod(t_start)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")
        interrupted = True

    print("[INFO] Stopping robot...")
    rtde_c.servoStop()
    rtde_c.forceModeStop()
    final_tcp = rtde_r.getActualTCPPose()
    print("[INFO] Final TCP Pose:", final_tcp)

    back_up_pose = final_tcp.copy()
    back_up_pose[2] += 0.039
    rtde_c.moveL(back_up_pose, 0.25, 0.5)
    rtde_c.stopScript()

    # Save data
    log_id = f"traj{args.traj_index}_len{len(trajectory_real)}_wrench{'-'.join(map(str, config['wrench']))}"
    os.makedirs("logs", exist_ok=True)

    np.save(f"logs/target_joint_positions_{log_id}.npy", np.array(target_qs))
    np.save(f"logs/real_joint_positions_{log_id}.npy", np.array(real_qs))
    np.save(f"logs/real_tcp_positions_{log_id}.npy", np.array(tcp_poses))
    np.save(f"logs/ft_sensor_data_{log_id}.npy", np.array(ft_data))
    np.save(f"logs/time_stamps_{log_id}.npy", np.array(time_stamps))

    with open(f"{log_id}_log.txt", "w") as f:
        f.write(f"Trajectory: {args.traj_file}\n")
        f.write(f"Velocity: {args.velocity}\n")
        f.write(f"Acceleration: {args.acceleration}\n")
        f.write(f"Interrupted: {interrupted}\n")
        f.write(f"Duration: {time_stamps[-1]:.2f}s, Steps: {step}\n")

    # Plot
    plot_joint_trajectory(time_stamps, target_qs, real_qs, log_id)
    plot_tcp_trajectory(time_stamps, tcp_poses, log_id)
    plot_ft_sensor(time_stamps, ft_data, log_id)

def plot_joint_trajectory(t, target_qs, real_qs, log_id):
    target_qs, real_qs = np.array(target_qs), np.array(real_qs)
    plt.figure(figsize=(12, 10))
    for i in range(target_qs.shape[1]):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, target_qs[:, i], label='Target', color='blue')
        plt.plot(t, real_qs[:, i], '--', label='Real', color='red')
        plt.title(f'Joint {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Rad')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"logs/joint_plot_{log_id}.png")
    plt.show()

def plot_tcp_trajectory(t, tcp_poses, log_id):
    tcp_poses = np.array(tcp_poses)
    labels = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, tcp_poses[:, i], label=labels[i], color='purple')
        plt.title(f"TCP {labels[i]}")
        plt.xlabel("Time (s)")
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"logs/tcp_plot_{log_id}.png")
    plt.show()

def plot_ft_sensor(t, ft_data, log_id):
    ft_data = np.array(ft_data)
    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, ft_data[:, i], label=labels[i], color='green')
        plt.title(labels[i])
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"logs/ft_plot_{log_id}.png")
    plt.show()
    
if __name__ == "__main__":
    main()