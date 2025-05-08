import argparse
import numpy as np
import time
import yaml
import os
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from utils import plot_joint_trajectory, plot_tcp_trajectory, plot_ft_sensor

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
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Time-based prefix
    prefix = time.strftime("%Y%m%d_%H%M%S")

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
    rtde_c = RTDEControl("192.168.5.1", config['rtde_frequency'])
    rtde_r = RTDEReceive("192.168.5.1")

    # Move to home
    rtde_c.moveJ(config['home_position'], 0.5, 0.3)
    target_tcp = rtde_r.getTargetTCPPose()
    print("tcp pose at home\n", target_tcp)
    print("[INFO] Sample trajectory_real.degrees \n", np.degrees(trajectory_real[:10]))
    input("[INPUT] At home. Press Enter to zero F/T sensor.")

    rtde_c.zeroFtSensor()
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
            target_q = getTrajectoryPoint(trajectory_real, step)
            actual_ft_base = rtde_r.getActualTCPForce()

            rtde_c.forceMode(rtde_r.getTargetTCPPose(), config['selection_vector'], config['wrench'],
                             config['force_type'], config['limits'])
            rtde_c.servoJ(target_q, config['velocity'], config['acceleration'],
                          config['dt'], config['lookahead_time'], config['gain'])

            time_stamps.append(step * 1 / config['rtde_frequency'])
            target_qs.append(target_q.copy())
            real_qs.append(rtde_r.getTargetQ())
            tcp_poses.append(rtde_r.getTargetTCPPose())
            ft_data.append(actual_ft_base)
            print(f"[STEP {step}] Target: {np.round(target_q, 3)}, Real: {rtde_r.getTargetQ()}")

            step += 1
            rtde_c.waitPeriod(t_start)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C) in the step: ", step)
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

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    np.save(f"{log_dir}/{prefix}_target_joint_positions.npy", np.array(target_qs))
    np.save(f"{log_dir}/{prefix}_real_joint_positions.npy", np.array(real_qs))
    np.save(f"{log_dir}/{prefix}_real_tcp_positions.npy", np.array(tcp_poses))
    np.save(f"{log_dir}/{prefix}_ft_sensor_data.npy", np.array(ft_data))
    np.save(f"{log_dir}/{prefix}_time_stamps.npy", np.array(time_stamps))

    with open(f"{log_dir}/{prefix}_log.txt", "w") as f:
        f.write(f"Trajectory file: {args.traj_file}\n")
        f.write(f"Trajectory length: {args.traj_length}\n")
        f.write(f"Trajectory index: {args.traj_index}\n")
        f.write(f"Interrupted: {interrupted}\n")
        f.write(f"Total duration: {time_stamps[-1]:.2f}s\n")
        f.write("\nConfig Parameters:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    plot_joint_trajectory(time_stamps, target_qs, real_qs, prefix)
    plot_tcp_trajectory(time_stamps, tcp_poses, prefix)
    plot_ft_sensor(time_stamps, ft_data, prefix, config['wrench'], config['selection_vector'])
    
if __name__ == "__main__":
    main()
