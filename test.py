# author RinYoshida
# email tororo1219@gmail.com

import mujoco
import mujoco.viewer
import numpy as np
import time

xml_path = "models/inverted_pendulum.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
data.qpos[0] = np.deg2rad(40)
 
# ゲイン　発散しまくりでおわおわり
# Kp = 35.0
# Kd = 5.0

Kp = 13.5
Kd = 1.0

target_angles_deg = [-40, 0, 40]
target_angles = [deg * 3.1416 / 180.0 for deg in target_angles_deg]
interval = 2.0
start_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        elapsed = time.time() - start_time

        index = int(elapsed // interval) % len(target_angles)
        target_theta = target_angles[index]

        theta = data.qpos[0]
        omega = data.qvel[0]

        error = theta - target_theta
        torque = -Kp * error - Kd * omega
        print("error:",error);
        print("omega:",omega);
        data.ctrl[0] = torque

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
