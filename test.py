# author RinYoshida
# email tororo1219@gmail.com

import mujoco
import mujoco.viewer
import numpy as np
import time

xml_path = "models/inverted_pendulum.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
data.qpos[0] = np.deg2rad(0)
# Kp = 2
# Kd = 2

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         theta = data.qpos[0]
#         omega = data.qvel[0]
#         torque = -Kp * theta - Kd * omega
#         print("deg:", theta)
#         print("torque:", torque)
#         torque = np.clip(torque, -2, 2)
#         data.ctrl[0] = torque
#         mujoco.mj_step(model, data)
        
#         viewer.sync()
#         time.sleep(model.opt.timestep)
 
# ゲイン
Kp = 13.0
Kd = 5.0
 
 
# Viewerつきループ
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        theta = data.qpos[0]
        omega = data.qvel[0]
        torque = -Kp * theta - Kd * omega
 
        disturbance = np.random.normal(loc=0.0, scale=1.0)
        data.ctrl[0] = torque + disturbance
 
        mujoco.mj_step(model, data)
 
        viewer.sync()                # ステップ後の状態をviewerに反映
        time.sleep(model.opt.timestep)  # 描画ループ安定化
