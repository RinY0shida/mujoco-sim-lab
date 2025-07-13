import mujoco
import mujoco.viewer
import numpy as np
import time

# XMLファイルのパス
xml_path = "models/inverted_pendulum.xml"

# モデルとデータのロード
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
data.qpos[0] = np.deg2rad(20)
Kp = 13.0
Kd =5.0
#ビューアを起動
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # theta = data.qpos[0]
        # omega = data.qvel[0]
        # torque = -Kp * theta - Kd * omega
 
        # disturbance = np.random.normal(loc=0.0, scale=1.0)
        # data.ctrl[0] = torque + disturbance
        data.ctrl[0] = np.random.uniform(-90.0, 90.0)
        mujoco.mj_step(model, data)
 
        viewer.sync()                # ステップ後の状態をviewerに反映
        time.sleep(model.opt.timestep)  # 描画ループ安定化

