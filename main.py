# author RinYoshida
# email tororo1219@gmail.com
# import mujoco
# import mujoco.viewer
# import numpy as np
# import time
# import math
# xml_path = "models/gripper_hand.xml"
# model = mujoco.MjModel.from_xml_path(xml_path)
# data = mujoco.MjData(model)

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while True:
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(model.opt.timestep)

import threading
import time
import mujoco
import mujoco.viewer

XML_PATH = "models/gripper_hand.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

actuator_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle1"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle2"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle3"),
]

hit_event = threading.Event()

hit_power = -100.0
hit_duration = 0.2
hit_end_time = 0.0


def input_thread():
    while True:
        input("Enterでtendon収縮: ")
        hit_event.set()


def set_tendon_force(value):
    for act_id in actuator_ids:
        data.ctrl[act_id] = value


threading.Thread(target=input_thread, daemon=True).start()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        sim_time = data.time

        if hit_event.is_set():
            hit_event.clear()
            hit_end_time = sim_time + hit_duration

        if sim_time < hit_end_time:
            hoge = hit_power
        else:
            hoge = 0
        
        print(hoge)
        set_tendon_force(hoge)

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)


