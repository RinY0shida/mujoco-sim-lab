import threading
import time
import mujoco
import mujoco.viewer
import math

XML_PATH = "models/gripper_hand.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

actuator_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle1"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle2"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle3"),
]

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

hit_event = threading.Event()

hit_power = 70.0
hit_duration = 0.2
hit_end_time = 0.0

p = 20
i = 1
d = 1

cx = -50
cy = -50
cz = -50




def input_thread():
    while True:
        input("Enterでtendon収縮: ")
        hit_event.set()


def set_tendon_force(x, y, z):
    data.ctrl[0] = x
    data.ctrl[1] = y
    data.ctrl[2] = z

threading.Thread(target=input_thread, daemon=True).start()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        sim_time = data.time

                # ボールのx, y, z座標を取得して表示
        ball_pos = data.xpos[ball_id]
        print("ball_x: ", ball_pos[0])
        print("ball_y: ", ball_pos[1])
        print("ball_z: ", ball_pos[2])
        # 極座標の導出
        r = math.sqrt(ball_pos[0]**2 + ball_pos[1]**2)
        theta = math.atan2(ball_pos[1], ball_pos[0])
        # 操作量の計算 とりあえずp制御だけ入れる
        print("theta: ", theta)
        control_axis_1 = r * (1.0 + math.cos(theta)) / 2.0 * p

        control_axis_2 = (
            r * (1.0 + math.cos(theta - 2.0 * math.pi / 3.0))
            / 2.0
            * p
        )

        control_axis_3 = (
            r * (1.0 + math.cos(theta - 4.0 * math.pi / 3.0))
            / 2.0
            * p
        )
        print("control_axis_1: ", control_axis_1)
        print("control_axis_2: ", control_axis_2)
        print("control_axis_3: ", control_axis_3)


        if hit_event.is_set():
            hit_event.clear()
            hit_end_time = sim_time + hit_duration

        if sim_time < hit_end_time:
            cx = -control_axis_1 - hit_power
            cy = -control_axis_2 - hit_power
            cz = -control_axis_3 - hit_power
        else:
            if (int(cx) < -50):
                cx = cx + 1
            if (int(cy) < -50):
                cy = cy + 1
            if (int(cz) < -50):
                cz = cz + 1
                

        print("cx: ", int(cx))
        print("cy: ", int(cy))
        print("cz: ", int(cz))
        
        set_tendon_force(cx, cy, cz)

        mujoco.mj_step(model, data)
        
        viewer.sync()

        time.sleep(model.opt.timestep)
