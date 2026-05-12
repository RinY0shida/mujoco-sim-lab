import threading
import time
import mujoco
import mujoco.viewer
import math
import sys
import termios
import tty
import select

XML_PATH = "models/gripper_hand.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

actuator_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle1"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle2"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle3"),
]

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# ball が freejoint を持っている前提
ball_joint_id = model.body_jntadr[ball_id]
ball_qpos_adr = model.jnt_qposadr[ball_joint_id]
ball_qvel_adr = model.jnt_dofadr[ball_joint_id]

# 制御パラメータ
base_power = 0.0
p = 200.0
alpha = 0.05

cx = base_power
cy = base_power
cz = base_power

# 矢印キーで動かす目標位置
target_x = 0.0
target_y = 0.0
move_step = 0.01
ball_follow_alpha = 0.08

running = True
lock = threading.Lock()


def set_tendon_force(x, y, z):
    data.ctrl[actuator_ids[0]] = x
    data.ctrl[actuator_ids[1]] = y
    data.ctrl[actuator_ids[2]] = z


def keyboard_thread():
    global target_x, target_y, running

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while running:
            if select.select([sys.stdin], [], [], 0.01)[0]:
                key = sys.stdin.read(1)

                if key == "\x1b":
                    seq = sys.stdin.read(2)

                    with lock:
                        if seq == "[A":      # ↑
                            target_y += move_step
                        elif seq == "[B":    # ↓
                            target_y -= move_step
                        elif seq == "[C":    # →
                            target_x += move_step
                        elif seq == "[D":    # ←
                            target_x -= move_step

                elif key == "q":
                    running = False

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


threading.Thread(target=keyboard_thread, daemon=True).start()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and running:
        ball_pos = data.xpos[ball_id]

        with lock:
            tx = target_x
            ty = target_y

        # ボールを矢印キーの目標位置へなめらかに移動
        data.qpos[ball_qpos_adr + 0] += ball_follow_alpha * (
            tx - data.qpos[ball_qpos_adr + 0]
        )
        data.qpos[ball_qpos_adr + 1] += ball_follow_alpha * (
            ty - data.qpos[ball_qpos_adr + 1]
        )

        # 手動移動による暴れを減らす
        data.qvel[ball_qvel_adr + 0] = 0.0
        data.qvel[ball_qvel_adr + 1] = 0.0

        # 現在位置を取得
        x = data.qpos[ball_qpos_adr + 0]
        y = data.qpos[ball_qpos_adr + 1]

        r = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x)

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

        target_cx = base_power - control_axis_1
        target_cy = base_power - control_axis_2
        target_cz = base_power - control_axis_3

        # 急激な入力変化を避ける
        cx += alpha * (target_cx - cx)
        cy += alpha * (target_cy - cy)
        cz += alpha * (target_cz - cz)

        set_tendon_force(cx, cy, cz)

        print(
            f"x={x:.3f}, y={y:.3f}, "
            f"target_x={tx:.3f}, target_y={ty:.3f}, "
            f"ctrl=({cx:.2f}, {cy:.2f}, {cz:.2f})"
        )

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)
