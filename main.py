"""Display the aligned monkey lower-body meshes in MuJoCo."""

from pathlib import Path
import math
import time

import mujoco
import mujoco.viewer


XML_PATH = Path(__file__).parent / "models" / "monkey_lower_body.xml"


def update_muscle_controls(data: mujoco.MjData) -> None:
    """Drive the left and right hip antagonistic muscle pairs alternately."""
    phase = 0.5 * (1.0 + math.sin(2.0 * math.pi * 0.7 * data.time))
    data.ctrl[0] = phase          # left flexor
    data.ctrl[1] = 1.0 - phase    # left extensor
    data.ctrl[2] = 1.0 - phase    # right flexor
    data.ctrl[3] = phase          # right extensor


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            update_muscle_controls(data)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
