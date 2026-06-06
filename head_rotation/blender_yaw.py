import bpy
import json
import math
import socket
import threading

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

ARMATURE_NAME    = "Armature"
HEAD_BONE        = "mixamorig:Head"
UPPER_ARM_BONE   = "mixamorig:LeftArm"
FOREARM_BONE     = "mixamorig:LeftForeArm"

HEAD_SMOOTHING = 0.15
ARM_SMOOTHING  = 0.35

# Scale how much arm raise (degrees) maps to bone rotation.
# Positive = arm raises forward/up. Flip sign if it goes the wrong way.
ARM_RAISE_SCALE = -1.5  # negative: camera arm up → bone axis 0 negative → arm goes up
ARM_RAISE_AXIS  = 0    # confirmed: -90=up, 0=horizontal, +90=down

# Wave: axis 1 = supination; when arm is raised this rotates hand left/right
WAVE_SCALE = 0.03
WAVE_AXIS  = 1

ns = bpy.app.driver_namespace

# --- cleanup previous run ---
if "arm_apply" in ns and bpy.app.timers.is_registered(ns["arm_apply"]):
    bpy.app.timers.unregister(ns["arm_apply"])
if "arm_sock" in ns:
    try:
        ns["arm_sock"].close()
    except Exception:
        pass

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
_sock.bind((UDP_IP, UDP_PORT))
ns["arm_sock"] = _sock

latest   = {"yaw": 0.0, "pitch": 0.0, "arm_raise": 0.0, "wave": 0.0}
smoothed = {"yaw": 0.0, "pitch": 0.0, "arm_raise": 0.0, "wave": 0.0}


def listen():
    while True:
        try:
            data, _ = ns["arm_sock"].recvfrom(512)
            parsed = json.loads(data.decode())
            for key in latest:
                if key in parsed:
                    latest[key] = float(parsed[key])
        except Exception:
            break


threading.Thread(target=listen, daemon=True).start()


def apply_pose():
    # Smooth all channels
    for key in smoothed:
        alpha = HEAD_SMOOTHING if key in ("yaw", "pitch") else ARM_SMOOTHING
        smoothed[key] += (latest[key] - smoothed[key]) * alpha

    obj = bpy.data.objects.get(ARMATURE_NAME)
    if not (obj and obj.pose):
        return 0.05

    # Head
    head = obj.pose.bones.get(HEAD_BONE)
    if head:
        head.rotation_mode = "XYZ"
        head.rotation_euler[1] = math.radians(smoothed["yaw"])
        head.rotation_euler[0] = math.radians(smoothed["pitch"])

    # Upper arm — raise / lower
    upper_arm = obj.pose.bones.get(UPPER_ARM_BONE)
    if upper_arm:
        upper_arm.rotation_mode = "XYZ"
        upper_arm.rotation_euler[ARM_RAISE_AXIS] = (
            math.radians(smoothed["arm_raise"]) * ARM_RAISE_SCALE
        )

    # Forearm — wave (side-to-side wrist movement)
    forearm = obj.pose.bones.get(FOREARM_BONE)
    if forearm:
        forearm.rotation_mode = "XYZ"
        forearm.rotation_euler[WAVE_AXIS] = smoothed["wave"] * WAVE_SCALE

    print(
        f"raise={smoothed['arm_raise']:.1f}° "
        f"wave={smoothed['wave']:.1f}  "
        f"yaw={smoothed['yaw']:.1f}°"
    )
    return 0.05


ns["arm_apply"] = apply_pose
bpy.app.timers.register(apply_pose, persistent=True)
print(f"Head + Arm receiver listening on UDP {UDP_IP}:{UDP_PORT}")
