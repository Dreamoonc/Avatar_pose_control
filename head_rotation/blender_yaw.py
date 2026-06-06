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
UPPER_ARM_BONE_R = "mixamorig:RightArm"
FOREARM_BONE_R   = "mixamorig:RightForeArm"

HEAD_SMOOTHING = 0.08
ARM_SMOOTHING  = 0.10

# Left arm — confirmed axes
ARM_RAISE_SCALE   = -1.5
ARM_RAISE_AXIS    = 0
ARM_FORWARD_SCALE = 1.0
ARM_FORWARD_AXIS  = 2
WAVE_SCALE        = 0.03
WAVE_AXIS         = 1

# Right arm — mirrored: flip raise & wave signs if wrong direction
R_ARM_RAISE_SCALE   = -1.5  # same sign as left
R_ARM_RAISE_AXIS    = 0
R_ARM_FORWARD_SCALE  = -1.0  # opposite sign to left
ARM_FORWARD_DEADZONE = 5.0   # degrees — zero out residual lean when arms are down
R_ARM_FORWARD_AXIS  = 2
R_WAVE_SCALE        = -0.03
R_WAVE_AXIS         = 1

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

latest   = {"yaw": 0.0, "pitch": 0.0, "arm_raise": 0.0, "wave": 0.0, "arm_forward": 0.0,
            "arm_raise_r": 0.0, "wave_r": 0.0, "arm_forward_r": 0.0}
smoothed = {"yaw": 0.0, "pitch": 0.0, "arm_raise": 0.0, "wave": 0.0, "arm_forward": 0.0,
            "arm_raise_r": 0.0, "wave_r": 0.0, "arm_forward_r": 0.0}


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

    # Upper arm — raise / lower + forward / backward
    upper_arm = obj.pose.bones.get(UPPER_ARM_BONE)
    if upper_arm:
        upper_arm.rotation_mode = "XYZ"
        upper_arm.rotation_euler[ARM_RAISE_AXIS] = (
            math.radians(smoothed["arm_raise"]) * ARM_RAISE_SCALE
        )
        fwd = smoothed["arm_forward"]
        fwd = 0.0 if abs(fwd) < ARM_FORWARD_DEADZONE else fwd
        upper_arm.rotation_euler[ARM_FORWARD_AXIS] = math.radians(fwd) * ARM_FORWARD_SCALE

    # Left forearm — wave
    forearm = obj.pose.bones.get(FOREARM_BONE)
    if forearm:
        forearm.rotation_mode = "XYZ"
        forearm.rotation_euler[WAVE_AXIS] = smoothed["wave"] * WAVE_SCALE

    # Right upper arm
    upper_arm_r = obj.pose.bones.get(UPPER_ARM_BONE_R)
    if upper_arm_r:
        upper_arm_r.rotation_mode = "XYZ"
        upper_arm_r.rotation_euler[R_ARM_RAISE_AXIS] = (
            math.radians(smoothed["arm_raise_r"]) * R_ARM_RAISE_SCALE
        )
        fwd_r = smoothed["arm_forward_r"]
        fwd_r = 0.0 if abs(fwd_r) < ARM_FORWARD_DEADZONE else fwd_r
        upper_arm_r.rotation_euler[R_ARM_FORWARD_AXIS] = math.radians(fwd_r) * R_ARM_FORWARD_SCALE

    # Right forearm — wave
    forearm_r = obj.pose.bones.get(FOREARM_BONE_R)
    if forearm_r:
        forearm_r.rotation_mode = "XYZ"
        forearm_r.rotation_euler[R_WAVE_AXIS] = smoothed["wave_r"] * R_WAVE_SCALE

    print(
        f"raise={smoothed['arm_raise']:.1f}° "
        f"fwd={smoothed['arm_forward']:.1f}°  "
        f"wave={smoothed['wave']:.1f}  "
        f"yaw={smoothed['yaw']:.1f}°"
    )
    return 0.05


ns["arm_apply"] = apply_pose
bpy.app.timers.register(apply_pose, persistent=True)
print(f"Head + Arm receiver listening on UDP {UDP_IP}:{UDP_PORT}")
