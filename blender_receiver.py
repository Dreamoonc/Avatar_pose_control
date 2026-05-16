import json
import math
import socket
import time

import bpy


UDP_IP = "127.0.0.1"
UDP_PORT = 5060
ARMATURE_NAME = "Armature"

# Mixamo bone axes can differ depending on import settings.
# Try each AXIS_INDEX value 0, 1, 2 if a segment bends on the wrong axis.
UPPER_ARM_BONE_NAME = "mixamorig:LeftArm"
UPPER_ARM_AXIS_INDEX = 2
UPPER_ARM_INVERT = False
UPPER_ARM_OFFSET_DEG = 0

FOREARM_BONE_NAME = "mixamorig:LeftForeArm"
FOREARM_AXIS_INDEX = 0
FOREARM_INVERT = False
FOREARM_OFFSET_DEG = 0

MIN_BEND_DEG = 0
MAX_BEND_DEG = 145
SMOOTHING = 0.35


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

last_upper_arm_rotation = 0.0
last_forearm_rotation = 0.0
last_packet_time = 0.0
last_wait_message_time = 0.0


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def get_pose_bone(bone_name):
    armature = bpy.data.objects.get(ARMATURE_NAME)
    if armature is None:
        print(f"Armature not found: {ARMATURE_NAME}")
        return None

    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        print(f"Bone not found: {bone_name}")
        print("Available bones:")
        for bone in armature.pose.bones:
            print("  ", bone.name)
        return None

    return pose_bone


def update_character():
    global last_forearm_rotation, last_packet_time
    global last_upper_arm_rotation, last_wait_message_time

    try:
        data, _addr = sock.recvfrom(1024)
    except BlockingIOError:
        now = time.monotonic()
        if now - last_packet_time > 2.0 and now - last_wait_message_time > 2.0:
            print(f"No UDP data received yet on {UDP_IP}:{UDP_PORT}")
            last_wait_message_time = now
        return 0.01

    last_packet_time = time.monotonic()

    try:
        values = json.loads(data.decode("utf-8"))
    except Exception as exc:
        print("Bad UDP data:", exc)
        return 0.01

    upper_arm = get_pose_bone(UPPER_ARM_BONE_NAME)
    forearm = get_pose_bone(FOREARM_BONE_NAME)
    if upper_arm is None or forearm is None:
        return 0.5

    if "left_shoulder" not in values or "left_elbow" not in values:
        return 0.01

    shoulder_angle = float(values["left_shoulder"])
    shoulder_lift = clamp(shoulder_angle - 90.0, -90.0, 90.0)
    upper_arm_target = math.radians(
        (-shoulder_lift if UPPER_ARM_INVERT else shoulder_lift)
        + UPPER_ARM_OFFSET_DEG
    )

    # MediaPipe elbow angle: about 180 when straight, lower when bent.
    elbow_angle = float(values["left_elbow"])
    bend_angle = clamp(180.0 - elbow_angle, MIN_BEND_DEG, MAX_BEND_DEG)
    forearm_target = math.radians(
        (-bend_angle if FOREARM_INVERT else bend_angle)
        + FOREARM_OFFSET_DEG
    )

    last_upper_arm_rotation = (
        last_upper_arm_rotation * (1.0 - SMOOTHING)
        + upper_arm_target * SMOOTHING
    )
    last_forearm_rotation = (
        last_forearm_rotation * (1.0 - SMOOTHING)
        + forearm_target * SMOOTHING
    )

    upper_arm.rotation_mode = "XYZ"
    upper_arm.rotation_euler[UPPER_ARM_AXIS_INDEX] = last_upper_arm_rotation

    forearm.rotation_mode = "XYZ"
    forearm.rotation_euler[FOREARM_AXIS_INDEX] = last_forearm_rotation

    bpy.context.view_layer.update()

    print(
        f"shoulder={shoulder_angle:.1f} lift={shoulder_lift:.1f} "
        f"elbow={elbow_angle:.1f} bend={bend_angle:.1f}"
    )
    return 0.01


bpy.app.timers.register(update_character)
print(
    f"Listening on UDP {UDP_IP}:{UDP_PORT} for "
    f"{ARMATURE_NAME}/{UPPER_ARM_BONE_NAME}+{FOREARM_BONE_NAME}"
)
