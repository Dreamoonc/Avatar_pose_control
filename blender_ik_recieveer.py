import json
import math
import socket
import time

import bpy
from mathutils import Vector


UDP_IP = "127.0.0.1"
UDP_PORT = 5063

ARMATURE_NAME = "Armature"
UPPER_ARM_BONE_NAME = "mixamorig:LeftArm"
FOREARM_BONE_NAME = "mixamorig:LeftForeArm"

WRIST_TARGET_NAME = "MP_LeftWrist_IK_Target"
ELBOW_POLE_NAME = "MP_LeftElbow_IK_Pole"
IK_CONSTRAINT_NAME = "MP_LeftArm_IK"

# Scale the MediaPipe normalized image movement into Blender world units.
# Increase this if the arm barely moves. Decrease if it moves too much.
MOTION_SCALE = 4.0
ELBOW_BEND_STRENGTH = 1.0

# The target is built around the current wrist position in Blender when the
# script starts. Face the camera in a T-pose or relaxed pose before running it.
SMOOTHING = 0.35
MIN_VISIBILITY = 0.45

# MediaPipe image x/y -> Blender coordinates.
# Usually: x = horizontal, z = vertical. Keep y/depth stable for 2D webcam input.
FLIP_X = False
FLIP_Z = False
DEPTH_OFFSET = 0.0

# Pole side controls which side the elbow bends toward.
# If the forearm goes behind the body, try -1.0 instead.
POLE_SIDE = -1.0
POLE_DISTANCE = 3.0


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

last_packet_time = 0.0
last_wait_message_time = 0.0
target_location = None
calibration = None


def cleanup_previous_ik():
    armature = bpy.data.objects.get(ARMATURE_NAME)
    if armature is not None:
        forearm = armature.pose.bones.get(FOREARM_BONE_NAME)
        if forearm is not None:
            constraint = forearm.constraints.get(IK_CONSTRAINT_NAME)
            if constraint is not None:
                forearm.constraints.remove(constraint)

    for object_name in (WRIST_TARGET_NAME, ELBOW_POLE_NAME):
        obj = bpy.data.objects.get(object_name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def get_armature():
    armature = bpy.data.objects.get(ARMATURE_NAME)
    if armature is None:
        print(f"Armature not found: {ARMATURE_NAME}")
    return armature


def get_pose_bone(armature, bone_name):
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        print(f"Bone not found: {bone_name}")
        print("Available bones:")
        for bone in armature.pose.bones:
            print("  ", bone.name)
    return pose_bone


def bone_world_location(armature, pose_bone, tail=False):
    point = pose_bone.tail if tail else pose_bone.head
    return armature.matrix_world @ point


def get_or_create_empty(name, location, display_type):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        bpy.context.collection.objects.link(obj)
    obj.empty_display_type = display_type
    obj.empty_display_size = 0.18
    obj.location = location
    return obj


def configure_ik(armature, forearm, target, pole):
    constraint = forearm.constraints.get(IK_CONSTRAINT_NAME)
    if constraint is None:
        constraint = forearm.constraints.new(type="IK")
        constraint.name = IK_CONSTRAINT_NAME

    constraint.target = target
    constraint.pole_target = pole
    constraint.chain_count = 2
    constraint.use_rotation = True

    # Mixamo rigs often need one of these pole angles. If elbow bends sideways,
    # change this to 0.0, 1.5708, -1.5708, or 3.14159 in Blender.
    constraint.pole_angle = 0.0
    armature.data.pose_position = "POSE"


def setup_scene():
    armature = get_armature()
    if armature is None:
        return None

    upper_arm = get_pose_bone(armature, UPPER_ARM_BONE_NAME)
    forearm = get_pose_bone(armature, FOREARM_BONE_NAME)
    if upper_arm is None or forearm is None:
        return None

    wrist_world = bone_world_location(armature, forearm, tail=True)
    elbow_world = bone_world_location(armature, forearm, tail=False)
    shoulder_world = bone_world_location(armature, upper_arm, tail=False)

    target = get_or_create_empty(WRIST_TARGET_NAME, wrist_world, "SPHERE")
    pole = get_or_create_empty(ELBOW_POLE_NAME, elbow_world, "CONE")

    configure_ik(armature, forearm, target, pole)

    return {
        "armature": armature,
        "target": target,
        "pole": pole,
        "wrist_rest": wrist_world.copy(),
        "elbow_rest": elbow_world.copy(),
        "shoulder_rest": shoulder_world.copy(),
    }


def point_ok(point):
    return point.get("visibility", 1.0) >= MIN_VISIBILITY


def media_pipe_delta(point, origin):
    x = point["x"] - origin["x"]
    y = point["y"] - origin["y"]
    if FLIP_X:
        x = -x
    if FLIP_Z:
        y = -y
    return Vector((x * MOTION_SCALE, DEPTH_OFFSET, -y * MOTION_SCALE))


def smooth_object_to(obj, current, desired):
    if current is None:
        current = obj.location.copy()
    current = current.lerp(desired, SMOOTHING)
    obj.location = current
    return current


def update_stable_pole(scene):
    # Use a fixed pole in front/behind the character instead of the webcam elbow.
    # This prevents the IK solver from flipping the forearm behind the body.
    armature = scene["armature"]
    shoulder = scene["shoulder_rest"]
    wrist = scene["target"].location
    arm_direction = wrist - shoulder

    if arm_direction.length < 0.001:
        arm_direction = Vector((1.0, 0.0, 0.0))

    side_direction = armature.matrix_world.to_quaternion() @ Vector((0.0, POLE_SIDE, 0.0))
    pole_position = shoulder + arm_direction * 0.5 + side_direction.normalized() * POLE_DISTANCE
    scene["pole"].location = pole_position


def apply_elbow_bend_distance(scene, desired_wrist, elbow_angle_deg):
    shoulder = scene["shoulder_rest"]
    arm_vector = desired_wrist - shoulder

    if arm_vector.length < 0.001:
        return desired_wrist

    upper_len = (scene["elbow_rest"] - scene["shoulder_rest"]).length
    forearm_len = (scene["wrist_rest"] - scene["elbow_rest"]).length
    max_reach = upper_len + forearm_len
    min_reach = abs(upper_len - forearm_len) + 0.02

    elbow_angle_deg = max(5.0, min(180.0, float(elbow_angle_deg)))
    elbow_angle_rad = math.radians(elbow_angle_deg)

    # Shoulder-to-wrist distance for this elbow angle.
    bent_reach = math.sqrt(
        upper_len * upper_len
        + forearm_len * forearm_len
        - 2.0 * upper_len * forearm_len * math.cos(elbow_angle_rad)
    )
    bent_reach = max(min_reach, min(max_reach, bent_reach))

    current_reach = min(arm_vector.length, max_reach)
    target_reach = (
        current_reach * (1.0 - ELBOW_BEND_STRENGTH)
        + bent_reach * ELBOW_BEND_STRENGTH
    )

    return shoulder + arm_vector.normalized() * target_reach


def update_character():
    global calibration, last_packet_time, last_wait_message_time
    global target_location

    try:
        data, _addr = sock.recvfrom(4096)
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

    if not all(
        key in values
        for key in (
            "left_elbow",
            "left_shoulder_point",
            "left_elbow_point",
            "left_wrist_point",
        )
    ):
        print("UDP data does not contain landmark points. Restart camera_sender.py.")
        return 0.1

    shoulder = values["left_shoulder_point"]
    wrist = values["left_wrist_point"]
    elbow_angle = values["left_elbow"]

    if not point_ok(shoulder) or not point_ok(wrist):
        return 0.01

    scene = setup_scene()
    if scene is None:
        return 0.5

    if calibration is None:
        calibration = {
            "shoulder": shoulder,
            "wrist": wrist,
            "wrist_rest": scene["wrist_rest"],
        }
        print("IK calibrated. Keep camera_sender.py running and move your left arm.")

    desired_wrist = (
        calibration["wrist_rest"]
        + media_pipe_delta(wrist, calibration["shoulder"])
    )
    desired_wrist = apply_elbow_bend_distance(scene, desired_wrist, elbow_angle)
    target_location = smooth_object_to(scene["target"], target_location, desired_wrist)
    update_stable_pole(scene)

    bpy.context.view_layer.update()
    print(
        "IK wrist="
        f"{desired_wrist.x:.2f},{desired_wrist.y:.2f},{desired_wrist.z:.2f} "
        f"elbow={float(elbow_angle):.1f}"
    )
    return 0.01


cleanup_previous_ik()
bpy.app.timers.register(update_character)
print(f"IK receiver listening on UDP {UDP_IP}:{UDP_PORT}")