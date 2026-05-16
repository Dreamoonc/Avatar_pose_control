import json
import math
import socket
import time

import bpy
from mathutils import Vector


UDP_IP = "127.0.0.1"
UDP_PORT = 5005
ARMATURE_NAME = "Armature"
UPPER_ARM_BONE_NAME = "mixamorig:LeftArm"
FOREARM_BONE_NAME   = "mixamorig:LeftForeArm"

WRIST_TARGET_NAME   = "MP_LeftWrist_IK_Target"
ELBOW_POLE_NAME     = "MP_LeftElbow_IK_Pole"
IK_CONSTRAINT_NAME  = "MP_LeftArm_IK"

# Scale MediaPipe normalized (0-1) movement to Blender world units.
# Raise if the arm barely moves, lower if it overshoots.
MOTION_SCALE = 4.0

# How strongly the elbow angle is used to pull the wrist closer (0=ignore, 1=full).
ELBOW_BEND_STRENGTH = 1.0

# 0=frozen, 1=instant. 0.2 is smooth with ~0.15s lag.
SMOOTHING = 0.2

# Skip landmarks whose confidence is below this (0.0 disables the check).
MIN_VISIBILITY = 0.0

# Flip horizontal or vertical if the avatar mirrors your movement.
FLIP_X = False
FLIP_Z = False

# Pole side: 1.0 = pole behind character, -1.0 = pole in front.
# Flip this if the elbow bends the wrong way.
POLE_SIDE     = 1.0
POLE_DISTANCE = 3.0

# Pole angle offset in degrees. Try 0, 90, -90, 180 if the arm still looks twisted.
POLE_ANGLE_DEG = 0


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

last_packet_time      = 0.0
last_wait_message_time = 0.0
target_location        = None
calibration            = None
scene_cache            = None


# ── Scene setup (runs once) ──────────────────────────────────────────────────

def cleanup_previous_ik():
    armature = bpy.data.objects.get(ARMATURE_NAME)
    if armature is not None:
        forearm = armature.pose.bones.get(FOREARM_BONE_NAME)
        if forearm is not None:
            c = forearm.constraints.get(IK_CONSTRAINT_NAME)
            if c is not None:
                forearm.constraints.remove(c)
    for name in (WRIST_TARGET_NAME, ELBOW_POLE_NAME):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def get_or_create_empty(name, location, display_type):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        bpy.context.collection.objects.link(obj)
    obj.empty_display_type = display_type
    obj.empty_display_size = 0.18
    obj.location = location
    return obj


def bone_world(armature, pose_bone, tail=False):
    pt = pose_bone.tail if tail else pose_bone.head
    return armature.matrix_world @ pt


def build_scene():
    armature = bpy.data.objects.get(ARMATURE_NAME)
    if armature is None:
        print(f"Armature not found: {ARMATURE_NAME}")
        return None

    upper_arm = armature.pose.bones.get(UPPER_ARM_BONE_NAME)
    forearm   = armature.pose.bones.get(FOREARM_BONE_NAME)
    if upper_arm is None or forearm is None:
        print("Arm bones not found. Check UPPER_ARM_BONE_NAME / FOREARM_BONE_NAME.")
        return None

    wrist_world    = bone_world(armature, forearm,   tail=True)
    elbow_world    = bone_world(armature, forearm,   tail=False)
    shoulder_world = bone_world(armature, upper_arm, tail=False)

    target = get_or_create_empty(WRIST_TARGET_NAME, wrist_world,  "SPHERE")
    pole   = get_or_create_empty(ELBOW_POLE_NAME,   elbow_world,  "CONE")

    constraint = forearm.constraints.get(IK_CONSTRAINT_NAME)
    if constraint is None:
        constraint = forearm.constraints.new(type="IK")
        constraint.name = IK_CONSTRAINT_NAME
    constraint.target       = target
    constraint.pole_target  = pole
    constraint.chain_count  = 2
    constraint.use_rotation = True
    constraint.pole_angle   = math.radians(POLE_ANGLE_DEG)
    armature.data.pose_position = "POSE"

    return {
        "armature":     armature,
        "target":       target,
        "pole":         pole,
        "wrist_rest":   wrist_world.copy(),
        "elbow_rest":   elbow_world.copy(),
        "shoulder_rest": shoulder_world.copy(),
        "upper_len":    (elbow_world - shoulder_world).length,
        "forearm_len":  (wrist_world - elbow_world).length,
    }


# ── Per-frame helpers ────────────────────────────────────────────────────────

def point_ok(pt):
    return pt.get("visibility", 1.0) >= MIN_VISIBILITY


def to_blender_delta(point, origin):
    x = point["x"] - origin["x"]
    y = point["y"] - origin["y"]
    if FLIP_X:
        x = -x
    if FLIP_Z:
        y = -y
    # MediaPipe x→Blender X, MediaPipe y (down)→Blender -Z
    return Vector((x * MOTION_SCALE, 0.0, -y * MOTION_SCALE))


def wrist_with_elbow_bend(scene, desired_wrist, elbow_angle_deg):
    shoulder = scene["shoulder_rest"]
    arm_vec  = desired_wrist - shoulder
    if arm_vec.length < 0.001:
        return desired_wrist

    upper_len   = scene["upper_len"]
    forearm_len = scene["forearm_len"]
    max_reach   = upper_len + forearm_len
    min_reach   = abs(upper_len - forearm_len) + 0.02

    angle_rad  = math.radians(max(5.0, min(180.0, float(elbow_angle_deg))))
    bent_reach = math.sqrt(
        upper_len ** 2 + forearm_len ** 2
        - 2.0 * upper_len * forearm_len * math.cos(angle_rad)
    )
    bent_reach = max(min_reach, min(max_reach, bent_reach))

    current_reach = min(arm_vec.length, max_reach)
    target_reach  = (
        current_reach * (1.0 - ELBOW_BEND_STRENGTH)
        + bent_reach  * ELBOW_BEND_STRENGTH
    )
    return shoulder + arm_vec.normalized() * target_reach


def update_pole(scene):
    shoulder     = scene["shoulder_rest"]
    wrist        = scene["target"].location
    arm_dir      = wrist - shoulder
    if arm_dir.length < 0.001:
        arm_dir = Vector((1.0, 0.0, 0.0))
    side_dir = (
        scene["armature"].matrix_world.to_quaternion()
        @ Vector((0.0, POLE_SIDE, 0.0))
    )
    scene["pole"].location = (
        shoulder + arm_dir * 0.5 + side_dir.normalized() * POLE_DISTANCE
    )


# ── Main timer callback ──────────────────────────────────────────────────────

def update_character():
    global calibration, last_packet_time, last_wait_message_time
    global target_location, scene_cache

    try:
        data, _addr = sock.recvfrom(4096)
    except BlockingIOError:
        now = time.monotonic()
        if now - last_packet_time > 2.0 and now - last_wait_message_time > 2.0:
            print(f"No UDP data on {UDP_IP}:{UDP_PORT} — is camera_sender.py running?")
            last_wait_message_time = now
        return 0.01

    last_packet_time = time.monotonic()

    try:
        values = json.loads(data.decode("utf-8"))
    except Exception as exc:
        print("Bad UDP data:", exc)
        return 0.01

    required = ("left_elbow", "left_shoulder_point", "left_wrist_point")
    if not all(k in values for k in required):
        print("Missing keys in UDP data:", [k for k in required if k not in values])
        return 0.1

    shoulder_pt  = values["left_shoulder_point"]
    wrist_pt     = values["left_wrist_point"]
    elbow_angle  = values["left_elbow"]

    if not point_ok(shoulder_pt) or not point_ok(wrist_pt):
        return 0.01

    # Build IK scene once (or rebuild if something was deleted).
    if scene_cache is None or bpy.data.objects.get(WRIST_TARGET_NAME) is None:
        scene_cache = build_scene()
    if scene_cache is None:
        return 0.5

    # Calibrate on first valid packet — stand in T-pose or relaxed pose.
    if calibration is None:
        calibration = {
            "shoulder": shoulder_pt,
            "wrist":    wrist_pt,
        }
        print("Calibrated. Move your left arm now.")

    desired_wrist = (
        scene_cache["wrist_rest"]
        + to_blender_delta(wrist_pt, calibration["shoulder"])
    )
    desired_wrist = wrist_with_elbow_bend(scene_cache, desired_wrist, elbow_angle)

    # Smooth the IK target location.
    if target_location is None:
        target_location = scene_cache["target"].location.copy()
    target_location = target_location.lerp(desired_wrist, SMOOTHING)
    scene_cache["target"].location = target_location

    update_pole(scene_cache)
    bpy.context.view_layer.update()

    print(
        f"wrist=({target_location.x:.2f},{target_location.y:.2f},{target_location.z:.2f})"
        f"  elbow={float(elbow_angle):.1f}°"
    )
    return 0.01


cleanup_previous_ik()
bpy.app.timers.register(update_character)
print(f"IK receiver listening on UDP {UDP_IP}:{UDP_PORT}")
print("Stand in T-pose and run camera_sender.py to calibrate.")
