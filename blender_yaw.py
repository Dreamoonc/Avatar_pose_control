import bpy
import json
import math
import socket
import threading

UDP_IP = "127.0.0.1"
UDP_PORT = 5008

ARMATURE_NAME    = "Armature"
HEAD_BONE        = "mixamorig:Head"
UPPER_ARM_BONE   = "mixamorig:LeftArm"
FOREARM_BONE     = "mixamorig:LeftForeArm"
UPPER_ARM_BONE_R = "mixamorig:RightArm"
FOREARM_BONE_R   = "mixamorig:RightForeArm"
WRIST_BONE_L     = "mixamorig:LeftHand"

WRIST_BONE_R     = "mixamorig:RightHand"

WRIST_ROLL_AXIS    = 2      # Z axis 
WRIST_ROLL_SCALE_L =  1.5    # roll direction
WRIST_ROLL_SCALE_R = 1.5

# Finger bones — bones 1-3 curled (4 is the fingertip, skip it)
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky", "Thumb"]
FINGER_CURL_AXIS    = 2      # Z axis
FINGER_CURL_SCALE_L =  2.0
FINGER_CURL_SCALE_R = -2.0
THUMB_CURL_SCALE_L  = -2.0  # thumb anatomy is mirrored
THUMB_CURL_SCALE_R  =  2.5
THUMB_OPEN_AXIS_R  = 2     # Z axis
THUMB_OPEN_AXIS_L  = 2     # Z axis
THUMB_OPEN_SCALE_R =  1.0   #  right thumb moves
THUMB_OPEN_SCALE_L =  -1.0   # left thumb moves 
THUMB_OPEN_REST_R  = 90.0   #  angle for right
THUMB_OPEN_REST_L  = 90.0   
FINGER_SMOOTHING    = 0.1

HEAD_SMOOTHING = 0.25
ARM_SMOOTHING  = 0.10

# Left arm — confirmed axes
ARM_RAISE_SCALE   = -1.5
ARM_RAISE_AXIS    = 0
ARM_FORWARD_SCALE = 1.0
ARM_FORWARD_AXIS  = 2
WAVE_SCALE        = -0.03
WAVE_AXIS         = 1

# Right arm 
R_ARM_RAISE_SCALE   = -1.5  # same sign as left
R_ARM_RAISE_AXIS    = 0
R_ARM_FORWARD_SCALE  = -1.0  # opposite sign to left
ARM_FORWARD_DEADZONE = 5.0   # zero out residual lean when arms are down
R_ARM_FORWARD_AXIS  = 2
R_WAVE_SCALE        = -0.03
R_WAVE_AXIS         = 1

# Elbow bend on axis 0, negative = natural forward curl 
ELBOW_BEND_AXIS  = 0
ELBOW_BEND_SCALE = -1.0

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

_finger_keys = [f"finger_{n.lower()}_{side}" for n in FINGER_NAMES for side in ("l", "r")]

latest   = {"yaw": 0.0, "pitch": 0.0, "arm_raise": 0.0, "wave": 0.0, "arm_forward": 0.0,
            "arm_raise_r": 0.0, "wave_r": 0.0, "arm_forward_r": 0.0,
            "elbow_bend": 0.0, "elbow_bend_r": 0.0,
            "wrist_roll_l": 0.0, "wrist_roll_r": 0.0,
            "thumb_open_l": THUMB_OPEN_REST_L, "thumb_open_r": THUMB_OPEN_REST_R,
            **{k: 0.0 for k in _finger_keys}}
smoothed = {"yaw": 0.0, "pitch": 0.0, "arm_raise": 0.0, "wave": 0.0, "arm_forward": 0.0,
            "arm_raise_r": 0.0, "wave_r": 0.0, "arm_forward_r": 0.0,
            "elbow_bend": 0.0, "elbow_bend_r": 0.0,
            "wrist_roll_l": 0.0, "wrist_roll_r": 0.0,
            "thumb_open_l": THUMB_OPEN_REST_L, "thumb_open_r": THUMB_OPEN_REST_R,
            **{k: 0.0 for k in _finger_keys}}


def listen():
    while True:
        try:
            data, _ = ns["arm_sock"].recvfrom(2048)
            parsed = json.loads(data.decode())
            for key in latest:
                if key in parsed:
                    latest[key] = float(parsed[key])
        except Exception:
            break


threading.Thread(target=listen, daemon=True).start()


def reset_to_tpose():
    obj = bpy.data.objects.get(ARMATURE_NAME)
    if not (obj and obj.pose):
        return
    bones_to_reset = [
        HEAD_BONE,
        UPPER_ARM_BONE, FOREARM_BONE, WRIST_BONE_L,
        UPPER_ARM_BONE_R, FOREARM_BONE_R, WRIST_BONE_R,
        "mixamorig:LeftHandThumb1", "mixamorig:RightHandThumb1",
    ]
    for fname in FINGER_NAMES:
        for seg in (1, 2, 3, 4):
            bones_to_reset.append(f"mixamorig:LeftHand{fname}{seg}")
            bones_to_reset.append(f"mixamorig:RightHand{fname}{seg}")
    for name in bones_to_reset:
        bone = obj.pose.bones.get(name)
        if bone:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)


reset_to_tpose()


def apply_pose():
    # Smooth all channels
    for key in smoothed:
        if key in ("yaw", "pitch"):
            alpha = HEAD_SMOOTHING
        elif key in _finger_keys:
            alpha = FINGER_SMOOTHING
        else:
            alpha = ARM_SMOOTHING
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

    # Upper arm : raise / lower + forward / backward
    upper_arm = obj.pose.bones.get(UPPER_ARM_BONE)
    if upper_arm:
        upper_arm.rotation_mode = "XYZ"
        upper_arm.rotation_euler[ARM_RAISE_AXIS] = (
            math.radians(smoothed["arm_raise"]) * ARM_RAISE_SCALE
        )
        fwd = smoothed["arm_forward"]
        fwd = 0.0 if abs(fwd) < ARM_FORWARD_DEADZONE else fwd
        upper_arm.rotation_euler[ARM_FORWARD_AXIS] = math.radians(fwd) * ARM_FORWARD_SCALE

    # Left forearm : elbow bend
    forearm = obj.pose.bones.get(FOREARM_BONE)
    if forearm:
        forearm.rotation_mode = "XYZ"
        forearm.rotation_euler[WAVE_AXIS] = smoothed["wave"] * WAVE_SCALE
        forearm.rotation_euler[ELBOW_BEND_AXIS] = math.radians(smoothed["elbow_bend"]) * ELBOW_BEND_SCALE

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

    # Right forearm / elbow bend
    forearm_r = obj.pose.bones.get(FOREARM_BONE_R)
    if forearm_r:
        forearm_r.rotation_mode = "XYZ"
        forearm_r.rotation_euler[R_WAVE_AXIS] = smoothed["wave_r"] * R_WAVE_SCALE
        forearm_r.rotation_euler[ELBOW_BEND_AXIS] = math.radians(smoothed["elbow_bend_r"]) * ELBOW_BEND_SCALE

    # Fingers — curl each finger independently, bones 1-3 per finger
    for fname in FINGER_NAMES:
        curl_l = smoothed[f"finger_{fname.lower()}_l"]
        curl_r = smoothed[f"finger_{fname.lower()}_r"]
        scale_l = THUMB_CURL_SCALE_L if fname == "Thumb" else FINGER_CURL_SCALE_L
        scale_r = THUMB_CURL_SCALE_R if fname == "Thumb" else FINGER_CURL_SCALE_R
        # Thumb1 (CMC joint) goes backward on Z — only flex Thumb2 and Thumb3
        segments = (2, 3) if fname == "Thumb" else (1, 2, 3)
        for segment in segments:
            bl = obj.pose.bones.get(f"mixamorig:LeftHand{fname}{segment}")
            if bl:
                bl.rotation_mode = "XYZ"
                bl.rotation_euler[FINGER_CURL_AXIS] = math.radians(curl_l) * scale_l
            br = obj.pose.bones.get(f"mixamorig:RightHand{fname}{segment}")
            if br:
                br.rotation_mode = "XYZ"
                br.rotation_euler[FINGER_CURL_AXIS] = math.radians(curl_r) * scale_r

    # Thumb abduction  bone drives the outward spread
    for side, axis, scale, rest, bone_name in (
        ("l", THUMB_OPEN_AXIS_L, THUMB_OPEN_SCALE_L, THUMB_OPEN_REST_L, "mixamorig:LeftHandThumb1"),
        ("r", THUMB_OPEN_AXIS_R, THUMB_OPEN_SCALE_R, THUMB_OPEN_REST_R, "mixamorig:RightHandThumb1"),
    ):
        bone = obj.pose.bones.get(bone_name)
        if bone:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler[axis] = (
                math.radians(smoothed[f"thumb_open_{side}"] - rest) * scale
            )

    # Wrist roll : rotates LeftHand/RightHand for thumbs-up / thumbs-down
    wrist_l = obj.pose.bones.get(WRIST_BONE_L)
    if wrist_l:
        wrist_l.rotation_mode = "XYZ"
        wrist_l.rotation_euler[WRIST_ROLL_AXIS] = math.radians(smoothed["wrist_roll_l"]) * WRIST_ROLL_SCALE_L
    wrist_r = obj.pose.bones.get(WRIST_BONE_R)
    if wrist_r:
        wrist_r.rotation_mode = "XYZ"
        wrist_r.rotation_euler[WRIST_ROLL_AXIS] = math.radians(smoothed["wrist_roll_r"]) * WRIST_ROLL_SCALE_R

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
