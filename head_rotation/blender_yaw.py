import bpy
import socket
import struct
import math
import threading

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

ARMATURE_NAME = "Armature"
HEAD_BONE_NAME = "mixamorig:Head"
SMOOTHING = 0.15  # lower = smoother but slower, higher = snappier

ns = bpy.app.driver_namespace

# Stop previous timer
if "apply_head" in ns and bpy.app.timers.is_registered(ns["apply_head"]):
    bpy.app.timers.unregister(ns["apply_head"])

# Close previous socket
if "head_sock" in ns:
    try:
        ns["head_sock"].close()
    except:
        pass

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
_sock.bind((UDP_IP, UDP_PORT))
ns["head_sock"] = _sock

latest = [0.0, 0.0]    # raw from camera
smoothed = [0.0, 0.0]  # smoothed values applied to bone


def listen():
    while True:
        try:
            data, _ = ns["head_sock"].recvfrom(8)
            latest[0], latest[1] = struct.unpack("ff", data)
        except:
            break


threading.Thread(target=listen, daemon=True).start()


def apply_head():
    try:
        smoothed[0] += (latest[0] - smoothed[0]) * SMOOTHING
        smoothed[1] += (latest[1] - smoothed[1]) * SMOOTHING

        obj = bpy.data.objects.get(ARMATURE_NAME)
        if obj and obj.pose:
            bone = obj.pose.bones.get(HEAD_BONE_NAME)
            if bone:
                bone.rotation_mode = "XYZ"
                bone.rotation_euler[1] = math.radians(smoothed[0])
                bone.rotation_euler[0] = math.radians(smoothed[1])
    except Exception as e:
        print("Timer error:", e)
    return 0.05


ns["apply_head"] = apply_head
bpy.app.timers.register(apply_head, persistent=True)
