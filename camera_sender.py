import cv2
import json
import math
import socket
import time
from pathlib import Path

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

UDP_IP = "127.0.0.1"
UDP_PORT = 5060

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

MODEL_PATH = Path(__file__).with_name("pose_landmarker_lite.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing MediaPipe model: {MODEL_PATH}\n"
        "Download it with:\n"
        "curl -L -o pose_landmarker_lite.task "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH),
    delegate=python.BaseOptions.Delegate.CPU,
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera 0. Check camera permission or try another camera index.")

def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) -
        math.atan2(ay - by, ax - bx)
    )

    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle

    return angle


def landmark_to_dict(landmark):
    return {
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": landmark.visibility,
    }


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(time.monotonic() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        shoulder = landmarks[11]  # LEFT_SHOULDER
        elbow = landmarks[13]     # LEFT_ELBOW
        wrist = landmarks[15]     # LEFT_WRIST
        hip = landmarks[23]       # LEFT_HIP

        a = (shoulder.x, shoulder.y)
        b = (elbow.x, elbow.y)
        c = (wrist.x, wrist.y)
        hip_point = (hip.x, hip.y)

        left_elbow_angle = calculate_angle(a, b, c)
        left_shoulder_angle = calculate_angle(hip_point, a, b)

        data = {
            "left_shoulder": left_shoulder_angle,
            "left_elbow": left_elbow_angle,
            "left_shoulder_point": landmark_to_dict(shoulder),
            "left_elbow_point": landmark_to_dict(elbow),
            "left_wrist_point": landmark_to_dict(wrist),
        }

        sock.sendto(json.dumps(data).encode("utf-8"), (UDP_IP, UDP_PORT))

        print(data)

    cv2.imshow("Camera Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
