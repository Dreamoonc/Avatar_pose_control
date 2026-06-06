import cv2
import json
import math
import socket

import mediapipe as mp

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/home/besedo-user/Desktop/Avatar_pose_control/pose_landmarker_lite.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
)

cap = cv2.VideoCapture(0)

# Head calibration
CALIBRATION_FRAMES = 30
calib_samples = []
neutral_x = 0.5
neutral_y = 0.5
calibrated = False


def arm_raise_degrees(shoulder, wrist):
    """Angle of arm raise: negative = arm down, 0 = horizontal, positive = arm up."""
    dx = wrist.x - shoulder.x
    dy = shoulder.y - wrist.y  # image y flipped: up = positive
    angle = math.degrees(math.atan2(dy, abs(dx) + 1e-6))
    return max(-90.0, min(90.0, angle))


with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect_for_video(rgb, timestamp_ms)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # --- head ---
            nose_x = lm[0].x
            nose_y = lm[0].y

            if not calibrated:
                calib_samples.append((nose_x, nose_y))
                remaining = CALIBRATION_FRAMES - len(calib_samples)
                cv2.putText(
                    frame, f"Look straight ahead... {remaining}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2,
                )
                if len(calib_samples) >= CALIBRATION_FRAMES:
                    neutral_x = sum(s[0] for s in calib_samples) / CALIBRATION_FRAMES
                    neutral_y = sum(s[1] for s in calib_samples) / CALIBRATION_FRAMES
                    calibrated = True
                cv2.imshow("Head + Arm Control", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            yaw   = (neutral_x - nose_x) * 200.0
            pitch = (nose_y - neutral_y) * 200.0

            # --- arm ---
            shoulder = lm[11]   # LEFT_SHOULDER
            elbow    = lm[13]   # LEFT_ELBOW
            wrist    = lm[15]   # LEFT_WRIST

            raise_deg = arm_raise_degrees(shoulder, wrist)

            # Wave: how far wrist is to the left/right of the elbow
            wave = (wrist.x - elbow.x) * 200.0  # degrees-like value

            data = {
                "yaw":       yaw,
                "pitch":     pitch,
                "arm_raise": raise_deg,
                "wave":      wave,
            }
            sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

            cv2.putText(
                frame,
                f"Yaw:{yaw:.0f} Pitch:{pitch:.0f} Raise:{raise_deg:.0f} Wave:{wave:.0f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        cv2.imshow("Head + Arm Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
