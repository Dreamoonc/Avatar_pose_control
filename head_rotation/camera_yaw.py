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

# Calibration (head + arm together)
CALIBRATION_FRAMES = 30
calib_samples = []
neutral_x = 0.5
neutral_y = 0.5
arm_fwd_samples = []
neutral_arm_fwd = 0.0
arm_fwd_r_samples = []
neutral_arm_fwd_r = 0.0
calibrated = False


def arm_angles(shoulder, wrist):
    """Returns (elevation_deg, azimuth_deg).
    elevation: negative=down, 0=horizontal, positive=up
    azimuth:   0=arm to side (T-pose), positive=arm forward
    """
    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y   # positive = wrist below shoulder
    dz = wrist.z - shoulder.z   # negative = wrist closer to camera (forward)

    arm_len = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    horiz   = math.sqrt(dx * dx + dz * dz) + 1e-6
    elevation = max(-90.0, min(90.0, math.degrees(math.atan2(-dy, horiz))))

    # Normalize dz by full arm length — stable when arm hangs down (dy large, dx→0)
    dz_norm = max(-1.0, min(1.0, -dz / arm_len))
    azimuth = max(-90.0, min(90.0, math.degrees(math.asin(dz_norm))))

    return elevation, azimuth


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
                shoulder = lm[11]
                wrist    = lm[15]
                _, raw_fwd = arm_angles(shoulder, wrist)
                calib_samples.append((nose_x, nose_y))
                arm_fwd_samples.append(raw_fwd)

                # right arm calibration
                r_shoulder_c = lm[12]
                r_wrist_c    = lm[16]
                _, r_raw_fwd = arm_angles(r_shoulder_c, r_wrist_c)
                arm_fwd_r_samples.append(r_raw_fwd)

                remaining = CALIBRATION_FRAMES - len(calib_samples)
                cv2.putText(
                    frame, f"Look ahead & both arms to side... {remaining}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2,
                )
                if len(calib_samples) >= CALIBRATION_FRAMES:
                    neutral_x         = sum(s[0] for s in calib_samples) / CALIBRATION_FRAMES
                    neutral_y         = sum(s[1] for s in calib_samples) / CALIBRATION_FRAMES
                    neutral_arm_fwd   = sum(arm_fwd_samples) / CALIBRATION_FRAMES
                    neutral_arm_fwd_r = sum(arm_fwd_r_samples) / CALIBRATION_FRAMES
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

            # --- left arm ---
            raise_deg, arm_forward_raw = arm_angles(shoulder, wrist)
            arm_forward = max(-90.0, min(90.0, (arm_forward_raw - neutral_arm_fwd) * 4.5))
            wave = (wrist.x - elbow.x) * 200.0

            # --- right arm ---
            r_shoulder = lm[12]
            r_elbow    = lm[14]
            r_wrist    = lm[16]
            raise_deg_r, arm_forward_raw_r = arm_angles(r_shoulder, r_wrist)
            arm_forward_r = max(-90.0, min(90.0, (arm_forward_raw_r - neutral_arm_fwd_r) * 4.5))
            wave_r = (r_wrist.x - r_elbow.x) * 200.0

            data = {
                "yaw":           yaw,
                "pitch":         pitch,
                "arm_raise":     raise_deg,
                "wave":          wave,
                "arm_forward":   arm_forward,
                "arm_raise_r":   raise_deg_r,
                "wave_r":        wave_r,
                "arm_forward_r": arm_forward_r,
            }
            sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

            cv2.putText(
                frame,
                f"Raise:{raise_deg:.0f} Fwd:{arm_forward:.0f} Wave:{wave:.0f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        cv2.imshow("Head + Arm Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
