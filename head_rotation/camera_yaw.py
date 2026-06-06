import cv2
import mediapipe as mp
import socket
import struct

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="/home/besedo-user/Desktop/Avatar_pose_control/pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.VIDEO,
)

cap = cv2.VideoCapture(0)

CALIBRATION_FRAMES = 30
calib_samples = []
neutral_x = 0.5
neutral_y = 0.5
calibrated = False

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(rgb, timestamp_ms)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            nose_x = lm[0].x
            nose_y = lm[0].y

            if not calibrated:
                calib_samples.append((nose_x, nose_y))
                remaining = CALIBRATION_FRAMES - len(calib_samples)
                cv2.putText(frame, f"Look straight ahead... {remaining}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
                if len(calib_samples) >= CALIBRATION_FRAMES:
                    neutral_x = sum(s[0] for s in calib_samples) / CALIBRATION_FRAMES
                    neutral_y = sum(s[1] for s in calib_samples) / CALIBRATION_FRAMES
                    calibrated = True
            else:
                yaw   = (neutral_x - nose_x) * 200.0
                pitch = (nose_y - neutral_y) * 200.0

                sock.sendto(struct.pack("ff", yaw, pitch), (UDP_IP, UDP_PORT))

                cv2.putText(frame, f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Head Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
