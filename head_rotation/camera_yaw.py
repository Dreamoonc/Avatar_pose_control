import cv2
import json
import math
import socket

import mediapipe as mp

UDP_IP = "127.0.0.1"
UDP_PORT = 5008

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

BaseOptions          = mp.tasks.BaseOptions
PoseLandmarker       = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/home/besedo-user/Desktop/Avatar_pose_control/pose_landmarker_lite.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/home/besedo-user/Desktop/Avatar_pose_control/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

# Webcam is front-facing: MediaPipe "Left" hand = user's RIGHT hand (mirror)
MIRROR_HANDEDNESS = True

# Hand landmark indices: (MCP, PIP, DIP, TIP) per finger
FINGER_LANDMARKS = {
    "Index":  (5,  6,  7,  8),
    "Middle": (9,  10, 11, 12),
    "Ring":   (13, 14, 15, 16),
    "Pinky":  (17, 18, 19, 20),
    "Thumb":  (1,  2,  3,  4),
}

cap = cv2.VideoCapture(0)

# Persistent finger/wrist state — keeps last known value when a hand leaves the frame
finger_state = {f"finger_{n.lower()}_{s}": 0.0
                for n in FINGER_LANDMARKS for s in ("l", "r")}
finger_state["wrist_roll_l"] = 0.0
finger_state["wrist_roll_r"] = 0.0
finger_state["thumb_open_l"] = 0.0
finger_state["thumb_open_r"] = 0.0

# Calibration (head + arm together)
CALIBRATION_FRAMES = 60
calib_samples = []
neutral_x = 0.5
neutral_y = 0.5
arm_fwd_samples = []
neutral_arm_fwd = 0.0
arm_fwd_r_samples = []
neutral_arm_fwd_r = 0.0
calibrated = False


def wrist_roll(lm, flip=False):
    """Hand roll angle in degrees from index-to-pinky MCP vector.
    0=horizontal, positive=thumb up, negative=thumb down.
    flip=True for the left hand so the neutral pose reads 0° instead of ±180°."""
    if flip:
        dx = lm[17].x - lm[5].x
        dy = lm[17].y - lm[5].y
    else:
        dx = lm[5].x - lm[17].x
        dy = lm[5].y - lm[17].y  # y increases downward in image
    return math.degrees(math.atan2(-dy, dx))


def thumb_abduction(lm):
    """Angle in degrees between the thumb-MCP and index-MCP vectors from wrist.
    ~20-30° = thumb adducted (closed), ~60-80° = thumb abducted (open)."""
    tx = lm[2].x - lm[0].x
    ty = lm[2].y - lm[0].y
    ix = lm[5].x - lm[0].x
    iy = lm[5].y - lm[0].y
    dot = tx*ix + ty*iy
    mt = math.sqrt(tx**2 + ty**2) + 1e-6
    mi = math.sqrt(ix**2 + iy**2) + 1e-6
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mt * mi)))))


def finger_curl(lm, mcp_i, pip_i, dip_i, tip_i):
    """Returns average bend angle in degrees: 0=straight, ~90=fully curled.
    Uses 2D (x,y) only — MediaPipe z in VIDEO mode is too noisy for reliable angles."""
    def v(a, b):
        return (lm[b].x - lm[a].x, lm[b].y - lm[a].y)
    def ang(a, b):
        dot = a[0]*b[0] + a[1]*b[1]
        ma  = math.sqrt(a[0]**2 + a[1]**2) + 1e-6
        mb  = math.sqrt(b[0]**2 + b[1]**2) + 1e-6
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (ma * mb)))))
    pip_bend = ang(v(mcp_i, pip_i), v(pip_i, dip_i))
    dip_bend = ang(v(pip_i, dip_i), v(dip_i, tip_i))
    return (pip_bend + dip_bend) / 2.0


def elbow_bend_angle(shoulder, elbow, wrist):
    """Signed elbow bend: positive=forearm up (wrist above elbow), negative=forearm down."""
    v1 = (elbow.x - shoulder.x, elbow.y - shoulder.y, elbow.z - shoulder.z)
    v2 = (wrist.x - elbow.x,    wrist.y - elbow.y,    wrist.z - elbow.z)
    dot  = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2) + 1e-6
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2) + 1e-6
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))
    sign = 1.0 if wrist.y < elbow.y else -1.0
    return max(-150.0, min(150.0, sign * angle))


def arm_angles(shoulder, wrist):
    """Returns (elevation_deg, azimuth_deg).
    elevation: negative=down, 0=horizontal, positive=up
    azimuth:   0=arm to side (T-pose), positive=arm forward
    """
    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y
    dz = wrist.z - shoulder.z

    arm_len   = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    horiz     = math.sqrt(dx * dx + dz * dz) + 1e-6
    elevation = max(-90.0, min(90.0, math.degrees(math.atan2(-dy, horiz))))
    dz_norm   = max(-1.0, min(1.0, -dz / arm_len))
    azimuth   = max(-90.0, min(90.0, math.degrees(math.asin(dz_norm))))

    return elevation, azimuth


with PoseLandmarker.create_from_options(pose_options) as pose_lm, \
     HandLandmarker.create_from_options(hand_options) as hand_lm:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        pose_result = pose_lm.detect_for_video(rgb, timestamp_ms)
        hand_result = hand_lm.detect_for_video(rgb, timestamp_ms)

        # --- parse finger curls from hand landmarks ---
        # Only update the side that is currently visible; keep last known value otherwise
        if hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                # Use wrist X position to determine which hand — more reliable than
                # handedness labels, which can flip when only one hand is visible.
                # Front-facing camera: wrist on left side of image = user's right hand.
                wrist_x = hand_lms[0].x
                if MIRROR_HANDEDNESS:
                    side = "r" if wrist_x < 0.5 else "l"
                else:
                    side = "l" if wrist_x < 0.5 else "r"
                for fname, (mcp_i, pip_i, dip_i, tip_i) in FINGER_LANDMARKS.items():
                    curl = finger_curl(hand_lms, mcp_i, pip_i, dip_i, tip_i)
                    finger_state[f"finger_{fname.lower()}_{side}"] = curl
                finger_state[f"wrist_roll_{side}"] = wrist_roll(hand_lms, flip=(side == "l"))
                finger_state[f"thumb_open_{side}"] = thumb_abduction(hand_lms)

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]

            nose_x = lm[0].x
            nose_y = lm[0].y

            if not calibrated:
                shoulder = lm[11]
                wrist    = lm[15]
                _, raw_fwd = arm_angles(shoulder, wrist)
                calib_samples.append((nose_x, nose_y))
                arm_fwd_samples.append(raw_fwd)

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

            yaw   = (neutral_x - nose_x) * 550.0
            pitch = (nose_y - neutral_y) * 550.0

            shoulder = lm[11]
            elbow    = lm[13]
            wrist    = lm[15]

            raise_deg, arm_forward_raw = arm_angles(shoulder, wrist)
            arm_forward  = max(-90.0, min(90.0, (arm_forward_raw - neutral_arm_fwd) * 4.5))
            wave         = (wrist.x - elbow.x) * 200.0
            elbow_bend   = elbow_bend_angle(shoulder, elbow, wrist)

            r_shoulder = lm[12]
            r_elbow    = lm[14]
            r_wrist    = lm[16]
            raise_deg_r, arm_forward_raw_r = arm_angles(r_shoulder, r_wrist)
            arm_forward_r = max(-90.0, min(90.0, (arm_forward_raw_r - neutral_arm_fwd_r) * 4.5))
            wave_r        = (r_wrist.x - r_elbow.x) * 200.0
            elbow_bend_r  = elbow_bend_angle(r_shoulder, r_elbow, r_wrist)

            data = {
                "yaw":           yaw,
                "pitch":         pitch,
                "arm_raise":     raise_deg,
                "wave":          wave,
                "arm_forward":   arm_forward,
                "arm_raise_r":   raise_deg_r,
                "wave_r":        wave_r,
                "arm_forward_r": arm_forward_r,
                "elbow_bend":    elbow_bend,
                "elbow_bend_r":  elbow_bend_r,
                **finger_state,
            }
            sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

            cv2.putText(
                frame,
                f"Raise:{raise_deg:.0f} Fwd:{arm_forward:.0f} Wave:{wave:.0f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        cv2.imshow("Head + Arm Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            calibrated = False
            calib_samples.clear()
            arm_fwd_samples.clear()
            arm_fwd_r_samples.clear()
            for k in finger_state:
                finger_state[k] = 0.0

cap.release()
cv2.destroyAllWindows()
