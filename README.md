# Avatar Pose Control

Control a 3D avatar in Blender using nothing but your webcam. Move your head, raise your arms, bend your elbows, and curl your fingers — the avatar mirrors you in real time.

---

## How it works

Two scripts talk to each other over a local UDP connection:

```
Your webcam
    ↓
camera_yaw.py       ← reads your pose via MediaPipe, sends data over UDP
    ↓  (UDP port 5008)
blender_yaw.py      ← runs inside Blender, moves the avatar bones
    ↓
3D Avatar in Blender
```

**`camera_yaw.py`** uses your webcam and Google's MediaPipe to track:
- Head position (yaw & pitch)
- Both arms (raise, swing forward, elbow bend)
- Both hands (wrist roll, all 5 fingers + thumb spread)

**`blender_yaw.py`** runs as a script inside Blender. It listens for that data and applies smooth bone rotations to a Mixamo-rigged character every 50 ms.

---

## Requirements

- Python 3.10+
- Blender 3.x or 4.x
- A webcam

Install Python dependencies:

```bash
pip install -r requirements.txt
```

You also need two MediaPipe model files in the project folder (already included):
- `pose_landmarker_lite.task`
- `hand_landmarker.task`

---

## Setup

### 1. Update the model paths in `camera_yaw.py`

Open `camera_yaw.py` and set the correct absolute paths to the two `.task` files:

```python
model_asset_path="/pose_landmarker_lite.task"
model_asset_path="/hand_landmarker.task"
```

### 2. Load your avatar in Blender

Import a **Mixamo** character (FBX) into Blender. The script expects the standard Mixamo bone names like `mixamorig:Head`, `mixamorig:LeftArm`, etc.

### 3. Run the Blender script

Open `blender_yaw.py` in Blender's text editor and click **Run Script**. You should see this in the console:

```
Head + Arm receiver listening on UDP 127.0.0.1:5008
```

### 4. Start the camera script

In a terminal:

```bash
python camera_yaw.py
```

A window will open showing your webcam feed.

---

## Calibration

When the camera script starts, it asks you to hold a neutral pose for about 2 seconds (60 frames):

> **Look straight ahead with both arms relaxed at your sides.**

This sets the baseline so your avatar starts in a natural resting position. Once calibrated, the avatar starts moving with you.

**Press `R`** at any time to recalibrate (useful if you moved your chair or camera).  
**Press `Q`** to quit.

---

## What gets tracked

| Body part | What it controls |
|---|---|
| Head | Left/right turn, tilt up/down |
| Upper arms | Raise up/down, swing forward/back |
| Forearms | Elbow bend |
| Wrists | Roll (thumbs-up / thumbs-down) |
| Fingers | Each finger curls independently |
| Thumbs | Curl + spread open |

---

## Troubleshooting

**Avatar doesn't move**
Make sure Blender's script is running *before* you start `camera_yaw.py`. Check the Blender console for the "listening on UDP" message.

**Bones move in the wrong direction**
The scale/axis constants at the top of `blender_yaw.py` (e.g. `ARM_RAISE_SCALE`, `FINGER_CURL_SCALE_L`) can be flipped by negating the value.

**Wrong hand is tracked**
If you're using a front-facing (mirror) webcam, `MIRROR_HANDEDNESS = True` in `camera_yaw.py` should handle this automatically.

**Jittery movement**
Increase the smoothing values in `blender_yaw.py`: `HEAD_SMOOTHING`, `ARM_SMOOTHING`, or `FINGER_SMOOTHING` (range 0–1; lower = smoother but more delayed).
