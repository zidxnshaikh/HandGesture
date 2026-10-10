# Hand Gesture Control

Control your Windows computer with hand gestures captured by a webcam. The app uses MediaPipe to track one hand and OpenCV to display the camera feed.

## Features

- Move the cursor with your index finger.
- Left-click by pinching your thumb and index finger.
- Right-click by pinching your thumb and middle finger.
- Adjust screen brightness and system volume using on-hand controls.
- Take a screenshot by making a fist.

## Requirements

- Windows
- Python 3.12
- A working webcam
- A display with brightness control for brightness adjustments

## Setup

Open PowerShell in the project directory and create/activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install opencv-python mediapipe pyautogui numpy pycaw screen-brightness-control
```

## Run

```powershell
python main.py
```

Show one right hand to the camera. Press **Q** while the camera window is active to quit. Screenshots are saved as `screenshot_<timestamp>.png` in the current working directory.

## Gesture controls

| Gesture/action | Result |
| --- | --- |
| Move index finger | Move the cursor |
| Pinch thumb and index finger | Left-click |
| Pinch thumb and middle finger | Right-click |
| Make a fist | Save a screenshot |
| Bring thumb near the control at landmark 13 | Decrease brightness |
| Bring thumb near the control at landmark 15 | Increase brightness |
| Bring thumb near the control at landmark 17 | Decrease volume |
| Bring thumb near the control at landmark 19 | Increase volume |

Hold the thumb near a brightness or volume control briefly to activate it. Volume and brightness change in small steps while the control remains selected.

## Notes

- The app expects one **right hand**; it will ask you to show only one hand if multiple hands are detected.
- Brightness control availability depends on your display and system.
- Hand tracking and gesture detection can vary with lighting, camera position, and hand orientation. See [limitations..txt](limitations..txt) for additional project notes.
