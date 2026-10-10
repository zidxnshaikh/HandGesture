import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from pycaw.pycaw import AudioUtilities
import screen_brightness_control as sbc


class HandDetector:
    def __init__(self, max_hands=1, detection_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return frame

    def find_position(self, frame, hand_no=0):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            h, w, _ = frame.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz = int(lm.z * w)  # z uses roughly the same scale as x, per MediaPipe docs
                landmark_list.append([id, cx, cy, cz])
        return landmark_list

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def fingers_up(self, landmark_list):
        if not landmark_list:
            return []

        tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        fingers = []

        # Thumb - angle-based: checks if thumb joint is straight or bent,
        # works regardless of whether palm or back of hand faces the camera
        thumb_mcp = landmark_list[2][1:]
        thumb_ip = landmark_list[3][1:]
        thumb_tip = landmark_list[4][1:]
        thumb_angle = self.calculate_angle(thumb_mcp, thumb_ip, thumb_tip)

        if thumb_angle > 150:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers - y-axis compare (tip above joint = up)
        for id in range(1, 5):
            if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_hand_count(self):
        if self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0

    def get_hand_label(self, hand_no=0):
        if self.results.multi_handedness:
            return self.results.multi_handedness[hand_no].classification[0].label
        return None


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def setup_volume_control():
    devices = AudioUtilities.GetSpeakers()
    return devices.EndpointVolume


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Gesture Control", cv2.WINDOW_NORMAL)
    detector = HandDetector()

    screen_w, screen_h = pyautogui.size()
    frame_reduction = 100     # margin so hand doesn't need to reach frame edges
    smoothening = 6
    prev_x, prev_y = screen_w // 2, screen_h // 2

    volume_ctrl = setup_volume_control()
    current_volume = volume_ctrl.GetMasterVolumeLevelScalar() * 100
    current_brightness = sbc.get_brightness()[0]
    last_click_time = 0
    click_cooldown = 1  # seconds - prevents one pinch from firing multiple clicks
    last_screenshot_time = 0
    screenshot_cooldown = 2  # seconds - prevents one fist from firing multiple screenshots
    last_vb_time = 0
    vb_cooldown = 0.3   # seconds between each step while the gesture is held
    vb_step = 5         # percent change per step
    hover_id = None
    hover_frames = 0
    HOVER_FRAMES_REQUIRED = 5   # thumb must stay near the same button this many frames before it fires

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # mirror view - makes Left/Right labels match your actual hand
        frame = detector.find_hands(frame)
        hand_count = detector.get_hand_count()
        h, w, _ = frame.shape
        status_text = "Ready"

        if hand_count > 1:
            status_text = "Please show only ONE hand!"

        elif hand_count == 1:
            label = detector.get_hand_label(0)

            if label != "Right":
                status_text = "Please use your RIGHT hand"
            else:
                landmark_list = detector.find_position(frame)
                if landmark_list:
                    fingers = detector.fingers_up(landmark_list)
                    pattern = tuple(fingers)

                    thumb = landmark_list[4][1:3]    # x,y only (screen-space, ignore z here)
                    index = landmark_list[8][1:3]
                    middle = landmark_list[12][1:3]

                    # CURSOR MOVE - always active whenever the right hand is detected
                    x_mapped = np.interp(index[0], [frame_reduction, w - frame_reduction], [0, screen_w])
                    y_mapped = np.interp(index[1], [frame_reduction, h - frame_reduction], [0, screen_h])
                    curr_x = prev_x + (x_mapped - prev_x) / smoothening
                    curr_y = prev_y + (y_mapped - prev_y) / smoothening
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                    status_text = "Cursor Moving"

                    # ON-HAND BUTTONS - drawn at fixed landmark IDs, "pressed" by bringing the thumb close
                    if pattern != (0, 0, 0, 0, 0):  # skip while making a fist (that's the screenshot gesture)
                        buttons = {
                            13: ("B-", "bright_down"),
                            15: ("B+", "bright_up"),
                            17: ("V-", "vol_down"),
                            19: ("V+", "vol_up"),
                        }
                        press_threshold = 35   # pixels - how close the thumb must get to count as a press
                        button_radius = 18

                        # find the closest button, with hysteresis so jitter doesn't flip-flop the target
                        distances = {}
                        for lm_id in buttons:
                            bx, by = landmark_list[lm_id][1], landmark_list[lm_id][2]
                            distances[lm_id] = get_distance(thumb, (bx, by))
                        candidate_id = min(distances, key=distances.get)
                        HYSTERESIS_MARGIN = 10  # px - a new button must be clearly closer to steal the target

                        if hover_id is not None and hover_id in distances and \
                                distances[candidate_id] + HYSTERESIS_MARGIN >= distances[hover_id]:
                            # keep tracking the current target unless the candidate is clearly closer
                            candidate_id = hover_id

                        within_range = distances[candidate_id] < press_threshold

                        if within_range and candidate_id == hover_id:
                            hover_frames += 1
                        elif within_range:
                            hover_id = candidate_id
                            hover_frames = 1
                        else:
                            hover_id = None
                            hover_frames = 0

                        for lm_id, (btn_label, action) in buttons.items():
                            bx, by = landmark_list[lm_id][1], landmark_list[lm_id][2]
                            hovering = (lm_id == hover_id) and within_range
                            confirmed = hovering and (hover_frames >= HOVER_FRAMES_REQUIRED)

                            if confirmed:
                                color = (0, 255, 0)       # green = confirmed, action will fire
                            elif hovering:
                                color = (0, 165, 255)     # orange = getting close, hold still
                            else:
                                color = (255, 255, 255)   # white = idle

                            cv2.circle(frame, (bx, by), button_radius, color,
                                       cv2.FILLED if confirmed else 2)
                            cv2.putText(frame, btn_label, (bx - 15, by - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            if confirmed and (time.time() - last_vb_time > vb_cooldown):
                                if action == "bright_down":
                                    current_brightness = max(0, current_brightness - vb_step)
                                    sbc.set_brightness(current_brightness)
                                    status_text = f"Brightness Down: {int(current_brightness)}%"
                                elif action == "bright_up":
                                    current_brightness = min(100, current_brightness + vb_step)
                                    sbc.set_brightness(current_brightness)
                                    status_text = f"Brightness Up: {int(current_brightness)}%"
                                elif action == "vol_down":
                                    current_volume = max(0, current_volume - vb_step)
                                    volume_ctrl.SetMasterVolumeLevelScalar(current_volume / 100, None)
                                    status_text = f"Volume Down: {int(current_volume)}%"
                                elif action == "vol_up":
                                    current_volume = min(100, current_volume + vb_step)
                                    volume_ctrl.SetMasterVolumeLevelScalar(current_volume / 100, None)
                                    status_text = f"Volume Up: {int(current_volume)}%"
                                last_vb_time = time.time()

                    if pattern == (0, 0, 0, 0, 0):
                        # SCREENSHOT - closed fist (checked before click, so a fist never fires a click)
                        current_time = time.time()
                        if current_time - last_screenshot_time > screenshot_cooldown:
                            img = pyautogui.screenshot()
                            filename = f"screenshot_{int(current_time)}.png"
                            img.save(filename)
                            last_screenshot_time = current_time
                            status_text = f"Screenshot Saved: {filename}"
                        else:
                            status_text = "Fist Detected"

                    else:
                        # CLICK - pinch check (runs whenever fist isn't matched)
                        current_time = time.time()
                        if current_time - last_click_time > click_cooldown:
                            if get_distance(thumb, index) < 40:
                                pyautogui.click()
                                last_click_time = current_time
                                status_text = "Left Click"
                            elif get_distance(thumb, middle) < 40:
                                pyautogui.click(button='right')
                                last_click_time = current_time
                                status_text = "Right Click"

        cv2.putText(frame, status_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 0), 3)
        cv2.imshow("Hand Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()