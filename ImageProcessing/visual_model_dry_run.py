import cv2
import numpy as np
import math
import time
from collections import deque

from stable_baselines3 import PPO


MODEL_PATH = "belt_stepper_pendulum_rl.zip"

model = PPO.load(MODEL_PATH)

STEP_RATE_TABLE = np.array(
    [-800, -400, -150, 0, 150, 400, 800],
    dtype=np.float32
)


CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try CAMERA_INDEX = 1 or 2.")

time.sleep(1)


LOWER_YELLOW = np.array([25, 120, 120])
UPPER_YELLOW = np.array([40, 255, 255])

LOWER_ORANGE = np.array([5, 150, 150])
UPPER_ORANGE = np.array([20, 255, 255])


def find_largest_blob(mask, min_area=100):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)

        if area < min_area:
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if area > best_area:
            best_area = area
            best = (cx, cy, area)

    return best


def angle_difference(theta_new, theta_old):
    dtheta = theta_new - theta_old
    return math.atan2(math.sin(dtheta), math.cos(dtheta))


def camera_theta_to_model_theta(theta_camera):
    return math.atan2(
        math.sin(theta_camera - math.pi),
        math.cos(theta_camera - math.pi)
    )


prev_theta_model = None
prev_time = None

theta_dot_history = deque(maxlen=10)

bob_x_smooth = None
bob_y_smooth = None

pivot_x_smooth = None
pivot_y_smooth = None

BOB_ALPHA = 0.35
PIVOT_ALPHA = 0.20


while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        continue

    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    orange_mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    kernel = np.ones((5, 5), np.uint8)

    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

    pivot = find_largest_blob(yellow_mask, min_area=100)
    bob = find_largest_blob(orange_mask, min_area=150)

    if pivot is not None and bob is not None:
        px_raw, py_raw, pivot_area = pivot
        bx_raw, by_raw, bob_area = bob

        if pivot_x_smooth is None:
            pivot_x_smooth = px_raw
            pivot_y_smooth = py_raw
        else:
            pivot_x_smooth = PIVOT_ALPHA * px_raw + (1.0 - PIVOT_ALPHA) * pivot_x_smooth
            pivot_y_smooth = PIVOT_ALPHA * py_raw + (1.0 - PIVOT_ALPHA) * pivot_y_smooth

        if bob_x_smooth is None:
            bob_x_smooth = bx_raw
            bob_y_smooth = by_raw
        else:
            bob_x_smooth = BOB_ALPHA * bx_raw + (1.0 - BOB_ALPHA) * bob_x_smooth
            bob_y_smooth = BOB_ALPHA * by_raw + (1.0 - BOB_ALPHA) * bob_y_smooth

        px = int(pivot_x_smooth)
        py = int(pivot_y_smooth)
        bx = int(bob_x_smooth)
        by = int(bob_y_smooth)

        dx = bx - px
        dy = by - py

        theta_camera = math.atan2(dx, dy)

        theta_model = camera_theta_to_model_theta(theta_camera)

        now = time.time()

        if prev_theta_model is not None and prev_time is not None:
            dt = now - prev_time

            if dt > 0:
                dtheta = angle_difference(theta_model, prev_theta_model)
                theta_dot_raw = dtheta / dt
            else:
                theta_dot_raw = 0.0
        else:
            theta_dot_raw = 0.0

        theta_dot_history.append(theta_dot_raw)
        theta_dot_model = sum(theta_dot_history) / len(theta_dot_history)

        prev_theta_model = theta_model
        prev_time = now

        obs = np.array(
            [
                math.cos(theta_model),
                math.sin(theta_model),
                theta_dot_model
            ],
            dtype=np.float32
        )

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        step_rate = float(STEP_RATE_TABLE[action])

        print(
            f"theta_model={math.degrees(theta_model):+.2f} deg, "
            f"theta_dot={theta_dot_model:+.3f} rad/s, "
            f"action={action}, "
            f"step_rate={step_rate:+.0f} steps/s, "
            f"pivot_area={pivot_area:.1f}, "
            f"bob_area={bob_area:.1f}"
        )

        cv2.circle(frame, (px_raw, py_raw), 4, (0, 255, 255), -1)
        cv2.circle(frame, (bx_raw, by_raw), 4, (0, 140, 255), -1)

        cv2.circle(frame, (px, py), 9, (0, 255, 255), 2)
        cv2.circle(frame, (bx, by), 9, (0, 140, 255), 2)
        cv2.line(frame, (px, py), (bx, by), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"theta_model: {math.degrees(theta_model):+.2f} deg",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"theta_dot: {theta_dot_model:+.2f} rad/s",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"action: {action}, step_rate: {step_rate:+.0f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"bob area: {bob_area:.0f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    else:
        if pivot is None:
            cv2.putText(
                frame,
                "Pivot not detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        if bob is None:
            cv2.putText(
                frame,
                "Bob not detected",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 140, 255),
                2
            )

    cv2.imshow("Vision + Model Dry Run", frame)
    cv2.imshow("Yellow Pivot Mask", yellow_mask)
    cv2.imshow("Orange Bob Mask", orange_mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()