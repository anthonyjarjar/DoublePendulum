import cv2
import numpy as np
import math
import time
import threading
from collections import deque


class FastCamera:

    def __init__(self, index=0, width=160, height=120, fps=60):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.running = True

        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

        time.sleep(0.5)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Try CAMERA_INDEX = 1 or 2.")

    def update(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None

            return True, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


CAMERA_INDEX = 0

FRAME_WIDTH = 160
FRAME_HEIGHT = 120
FPS = 60

FLIP_FRAME = True

SHOW_MASKS = False

PRINT_EVERY = 10


LOWER_BLUE = np.array([90, 30, 15])
UPPER_BLUE = np.array([140, 255, 180])


LOWER_ORANGE = np.array([5, 100, 70])
UPPER_ORANGE = np.array([28, 255, 255])


PIVOT_MIN_AREA = 5
BOB_MIN_AREA = 5


PIVOT_ALPHA = 1.00
BOB_ALPHA = 1.00


THETA_DOT_HISTORY_LEN = 1


def find_largest_blob(mask, min_area=5):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0.0

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


prev_theta = None
prev_time = None

theta_dot_history = deque(maxlen=THETA_DOT_HISTORY_LEN)

pivot_x_smooth = None
pivot_y_smooth = None

bob_x_smooth = None
bob_y_smooth = None

frame_count = 0

last_loop_time = time.perf_counter()
fps_estimate = 0.0


cam = FastCamera(
    index=CAMERA_INDEX,
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT,
    fps=FPS
)

try:
    while True:
        ret, frame = cam.read()

        if not ret:
            continue

        frame_count += 1

        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        pivot_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        orange_mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

        kernel = np.ones((3, 3), np.uint8)

        pivot_mask = cv2.morphologyEx(pivot_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

        pivot = find_largest_blob(pivot_mask, min_area=PIVOT_MIN_AREA)
        bob = find_largest_blob(orange_mask, min_area=BOB_MIN_AREA)

        if pivot is not None and bob is not None:
            px_raw, py_raw, pivot_area = pivot
            bx_raw, by_raw, bob_area = bob

            if pivot_x_smooth is None:
                pivot_x_smooth = float(px_raw)
                pivot_y_smooth = float(py_raw)
            else:
                pivot_x_smooth = PIVOT_ALPHA * px_raw + (1.0 - PIVOT_ALPHA) * pivot_x_smooth
                pivot_y_smooth = PIVOT_ALPHA * py_raw + (1.0 - PIVOT_ALPHA) * pivot_y_smooth

            if bob_x_smooth is None:
                bob_x_smooth = float(bx_raw)
                bob_y_smooth = float(by_raw)
            else:
                bob_x_smooth = BOB_ALPHA * bx_raw + (1.0 - BOB_ALPHA) * bob_x_smooth
                bob_y_smooth = BOB_ALPHA * by_raw + (1.0 - BOB_ALPHA) * bob_y_smooth

            px = int(pivot_x_smooth)
            py = int(pivot_y_smooth)

            bx = int(bob_x_smooth)
            by = int(bob_y_smooth)

            dx = bx - px
            dy = by - py

            theta = math.atan2(dx, dy)

            now = time.perf_counter()

            if prev_theta is not None and prev_time is not None:
                dt = now - prev_time

                if dt > 0:
                    dtheta = angle_difference(theta, prev_theta)
                    theta_dot_raw = dtheta / dt
                else:
                    theta_dot_raw = 0.0
            else:
                theta_dot_raw = 0.0

            theta_dot_history.append(theta_dot_raw)
            theta_dot = sum(theta_dot_history) / len(theta_dot_history)

            prev_theta = theta
            prev_time = now

            loop_dt = now - last_loop_time
            last_loop_time = now

            if loop_dt > 0:
                fps_estimate = 1.0 / loop_dt

            state = np.array(
                [math.sin(theta), math.cos(theta), theta_dot],
                dtype=np.float32
            )

            if frame_count % PRINT_EVERY == 0:
                print(
                    f"theta={theta:+.4f} rad, "
                    f"theta_deg={math.degrees(theta):+.2f}, "
                    f"theta_dot={theta_dot:+.4f} rad/s, "
                    f"fps={fps_estimate:.1f}, "
                    f"pivot_area={pivot_area:.1f}, "
                    f"bob_area={bob_area:.1f}, "
                    f"state=[{state[0]:+.3f}, {state[1]:+.3f}, {state[2]:+.3f}]"
                )

            cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)
            cv2.circle(frame, (bx, by), 4, (0, 140, 255), -1)
            cv2.line(frame, (px, py), (bx, by), (0, 255, 0), 1)

            cv2.putText(
                frame,
                f"theta={math.degrees(theta):+.1f} deg",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                f"tdot={theta_dot:+.2f}",
                (5, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                f"fps={fps_estimate:.0f}",
                (5, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

        else:
            if pivot is None:
                cv2.putText(
                    frame,
                    "No pivot",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1
                )

            if bob is None:
                cv2.putText(
                    frame,
                    "No bob",
                    (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 140, 255),
                    1
                )

        cv2.imshow("Fast Pendulum Tracking", frame)

        if SHOW_MASKS:
            cv2.imshow("Pivot Mask - Blue", pivot_mask)
            cv2.imshow("Bob Mask - Orange", orange_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:
    cam.release()
    cv2.destroyAllWindows()