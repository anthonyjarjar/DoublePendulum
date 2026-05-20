import math
import time
import serial
import numpy as np
from stable_baselines3 import PPO


ARDUINO_PORT = "/dev/cu.usbmodem101"
BAUD_RATE = 115200

MODEL_PATH = "belt_stepper_pendulum_rl.zip"

BASE_STEP_RATE_TABLE = np.array(
    [-9000, -6000, -3000, 0, 3000, 6000, 9000],
    dtype=np.float32
)

SPEED_SCALE = 0.75
STEP_RATE_TABLE = SPEED_SCALE * BASE_STEP_RATE_TABLE

MOTOR_SIGN = +1

ANGLE_CUTOFF_DEG = 12.0
START_TOLERANCE_DEG = 5.0

SEND_INTERVAL = 0.02

THETA_DOT_ALPHA = 1.0
PYTHON_ENCODER_SIGN = +1


def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def encoder_to_model_angle(theta_encoder):
    return wrap_angle(theta_encoder - math.pi)


def parse_encoder_line(line):
    line = line.strip()

    if not line.startswith("E,"):
        return None

    parts = line.split(",")

    if len(parts) != 4:
        return None

    try:
        theta_encoder = float(parts[1])
        theta_dot_encoder = float(parts[2])
        count = int(parts[3])
        return theta_encoder, theta_dot_encoder, count

    except ValueError:
        return None


def send_step_rate(ser, step_rate):
    ser.write(f"S:{int(step_rate)}\n".encode())


def stop_motor(ser):
    send_step_rate(ser, 0)


def read_latest_state(ser, timeout=1.0):
    latest = None
    start = time.time()

    while time.time() - start < timeout:
        while ser.in_waiting > 0:
            raw_line = ser.readline().decode(errors="ignore").strip()
            parsed = parse_encoder_line(raw_line)

            if parsed is not None:
                theta_encoder, theta_dot_encoder, count = parsed

                theta_encoder = wrap_angle(PYTHON_ENCODER_SIGN * theta_encoder)
                theta_dot_encoder = PYTHON_ENCODER_SIGN * theta_dot_encoder

                theta_model = encoder_to_model_angle(theta_encoder)
                theta_dot_model = theta_dot_encoder

                latest = theta_model, theta_dot_model, count

        if latest is not None:
            return latest

        time.sleep(0.001)

    return None


def wait_for_zeroed_response(ser, timeout=2.0):
    start = time.time()

    while time.time() - start < timeout:
        line = ser.readline().decode(errors="ignore").strip()

        if line:
            print("Arduino:", line)

        if line == "ZEROED":
            return True

    return False


print("Loading PPO model...")
model = PPO.load(MODEL_PATH)
print("Model loaded.")


print("Opening Arduino serial port...")
ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.01)
time.sleep(2)

print("Connected to Arduino.")


start_time = time.time()
while time.time() - start_time < 1.0:
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print("Arduino:", line)

stop_motor(ser)


print()
print("STEP 1: ZERO ENCODER")
print("Let the pendulum hang straight down and stop moving.")
input("Press Enter to zero at hanging-down position...")

stop_motor(ser)
time.sleep(0.1)

ser.write(b"Z\n")

if not wait_for_zeroed_response(ser):
    print("WARNING: Did not see ZEROED response, but continuing.")

time.sleep(0.2)
stop_motor(ser)

print()
print("Encoder zeroed.")
print("Now lift the pendulum upright.")
print("Motor is still OFF.")
print()


print("STEP 2: MOVE TO UPRIGHT")
print(f"Get theta_model close to 0 degrees, within about ±{START_TOLERANCE_DEG} deg.")
print("Live angle preview will print below.")
print("Press Enter to START RL only when pendulum is upright.")
print()

last_preview = 0.0

while True:
    state = read_latest_state(ser, timeout=0.05)

    if state is not None:
        theta_model, theta_dot_model, count = state
        theta_deg = math.degrees(theta_model)

        now = time.time()

        if now - last_preview >= 0.15:
            last_preview = now

            ok_text = "OK TO START" if abs(theta_deg) <= START_TOLERANCE_DEG else "NOT UPRIGHT"

            print(
                f"theta_model={theta_deg:+.2f} deg, "
                f"theta_dot={theta_dot_model:+.3f}, "
                f"count={count}, "
                f"{ok_text}"
            )

    if abs(math.degrees(state[0])) <= START_TOLERANCE_DEG if state is not None else False:
        cmd = input("Pendulum is near upright. Press Enter to START RL, or type n then Enter to keep previewing: ")
        if cmd.strip().lower() != "n":
            break

print()
print("Starting RL loop now.")
print("Keep your hand nearby. Ctrl+C stops.")
print()


latest_theta_model = None
latest_theta_dot_model = None
latest_count = None

filtered_theta_dot = None

last_send_time = 0.0
last_print_time = 0.0

try:
    while True:
        while ser.in_waiting > 0:
            raw_line = ser.readline().decode(errors="ignore").strip()
            parsed = parse_encoder_line(raw_line)

            if parsed is not None:
                theta_encoder, theta_dot_encoder, count = parsed

                theta_encoder = wrap_angle(PYTHON_ENCODER_SIGN * theta_encoder)
                theta_dot_encoder = PYTHON_ENCODER_SIGN * theta_dot_encoder

                latest_theta_model = encoder_to_model_angle(theta_encoder)
                latest_theta_dot_model = theta_dot_encoder
                latest_count = count

        if latest_theta_model is None or latest_theta_dot_model is None:
            continue

        if filtered_theta_dot is None:
            filtered_theta_dot = latest_theta_dot_model
        else:
            filtered_theta_dot = (
                THETA_DOT_ALPHA * latest_theta_dot_model
                + (1.0 - THETA_DOT_ALPHA) * filtered_theta_dot
            )

        theta_model = latest_theta_model
        theta_dot_model = filtered_theta_dot

        theta_deg = math.degrees(theta_model)
        now = time.time()

        if abs(theta_deg) > ANGLE_CUTOFF_DEG:
            step_rate = 0
            action = -1
            status = "ANGLE_CUTOFF"

            if now - last_send_time >= SEND_INTERVAL:
                stop_motor(ser)
                last_send_time = now

        else:
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

            step_rate = MOTOR_SIGN * float(STEP_RATE_TABLE[action])

            if now - last_send_time >= SEND_INTERVAL:
                send_step_rate(ser, step_rate)
                last_send_time = now

            status = "BALANCING"

        if now - last_print_time >= 0.10:
            last_print_time = now

            print(
                f"theta_model={theta_deg:+.2f} deg, "
                f"theta_dot={theta_dot_model:+.3f}, "
                f"action={action}, "
                f"step_rate={step_rate:+.0f}, "
                f"count={latest_count}, "
                f"status={status}"
            )

except KeyboardInterrupt:
    print()
    print("Stopping...")

finally:
    stop_motor(ser)
    time.sleep(0.2)
    ser.close()
    print("Motor stopped. Serial closed.")