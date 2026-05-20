import math
import tkinter as tk
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO


class HardwarePendulumEnv(gym.Env):
    def __init__(self, render_mode=False):
        super().__init__()

        self.render_mode = render_mode

        self.g = 9.81

        self.base_cart_mass = 1.0
        self.base_bob_mass = 1.0
        self.base_rod_mass = 0.5
        self.base_length = 1.0

        self.base_b_pivot = 0.05
        self.base_c_air = 0.02
        self.base_b_cart = 0.05

        self.dt = 0.02

        self.x_limit = 1.0
        self.theta_limit = math.radians(35)
        self.max_steps = 1000

        self.max_motor_speed = 1.2
        self.motor_force_gain = 55.0
        self.max_motor_force = 25.0

        self.delay_steps = 2
        self.action_queue = deque([1] * self.delay_steps, maxlen=self.delay_steps)

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array(
                [-self.x_limit, -np.inf, -1.0, -1.0, -np.inf],
                dtype=np.float32
            ),
            high=np.array(
                [self.x_limit, np.inf, 1.0, 1.0, np.inf],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.state = None
        self.steps = 0

        self.cart_mass = self.base_cart_mass
        self.bob_mass = self.base_bob_mass
        self.rod_mass = self.base_rod_mass
        self.length = self.base_length

        self.b_pivot = self.base_b_pivot
        self.c_air = self.base_c_air
        self.b_cart = self.base_b_cart

    def angle_normalize(self, theta):
        return ((theta + math.pi) % (2 * math.pi)) - math.pi

    def get_observation(self):
        x, x_dot, theta, theta_dot = self.state

        return np.array(
            [
                x,
                x_dot,
                math.cos(theta),
                math.sin(theta),
                theta_dot
            ],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        x = self.np_random.uniform(-0.04, 0.04)
        x_dot = self.np_random.uniform(-0.05, 0.05)
        theta = self.np_random.uniform(-0.12, 0.12)
        theta_dot = self.np_random.uniform(-0.08, 0.08)

        self.state = np.array(
            [x, x_dot, theta, theta_dot],
            dtype=np.float64
        )

        self.steps = 0
        self.action_queue = deque([1] * self.delay_steps, maxlen=self.delay_steps)

        return self.get_observation(), {}

    def action_to_target_speed(self, action):
        if action == 0:
            return -self.max_motor_speed

        if action == 1:
            return 0.0

        return self.max_motor_speed

    def motor_force(self, x_dot, action):
        target_speed = self.action_to_target_speed(action)

        force = self.motor_force_gain * (target_speed - x_dot)

        force = max(
            -self.max_motor_force,
            min(self.max_motor_force, force)
        )

        return force

    def derivatives(self, state, action):
        x, x_dot, theta, theta_dot = state

        F = self.motor_force(x_dot, action)

        Mc = self.cart_mass
        m = self.bob_mass
        M = self.rod_mass
        L = self.length

        pendulum_mass = m + M

        center_distance = (
            m * L + M * (L / 2)
        ) / pendulum_mass

        I_pivot = (
            m * L**2 + (1 / 3) * M * L**2
        )

        A = Mc + pendulum_mass
        B = pendulum_mass * center_distance
        I = I_pivot

        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        gravity_torque = B * self.g * sin_theta
        pivot_drag_torque = -self.b_pivot * theta_dot
        air_drag_torque = -self.c_air * theta_dot * abs(theta_dot)

        total_theta_torque = (
            gravity_torque
            + pivot_drag_torque
            + air_drag_torque
        )

        cart_drag_force = -self.b_cart * x_dot
        total_cart_force = F + cart_drag_force

        rhs_1 = (
            total_cart_force
            + B * sin_theta * theta_dot**2
        )

        rhs_2 = total_theta_torque

        matrix = np.array(
            [
                [A, B * cos_theta],
                [B * cos_theta, I]
            ],
            dtype=np.float64
        )

        rhs = np.array(
            [rhs_1, rhs_2],
            dtype=np.float64
        )

        x_ddot, theta_ddot = np.linalg.solve(matrix, rhs)

        return np.array(
            [
                x_dot,
                x_ddot,
                theta_dot,
                theta_ddot
            ],
            dtype=np.float64
        )

    def rk4_update(self, state, action):
        dt = self.dt

        k1 = self.derivatives(state, action)

        k2 = self.derivatives(
            state + 0.5 * dt * k1,
            action
        )

        k3 = self.derivatives(
            state + 0.5 * dt * k2,
            action
        )

        k4 = self.derivatives(
            state + dt * k3,
            action
        )

        new_state = state + (dt / 6.0) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

        new_state[2] = self.angle_normalize(new_state[2])

        return new_state

    def step(self, action):
        self.action_queue.append(int(action))
        delayed_action = self.action_queue[0]

        self.state = self.rk4_update(self.state, delayed_action)
        self.steps += 1

        x, x_dot, theta, theta_dot = self.state

        reward = (
            3.0 * math.cos(theta)
            - 0.4 * x**2
            - 0.04 * x_dot**2
            - 0.02 * theta_dot**2
        )

        terminated = (
            abs(x) > self.x_limit
            or abs(theta) > self.theta_limit
        )

        truncated = self.steps >= self.max_steps

        return self.get_observation(), reward, terminated, truncated, {}


class PendulumViewer:
    def __init__(self, env, model):
        self.env = env
        self.model = model

        self.root = tk.Tk()
        self.root.title("View Saved Hardware-Ready RL Model")

        self.canvas_width = 700
        self.canvas_height = 500

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white"
        )
        self.canvas.pack()

        self.scale = 220
        self.cart_y = 300
        self.cart_width = 80
        self.cart_height = 35

        self.cart = self.canvas.create_rectangle(
            0, 0, 0, 0,
            fill="gray",
            outline="black"
        )

        self.pole = self.canvas.create_line(
            0, 0, 0, 0,
            width=5,
            fill="brown"
        )

        self.bob = self.canvas.create_oval(
            0, 0, 0, 0,
            fill="royalblue",
            outline="black"
        )

        self.info = self.canvas.create_text(
            20,
            20,
            anchor="nw",
            font=("Arial", 14),
            text=""
        )

        self.obs, _ = self.env.reset()
        self.animate()

    def draw(self, action):
        x, x_dot, theta, theta_dot = self.env.state

        cart_x = self.canvas_width / 2 + x * self.scale
        cart_y = self.cart_y

        left = cart_x - self.cart_width / 2
        right = cart_x + self.cart_width / 2
        top = cart_y - self.cart_height / 2
        bottom = cart_y + self.cart_height / 2

        self.canvas.coords(
            self.cart,
            left,
            top,
            right,
            bottom
        )

        pivot_x = cart_x
        pivot_y = top

        pole_length_px = self.env.length * self.scale

        end_x = pivot_x + pole_length_px * math.sin(theta)
        end_y = pivot_y - pole_length_px * math.cos(theta)

        self.canvas.coords(
            self.pole,
            pivot_x,
            pivot_y,
            end_x,
            end_y
        )

        bob_radius = 12

        self.canvas.coords(
            self.bob,
            end_x - bob_radius,
            end_y - bob_radius,
            end_x + bob_radius,
            end_y + bob_radius
        )

        if action == 0:
            action_text = "LEFT"
        elif action == 1:
            action_text = "STOP"
        else:
            action_text = "RIGHT"

        self.canvas.itemconfig(
            self.info,
            text=(
                f"x = {x:.3f} m\n"
                f"x_dot = {x_dot:.3f} m/s\n"
                f"theta = {math.degrees(theta):.2f} deg\n"
                f"theta_dot = {theta_dot:.3f} rad/s\n"
                f"action = {action_text}"
            )
        )

    def animate(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        action = int(action)

        self.obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self.obs, _ = self.env.reset()

        self.draw(action)

        self.root.after(20, self.animate)

    def run(self):
        self.root.mainloop()


print("Loading saved model...")

model = PPO.load("hardware_ready_pendulum_rl")

print("Model loaded.")
print("Opening viewer...")

env = HardwarePendulumEnv(render_mode=True)
viewer = PendulumViewer(env, model)
viewer.run()