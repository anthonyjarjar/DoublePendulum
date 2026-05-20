import tkinter as tk
from tkinter import ttk
import math
import time


class Pendulum:
    def __init__(self, root):
        self.root = root
        self.root.title("Pendulum Physics")

        self.canvas = tk.Canvas(root, width=500, height=400, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.panel = ttk.Frame(root, padding=12)
        self.panel.grid(row=0, column=1, sticky="ns")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.pivot_x, self.pivot_y = 250, 60

        self.g = 9.81
        self.scale = 100

        self.theta = math.radians(45)
        self.omega = 0.0

        self.bob_mass = 1.0
        self.rod_mass = 0.5

        self.b_pivot = 0.02
        self.b_air = 0.00
        self.c_air = 0.01

        self.last_time = time.perf_counter()

        self.angle_frame = ttk.Frame(self.panel)
        self.angle_frame.pack(fill="x", pady=8)

        self.angle_label = ttk.Label(
            self.angle_frame,
            text="Initial angle theta0 (deg)"
        )
        self.angle_label.pack(anchor="w")

        self.angle_value_label = ttk.Label(
            self.angle_frame,
            text="45.00"
        )
        self.angle_value_label.pack(anchor="e")

        self.angle_slider = ttk.Scale(
            self.angle_frame,
            from_=0,
            to=360,
            orient="horizontal",
            length=250,
            command=self.update_angle_label
        )
        self.angle_slider.set(45)
        self.angle_slider.pack(fill="x")

        self.velocity_frame = ttk.Frame(self.panel)
        self.velocity_frame.pack(fill="x", pady=8)

        self.velocity_label = ttk.Label(
            self.velocity_frame,
            text="Initial velocity omega0 (rad/s)"
        )
        self.velocity_label.pack(anchor="w")

        self.velocity_value_label = ttk.Label(
            self.velocity_frame,
            text="0.00"
        )
        self.velocity_value_label.pack(anchor="e")

        self.velocity_slider = ttk.Scale(
            self.velocity_frame,
            from_=-10.0,
            to=10.0,
            orient="horizontal",
            length=250,
            command=self.update_velocity_label
        )
        self.velocity_slider.set(0.0)
        self.velocity_slider.pack(fill="x")

        self.length_frame = ttk.Frame(self.panel)
        self.length_frame.pack(fill="x", pady=8)

        self.length_label = ttk.Label(
            self.length_frame,
            text="Length L (m)"
        )
        self.length_label.pack(anchor="w")

        self.length_value_label = ttk.Label(
            self.length_frame,
            text="2.00"
        )
        self.length_value_label.pack(anchor="e")

        self.length_slider = ttk.Scale(
            self.length_frame,
            from_=0.5,
            to=5.0,
            orient="horizontal",
            length=250,
            command=self.update_length_label
        )
        self.length_slider.set(2.0)
        self.length_slider.pack(fill="x")

        self.bob_mass_frame = ttk.Frame(self.panel)
        self.bob_mass_frame.pack(fill="x", pady=8)

        self.bob_mass_label = ttk.Label(
            self.bob_mass_frame,
            text="Bob mass m (kg)"
        )
        self.bob_mass_label.pack(anchor="w")

        self.bob_mass_value_label = ttk.Label(
            self.bob_mass_frame,
            text="1.00"
        )
        self.bob_mass_value_label.pack(anchor="e")

        self.bob_mass_slider = ttk.Scale(
            self.bob_mass_frame,
            from_=0.1,
            to=20.0,
            orient="horizontal",
            length=250,
            command=self.update_bob_mass_label
        )
        self.bob_mass_slider.set(self.bob_mass)
        self.bob_mass_slider.pack(fill="x")

        self.rod_mass_frame = ttk.Frame(self.panel)
        self.rod_mass_frame.pack(fill="x", pady=8)

        self.rod_mass_label = ttk.Label(
            self.rod_mass_frame,
            text="Rod mass M (kg)"
        )
        self.rod_mass_label.pack(anchor="w")

        self.rod_mass_value_label = ttk.Label(
            self.rod_mass_frame,
            text="0.50"
        )
        self.rod_mass_value_label.pack(anchor="e")

        self.rod_mass_slider = ttk.Scale(
            self.rod_mass_frame,
            from_=0.0,
            to=20.0,
            orient="horizontal",
            length=250,
            command=self.update_rod_mass_label
        )
        self.rod_mass_slider.set(self.rod_mass)
        self.rod_mass_slider.pack(fill="x")

        self.pivot_friction_frame = ttk.Frame(self.panel)
        self.pivot_friction_frame.pack(fill="x", pady=8)

        self.pivot_friction_label = ttk.Label(
            self.pivot_friction_frame,
            text="Pivot friction b_pivot"
        )
        self.pivot_friction_label.pack(anchor="w")

        self.pivot_friction_value_label = ttk.Label(
            self.pivot_friction_frame,
            text="0.020"
        )
        self.pivot_friction_value_label.pack(anchor="e")

        self.pivot_friction_slider = ttk.Scale(
            self.pivot_friction_frame,
            from_=0.0,
            to=5.0,
            orient="horizontal",
            length=250,
            command=self.update_pivot_friction_label
        )
        self.pivot_friction_slider.set(self.b_pivot)
        self.pivot_friction_slider.pack(fill="x")

        self.air_drag_frame = ttk.Frame(self.panel)
        self.air_drag_frame.pack(fill="x", pady=8)

        self.air_drag_label = ttk.Label(
            self.air_drag_frame,
            text="Quadratic air drag c_air"
        )
        self.air_drag_label.pack(anchor="w")

        self.air_drag_value_label = ttk.Label(
            self.air_drag_frame,
            text="0.010"
        )
        self.air_drag_value_label.pack(anchor="e")

        self.air_drag_slider = ttk.Scale(
            self.air_drag_frame,
            from_=0.0,
            to=5.0,
            orient="horizontal",
            length=250,
            command=self.update_air_drag_label
        )
        self.air_drag_slider.set(self.c_air)
        self.air_drag_slider.pack(fill="x")

        self.apply_button = ttk.Button(
            self.panel,
            text="Apply Initial Conditions",
            command=self.apply_initial_conditions
        )
        self.apply_button.pack(fill="x", pady=10)

        self.reset_button = ttk.Button(
            self.panel,
            text="Reset",
            command=self.reset
        )
        self.reset_button.pack(fill="x", pady=5)

        self.pole = self.canvas.create_line(
            0, 0, 0, 0,
            width=5,
            fill="brown",
            capstyle=tk.ROUND
        )

        self.bob = self.canvas.create_oval(
            0, 0, 0, 0,
            fill="royalblue",
            outline="black"
        )

        self.pivot = self.canvas.create_oval(
            self.pivot_x - 5,
            self.pivot_y - 5,
            self.pivot_x + 5,
            self.pivot_y + 5,
            fill="black"
        )

        self.info_label = ttk.Label(self.panel, text="", justify="left")
        self.info_label.pack(pady=15)

        self.animate()

    def update_angle_label(self, value):
        self.angle_value_label.config(text=f"{float(value):.2f}")

    def update_velocity_label(self, value):
        self.velocity_value_label.config(text=f"{float(value):.2f}")

    def update_length_label(self, value):
        self.length_value_label.config(text=f"{float(value):.2f}")

    def update_bob_mass_label(self, value):
        self.bob_mass_value_label.config(text=f"{float(value):.2f}")

    def update_rod_mass_label(self, value):
        self.rod_mass_value_label.config(text=f"{float(value):.2f}")

    def update_pivot_friction_label(self, value):
        self.pivot_friction_value_label.config(text=f"{float(value):.3f}")

    def update_air_drag_label(self, value):
        self.air_drag_value_label.config(text=f"{float(value):.3f}")

    def apply_initial_conditions(self):
        self.theta = math.radians(self.angle_slider.get())
        self.omega = self.velocity_slider.get()
        self.last_time = time.perf_counter()

    def reset(self):
        self.apply_initial_conditions()

    def theta_double_dot(self, theta, omega):
        L = self.length_slider.get()
        m = self.bob_mass_slider.get()
        M = self.rod_mass_slider.get()

        b_pivot = self.pivot_friction_slider.get()
        b_air = self.b_air
        c_air = self.air_drag_slider.get()

        numerator = (
            -self.g * L * (m + M / 2) * math.sin(theta)
            - (b_pivot + b_air) * omega
            - c_air * omega * abs(omega)
        )

        denominator = L**2 * (m + M / 3)

        return numerator / denominator

    def derivatives(self, theta, omega):
        dtheta_dt = omega
        domega_dt = self.theta_double_dot(theta, omega)

        return dtheta_dt, domega_dt

    def update_physics(self, dt):
        theta0 = self.theta
        omega0 = self.omega

        k1_theta, k1_omega = self.derivatives(theta0, omega0)

        k2_theta, k2_omega = self.derivatives(
            theta0 + 0.5 * dt * k1_theta,
            omega0 + 0.5 * dt * k1_omega
        )

        k3_theta, k3_omega = self.derivatives(
            theta0 + 0.5 * dt * k2_theta,
            omega0 + 0.5 * dt * k2_omega
        )

        k4_theta, k4_omega = self.derivatives(
            theta0 + dt * k3_theta,
            omega0 + dt * k3_omega
        )

        self.theta += (dt / 6) * (
            k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta
        )

        self.omega += (dt / 6) * (
            k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega
        )

    def draw(self):
        L = self.length_slider.get()
        m = self.bob_mass_slider.get()
        M = self.rod_mass_slider.get()

        b_pivot = self.pivot_friction_slider.get()
        c_air = self.air_drag_slider.get()

        length_px = L * self.scale

        end_x = self.pivot_x + length_px * math.sin(self.theta)
        end_y = self.pivot_y + length_px * math.cos(self.theta)

        rod_width = 3 + 4 * math.sqrt(M)
        bob_radius = 8 + 7 * math.sqrt(m)

        self.canvas.coords(
            self.pole,
            self.pivot_x,
            self.pivot_y,
            end_x,
            end_y
        )

        self.canvas.itemconfig(
            self.pole,
            width=rod_width
        )

        self.canvas.coords(
            self.bob,
            end_x - bob_radius,
            end_y - bob_radius,
            end_x + bob_radius,
            end_y + bob_radius
        )

        self.info_label.config(
            text=(
                f"theta = {math.degrees(self.theta):.2f} deg\n"
                f"omega = {self.omega:.3f} rad/s\n"
                f"L = {L:.2f} m\n"
                f"m = {m:.2f} kg\n"
                f"M = {M:.2f} kg\n"
                f"b_pivot = {b_pivot:.3f}\n"
                f"c_air = {c_air:.3f}"
            )
        )

    def animate(self):
        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time

        dt = min(dt, 0.03)

        self.update_physics(dt)
        self.draw()

        self.canvas.after(16, self.animate)


root = tk.Tk()
app = Pendulum(root)
root.mainloop()