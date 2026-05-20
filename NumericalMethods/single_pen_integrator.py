import math

g = 9.81
L = 2.0

theta = math.radians(45)
omega = 0.0

t = 0.0
history = []


def theta_double_dot(theta, omega):
    return -(g / L) * math.sin(theta)


def euler_update(theta, omega, dt):
    alpha = theta_double_dot(theta, omega)

    omega_new = omega + alpha * dt
    theta_new = theta + omega_new * dt

    return theta_new, omega_new, alpha


dt = 0.01
total_time = 10.0
steps = int(total_time / dt)

for _ in range(steps):
    theta, omega, alpha = euler_update(theta, omega, dt)
    t += dt

    history.append([
        t,
        theta,
        omega,
        alpha
    ])

print(history[-1])