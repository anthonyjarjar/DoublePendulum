import math

g = 9.81
L = 2.0

theta = math.radians(45)
omega = 0.0

t = 0.0
history = []


def theta_double_dot(theta, omega):
    return -(g / L) * math.sin(theta)


def derivatives(theta, omega):
    theta_dot = omega
    omega_dot = theta_double_dot(theta, omega)

    return theta_dot, omega_dot


def rk4_update(theta, omega, dt):
    theta0 = theta
    omega0 = omega

    k1_theta, k1_omega = derivatives(theta0, omega0)

    k2_theta, k2_omega = derivatives(
        theta0 + 0.5 * dt * k1_theta,
        omega0 + 0.5 * dt * k1_omega
    )

    k3_theta, k3_omega = derivatives(
        theta0 + 0.5 * dt * k2_theta,
        omega0 + 0.5 * dt * k2_omega
    )

    k4_theta, k4_omega = derivatives(
        theta0 + dt * k3_theta,
        omega0 + dt * k3_omega
    )

    theta_new = theta + (dt / 6) * (
        k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta
    )

    omega_new = omega + (dt / 6) * (
        k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega
    )

    alpha = theta_double_dot(theta_new, omega_new)

    return theta_new, omega_new, alpha


dt = 0.01
total_time = 10.0
steps = int(total_time / dt)

for _ in range(steps):
    theta, omega, alpha = rk4_update(theta, omega, dt)
    t += dt

    history.append([
        t,
        theta,
        omega,
        alpha
    ])

print(history[-1])