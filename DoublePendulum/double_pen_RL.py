import math
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class BeltStepperDoublePendulumEnv(gym.Env):
    """
    Upright-balancing environment for a belt + stepper driven DOUBLE pendulum.

    Angles:
        theta1 = 0 means first pendulum link is upright.
        theta2 = 0 means second pendulum link is upright.

    Observation:
        [
            cos(theta1), sin(theta1), theta1_dot,
            cos(theta2), sin(theta2), theta2_dot,
            actual_step_rate
        ]

    Action:
        Discrete stepper speed command.

        action 0 -> fast left
        action 1 -> medium left
        action 2 -> slow left
        action 3 -> stop
        action 4 -> slow right
        action 5 -> medium right
        action 6 -> fast right

    Physics:
        Double pendulum EOM with horizontal base acceleration.
        No friction.
        No air resistance.
        Euler method integration.
    """

    def __init__(self):
        super().__init__()

        self.dt = 0.005
        self.max_steps = 1000

        # -----------------------------
        # Double pendulum parameters
        # -----------------------------
        self.g = 9.81

        self.l1 = 0.35
        self.l2 = 0.35

        self.m1 = 0.10
        self.m2 = 0.10

        # Terminate if either link falls too far from upright.
        self.theta_limit = math.radians(20)

        # -----------------------------
        # Stepper / belt model
        # -----------------------------
        self.step_rate_table = np.array(
            [-800, -400, -150, 0, 150, 400, 800],
            dtype=np.float32
        )

        # Stepper cannot instantly change speed.
        self.max_step_accel = 9000.0  # steps/s^2

        # Converts stepper speed into horizontal base acceleration.
        self.step_to_accel_gain = 0.0020

        # Delay: 1 step = 20 ms
        self.delay_steps = 1
        self.action_queue = deque([3] * self.delay_steps, maxlen=self.delay_steps)

        # -----------------------------
        # Gym spaces
        # -----------------------------
        self.action_space = spaces.Discrete(len(self.step_rate_table))

        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, -1.0, -np.inf,
                 -1.0, -1.0, -np.inf,
                 -np.inf],
                dtype=np.float32
            ),
            high=np.array(
                [1.0, 1.0, np.inf,
                 1.0, 1.0, np.inf,
                 np.inf],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # State:
        # theta1, theta1_dot, theta2, theta2_dot, actual_step_rate
        self.state = None
        self.steps = 0

    def angle_normalize(self, theta):
        return math.atan2(math.sin(theta), math.cos(theta))

    def get_obs(self):
        theta1, theta1_dot, theta2, theta2_dot, actual_step_rate = self.state

        theta1 = self.angle_normalize(theta1)
        theta2 = self.angle_normalize(theta2)

        return np.array(
            [
                math.cos(theta1),
                math.sin(theta1),
                theta1_dot,
                math.cos(theta2),
                math.sin(theta2),
                theta2_dot,
                actual_step_rate
            ],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start both links almost upright.
        theta1 = self.np_random.uniform(
            -math.radians(0.5),
            math.radians(0.5)
        )

        theta2 = self.np_random.uniform(
            -math.radians(0.5),
            math.radians(0.5)
        )

        theta1_dot = self.np_random.uniform(-0.01, 0.01)
        theta2_dot = self.np_random.uniform(-0.01, 0.01)

        actual_step_rate = 0.0

        self.state = np.array(
            [theta1, theta1_dot, theta2, theta2_dot, actual_step_rate],
            dtype=np.float64
        )

        self.steps = 0
        self.action_queue = deque([3] * self.delay_steps, maxlen=self.delay_steps)

        return self.get_obs(), {}

    def dynamics(self, state, action):
        theta1, theta1_dot, theta2, theta2_dot, actual_step_rate = state

        target_step_rate = float(self.step_rate_table[int(action)])

        # Limit how quickly the stepper speed can change.
        max_delta_rate = self.max_step_accel * self.dt
        rate_error = target_step_rate - actual_step_rate

        rate_change = np.clip(
            rate_error,
            -max_delta_rate,
            max_delta_rate
        )

        new_step_rate = actual_step_rate + rate_change

        # Horizontal base acceleration from the belt/stepper.
        base_accel = self.step_to_accel_gain * new_step_rate

        # -----------------------------
        # Double pendulum equations
        # -----------------------------
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g

        delta = theta1 - theta2

        # Mass matrix
        M11 = (m1 + m2) * l1**2
        M12 = m2 * l1 * l2 * math.cos(delta)
        M21 = M12
        M22 = m2 * l2**2

        M = np.array(
            [
                [M11, M12],
                [M21, M22]
            ],
            dtype=np.float64
        )

        # Right-hand side for upright-angle double pendulum with moving base.
        rhs1 = (
            (m1 + m2) * g * l1 * math.sin(theta1)
            - m2 * l1 * l2 * math.sin(delta) * theta2_dot**2
            - (m1 + m2) * base_accel * l1 * math.cos(theta1)
        )

        rhs2 = (
            m2 * g * l2 * math.sin(theta2)
            + m2 * l1 * l2 * math.sin(delta) * theta1_dot**2
            - m2 * base_accel * l2 * math.cos(theta2)
        )

        rhs = np.array([rhs1, rhs2], dtype=np.float64)

        theta1_ddot, theta2_ddot = np.linalg.solve(M, rhs)

        step_rate_dot = rate_change / self.dt

        return np.array(
            [
                theta1_dot,
                theta1_ddot,
                theta2_dot,
                theta2_ddot,
                step_rate_dot
            ],
            dtype=np.float64
        )

    def euler_update(self, state, action):
        dt = self.dt

        deriv = self.dynamics(state, action)

        new_state = state + dt * deriv

        new_state[0] = self.angle_normalize(new_state[0])
        new_state[2] = self.angle_normalize(new_state[2])

        max_rate = float(np.max(np.abs(self.step_rate_table)))
        new_state[4] = np.clip(new_state[4], -max_rate, max_rate)

        return new_state

    def step(self, action):
        self.action_queue.append(int(action))
        delayed_action = self.action_queue[0]

        self.state = self.euler_update(self.state, delayed_action)
        self.steps += 1

        theta1, theta1_dot, theta2, theta2_dot, actual_step_rate = self.state

        # -----------------------------
        # Reward
        # -----------------------------
        reward = 0.0

        # Main reward: keep both links upright.
        reward += 6.0 * math.cos(theta1)
        reward += 6.0 * math.cos(theta2)

        # Penalize angular velocity.
        reward -= 0.20 * theta1_dot**2
        reward -= 0.20 * theta2_dot**2

        # Penalize motor effort.
        reward -= 0.0000005 * actual_step_rate**2

        # Bonus if both links are close to upright.
        if abs(theta1) < math.radians(5) and abs(theta2) < math.radians(5):
            reward += 3.0

        if abs(theta1) < math.radians(2) and abs(theta2) < math.radians(2):
            reward += 5.0

        if abs(theta1) < math.radians(1) and abs(theta2) < math.radians(1):
            reward += 5.0

        # Penalize near failure.
        if abs(theta1) > math.radians(10) or abs(theta2) > math.radians(10):
            reward -= 4.0

        if abs(theta1) > math.radians(15) or abs(theta2) > math.radians(15):
            reward -= 8.0

        # -----------------------------
        # Termination
        # -----------------------------
        terminated = False

        if abs(theta1) > self.theta_limit or abs(theta2) > self.theta_limit:
            terminated = True
            reward -= 40.0

        truncated = self.steps >= self.max_steps

        return self.get_obs(), reward, terminated, truncated, {}



class TrainingPrinter(BaseCallback):
    def __init__(self, print_every=10_000):
        super().__init__()
        self.print_every = print_every
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0
        self.best_episode_reward = -float("inf")
        self.best_episode_length = 0

    def _on_step(self):
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self.episode_reward += reward
        self.episode_length += 1

        if done:
            self.episode_count += 1

            if self.episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.episode_reward
                self.best_episode_length = self.episode_length

            self.episode_reward = 0.0
            self.episode_length = 0

        if self.num_timesteps % self.print_every == 0:
            print(
                f"timesteps={self.num_timesteps}, "
                f"episodes={self.episode_count}, "
                f"best_reward={self.best_episode_reward:.2f}, "
                f"best_length={self.best_episode_length}"
            )

        return True



def train():
    print("Creating upright belt-stepper DOUBLE pendulum environment...")
    env = BeltStepperDoublePendulumEnv()

    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=[256, 256]
        )
    )

    print("Training started...")
    callback = TrainingPrinter(print_every=10_000)

    model.learn(
        total_timesteps=2_000_000,
        callback=callback
    )

    print("Training finished.")

    model.save("belt_stepper_double_pendulum_rl")

    print("Saved model as belt_stepper_double_pendulum_rl.zip")



if __name__ == "__main__":
    train()