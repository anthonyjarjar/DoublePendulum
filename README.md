# Self-Balancing Double Pendulum

## Project Overview

This project explores the modeling, simulation, reinforcement learning control, and hardware implementation of a self-balancing pendulum system. The work began with an idealized double pendulum simulation, then moved toward reinforcement learning control using PPO, and finally toward a physical single-pendulum balancing setup driven by an Arduino, a DRV8825 stepper driver, and encoder feedback.

The overall goal is to train a model that maps the pendulum state to motor actions so the cart can move in a way that keeps the pendulum balanced near its unstable upright equilibrium.

## Numerical Modeling

The double pendulum was modeled as two pendulums connected in tandem. The system was derived using Lagrangian mechanics, where

```text
L = T - U
```

The kinetic and potential energies were used to obtain the equations of motion through the Euler-Lagrange equation:

```text
d/dt(∂L/∂θ_dot_i) - ∂L/∂θ_i = 0
```

An initial idealized double pendulum simulation was implemented using the Euler method. This helped visualize the basic physics of the system, although the animation was less stable and less accurate than later numerical approaches.

## Simulation Methods

The project used multiple simulation approaches:

- Euler method for an early idealized double pendulum trajectory simulation
- RK4 numerical integration for improved single-pendulum simulation
- PPO reinforcement learning environments for both single-pendulum and double-pendulum balancing

The Euler method was useful for understanding the system, but RK4 was introduced because it gives a more accurate approximation by averaging slopes across each time step rather than assuming a constant slope.

## Reinforcement Learning

Reinforcement learning was used to train an agent to choose motor actions based on the state of the pendulum. The model acts as a map from the pendulum state to a stepper motor command.

The reward function was designed to encourage:

- Keeping the pendulum upright
- Reducing excessive angular velocity
- Reducing unnecessary motor effort
- Producing smoother motion

The model was trained using Proximal Policy Optimization, or PPO. After training, the policy acts deterministically by choosing the action that the trained model predicts will best stabilize the system.

## Testing Results

Both the single-pendulum and double-pendulum PPO models were tested in simulation.

The single-pendulum model successfully survived the full 1000 time steps for 10 out of 10 test runs.

The double-pendulum model also successfully survived the full 1000 time steps for 10 out of 10 test runs.

These tests showed that both trained models were able to maintain balance for the complete episode length under the simulation conditions used during evaluation.

## Computer Vision and Angle Measurement

Two angle-measurement approaches were explored:

### Optical Encoder

The optical encoder was intended to provide fast and reliable angle feedback. This was the preferred approach for hardware control because it can provide near-instantaneous measurements and integrates directly with the Arduino.

### Image Processing

A Python OpenCV-based image processor was also tested. It used HSV color thresholding to detect the pivot and bob markers. This approach was useful for experimentation, but it introduced delay and was less reliable than the encoder-based approach.

## Hardware Setup

The physical system was built around a cart moving along a V-slot aluminum rail. The pendulum arm is attached to the moving cart, and the cart is driven by a NEMA17 stepper motor through a GT2 timing belt and pulley system.

Main hardware components included:

- Arduino Mega 2560
- DRV8825 stepper motor driver
- NEMA17 stepper motor
- GT2 timing belt and pulley system
- V-slot aluminum rail
- V-wheels and 625RS bearings
- Pendulum arm
- HEDS-9140 optical encoder
- External 12V motor power supply
- Pull-up resistors for encoder channels

## Arduino and Driver Wiring

The Arduino and DRV8825 were wired so that the Arduino could send step and direction signals to the motor driver.

Basic wiring:

```text
Arduino D2 -> DRV8825 STEP
Arduino D3 -> DRV8825 DIR
Arduino D4 -> DRV8825 ENABLE
DRV8825 RESET tied to SLEEP
DRV8825 SLEEP -> 5V
VMOT -> 12V+
DRV8825 GND -> 12V-
Arduino GND -> DRV8825 / power supply GND
```

The encoder was connected to the Arduino for angle feedback:

```text
Encoder Index -> D20
Encoder A     -> D18
Encoder B     -> D19
Encoder VCC   -> 5V
Encoder GND   -> GND
```

The encoder channels used pull-up resistors to 5V.

## Physical Assembly

The prototype consisted of:

- A carriage moving along the rail using V-wheels
- A belt-driven cart powered by the stepper motor
- A pendulum arm mounted to the moving cart
- An Arduino and DRV8825 driver circuit
- An optical encoder mounted at the pivot for feedback

The physical system demonstrated that the cart and motor system could move the pendulum, although the final self-balancing hardware behavior was limited by encoder and hardware issues.

## What Worked

- Motor control
- Arduino communication
- Stepper driver setup
- Basic cart movement
- 3D-printed parts
- Single-pendulum hardware motion
- PPO simulation for single and double pendulum balancing

## What Did Not Work Fully

- Image processing was delayed and not reliable enough for final balancing
- The optical encoder had issues during hardware testing
- Some hardware parts lacked consistency
- The physical self-balancing system needed additional tuning and better encoder reliability

## Future Improvements

Future work includes:

- Replacing faulty hardware
- Improving encoder mounting and alignment
- Acquiring a code-wheel alignment tool
- Further tuning the motor and belt system
- Improving the smoothness of mechanical parts
- Refining the physical balancing controller
- Implementing a reliable swing-up motion into the unstable upright equilibrium

## Dependencies

The Python portions of this project use:

```text
numpy
opencv-python
gymnasium
stable-baselines3
pyserial
```

Install them with:

```bash
pip install -r requirements.txt
```

## References

- J. Murad, *The double pendulum: Equations of motion & Lagrangian mechanics*, Engineered Mind.
- D. Baden, *Double pendulum*, University of Maryland Department of Physics, 2016.
- E. Neumann, *DoublePendulumApp.ts*, myPhysicsLab GitHub source code, 2016.
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, *Proximal Policy Optimization Algorithms*, arXiv, 2017.
