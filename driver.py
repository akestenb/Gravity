import numpy as np
from scipy.integrate import odeint
from typing import Any
import matplotlib.pyplot as plt


# establish array as datatype
NDArray = np.ndarray[Any, np.dtype[np.float64]]

G = 6.67 * 10 ** (-11)

samples_per_time = 128
time_steps = 100000 * 1
t = np.linspace(0, time_steps, time_steps * samples_per_time)

def one_body(IVP, t, G, M): 
    
    x, y, vx, vy = IVP

    r = np.sqrt(x**2 + y**2)

    ax = -G * M * x / r**3
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]

def two_body(IVP, t, G, M, m1, m2):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP

    r1 = np.sqrt(x1**2 + y1**2)  # distance of 1st body to sun
    r2 = np.sqrt(x2**2 + y2**2)  # distance of 2nd body to sun
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)  # distance between 1st and 2nd body

    ax1 = (-G * M * x1 / r1**3) + (-G * m2 * (x2 - x1) / d**3)
    ay1 = (-G * M * y1 / r1**3) + (-G * m2 * (y2 - y1) / d**3)
    ax2 = (-G * M * x2 / r2**3) + (-G * m1 * (x1 - x2) / d**3)
    ay2 = (-G * M * y2 / r2**3) + (-G * m1 * (y1 - y2) / d**3)

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

def orbital_elements(x, y, vx, vy, G, M):
    r = np.sqrt(x**2 + y**2)
    v = np.sqrt(vx**2 + vy**2)
    
    # Specific angular momentum (scalar, since 2D)
    h = x * vy - y * vx
    
    # Specific mechanical energy
    energy = 0.5 * v**2 - G * M / r
    
    # Semi-major axis
    a = -G * M / (2 * energy)
    
    # Eccentricity vector magnitude
    e = np.sqrt(1 - (h**2) / (G * M * a))
    return a, e
## driver 

M = 333000
x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = [1, 0, 0, np.sqrt(G*M/1), 2, 0, 0, np.sqrt(G*M/2), M, 1, 1]

two_body_solution = odeint(two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
one_body_solution_mass1 = odeint(one_body, [x1, y1, vx1, vy1], t, args=(G, m0))
one_body_solution_mass2 = odeint(one_body, [x2, y2, vx2, vy2], t, args=(G, m0))


## Deviations in Body 1 Orbit as a Result of Body 2 

# Extract Body 1 positions from both simulations
x1_two = two_body_solution[:, 0]
y1_two = two_body_solution[:, 1]

x1_one = one_body_solution_mass1[:, 0]
y1_one = one_body_solution_mass1[:, 1]

# Calculate displacement (distance) between trajectories over time
dx = x1_two - x1_one
dy = y1_two - y1_one
displacement = np.sqrt(dx**2 + dy**2)

# Plot the deviation over time
plt.figure(figsize=(10, 5))
plt.plot(t, displacement)
plt.xlabel("Time")
plt.ylabel("Deviation of Body 1 Position")
plt.title("Deviation of Body 1 due to Presence of Body 2")
plt.grid(True)
plt.show()


## Orbital Element Comparison ## (how much does the shape of the orbit change over time)

# Extract positions and velocities for Body 1 from two-body simulation
x1 = two_body_solution[:, 0]
y1 = two_body_solution[:, 1]
vx1 = two_body_solution[:, 2]
vy1 = two_body_solution[:, 3]

# Compute eccentricity over time
a_vals = np.zeros_like(t)
e_vals = np.zeros_like(t)
for i in range(len(t)):
    a_vals[i], e_vals[i] = orbital_elements(x1[i], y1[i], vx1[i], vy1[i], G, m0)

# Plot eccentricity over time
plt.figure(figsize=(10,5))
plt.plot(t, e_vals)
plt.xlabel("Time")
plt.ylabel("Eccentricity of Body 1")
plt.title("Eccentricity of Body 1 over Time (Two-Body)")
plt.grid(True)
plt.show()

## Energy Analysis ## (checks the conservation of energy, the pertubations should only cause small oscialltions)
m1_val = m1  # mass of Body 1

# Velocity magnitude for Body 1 (two-body)
v1 = np.sqrt(vx1**2 + vy1**2)

# Distance from central mass
r1 = np.sqrt(x1**2 + y1**2)

# Kinetic energy
KE1 = 0.5 * m1_val * v1**2

# Potential energy (only central mass)
PE1 = -G * m0 * m1_val / r1

#  Distance from Body 2
x2 = two_body_solution[:, 4]
y2 = two_body_solution[:, 5]
r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

PE_perturb = -G * m1_val * m2 / r12
PE1_total = PE1 + PE_perturb

# Total energy
E1 = KE1 + PE1_total

plt.figure(figsize=(10,5))
plt.plot(t, E1)
plt.xlabel("Time")
plt.ylabel("Total Energy of Body 1")
plt.title("Total Energy of Body 1 over Time (Two-Body)")
plt.grid(True)
plt.show()


## Orbital Angle ## (checks for precession, is the orbit rotating over time)

theta = np.arctan2(y1, x1)
theta_unwrapped = np.unwrap(theta)

plt.figure(figsize=(10,5))
plt.plot(t, theta_unwrapped)
plt.xlabel("Time")
plt.ylabel("Orbital Angle (radians)")
plt.title("Orbital Angle of Body 1 Over Time (Two-Body)")
plt.grid(True)
plt.show()