import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.signal import find_peaks


# Constants
G = 6.67430e-11 # gravitiaional constant in m^3 kg^-1 s^-2

# Sympletic integrator coefficents (Yoshida 4th order)
# note: the coefficients control the integration steps to preserve physical properties
w0 = -np.power(2, 1/3) / (2 - np.power(2,1/3))
w1 = 1 / (2 - np.power(2, 1/3))
c = [w1 / 2, (w0 + w1) / 2, (w0 + w1) / 2, w1 / 2] # position update weights 
d = [w1, w0, w1] # velocity update weights 

# Function to compute accelerations on two orbiting bodies affected by central mass and each other
def compute_acceleration_two_orbiting_bodies(x1,y1,x2,y2,M,m1,m2):
    
    r1 = np.sqrt(x1**2 + y1**2) # distance between m1 and M
    r2 = np.sqrt(x2**2 + y2**2) # distance between m2 and M
    d= np.sqrt((x2 - x1)**2 + (y2 - y1)**2) # distance between m1 and m2

    # Compute acceleration on mass 1 (e.g., Earth):
    # First term: attraction to central mass (Sun)
    # Second term: gravitational pull from mass 2 (e.g., Mars)
    ax1 = -G * M * x1 / r1**3 + (G * m2 * (x2 - x1) / d**3) # acceleration in x direction on mass 1
    ay1 = -G * M * y1 / r1**3 + (G * m2 * (y2 - y1) / d**3) # acceleration in y direction on mass 1
   
    # Compute acceleration on mass 1 (e.g., Mars):
    ax2 = -G * M * x2 / r2**3 + (G * m1 * (x1 - x2) / d**3) # acceleration in x direction on mass 2
    ay2 = -G * M * y2 / r2**3 + (G * m1 * (y1 - y2) / d**3) # acceleration in y direction on mass 2

    return ax1, ay1, ax2, ay2

# Function to compute acceleration on a single orbiting body affected only by the central mass
def compute_acceleration_one_orbiting_body(x1,y1,M,m):
    
    r = np.sqrt(x1**2 + y1**2) # distance between m1 and M
  
    ax = -G * M * x1 / r**3 # acceleration in x direction on mass 1
    ay = -G * M * y1 / r**3  # acceleration in y direction on mass 1
   
    return ax, ay


# Symplectic integrator for two orbiting bodies under mutual and central gravitational forces
def symplectic_integrate_two_body(IVP, dt, steps, M, m1, m2):

    # Unpack initial conditions: positions and velocities of both bodies
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP # initial value conditions 
   
    solution = np.zeros((steps,8)) # initialze array of solutions, this should be the size of the intial conditons by the number of time steps 
    solution[0] = IVP # the first time step soltuion is set to the initial values

    for i in range (1,steps):
        # First partial position update with coefficient c[0]
        x1 += c[0] * dt * vx1
        y1 += c[0] * dt * vy1
        x2 += c[0] * dt * vx2
        y2 += c[0] * dt * vy2

        # Loop through the three substeps of the 4th order symplectic integration
        for j in range(3): #  there are 3 substeps for 4th order integration

            # Compute accelerations on both bodies due to gravitational forces
            ax1, ay1, ax2, ay2 = compute_acceleration_two_orbiting_bodies(x1, y1, x2, y2, M, m1, m2)

            # Update velocities using the computed accelerations weighted by d[j]
            vx1 += d[j] * dt * ax1
            vy1 += d[j] * dt * ay1
            vx2 += d[j] * dt * ax2
            vy2 += d[j] * dt * ay2

            # Update positions again with updated velocities weighted by c[j+1]
            x1 += c[j + 1] * dt * vx1
            y1 += c[j + 1] * dt * vy1
            x2 += c[j + 1] * dt * vx2
            y2 += c[j + 1] * dt * vy2

       # Save the positions and velocities at this timestep
        solution[i] = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]

    return solution

# Symplectic integrator for a single orbiting body around a central mass
def symplectic_integrate_one_body(IVP, dt, steps, M, m):

    # Unpack initial position and velocity
    x, y, vx, vy = IVP # initial value conditions 

    solution = np.zeros((steps,4)) # initialze array of solutions, this should be the size of the intial conditons by the number of time steps 
    solution[0] = IVP # the first time step soltuion is set to the initial values

    for i in range (1,steps):

        # initial position update with c[0], first partial position update
        x += c[0] * dt * vx
        y += c[0] * dt * vy 

        # Perform the three substeps of the integrator
        for j in range(3): #  there are 3 substeps for 4th order integration

             # Calculate acceleration due to central mass gravity           
            ax, ay = compute_acceleration_one_orbiting_body(x, y,  M, m)

            # Update velocities
            vx += d[j] * dt * ax
            vy += d[j] * dt * ay
            
            # Update positions
            x += c[j + 1] * dt * vx
            y += c[j + 1] * dt * vy
           
        # stoe the solution at the current time step
        solution[i] = [x, y, vx, vy]

    return solution


# -------------------- Example Usage ----------------------------

# Masses in kilograms 
M = 1.989e30 # mass of the central body, this position is assumed to be (0,0) (Sun)
m1 = 5.972e24 # mass of secondary body 1 (Earth) 5.972e24
m2 = 6.39e23  # mass of secondary body 2 (Mars) 6.39e23 


# Initial positions (meters) and velocities (meters per second)
x1 = 1.5e11  # Earth initial x-position (~1 AU)
y1 = 0
r1 = np.sqrt(x1**2 + y1 **2)
vx1 = 0 # velocity is purley tangential
vy1 = np.sqrt(G * M/ x1) # velocity is set to ensure intially ciruclar motion

x2 = 2.28e11 # Mars initial x-position (~1.5 AU) 2.28e11
y2 = 0
r2 = np.sqrt(x2**2 + y2 **2)
vx2 = 0 # velocity is purley tangential
vy2 = np.sqrt(G* M/ x2) # velocity is set to ensure intially ciruclar motion

# Orbital Period 
T_earth_theoretical = np.sqrt((4*np.pi**2 * r1 **3) / (G * M)) /  3.154e+7
T_mars_theoretical = np.sqrt((4*np.pi**2 * r2 **3) / (G * M)) /  3.154e+7

print(f"Theoretical Orbital Period Earth {T_earth_theoretical:.4f}")
print(f"Theoretical Orbital Period Mars {T_mars_theoretical:.4f}")

# Combine initial conditions into arrays for integrators
IVP_2body= [x1, y1, vx1, vy1, x2, y2, vx2, vy2 ] # set Two- body intial conditions
IVP_Earth= [x1, y1, vx1, vy1] # set One-body Earth intial conditions
IVP_Mars = [x2, y2, vx2, vy2 ] # set One-body Mars intial conditions

# Time
dt = (60 ** 2)*24  # time step value (duration of each time step in seconds), initall set to 1 day
total_time = 300 # in years 
total_time_seconds = total_time * 31556952
steps = int(total_time_seconds / dt)

# Run the simulations
sol_2body = symplectic_integrate_two_body(IVP_2body, dt, steps, M, m1, m2)
sol_Earth = symplectic_integrate_one_body(IVP_Earth, dt, steps, M, m1)
sol_Mars = symplectic_integrate_one_body(IVP_Mars, dt, steps, M, m2)

# Plot Orbits (includes graviaitional relationship between the two secondary bodies)

# Extract Positions for plotting
x1s, y1s, vx1s, vy1s = sol_2body[:, 0], sol_2body[:, 1], sol_2body[:, 2], sol_2body[:, 3] # two body Earth 
x2s, y2s, vx2s, vy2s = sol_2body[:, 4], sol_2body[:, 5], sol_2body[:, 6], sol_2body[:, 7] # two body Mars

xE, yE, vxE, vyE = sol_Earth[:,0], sol_Earth[:,1], sol_Earth[:,2], sol_Earth[:,3] # one body Earth
xM, yM, vxM, vyM = sol_Mars[:,0], sol_Mars[:,1], sol_Mars[:,2], sol_Mars[:,3]  # one body Mars

# Plot Orbital Divergence
diff_Earth = np.hypot(x1s - xE, y1s - yE)
diff_Mars =  np.hypot(x2s - xM, y2s - yM)

t = np.arange(steps) * dt / (60*60*24*365.25)   # years for the x-axis

# Line up the data
slopeE, interceptE = np.polyfit(t,diff_Earth,1)
best_fit_lineE = slopeE * t + interceptE

slopeM, interceptM= np.polyfit(t,diff_Mars,1)
best_fit_lineM = slopeM * t + interceptM

adjustedE = diff_Earth - best_fit_lineE
adjustedM = diff_Mars - best_fit_lineM

vxE, vyE = vx1s, vy1s

dx = x2s - x1s 
dy = y2s - y1s

# Dot and cross products
dot = vxE * dx + vyE * dy
cross = vxE * dy - vyE * dx  # scalar 2D cross product

# Angle in radians (signed)
theta_rad = np.arctan2(cross, dot)

# Convert to degrees
theta_deg = np.degrees(theta_rad)

# Find peaks in the oscillating signal
peaks, _ = find_peaks(adjustedE, distance=1)  # tweak distance based on your data

# Estimate period between peaks
peak_times = t[peaks]
T_syn =  2 * np.mean(np.diff(peak_times))
print(f"Estimated synodic period: {T_syn:.2f}")

# Calculate the Simulation Period of Earth
zero_crossings = (y1s[:-1] < 0) & (y1s[1:] >= 0)
x_positive = x1s[1:] > 0  # shift by 1 to align with y[1:]
valid_crossings = zero_crossings & x_positive
crossing_times = (t[:-1][valid_crossings] + t[1:][valid_crossings]) / 2
if len(crossing_times) >= 2:
    orbital_periods = np.diff(crossing_times)
    T_sim = np.mean(orbital_periods)
    print(f"Simulated Earth Orbital Period: {T_sim:.6f} years")
else:
    print("Not enough crossings to estimate orbital period.")

# Calculate the Simulation Period of Mars
zero_crossingsM = (y2s[:-1] < 0) & (y2s[1:] >= 0)
x_positiveM = x2s[1:] > 0  # shift by 1 to align with y[1:]
valid_crossingsM = zero_crossingsM & x_positiveM
crossing_timesM = (t[:-1][valid_crossingsM] + t[1:][valid_crossingsM]) / 2
if len(crossing_timesM) >= 2:
    orbital_periodsM = np.diff(crossing_timesM)
    T_simM = np.mean(orbital_periodsM)
    print(f"Simulated Mars Orbital Period: {T_simM:.6f} years")
else:
    print("Not enough crossings to estimate orbital period.")


# Calculate the Theoretical Mars Period 
T_M = 1 / (1/T_sim - 1/T_syn)
print(f"Calculated Mars Orbital Period: {T_M:.3f} years")


print(f"Difference Between Simulated and Calculated Mars Orbital Period: {abs(T_M - T_simM):.3f} years")







# INTERACTIVE PLOT
# ==== Create figure and 3 vertically stacked plots ====
fig, (ax_orbit, ax_deviation, ax_angle) = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
plt.subplots_adjust(bottom=0.25, hspace=0.4)

# === Top: Orbit Plot ===
ax_orbit.plot(x1s, y1s, label='Earth (2-body)', alpha=0.6)
ax_orbit.plot(x2s, y2s, label='Mars (2-body)', alpha=0.6)
ax_orbit.plot(0, 0, 'yo', label='Sun')
earth_marker, = ax_orbit.plot([], [], 'bo', markersize=8, label='Earth')
mars_marker, = ax_orbit.plot([], [], 'ro', markersize=8, label='Mars')
ax_orbit.set_aspect('equal')
max_range = max(np.max(np.abs(x1s)), np.max(np.abs(x2s)))
ax_orbit.set_xlim(-1.2 * max_range, 1.2 * max_range)
ax_orbit.set_ylim(-1.2 * max_range, 1.2 * max_range)
ax_orbit.set_xlabel("x position (m)")
ax_orbit.set_ylabel("y position (m)")
ax_orbit.set_title("Planetary Orbits")
ax_orbit.grid(True)
#ax_orbit.legend()

# === Middle: Orbital Deviation ===
dev_earth_line, = ax_deviation.plot(t, adjustedE, label='Earth deviation (adjusted)')
#dev_mars_line, = ax_deviation.plot(t, adjustedM, label='Mars deviation (adjusted)')
time_marker_dev = ax_deviation.axvline(0, color='k', linestyle='--', label='Current time')
ax_deviation.set_xlabel("Time (years)")
ax_deviation.set_ylabel("Position Difference (m)")
ax_deviation.set_title("Orbital Deviations from 1-body Orbits")
ax_deviation.grid(True)
#ax_deviation.legend()

# === Bottom: Angle Between Velocity and Mars Position ===
angle_line, = ax_angle.plot(t, theta_deg, label='Angle (deg)')
time_marker_angle = ax_angle.axvline(0, color='k', linestyle='--')
ax_angle.set_xlabel("Time (years)")
ax_angle.set_ylabel("Signed Angle (deg)")
ax_angle.set_title("Angle Between Earth's Velocity and Mars Position")
ax_angle.grid(True)
#ax_angle.legend()

# === Slider and TextBox ===
slider_ax = plt.axes([0.2, 0.12, 0.6, 0.03])
time_slider = Slider(slider_ax, 'Time (years)', 0, t[-1], valinit=0, valstep=0.01)

text_ax = plt.axes([0.83, 0.12, 0.1, 0.03])
time_text = TextBox(text_ax, '', initial="0.00")

# === Update Function ===
def update(val):
    idx = min(int(val / (t[1] - t[0])), len(x1s) - 1)

    # Update orbit markers
    earth_marker.set_data([x1s[idx]], [y1s[idx]])
    mars_marker.set_data([x2s[idx]], [y2s[idx]])

    # Update time markers on deviation and angle plots
    time_marker_dev.set_xdata([val])
    time_marker_angle.set_xdata([val])

    # Update text box
    time_text.set_val(f"{val:.2f}")

    fig.canvas.draw_idle()

# === TextBox Submit ===
def submit_text(text):
    try:
        val = float(text)
        val = max(0, min(val, t[-1]))
        time_slider.set_val(val)  # Triggers update
    except ValueError:
        pass

# Register callbacks
time_slider.on_changed(update)
time_text.on_submit(submit_text)

# Initialize plot
update(0)
plt.show()
