import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.ndimage import gaussian_filter1d
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
    accelerations = np.zeros((steps,4))
    accelerations[0] = [0,0,0,0]
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
        accelerations[i] = [ax1,ay1,ax2,ay2]

    return solution, accelerations

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

angleE = 0 * (np.pi/180)
rad1 = 1.5e11
x1 = rad1 * np.cos(angleE)  # Earth initial x-position (~1 AU)
y1 = rad1 * np.sin(angleE)

'''x1 = 149999210088.4012
y1 = 94454630.1145
rad1 = np.hypot(x1,y1)
angleE = np.arctan2(y1,x1)
'''
r1 = np.sqrt(x1**2 + y1 **2)
velE =  np.sqrt(abs(G * M/ r1))

vx1 = velE * -np.sin(angleE) # velocity is purley tangential
vy1 = velE * np.cos(angleE) # velocity is set to ensure intially ciruclar motion


angleM = 52 * (np.pi/180)
rad2 = 2.28e11

x2 = rad2*np.cos(angleM)
y2 = rad2*np.sin(angleM)

'''x2 = 141982228916.6104
y2 =  178395210169.4843
rad2 = np.hypot(x2,y2)

angleM = 51 * np.arctan2(y2,x2)
'''
r2 = np.sqrt(x2**2 + y2 **2)
velM = np.sqrt(abs(G * M/rad2))

vx2 =velM * -np.sin(angleM) # velocity is purley tangential
vy2 = velM * np.cos(angleM) # velocity is set to ensure intially ciruclar motion

# Orbital Period 
T_earth_theoretical = np.sqrt((4*np.pi**2 * r1 **3) / (G * M)) /  3.154e+7
T_mars_theoretical = np.sqrt((4*np.pi**2 * r2 **3) / (G * M)) /  3.154e+7

# Combine initial conditions into arrays for integrators
IVP_2body= [x1, y1, vx1, vy1, x2, y2, vx2, vy2 ] # set Two- body intial conditions
IVP_Earth= [x1, y1, vx1, vy1] # set One-body Earth intial conditions
IVP_Mars = [x2, y2, vx2, vy2 ] # set One-body Mars intial conditions

# Time
dt = (60 ** 2) * 12 # time step value (duration of each time step in seconds), initall set to 1 day
total_time = 600 # in years 
total_time_seconds = total_time * 31556952
steps = int(total_time_seconds / dt)

# Run the simulations
sol_2body, acc_2body = symplectic_integrate_two_body(IVP_2body, dt, steps, M, m1, m2)
sol_Earth = symplectic_integrate_one_body(IVP_Earth, dt, steps, M, m1)
sol_Mars = symplectic_integrate_one_body(IVP_Mars, dt, steps, M, m2)

# Plot Orbits (includes graviaitional relationship between the two secondary bodies)

# Extract Positions for plotting
x1s, y1s, vx1s, vy1s = sol_2body[:, 0], sol_2body[:, 1], sol_2body[:, 2], sol_2body[:, 3] # two body Earth 
x2s, y2s, vx2s, vy2s = sol_2body[:, 4], sol_2body[:, 5], sol_2body[:, 6], sol_2body[:, 7] # two body Mars
ax1, ay1, ax2, ay2 = acc_2body[:,0],acc_2body[:,1],acc_2body[:,2],acc_2body[:,3]


xE, yE, vxE, vyE = sol_Earth[:,0], sol_Earth[:,1], sol_Earth[:,2], sol_Earth[:,3] # one body Earth
xM, yM, vxM, vyM = sol_Mars[:,0], sol_Mars[:,1], sol_Mars[:,2], sol_Mars[:,3]  # one body Mars


# Plot Orbital Divergence
diff_Earth = np.hypot(x1s - xE, y1s - yE)
diff_Mars =  np.hypot(x2s - xM, y2s - yM)

t = np.arange(steps) * dt / (60*60*24*365.25)   # years for the x-axis

plt.figure(figsize=(10, 4))
plt.plot(t, np.hypot(ax1,ay1), label = 'Earth')
plt.plot(t, np.hypot(ax2,ay2), label = 'Mars')
plt.xlabel('Time (years)')
plt.ylabel('Acceleration Magnitude (m/s²)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


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
dot = (vxE * dx) + (vyE * dy)
cross =(vxE * dy) - (vyE * dx)  # scalar 2D cross product

# Angle in radians (signed)
theta_rad = np.arctan2(cross, dot)

# Convert to degrees
theta_deg = np.degrees(theta_rad)

angle_ahead = np.rad2deg(np.arctan2(y2s,x2s) - np.arctan2(y1s,x1s))



'''for k in range(len(adjustedE)):
    if angle_ahead[k] > 51 and angle_ahead[k] < 52 and x1s[k] > 0 and y1s[k] < 100000000 and y1s[k] > -100000000 :
        earth_pos_x= x1s[k]
        earth_pos_y = y1s[k]
        mars_pos_x = x2s[k]
        mars_pos_y = y2s[k]
        stop_time = t[k]
        print(f"Earth Postion = {earth_pos_x:.4f}, {earth_pos_y:.4f}, Mars Postion = {mars_pos_x:.4f}, {mars_pos_y:.4f}, Stop time = {stop_time:.4f}" )
        break'''
    

# Cycle Time 
yf = rfft(adjustedE)
xf = rfftfreq(len(t),d =(t[1]-t[0]))
peaks_fft, _ = find_peaks(np.abs(yf))
cuttoff = 1
peak_freq_fft = xf[peaks_fft]
peak_freq_fft_low = []
peaks_fft_low = []

for j in range(len(peak_freq_fft)):
    if peak_freq_fft[j]< cuttoff:
        peak_freq_fft_low.append(peak_freq_fft[j])
        peaks_fft_low.append(peaks_fft[j])

for i in range (len(peak_freq_fft_low)):
    print(f"Freqeuncy = {peak_freq_fft_low[i]:.4f} Hz,  Amplitude = {np.abs(yf[peaks_fft_low[i]]):.4f}, Period = {1/peak_freq_fft_low[i]:.4f} years")

plt.figure(figsize=(10, 4))
plt.plot(xf,np.abs(yf))
plt.grid(True)
#plt.legend()
plt.tight_layout()
plt.show()




# --- Find valleys (local minima) ---
maxDeviationE = max(adjustedE)
minDeviationE = min(adjustedE)
maxDeviationM = max(adjustedM)

peaks, _ = find_peaks(adjustedE)
valleys, _ = find_peaks(-adjustedE)

sig_peaks1 = []

# We start from 1 and end at len(peaks) - 1 to avoid out-of-bounds
for i in range(1, len(peaks) - 1):
        prev_peak = peaks[i - 1]
        curr_peak = peaks[i]
        next_peak = peaks[i + 1]
        
        if adjustedE[curr_peak] > adjustedE[prev_peak] and adjustedE[curr_peak] > adjustedE[next_peak]:
         sig_peaks1.append(curr_peak)

sig_peaks2 = []

for j in range(1,len(sig_peaks1) - 1) :
    prev_peak = sig_peaks1[j -1]
    curr_peak = sig_peaks1[j]
    next_peak = sig_peaks1[j+1]

    if adjustedE[curr_peak] > adjustedE[prev_peak] and adjustedE[curr_peak] > adjustedE[next_peak]:
        sig_peaks2.append(curr_peak)

sig_peaks3 = []

for k in range(1,len(sig_peaks2) - 1) :
    prev_peak = sig_peaks2[k -1]
    curr_peak = sig_peaks2[k]
    next_peak = sig_peaks2[k+1]

    if adjustedE[curr_peak] > adjustedE[prev_peak] and adjustedE[curr_peak] > adjustedE[next_peak]:
        sig_peaks3.append(curr_peak)

sig_peaks = sig_peaks2



'''sigma = 100
smoothedE = gaussian_filter1d(adjustedE, sigma = sigma)

peaks, _ = find_peaks(smoothedE, distance = 500)
peak_times = t[peaks]

cycle_times = np.diff(peak_times)
mean_cycle_time = np.mean(cycle_times)

print(f"Estimated synodic cycle time: {mean_cycle_time:.2f} years")

# Plot the result
plt.figure(figsize=(12,5))
plt.plot(t, adjustedE, label='Original Earth Deviation', alpha=0.5)
plt.plot(t, smoothedE, label='Gaussian Smoothed', linewidth=2)
plt.scatter(peak_times, smoothedE[peaks], color='red', label='Detected Peaks')
plt.xlabel('Time (years)')
plt.ylabel('Deviation (m)')
plt.legend()
plt.show()
'''








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
ax_orbit.set_title(f"Planetary Orbits E: {angleE * 180/np.pi :2f} M: {angleM * 180/np.pi :2f}")
ax_orbit.grid(True)
#ax_orbit.legend()

# === Middle: Orbital Deviation ===
dev_earth_line, = ax_deviation.plot(t, adjustedE, label='Earth deviation (adjusted)')
#dev_mars_line, = ax_deviation.plot(t, adjustedM, label='Mars deviation (adjusted)')
time_marker_dev = ax_deviation.axvline(0, color='k', linestyle='--', label='Current time')
ax_deviation.set_xlabel("Time (years)")
ax_deviation.set_ylabel("Position Difference (m)")
ax_deviation.set_title("Orbital Deviations from 1-body Orbits")


#ax_deviation.legend()

# --- Get times and angles at peaks and valleys ---
peak_times = t[sig_peaks]
##valley_times = t[sig_valleys]
peak_angles = theta_deg[sig_peaks]
#valley_angles = theta_deg[sig_valleys]

ax_deviation.plot(peak_times, adjustedE[sig_peaks], 'gx', label='Peaks', markersize=8, zorder=5)  # Green 'x' for peaks
#ax_deviation.plot(valley_times, adjustedE[sig_valleys], 'rx', label='Valleys', markersize=8, zorder=5)  # Red 'x' for valleys
ax_deviation.grid(True)

angle_scatter = ax_angle.scatter(peak_times, peak_angles, color='g', label='Angle @ Peaks')
#valley_scatter = ax_angle.scatter(valley_times, valley_angles, color='m', label='Angle @ Valleys')
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