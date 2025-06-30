import numpy as np 
import matplotlib.pyplot as plt
import scipy as sci
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

# Helper function to calcualte the energy of the two body problem
def energy_two_body(x1, y1, vx1, vy1, x2, y2, vx2, vy2, M, m1, m2, G=G):
    
    # Distances 
    r1 = np.hypot(x1, y1)
    r2 = np.hypot(x2, y2)
    d12 = np.hypot(x2 - x1, y2 - y1)

    # Kinetic Energy
    KE = (0.5 * m1 * (vx1**2 + vy1**2)) + (0.5 * m2 * (vx2**2 + vy2**2))

    # Potential Energy
    PE = (-(G * M * m1) / r1 )- ((G * M * m2) / r2) - ((G * m1 * m2) / d12)

    # Return total energy
    return KE + PE


# Helper function to calculate the energy of the one body problem 
def energy_one_body(x, y, vx, vy, M, m, G=G):
        # Distance 
        r = np.hypot(x, y)
        
        # Kinetic Energy 
        KE = 0.5 * m * (vx**2 + vy**2)

        # Potential Energy 
        PE = -(G * M * m)/r

        # Total Energy
        return KE + PE

def angular_momentum_two_body(x1, y1, vx1, vy1, x2, y2, vx2, vy2, m1, m2):

    L1 = m1 * (x1 * vy1 - y1 * vx1) # angular momentum of body 1
    L2 = m2 * (x2 * vy2 - y2 * vx2) # angular momentum of body 2 

    return L1 + L2


# -------------------- Example Usage ----------------------------

# Masses in kilograms 
M = 1.989e30 # mass of the central body, this position is assumed to be (0,0) (Sun)
m1 = 5.972e24 # mass of secondary body 1 (Earth)
m2 = 6.39e23 # mass of secondary body 2 (Mars)

print(f"Mass Ratio {m2/m1:.4f}")



# Initial positions (meters) and velocities (meters per second)
x1 = 1.5e11  # Earth initial x-position (~1 AU)
y1 = 0
vx1 = 0 # velocity is purley tangential
vy1 = np.sqrt(G * M/ x1) # velocity is set to ensure intially ciruclar motion

x2 = 2.28e11 # Mars initial x-position (~1.5 AU)
y2 = 0
vx2 = 0 # velocity is purley tangential
vy2 = np.sqrt(G* M/ x2) # velocity is set to ensure intially ciruclar motion
print(f"Orbital Radius Ratio {x2/x1:.4f}")

# Combine initial conditions into arrays for integrators
IVP_2body= [x1, y1, vx1, vy1, x2, y2, vx2, vy2 ] # set Two- body intial conditions
IVP_Earth= [x1, y1, vx1, vy1] # set One-body Earth intial conditions
IVP_Mars = [x2, y2, vx2, vy2 ] # set One-body Mars intial conditions

# Time
dt = (60 ** 2)*24  # time step value (duration of each time step in seconds), initall set to 1 day
total_time = 100 # in years 
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

# Two Body Energy Analysis
E_2body = energy_two_body(x1s, y1s, vx1s, vy1s, x2s, y2s, vx2s, vy2s, M, m1, m2)

# One Body Energy Analysis
E_Earth = energy_one_body(xE, yE, vxE, vyE, M, m1)
E_Mars = energy_one_body(xM, yM, vxM, vyM, M, m2)

# Angular Momentum 
L_2body = angular_momentum_two_body(x1s, y1s, vx1s, vy1s, x2s, y2s, vx2s, vy2s, m1, m2)

# Statistical Relationship
diff_Earth = np.hypot(x1s - xE, y1s - yE)
diff_Mars =  np.hypot(x2s - xM, y2s - yM)

correlation, _ = pearsonr(diff_Earth, diff_Mars)
print(f"Correlation between Earth and Mars deviations: {correlation:.4f}")

# Cross-correlation
cross_corr = correlate(diff_Earth - np.mean(diff_Earth), diff_Mars - np.mean(diff_Mars), mode='full')
lags = np.arange(-len(diff_Earth)+1, len(diff_Earth))
# ------------------------------ Plotting the Results --------------------------- 

'''# Plot the Two Body Solution with what the one body solution should be , one two seperate subplots 
plt.figure(figsize=(14, 6))

# --------- Subplot for Earth ---------
plt.subplot(1, 2, 1)
plt.plot(xE, yE, '--', label='Earth (1-body)', color='cyan')
plt.plot(x1s, y1s, label='Earth (2-body)', color='blue')
plt.scatter(0, 0, color='yellow', s=100, edgecolors='black', label='Sun')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Earth: 1-body vs 2-body Orbit")
plt.axis('equal')
plt.grid(True)
plt.legend()

# --------- Subplot for Mars ---------
plt.subplot(1, 2, 2)
plt.plot(xM, yM, '--', label='Mars (1-body)', color='salmon')
plt.plot(x2s, y2s, label='Mars (2-body)', color='orange')
plt.scatter(0, 0, color='yellow', s=100, edgecolors='black', label='Sun')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Mars: 1-body vs 2-body Orbit")
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()'''

'''# Plot Earh and Mars on the Same Graph
plt.figure(figsize=(8, 8))              # square aspect makes it easier to judge the shapes

# Earth
plt.plot(xE, yE,  '--',  label='Earth (1-body)', color='cyan')
plt.plot(x1s, y1s,      label='Earth (2-body)', color='blue')

# Mars
plt.plot(xM, yM,  '--',  label='Mars (1-body)',  color='salmon')
plt.plot(x2s, y2s,      label='Mars (2-body)',  color='orange')

# Sun at the origin
plt.scatter(0, 0, color='yellow', edgecolors='black', s=120, zorder=5, label='Sun')

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Earth & Mars: 1-body vs 2-body orbits (same axes)')
plt.axis('equal')
plt.grid(True)
#plt.legend()
plt.tight_layout()
plt.show()'''

# --------------------------- Result Validation ------------------------------------

#  Plot the Energy Error ( ~ e-14 ) 
t = np.arange(steps) * dt / (60*60*24*365.25)   # years for the x-axis

'''plt.figure(figsize=(10,4))
plt.plot(t, (E_2body - E_2body[0]) / abs(E_2body[0]), label='Earth+Mars (2-body)')
plt.xlabel('Time (years)')
plt.ylabel('Relative energy error ΔE / E₀')
plt.title('Energy conservation of 4th-order symplectic integrator')
plt.grid(True)
#plt.legend()
plt.tight_layout()
plt.show()

# Plot Angular Momentum
plt.figure(figsize=(10,4))
plt.plot(t, (L_2body - L_2body[0]) / abs(L_2body[0]), label='Angular momentum error')
plt.xlabel('Time (years)')
plt.ylabel('ΔL / L₀')
plt.title('Angular momentum conservation (2-body)')
plt.grid(True)
#plt.legend()
plt.tight_layout()
plt.show()'''


# 1. Compute best-fit lines first
slopeE, interceptE = np.polyfit(t, diff_Earth, 1)
best_fit_lineE = slopeE * t + interceptE

slopeM, interceptM = np.polyfit(t, diff_Mars, 1)
best_fit_lineM = slopeM * t + interceptM

# 2. Compute adjusted deviations
adjustedE = diff_Earth - best_fit_lineE
adjustedM = diff_Mars - best_fit_lineM

# 3. Find peaks and valleys in Earth's adjusted deviation
peak_indices, _ = find_peaks(adjustedE)
valley_indices, _ = find_peaks(-adjustedE)
all_extrema_indices = np.concatenate((peak_indices, valley_indices))
all_extrema_indices.sort()

# 5. Extract Earth and Mars 2-body positions at extrema
xE_pts = x1s[all_extrema_indices]
yE_pts = y1s[all_extrema_indices]
xM_pts = x2s[all_extrema_indices]
yM_pts = y2s[all_extrema_indices]

angles_rad = []
angles_deg = []

orientation_degs = []
orientation_rads = []


for i in all_extrema_indices:
    #solves for dot product of vectors over product of magnitude of vectors (inverse Cosine definition)
    # Earth's velocity vector
    #TESTS
    #90 deg
#     vE = np.array([1000000, 0])         
#     r_EM = np.array([0, 10000000000])
#random numbers i picked result was 11.31
    #vE = np.array([-1000, 5000])          #
    #r_EM = np.array([1000000, 10^11])
#0 deg
#     vE = np.array([1000, 0])
#     r_EM = np.array([1000000, 0])
#180 deg
#     vE = np.array([1000, 0])
#     r_EM = np.array([-1000000, 0]) 
    #ACTUAL DATA
    vE = np.array([vx1s[i], vy1s[i]])
    ## Vector from Earth to Mars
    r_EM = np.array([x2s[i] - x1s[i], y2s[i] - y1s[i]])
    # Normalize direction matters (angle is sensitive to direction)
    dot_product = np.dot(vE, r_EM) #dot product of vectors
    norm_product = np.linalg.norm(vE) * np.linalg.norm(r_EM) #norm is the function for magnitude in numpy

   # Orientation of Earth→Mars vector (If Earth is at (1 AU, 0) and Mars is at (0, 1.5 AU), the vector points straight up, and the orientation is 90°.)
    orientation_rad = np.arctan2(y2s[i]-y1s[i], x2s[i]-x1s[i])
    orientation_deg = np.degrees(orientation_rad)

    orientation_rads.append(orientation_rad)
    orientation_degs.append(orientation_deg)
    
    # Angle between vectors
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    angles_rad.append(angle_rad)
    angles_deg.append(angle_deg)

    print(f"Index: {i}")
    print(f"  Time (years): {t[i]:.4f}")
    print(f"  Earth Position: ({x1s[i]:.3e}, {y1s[i]:.3e})")
    print(f"  Mars Position:  ({x2s[i]:.3e}, {y2s[i]:.3e})")
    print(f"  Velocity Vector (Earth): {vE}")
    print(f"  Earth→Mars Vector:       {r_EM}")
    print(f"  Angle Between Vectors (degrees): {angle_deg:.4f}")
    print(f"  Orientation Between Earth and Mars (degrees): {orientation_deg:.4f}")
    print("-" * 80)

##########plotting the angles
# Time at each deviation point
t_extrema = t[all_extrema_indices]

plt.figure(figsize=(10, 5))
plt.scatter(t_extrema, angles_rad, marker='o', linestyle='-', color='purple')
plt.scatter(t_extrema, orientation_rads, marker='o', linestyle='-', color='blue')
plt.xlabel('Time (years)')
plt.ylabel('Angle (rad)')
plt.title('Angle between Earth velocity vector and Earth–Mars line at deviations')
plt.grid(True)
plt.tight_layout()
plt.show()
#########differentiate valleys and peaks
# Extract time and angle values for each group
t_peaks = t[peak_indices]
angles_peaks = np.array(angles_rad)[np.searchsorted(all_extrema_indices, peak_indices)]

t_valleys = t[valley_indices]
angles_valleys = np.array(angles_rad)[np.searchsorted(all_extrema_indices, valley_indices)]

# Plotting peaks and valleys separately
plt.figure(figsize=(10, 5))
plt.scatter(t_peaks, angles_peaks, color='red', marker='^', label='Peaks')
plt.scatter(t_valleys, angles_valleys, color='blue', marker='v', label='Valleys')

plt.xlabel('Time (years)')
plt.ylabel('Angle (rads)')
plt.title('Angle between Earth velocity vector and Earth–Mars line at extrema')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
