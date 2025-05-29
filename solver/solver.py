import numpy as np 
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11 # gravitiaional constant 

# Sympletic integrator coefficents (Yoshida 4th order)
w0 = -np.power(2, 1/3) / (2 - np.power(2,1/3))
w1 = 1 / (2 - np.power(2, 1/3))
c = [w1 / 2, (w0 + w1) / 2, (w0 + w1) / 2, w1 / 2]
d = [w1, w0, w1]

def compute_acceleration_two_orbiting_bodies(x1,y1,x2,y2,M,m1,m2):
    
    r1 = np.sqrt(x1**2 + y1**2) # distance between m1 and M
    r2 = np.sqrt(x2**2 + y2**2) # distance between m2 and M
    d= np.sqrt((x2 - x1)**2 + (y2 - y1)**2) # distance between m1 and m2


    ax1 = -G * M * x1 / r1**3 + (G * m2 * (x2 - x1) / d**3) # acceleration in x direction on mass 1
    ay1 = -G * M * y1 / r1**3 + (G * m2 * (y2 - y1) / d**3) # acceleration in y direction on mass 1
    ax2 = -G * M * x2 / r2**3 + (G * m1 * (x1 - x2) / d**3) # acceleration in x direction on mass 2
    ay2 = -G * M * y2 / r2**3 + (G * m1 * (y1 - y2) / d**3) # acceleration in y direction on mass 2

    return ax1, ay1, ax2, ay2

def compute_acceleration_one_orbiting_body(x1,y1,M,m):
    
    r = np.sqrt(x1**2 + y1**2) # distance between m1 and M
  
    ax = -G * M * x1 / r**3 # acceleration in x direction on mass 1
    ay = -G * M * y1 / r**3  # acceleration in y direction on mass 1
   
    return ax, ay

def symplectic_integrate_two_body(IVP, dt, steps, M, m1, m2):

    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP # initial value conditions 
    solution = np.zeros((steps,8)) # initialze array of solutions, this should be the size of the intial conditons by the number of time steps 
    solution[0] = IVP # the first time step soltuion is set to the initial values

    for i in range (1,steps):

        x1 += c[0] * dt * vx1
        y1 += c[0] * dt * vy1
        x2 += c[0] * dt * vx2
        y2 += c[0] * dt * vy2

        for j in range(3): #  there are 3 substeps for 4th order integration
            
            ax1, ay1, ax2, ay2 = compute_acceleration_two_orbiting_bodies(x1, y1, x2, y2, M, m1, m2)

            vx1 += d[j] * dt * ax1
            vy1 += d[j] * dt * ay1
            vx2 += d[j] * dt * ax2
            vy2 += d[j] * dt * ay2

            x1 += c[j + 1] * dt * vx1
            y1 += c[j + 1] * dt * vy1
            x2 += c[j + 1] * dt * vx2
            y2 += c[j + 1] * dt * vy2

        solution[i] = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]

    return solution

def symplectic_integrate_one_body(IVP, dt, steps, M, m):

    x, y, vx, vy = IVP # initial value conditions 
    solution = np.zeros((steps,4)) # initialze array of solutions, this should be the size of the intial conditons by the number of time steps 
    solution[0] = IVP # the first time step soltuion is set to the initial values

    for i in range (1,steps):

        # initial position update with c[0]
        x += c[0] * dt * vx
        y += c[0] * dt * vy 

        for j in range(3): #  there are 3 substeps for 4th order integration
            
            ax, ay = compute_acceleration_one_orbiting_body(x, y,  M, m)

            vx += d[j] * dt * ax
            vy += d[j] * dt * ay
            
            x += c[j + 1] * dt * vx
            y += c[j + 1] * dt * vy
           

        solution[i] = [x, y, vx, vy]

    return solution


# Example Usage 

M = 1.989e30 # mass of the central body, this position is assumed to be (0,0) (Sun)
m1 = 5.972e24 # mass of secondary body 1 (Earth)
m2 = 6.39e23 # mass of secondary body 2 (Mars)

x1 = 1.5e11 
y1 = 0
vx1 = 0 # velocity is purley tangential
vy1 = np.sqrt(G * M/ x1) # velocity is set to ensure intially ciruclar motion

x2 = 2.28e11 
y2 = 0
vx2 = 0 # velocity is purley tangential
vy2 = np.sqrt(G* M/ x2) # velocity is set to ensure intially ciruclar motion


IVP_2body= [x1, y1, vx1, vy1, x2, y2, vx2, vy2 ] # set intial conditions
IVP_Earth= [x1, y1, vx1, vy1] # set intial conditions
IVP_Mars = [x2, y2, vx2, vy2 ] # set intial conditions

dt = 60 ** 2 # time step value (intially set to 1 hour)
steps = 100000 # number of time steps 

sol_2body = symplectic_integrate_two_body(IVP_2body, dt, steps, M, m1, m2)
sol_Earth = symplectic_integrate_one_body(IVP_Earth, dt, steps, M, m1)
sol_Mars = symplectic_integrate_one_body(IVP_Mars, dt, steps, M, m2)

# Plot Orbits (includes graviaitional relationship between the two secondary bodies)

x1s, y1s = sol_2body[:, 0], sol_2body[:, 1] # two body Earth 
x2s, y2s = sol_2body[:, 4], sol_2body[:, 5] # two body Mars

xE, yE = sol_Earth[:,0], sol_Earth[:,1] # one body Earth
xM, yM= sol_Mars[:,0], sol_Mars[:,1] # one body Mars


# Plot the Two Body Solution with what the one body solution should be 
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
plt.show()




