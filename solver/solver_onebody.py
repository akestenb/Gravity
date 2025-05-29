import numpy as np 
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11 # gravitiaional constant 

# Sympletic integrator coefficents (Yoshida 4th order)
w0 = -np.power(2, 1/3) / (2 - np.power(2,1/3))
w1 = 1 / (2 - np.power(2, 1/3))
c = [w1 / 2, (w0 + w1) / 2, (w0 + w1) / 2, w1 / 2]
d = [w1, w0, w1]

def compute_acceleration_one_orbiting_body(x1,y1,M,m):
    
    r = np.sqrt(x1**2 + y1**2) # distance between m1 and M
  
    ax = -G * M * x1 / r**3 # acceleration in x direction on mass 1
    ay = -G * M * y1 / r**3  # acceleration in y direction on mass 1
   
    return ax, ay

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

def plot_single_orbit(x, y, label='Orbiting Body', color='blue'):
    plt.plot(x, y, label=label, color=color)
    plt.scatter(0, 0, color='yellow', label='Central Mass (Sun)')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Orbit Around Central Mass")
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.show()

# Example Usage 

M = 1.989e30 # mass of the central body, this position is assumed to be (0,0) (Sun)
m1 = 5.972e24 # mass of secondary body 1 (Earth)


x1 = 1.5e11 
y1 = 0
vx1 = 0 # velocity is purley tangential
vy1 = np.sqrt(G * M/ x1) # velocity is set to ensure intially ciruclar motion

dt = 60 ** 2 # time step value (intially set to 1 hour)
steps = 100000 # number of time steps 


# Compare to only one orbiting body 
IVP_earth = [x1, y1, vx1, vy1]
sol_earth = symplectic_integrate_one_body(IVP_earth, dt, steps, M, m1)
x_earth, y_earth = sol_earth[:, 0], sol_earth[:, 1]
plot_single_orbit(x_earth, y_earth, label='Earth (No Interaction)', color='blue')