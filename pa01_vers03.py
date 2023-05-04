'''TODO:
###
    - implement other two approx methords
    - also visualize them
    - Visualize f2
    - ! How to do 2D?'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%% funkctions
class f1:
    def __call__(self, x):
        return 5*(x[0]**2) - 6*x[0]*x[1] + 5*(x[1]**2)
    def derivative(self, x):
        return np.array([10*x[0] - 6*x[1], -6*x[0] + 10*x[1]])

class f2:
    def __call__(self, x):
        return 100*(x[1]-x[0]^2)^2 +(1-x[0])^2
    def derivative(self, x):
        return np.array([2*(200*x[0]^3 - 200*x[0]*x[1] + x[0] -1), 200*(x[1]-x[0])^2])
        
#%% optimisation
#x0 = startvalue, a0 = initial stepwith, o = adaptive factor for a0, e = approx-closeness

def gradientDescent(f, x0, a0, o, e):
    ak = a0
    xk = x0
    fk1 = np.inf
    fk = f(x0)
    while np.linalg.norm(fk1 - fk) >= e:
        df = f.derivative(xk)
        df = (1/np.linalg.norm(df)) * df
        while f(xk - (ak * df)) >= fk:
            ak *= o
        xk = xk - ak * df
        fk1 = np.copy(fk)
        fk = f(xk)
    return [xk, fk]

def coordinateDescent(F,x0,a0,o,e):
    return None

def stochasticGradientDescent(F,x0,a0,o,e):
    return None


        
                
#%% visualisation: 
'''
Given Information to remember:
    - (x,y) is in [-10,10]^2 - DONE
    - implement optimisations(opt) for f1 with x0 = (-5,-5) and x0 = (-3,2)
    - implement opt for f2 with x0 = (0,3) and x0 = (2,1)
    - use quiver function to show the opt
    - run opt until e <= 10^-2
    
    TODO whisches: 
        -slider for o?
        -better colourscheme
        - turn the 3D Object automaticaly?
'''

# construct x- & y-Values
x_values = np.linspace(-10, 10, 100)
y_values = np.linspace(-10, 10, 100)
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Calculate z according to x- & y-Values
f1_ = f1()
z_grid = np.array([[f1_(np.array([x, y])) for x in x_values] for y in y_values])

# Drwa 3D-Surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.PuBu, alpha=0.8)

# Gradient Descent
a0 = 1
o = 0.5
e = 1e-2

x0_0 = np.array([-5, -5])
gd = gradientDescent(f1_, x0_0, a0, o, e)
pnt_x, pnt_y = [x0_0[0], gd[0][0]], [x0_0[1], gd[0][1]]
pnt_z = np.array([f1_(p) for p in np.column_stack((pnt_x, pnt_y))]) # itteration over the column_stack and then applies the f1_(p) on it. p is a point in the given stack)
ax.plot(pnt_x, pnt_y, pnt_z, 'b-o', linewidth=1, markersize=1)

x0_1 = np.array([-3, 2])
gd = gradientDescent(f1_, x0_1, a0, o, e)
pnt_x, pnt_y = [x0_1[0], gd[0][0]], [x0_1[1], gd[0][1]]
pnt_z = np.array([f1_(p) for p in np.column_stack((pnt_x, pnt_y))]) # itteration over the column_stack and then applies the f1_(p) on it. p is a point in the given stack)
ax.plot(pnt_x, pnt_y, pnt_z, 'y-o', linewidth=1, markersize=1)

# Labels
ax.set_xlabel('X-Achse')
ax.set_ylabel('Y-Achse')
ax.set_zlabel('Z-Achse')

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


