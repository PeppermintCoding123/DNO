import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%% implement as lamda functions

f = lambda x: (x[0] + x[1]) ** 2 + (np.sin(x[1]) * 3)
f_derivative = lambda x: np.array([(x[0] + x[1]) * 2, (x[0] + x[1]) * 2 + 3 * np.cos(x[1])])

#%%
#x0 = startvalue, H0 = initial Hessematrixapprox, a0 = initial stepwith, o = adaptive factor for a0, e = approx-closeness
def BFGS(x0, H0, a0, o, e):
    # initialize iterations
    xk = x0
    Hk = H0
    fk = f(xk)
    fk_plus1 = np.inf
    res = []
    res.append(np.hstack((xk,fk)))

    while np.linalg.norm(fk_plus1 - fk) > e:
        ak = a0
        fk = f(xk)
        # adjust ak to take step
        dfk = f_derivative(xk)

        xk_plus1 = xk - ak*Hk@dfk #TODO: Merschweinchen: Is this the right function to create a new vector or should we usw something better?
        #both Hk.dot(dfk) and Hk@dfk work, but I can't find information to .dot, so maybe using @ is better
        fk_plus1 = f(xk_plus1)
        
        while fk_plus1 >= fk:
                ak *= o
                xk_plus1 = xk - ak*Hk@dfk
                fk_plus1 = f(xk_plus1)
                
        res.append(np.hstack((xk_plus1,fk_plus1)))
        
        # calculate Hk+1
        sk = xk_plus1 - xk
        yk = f_derivative(xk_plus1) - f_derivative(xk)
        rohk = 1/(np.dot(yk,sk))
        
        rys = rohk * np.outer(yk, sk.T)
        rsy = rys.T
        Hk = Hk - (Hk @ rys) - (rsy @ Hk) + (rsy @ Hk @ rys) + (rohk * np.outer(sk,sk.T))
        
        # move for next iteration
        xk = np.copy(xk_plus1)
        
    return np.array(res)
    
    
#%%
'''TODO:
    - test BFSG with startingvalues of x0 = [-5,-5], a0 = 1, H0 = 1n
    o = 0.5
    
    PLOT:
        - pathing with quiver
        - in 'Konturenplot' 
        - presision: e-6
  '''
#%% visualisation: 
plt.close('all')

# construct x- & y-Values
x_values = np.linspace(-10, 10, 100)
y_values = np.linspace(-10, 10, 100)
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Define the starting points and functions
x0 = np.array([-5, -5])
a0 = 1;
H0 = np.eye(2)
o = 0.5
e = 1e-2

     
# Calculate z according to x- & y-Values
z_grid = np.array([[f(np.array([x, y])) for x in x_values] for y in y_values])

# Draw 3D-Surface
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
surf = ax1.plot_surface(x_grid, y_grid, z_grid, cmap=cm.YlOrRd, alpha=0.8)

fig.colorbar(surf, shrink=0.5, aspect=5)
        
#BFGS
bfgs_points = BFGS(x0, H0, a0, o, e)
drc = np.diff(bfgs_points, axis=0)
ax1.quiver(bfgs_points[:-1, 0], bfgs_points[:-1, 1], bfgs_points[:-1, 2], drc[:, 0], drc[:, 1], drc[:, 2], color='blue', label='BFGS')
# TODO: Merschweinchen: How to fix the issue with the quiver not functioning?
# !!!The attribute scale (apparently) can't be used for 3D plots, only for 2D plots      
                
# Labels
plt.title(f'Figure BFGS with point x0 = {x0} in 3D' )
ax1.set_xlabel('X-Achse')
ax1.set_ylabel('Y-Achse')
ax1.set_zlabel('Z-Achse')
        
ax1.legend()
        
plt.show()

def it_steps(bfgs_points):
    return bfgs_points.T[0].size
#print('Total number of itterations to reach |xk - x*| < ' + str(e) + ' is: ' + str(bfgs_points.T[0].size))
print('Total number of iterations to reach |xk - x*| < ' + str(e) + ' is: ' + str(it_steps(bfgs_points)))
# !!!If I understood the explanation right, he wants a function, you can also call over the commando line, so I think this should be a good way of implementation
print('x* = ' + str(bfgs_points[-1]))

    
    