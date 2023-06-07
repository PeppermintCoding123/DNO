
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%% implement as lamda functions

f = lambda x: (x[0] + x[1]) ** 2 + (np.sin(x[1]) * 3)
f_derivative = lambda x: np.array([(x[0] + x[1]) * 2, (x[0] + x[1]) * 2 + 3 * np.cos(x[1])])

#%%
#x0 = startvalue, A = spd-Matrix, a0 = Startingvalue

# see Algorithm 2.42 from lecture
def conjugateGradient(A, b, x0):   
    # initialise step
    n = np.shape(b)[0]
    xk = x0
    dk = rk = b - A@xk
    
    res = []
    res.append(xk)
    
    for k in range(n):
        # calculate step
        ak = np.inner(rk, dk) / np.inner(dk, A@dk)
        
        xk = xk + ak*dk
        res.append(xk)
        
        # prepare for next step
        if k < n-1:
            rk_plus1 = rk - ak* A@dk
            
            bk_plus1 = np.inner(rk_plus1, rk_plus1) / np.inner(rk, rk)
            
            dk = rk_plus1 + bk_plus1 * dk
            rk = rk_plus1
    
    return 'conjugateGradient: x* = ' + str(res[-1])

def conjugateGradientNL(x0, a0, o, e):
    # initialise step
    n = np.shape(x0)[0]
    xk = x0
    ak = a0
    dk = - f_derivative(xk)
    pk = dk
    
    # for Armijo-Goldstein
    c1 = 0.1
    c2 = 0.9 # values from lecture
    
    res = []
    res.append(np.hstack((xk,f(xk))))
    
    
    xk_plus1 = xk + ak * dk
    
    while np.linalg.norm(f(xk)-f(xk_plus1)) > e:
        ak = a0
        
        # Armijo Goldstein
        Ek = ak * np.inner(-dk, pk)
        Dk = f(xk_plus1) - f(xk)
        while not c1*Ek > Dk or Dk > c2*Ek:
            ak = o * ak
            xk_plus1 = xk + ak * pk
            Ek = ak * np.inner(-dk, pk)
            Dk = f(xk_plus1) - f(xk)
        
        # prepare for next step
        df_xkplus1 = - f_derivative(xk_plus1)
            
        bk_plus1 = np.inner(df_xkplus1, df_xkplus1) / np.inner(dk, dk)
        pk = df_xkplus1 + bk_plus1 * pk
            
        xk = xk_plus1
        xk_plus1 = xk + ak * pk
        dk = - f_derivative(xk)
        res.append(np.hstack((xk,f(xk))))
        
    return np.array(res)
#%% Test for conjugateGradient
A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])
x0 = np.array([-2, 2])

print(conjugateGradient(A, b, x0))

#%% visualisation: 
plt.close('all')

# construct x- & y-Values
x_values = np.linspace(-10, 10, 100)
y_values = np.linspace(-10, 10, 100)
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Define the starting points and functions
x0 = np.array([-5, -5])
a0 = 1;
o = 0.9
e = 1e-2

     
# Calculate z according to x- & y-Values
z_grid = np.array([[f(np.array([x, y])) for x in x_values] for y in y_values])

# Draw 3D-Surface
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
surf = ax1.plot_surface(x_grid, y_grid, z_grid, cmap=cm.YlOrRd, alpha=0.8)

fig.colorbar(surf, shrink=0.5, aspect=5)
             
# Labels
plt.title(f'Figure conjugateGradientNL with point x0 = {x0} in 3D' )
ax1.set_xlabel('X-Achse')
ax1.set_ylabel('Y-Achse')
ax1.set_zlabel('Z-Achse')

fig2, ax2 = plt.subplots()
contour = ax2.contourf(x_grid, y_grid, z_grid, cmap=cm.YlOrRd, levels=20)
fig2.colorbar(contour)

cgNL_points = conjugateGradientNL(x0, a0, o, e)
drc = np.diff(cgNL_points, axis=0)
ax2.quiver(cgNL_points[:-1,0], cgNL_points[:-1,1], drc[:,0], drc[:,1], color='blue', angles='xy', scale_units='xy', scale=1, label='cgNL')

plt.title(f'Figure conjugateGradientNL with point x0 = {x0} in 2D')
ax2.set_xlabel('X-Achse')
ax2.set_ylabel('Y-Achse')

ax2.legend()

plt.show()

print('x* in cgNL = ' + str(cgNL_points[-1][:-1]))
