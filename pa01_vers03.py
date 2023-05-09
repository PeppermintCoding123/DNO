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
    def part_derivative(self, x, i):    #i = {0,1}
        if i == 0:
            return 10*x[0] - 6*x[1]
        if i == 1:
            return 10*x[1] - 6*x[0]
        else:
            raise ValueError('i must be 0 or 1')
    #Aufteilen der Funktion für stochasticGradientDecent, direkt Bildung des Gradienten
    def sgd_parts(self):
        return 3
    def sgd_derivative(self, x, i):
        if i == 0:
            return np.array([10*x[0], 0])
        if i == 1:
            return np.array([-6*x[1], -6*x[0]])
        if i == 2:
            return np.array([0,10*x[1]])
        else:
            return ValueError('Index out of range')

class f2:
    def __call__(self, x):
        return 100*(x[1]-x[0]**2)**2 +(1-x[0])**2
    def derivative(self, x):
        return np.array([2*(200*x[0]**3 - 200*x[0]*x[1] + x[0] -1), 200*(x[1]-x[0])**2])
    def part_derivative(self, x, i):    #i = {0,1}
        if i == 0:
            return -400*x[0]*(x[1]-(x[0])**2) - 2*(1-x[0])
        if i == 1:
            return 200*(x[1]-(x[0])**2)
        else:
            raise ValueError('i must be 0 or 1')
    #Aufteilen der Funktion für stochasticGradientDecent, direkt Bildung des Gradienten
    def sgd_parts(self):
        return 2
    def sgd_derivative(self, x, i):
        if i == 0:
            return np.array([400*x[0]*(x[1]-(x[0])**2), 200*(x[1]-(x[0])**2)])
        if i == 1:
            return np.array([-2*(1-x[0])])
        else:
            return ValueError('Index out of range')

#!!!can someone maybe check the partial derivatives?
#!!!Did I understand sgd correctly?
        
#%% optimisation
#x0 = startvalue, a0 = initial stepwith, o = adaptive factor for a0, e = approx-closeness

'''---gradientDescent---
Takes the arguments and itterates over them until the x* value is reaced in range of e.
The result should be a np.array with all the steps, beginning with the starting coordinates. 
    [[x0,f(x0)],[x1,f(x1)],[x2,f(x2)],...]
'''
def gradientDescent(f, x0, a0, o, e):
    ak = a0
    xk = x0
    fk1 = np.inf
    f_ = f
    fk = f_(xk)
    res = []
    res.append(np.hstack((xk,fk)))
    while np.linalg.norm(fk1 - fk) >= e:
        df = f_.derivative(xk)
        df = (1/np.linalg.norm(df)) * df
        while f_(xk - (ak * df)) >= fk:
            ak *= o
        xk = xk - ak * df
        fk1 = np.copy(fk)
        fk = f_(xk)
        res.append(np.hstack((xk,fk)))
    return np.array(res)

def coordinateDescent(f, x0, a0, o, e):#in progress Lisa
    ak = a0
    xk = x0
    xk1 = np.inf
    f_ = f()
    fk1 = f_(xk1)
    fk = f_(xk)
    while np.linalg.norm(fk1 - fk) > e:
        xk = xk1.copy#!!!is what I'm doing here correct?
        i = np.random.randint(np.shape(x0)[0])
        pk = f.part_derivative(xk, i)
        
        while f_(xk - ak*pk/np.linalg.norm(pk)) > f_(xk):
            ak = o*ak
        xk1 = xk - ak*pk/np.linalg.norm(pk)
        
    return xk1, f_(xk1)#!!!is it correct to return xk1?
    #Answer: Yes, but we want an array of xk for all 0 to k+1 and f(xk) from 0 to k+1, see gradientDescent

def stochasticGradientDescent(f,x0,a0,o,e):
    ak = a0
    xk = x0
    xk1 = np.inf
    f_ = f()
    fk1 = f_(xk1)
    fk = f_(xk)
    while np.linalg.norm(fk1 - fk) > e:
        xk = xk1.copy
        i = np.random.randint(f.sgd_parts)
        pk = f.sgd_derivative(xk, i)
        
        while f_(xk - ak*pk/np.linalg.norm(pk)) > f_(xk):
            ak = o*ak
        xk1 = xk - ak*pk/np.linalg.norm(pk)
    return xk1, f_(xk1)


        
                
#%% visualisation: 
'''
Given Information to remember:
    - (x,y) is in [-10,10]^2 - DONE
    - implement optimisations(opt) for f1 with x0 = (-5,-5) and x0 = (-3,2) - Done for gradient-Decent
    - implement opt for f2 with x0 = (0,3) and x0 = (2,1) - TODO:Lisa
    - use quiver function to show the opt -DONE
    - run opt until e <= 10^-2 - DONE
    
    TODO whisches: 
        -slider for o? - NOT NESSACARY
        -better colourscheme - Done
        - turn the 3D Object automaticaly? - NOT NESSACARY
      
    quiver: Alle Startpunkte von Pfeilen, Richtungen von Pfeilen übergeben --> plottet Pfeil von aktueller iterierter zu nächster
'''

# construct x- & y-Values
x_values = np.linspace(-10, 10, 100)
y_values = np.linspace(-10, 10, 100)
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Define the starting points and functions
starting_points_map = {
    f1(): [np.array([-5, -5]), np.array([-3, 2])] 
    #,  f2(): [np.array([0, 3]), np.array([2, 1])]
    #TODO: LISA: this line is commented out, because gradientDecent dont work on it jet. Debugging later tonight.
}

# Loop through the functions and their corresponding starting points
for func, starting_points in starting_points_map.items():
    for x0 in starting_points:
        
        # Calculate z according to x- & y-Values
        f_ = func
        z_grid = np.array([[f_(np.array([x, y])) for x in x_values] for y in y_values])

        # Drwa 3D-Surface
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        surf = ax1.plot_surface(x_grid, y_grid, z_grid, cmap=cm.YlOrRd, alpha=0.8)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        # Gradient Descent
        a0 = 1
        o = 0.5
        e = 1e-2
        
'''TODO: Merschweinchen: Implement the next 3 lines for the other two functions '''
        gd = gradientDescent(f_, x0, a0, o, e)
        drc = np.diff(gd, axis=0) # calculated the directiojn between points
        ax1.quiver(gd[:-1, 0], gd[:-1, 1], gd[:-1, 2], drc[:, 0], drc[:, 1], drc[:, 2], color='lightcoral')#x ist the first two coordinates, f(x) should be the third one. The arrowshould point in the direction orthogonal to the 
        
        '''TODO: Merschweinchen: Probbably like this:
        cd = coordinateDescent(f, x0, a0, o, e)
        drc = np.diff(cd, axis=0) # calculated the directiojn between points
        ax1.quiver(cd[:-1, 0], cd[:-1, 1], cd[:-1, 2], drc[:, 0], drc[:, 1], drc[:, 2], color='crimson')
        
        sd = stochasticGradientDescent(f,x0,a0,o,e)
        drc = np.diff(sd, axis=0) # calculated the directiojn between points
        ax1.quiver(sd[:-1, 0], sd[:-1, 1], sd[:-1, 2], drc[:, 0], drc[:, 1], drc[:, 2], color='deeppink')
        '''
        
        # Labels
        plt.title(f'Figure {func.__class__.__name__} with point x0 = {x0} in 3D' )
        ax1.set_xlabel('X-Achse')
        ax1.set_ylabel('Y-Achse')
        ax1.set_zlabel('Z-Achse')

        plt.show()


        # Create a new figure and axis for the 2D plot
        fig2, ax2 = plt.subplots()
        contour = ax2.contourf(x_grid, y_grid, z_grid, cmap=cm.YlOrRd, levels=20)
        fig2.colorbar(contour)

        # Add quiver arrows to the 2D plot
        ax2.quiver(gd[:-1, 0], gd[:-1, 1], drc[:, 0], drc[:, 1], color='lightcoral', angles='xy', scale_units='xy', scale=1)
        '''TODO: Merschweinchen: the rest
        ax2.quiver(cd[:-1, 0], cd[:-1, 1], cd[:-1, 2], drc[:, 0], drc[:, 1], drc[:, 2], color='crimson', angles='xy', scale_units='xy', scale=1)
        ax1.quiver(sd[:-1, 0], sd[:-1, 1], sd[:-1, 2], drc[:, 0], drc[:, 1], drc[:, 2], color='deeppink', angles='xy', scale_units='xy', scale=1)
        '''
        
        
        plt.title(f'Figure {func.__class__.__name__} with point x0 = {x0} in 2D' )
        ax2.set_xlabel('X-Achse')
        ax2.set_ylabel('Y-Achse')

        # Display the 2D plot
        plt.show()

