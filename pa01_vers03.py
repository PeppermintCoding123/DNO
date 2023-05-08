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
        return 100*(x[1]-x[0]^2)^2 +(1-x[0])^2
    def derivative(self, x):
        return np.array([2*(200*x[0]^3 - 200*x[0]*x[1] + x[0] -1), 200*(x[1]-x[0])^2])
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
    f_ = f()
    fk = f_(xk)
    res = []
    res.append((xk,fk))
    while np.linalg.norm(fk1 - fk) >= e:#!!!Shouldn't it be > and not >= ?
        df = f_.derivative(xk)
        df = (1/np.linalg.norm(df)) * df
        while f_(xk - (ak * df)) >= fk:
            ak *= o
        xk = xk - ak * df
        fk1 = np.copy(fk)
        fk = f_(xk)
        res.append((xk,fk))
    return np.array(res)

def coordinateDescent(f, x0, a0, o, e):
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
    - implement optimisations(opt) for f1 with x0 = (-5,-5) and x0 = (-3,2)
    - implement opt for f2 with x0 = (0,3) and x0 = (2,1)
    - use quiver function to show the opt
    - run opt until e <= 10^-2
    
    TODO whisches: 
        -slider for o?
        -better colourscheme
        - turn the 3D Object automaticaly?
      
    quiver: Alle Startpunkte von Pfeilen, Richtungen von Pfeilen übergeben --> plottet Pfeil von aktueller iterierter zu nächster
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
'''
o = float(input("Wählen Sie sigma zwischen 0 und 1."))
if o >= 1:
    raise ValueError('sigma ist zu groß')
if o < 0:
    raise ValueError('sigma muss positiv sein')
'''
o = 0.5 #Wir setzen o hier 0.5, wenn das Programm funktioniert, erfolgt die Wahl von o durch Eingabe eines Wertes. (Diese Zeile wird gelöscht und dafür der obere Block genommen.)
e = 1e-2
'''TODO:
- Lisa:
- use quiver function of numpy
- have arrows as previous version'''
'''
x0_0 = np.array([-5, -5])
#gd = gradientDescent(f1_, x0_0, a0, o, e)
gd = gradientDescent(f1, x0_0, a0, o, e)#should be f1, not f1_. Otherwise class can't be called correclty.
pnt_x, pnt_y = [x0_0[0], gd[0][0]], [x0_0[1], gd[0][1]]
pnt_z = np.array([f1_(p) for p in np.column_stack((pnt_x, pnt_y))]) # itteration over the column_stack and then applies the f1_(p) on it. p is a point in the given stack)
ax.plot(pnt_x, pnt_y, pnt_z, 'b-o', linewidth=1, markersize=1)

x0_1 = np.array([-3, 2])
#gd = gradientDescent(f1_, x0_1, a0, o, e)
gd = gradientDescent(f1, x0_1, a0, o, e)#The same thing here.
pnt_x, pnt_y = [x0_1[0], gd[0][0]], [x0_1[1], gd[0][1]]
pnt_z = np.array([f1_(p) for p in np.column_stack((pnt_x, pnt_y))]) # itteration over the column_stack and then applies the f1_(p) on it. p is a point in the given stack)
ax.plot(pnt_x, pnt_y, pnt_z, 'y-o', linewidth=1, markersize=1)'''

# Labels
ax.set_xlabel('X-Achse')
ax.set_ylabel('Y-Achse')
ax.set_zlabel('Z-Achse')

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


