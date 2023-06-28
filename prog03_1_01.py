import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

#%%
def F_call (u, aufgabe):
    if aufgabe == 'a':
        A = np.array([[0, 1], [-1, 0]])
        delta_u = A @ u
        return delta_u
    if aufgabe == 'b':
        A = np.array([[0.5, -1],[1, -1]])
        delta_u = A @ u
        return delta_u
    if aufgabe == 'c':
        if u[1] < 0:
            return np.array([0,0])
        else:
            delta_u_up = np.sqrt(u[1])
            delta_u_down = -2 * u[0] * delta_u_up
            delta_u = np.array([delta_u_up, delta_u_down])
            return delta_u
    else:
        raise ValueError('Not known Aufgabe')

def F_derivative(aufgabe):
    if aufgabe == 'a':
        A = np.array([[0, 1], [-1, 0]])
        A_invers = np.linalg.solve(A, np.eye(2))
        return A_invers
    if aufgabe == 'b':
        A = np.array([[0.5, -1],[1, -1]])
        A_invers = np.linalg.solve(A, np.eye(2))
        return A_invers
    if aufgabe == 'c':
        pass
#%%
def explisiteEuler(u0, tau, nbr_Steps, aufgabe):
    res = []
    u_tk = u0
    
    u_tk_x = u_tk[0]
    
    res.append(u_tk) # Step for tau = 0
    
    for k in range(1, nbr_Steps+1):
        u_tkPlus1_x = u_tk_x + tau
        F_tk = F_call(u_tk, aufgabe)
        Ftk = F_tk * tau
        u_tkPlus1 = (u_tk + Ftk)
        u_tkPlus1_y = u_tkPlus1[1]
        
        # Prepare for next step 
        u_tk_x = u_tkPlus1_x
        u_tk = u_tkPlus1
        res.append(np.hstack((u_tkPlus1_x, u_tkPlus1_y)))
          
    return np.array(res)

def impliciteEuler(u0, tau, nbr_Steps, aufgabe):
    u0_shape = np.shape(u0)
    res = []
    u_tk = u0
    res.append(u_tk)
    
    u_tk_x = u_tk[0]
    
    for k in range(1, nbr_Steps+1):
        u_tk_xPlus1 = u_tk_x + tau
        tmp = lambda u_tkPlus1: u_tkPlus1 - u_tk - tau * F_call(u_tkPlus1, aufgabe)
        u_tkPlus1 = fsolve(tmp, np.zeros(u0_shape))
        u_tkPlus1_y = u_tkPlus1[1]
        res.append(np.hstack((u_tk_xPlus1, u_tkPlus1_y)))
        
        # prepare for next step
        u_tk_x = u_tk_xPlus1
        u_tk = u_tkPlus1
        
    return np.array(res)
#%%
# choose exact exersise to display from ('a', 'b', 'c')
'''
chosen_aufgabe = str(input('Wählen Sie zwischen den Anfangswertproblemen a), b) und c).'))
if chosen_aufgabe != 'a':
    if chosen_aufgabe != 'b':
        if chosen_aufgabe != 'c':
            print("Es gibt nur die Möglichkeiten a), b) und c).")
            chosen_aufgabe = str(input('Wählen Sie zwischen den Anfangswertproblemen a), b) und c).'))
'''
chosen_aufgabe = 'a'

#%% implementation of analytic solution

def analytic_solution(aufgabe):
    if aufgabe == 'a':
        res = []
        for i in range(nbr_Steps+1):
            res.append(np.array([np.sin(i), np.cos(i)]))
        return np.array(res)
    if aufgabe == 'b':
        pass
    if aufgabe == 'c':
        pass


#%% visualisation: 
plt.close('all')

u0 = np.array([0,1])
tau = 0.1
nbr_Steps = 100

x = []
y = []
res = explisiteEuler(u0, tau, nbr_Steps, chosen_aufgabe)

figure, ax = plt.subplots()

ax.set_xlim(res[0,0],res[-1,0])
ax.set_ylim(-5,5)

line, = ax.plot(0,0)

def animation_function(i):
    x.append(res[i,0])
    y.append(res[i,1])
    
    line.set_xdata(x)
    line.set_ydata(y)
    return line,
 
animation = FuncAnimation(figure, animation_function, frames = np.arange(0,100,1), interval = 70, repeat = False)


plt.show()
