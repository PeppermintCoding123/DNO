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
        #return np.array[[np.cos(i)], [-2 * np.cos(i) * np.sin(i)]]
        pass
#%%
def explisiteEuler(u0, tau, nbr_Steps, aufgabe):
    res = []
    u_tk = u0
    
    u_tk_x = u_tk[0]
    
    res.append(u_tk) # Step for tau = 0
    
    for t in range(1, nbr_Steps+1):
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

chosen_aufgabe = str(input('Wählen Sie zwischen den Anfangswertproblemen a), b) und c).'))
if chosen_aufgabe != 'a':
    if chosen_aufgabe != 'b':
        if chosen_aufgabe != 'c':
            print("Es gibt nur die Möglichkeiten a), b) und c).")
            chosen_aufgabe = str(input('Wählen Sie zwischen den Anfangswertproblemen a), b) und c).'))
            
#chosen_aufgabe = 'c'

#%% implementation of analytic solution

def analytic_solution(aufgabe, tau, nbr_Steps):
    if aufgabe == 'a':
        res = []
        for i in range(nbr_Steps+1):
            i *= tau
            #res.append(np.array([np.sin(i), np.cos(i)]))
            res.append(np.array([i, np.cos(i)]))
        return np.array(res)
    if aufgabe == 'b':
        res = []
        for i in range(nbr_Steps+1):
            i *= tau
            #res.append(np.array([1/np.sqrt(7) * 4 * np.exp(-i/4) * np.sin(np.sqrt(7)*i/4), np.exp(-i/4) * np.cos(np.sqrt(7)*i/4) - 3*np.sqrt(7)/7 * np.exp(-i/4) * np.sin(np.sqrt(7)*i/4)]))
            res.append(np.array([i, np.exp(-i/4) * np.cos(np.sqrt(7)*i/4) - 3*np.sqrt(7)/7 * np.exp(-i/4) * np.sin(np.sqrt(7)*i/4)]))
        return np.array(res)
    if aufgabe == 'c':
        res = []
        for i in range(nbr_Steps+1):
            i *= tau
            res.append(np.array([i, np.cos(i)**2]))
        return np.array(res)
    
    
#%% visualisation: 
'''
plt.close('all')

u0 = np.array([0,1])
tau = 0.1
nbr_Steps = 100

x = []
y = []
#res = explisiteEuler(u0, tau, nbr_Steps, chosen_aufgabe)
#res2 = impliciteEuler(u0, tau, nbr_Steps, chosen_aufgabe)
res = analytic_solution(chosen_aufgabe, tau, nbr_Steps)
figure, ax = plt.subplots()

ax.set_xlim(res[0,0],res[-1,0])
ax.set_ylim(-5,5)

line, = ax.plot(0,0)
#line_imp, = ax.plot(0,0)

def animation_function(i):
    x.append(res[i,0])
    y.append(res[i,1])
    
    x.append(res2[i,0])
    y.append(res2[i,1])

    line.set_xdata(x)
    line.set_ydata(y)
    return line,
 
animation = FuncAnimation(figure, animation_function, frames = np.arange(0,100,1), interval = 70, repeat = False)

'''
'''
res2 = analytic_solution(chosen_aufgabe)
def animation_function(i):
    x.append(res2[i,0])
    y.append(res2[i,1])
    
    line.set_xdata(x)
    line.set_ydata(y)
    return line,
 
animation = FuncAnimation(figure, animation_function, frames = np.arange(0,100,1), interval = 70, repeat = False)
'''
'''
plt.show()
'''

#other version of plot that should work for 2 values at the same time:
#%% visualisation: 
plt.close('all')

u0 = np.array([0,1])
tau = 0.1
nbr_Steps = 100

figure, ax = plt.subplots()
x = []
y_analytical = []
y_explEuler = []
y_implEuler = []
line_ana, = plt.plot([], [], 'r', label='analytical solution')
line_expEul, = plt.plot([], [], 'b', label='explicite Euler')
line_impEul, = plt.plot([], [], 'c', label='implicite Euler')


# apply functions
expEuler_res = explisiteEuler(u0, tau, nbr_Steps, chosen_aufgabe)
impEuler_res = impliciteEuler(u0, tau, nbr_Steps, chosen_aufgabe)
analytical_res = analytic_solution(chosen_aufgabe, tau, nbr_Steps)

ax.set_xlim(expEuler_res[0,0],expEuler_res[-1,0])
ax.set_ylim(-5,5)

def animation_function(i):
    ax.set_xlabel('Zeitschritte')
    x.append(expEuler_res[i,0])
    y_analytical.append(analytical_res[i,1])
    y_explEuler.append(expEuler_res[i,1])
    y_implEuler.append(impEuler_res[i,1])
    
    line_ana.set_xdata(x)
    line_expEul.set_xdata(x)
    line_impEul.set_xdata(x)
    line_ana.set_ydata(y_analytical)
    line_expEul.set_ydata(y_explEuler)
    line_impEul.set_ydata(y_implEuler)
    return line_ana, line_expEul, line_impEul,
 
animation = FuncAnimation(figure, animation_function, frames = np.arange(0,100,1), interval = 70, repeat = False)

ax.legend()

plt.show()