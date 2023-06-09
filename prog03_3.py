

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

#%%
class F_a:
    def __call__(u): 
        A = np.array([[0, 1], [-1, 0]])
        delta_u = A @ u
        return delta_u

    def derivative():
        A = np.array([[0, 1], [-1, 0]])
        return A
    
class F_b:
    def __call__(u):
        A = np.array([[0.5, -1],[1, -1]])
        delta_u = A @ u
        return delta_u

    def derivative():
        A = np.array([[0.5, -1],[1, -1]])
        return A

class F_c:
    def __call__(u):
        delta_u_up = np.sqt(u[1])
        delta_u_down = -2 * u[0] * delta_u_up
        delta_u = np.array([[delta_u_up],[delta_u_down]])
        return delta_u
    
    def derivative():
        #TODO: is stil in calculation
        pass

def newton(F, x0):
    F_ = F
    xk = x0
    Fd = F_.derivative()
    Fd_invers = (np.linalg.solve(Fd, np.eye(Fd[0].size)))
    for i in range (10):
        if np.linalg.norm(xk) != 0:
            xk1 = xk - (Fd_invers * F_.__call__(xk))
            xk = xk1
        else:
            break
    return xk
# %%
# u0 ist anfangswert, tau ist schrittweite, anzahl Schritte, F ist u'(t)
def explisiteEuler(u0, tau, nbr_Steps, F):
    res = []
    u_tk = u0
    F_ = F
    
    u_tk_x = u_tk[0]
    
    res.append(u_tk) # Step for tau = 0
    
    for t in range(1, nbr_Steps):
        u_tkPlus1_x = u_tk_x + tau
        F_tk = F_.__call__(u_tk)
        Ftk = F_tk * tau
        u_tkPlus1 = (u_tk + Ftk)
        u_tkPlus1_y = u_tkPlus1[1]
        
        # Prepare for next step 
        u_tk_x = u_tkPlus1_x
        u_tk = u_tkPlus1
        res.append(np.hstack((u_tkPlus1_x, u_tkPlus1_y)))
          
    return np.array(res)
        
#%%
def impliciteEuler(u0, tau, nbr_Steps, F):
    # reformulate the problem as a fiffarent equasion, with x = utkplus1, u = utk
    
    class G:
        def __call__(u, x, tau, F):
            return x - tau * F.__call__(x) - u
        def derivative():
            pass
    
    res = []
    u_tk = u0
    
    u_tk_x = u_tk[0]
    
    res.append(u_tk) # Step for tau = 0
    
    for t in range(1, nbr_Steps):
        u_tkPlus1_x  = u_tk_x + tau
        u_tkPlus1 = u_tk
        for i in range(10): # with newton we solve the equation of G, therefore finding the x, that is closest to u_ktplus1
            u_tkPlus1 = newton(G, u_tkPlus1)
            
        res.append(np.hstack((u_tkPlus1_x, u_tkPlus1[1])))
        
        # Prepare for next step
        u_tk = u_tkPlus1
        u_tk_x = u_tkPlus1_x
        
    return np.array(res)
#%% 
''' TODO: Implement Implicite 
    - probably implement a function like f_explicite and do the strange turn-around calculations there

'''
# %% 
'''TODO: Implement viualisation
    - for each aufgabe: 
        - calculate the analytical function of 'a' & 'b'
        - plot the explicite and implicite in one graph
        - also have the analytical cunction in the same graph.
        
        '''
# choose exact exersise to display from ('a', 'b', 'c')
chosen_aufgabe = 'a'


#%% visualisation: 
plt.close('all')

if chosen_aufgabe == 'a':
    u0 = np.array([0,1])
    tau = 0.1
    nbr_Steps = 100
    F = F_a
    
    x = []
    y = []
    res = explisiteEuler(u0, tau, nbr_Steps, F)

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
     
    animation = FuncAnimation(figure, animation_function, frames = np.arange(0,100,1), repeat = False)

    plt.show()

# %% 
'''TODO: write a written evaluation of the differances between explicite and implicite in .txt File, with the results of the given graph'''
