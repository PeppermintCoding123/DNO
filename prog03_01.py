import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%%
class F_a:
    def __call__(u): 
        A = np.array([0, 1], [-1, 0])
        delta_u = A * u
        return delta_u

    def derivative():
        A = np.array([0, 1], [-1, 0])
        return A
    
class F_b:
    def __call__(u):
        A = np.array([0.5, -1],[1, -1])
        delta_u = A * u
        return delta_u

    def derivative():
        A = np.array([0.5, -1],[1, -1])
        return A

class F_c(u):
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
    for i in range (10):
        if xk != 0:
            xk1 = xk - 1/F_.derivative() * F_.__call__(xk)
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
    
    res.append(u_tk) # Step for tau = 0
    
    for t in range(tau, tau * nbr_Steps, tau):
        F_tk = F_.__call__(u_tk)
        F_tk *= tau
        u_tkPlut1 = u_tk + F_tk
        u_tk = u_tkPlut1
        res.append(u_tk)
          
    return res
        
#%%
def impliciteEuler(u0, tau, nbr_Steps, F):
    # TODO: do as newton
    pass
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


# %% 
'''TODO: write a written evaluation of the differances between explicite and implicite in .txt File, with the results of the given graph'''
