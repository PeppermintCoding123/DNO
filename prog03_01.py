import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%%
# f is the function of delta u. u should be called at position tk and then handed over. Thus far, just the explisite Implementation.
def f_explicite(aufgabe, u):    
    if aufgabe == 'a':
        A = np.array([0, 1], [-1, 0])
        delta_u = A * u
        return delta_u
    if aufgabe == 'b':
        A = np.array([0.5, -1],[1, -1])
        delta_u = A * u
        return delta_u
    if aufgabe == 'c':
        # is it nessasary to do errorhandeling for the case that u2 is smaller than 
        delta_u_up = np.sqt(u[1])
        delta_u_down = -2 * u[0] * delta_u_up
        delta_u = np.array([[delta_u_up],[delta_u_down]])
        return delta_u
    raise ValueError('unknown aufgabe given')

def f_implicite(aufgabe, tau):
    # this should be multiplied to u(tk), not added as in the explicit case
    # TODO: actually calculate by hand and check if my calculations are correct...
    if aufgabe == 'a':
        A = np.array([1, tau], [-tau, 1])
        A = 1/(1 + tau**2) * A
        return A
    if aufgabe == 'b':
        # TODO:
        A = np.array([2 * (tau + 1), -2 * tau],[2 * tau, 2 - tau])
        factor = 1/(tau**2 + tau + 2)
        A = factor * A
        return A
    if aufgabe == 'c':
        # TODO: not jet finished with calculations
        pass
        
    raise ValueError('unknown aufgabe given')
# %%
# u0 ist anfangswert, tau ist schrittweite, anzahl Schritte, F ist u'(t)
def explisiteEuler(u0, tau, nbr_Steps, aufgabe):
    res = []
    u_tk = u0
    if aufgabe == 'a' or aufgabe == 'b':
        res.append(u_tk) # Step for tau = 0
    
        for t in range(tau, tau * nbr_Steps, tau):
            F_tk = f_explicite(aufgabe, t, u_tk)
            F_tk *= tau
            u_tkPlut1 = u_tk + F_tk
        
            u_tk = u_tkPlut1
            res.append(u_tk)
        
        return res
    elif aufgabe == 'c':
        pass
    else:
        raise ValueError('aufgabe is unknown')

def impliciteEuler(u0, tau, nbr_Steps, aufgabe):
    res = []
    u_tk = u0
    res.append(u_tk) # Step for tau = 0
    
    factor = f_implicite(aufgabe, tau)
    
    for t in range(tau, tau * nbr_Steps, tau):
        u_tkPlut1 = u_tk * factor
        
        u_tk = u_tkPlut1
        res.append(u_tk)
        
    return res
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
