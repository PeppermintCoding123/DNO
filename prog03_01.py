import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%%
# f is the function of delta u. u1 and u2 should be called at position t and then handed over. Thus far, just the explisite Implementation.
def f_explicite(aufgabe, t, u):    
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
        pass
    raise ValueError('unknown aufgabe given')

# %%
# u0 ist anfangswert, tau ist schrittweite, anzahl Schritte, F ist u'(t)
def explisiteEuler(u0, tau, nbr_Steps, aufgabe):
    res = []
    u_tk = u0
    res.append(u_tk) # Step for tau = 0
    
    for t in range(tau, tau * nbr_Steps, tau):
        F_tk = f_explicite(aufgabe, t, u_tk)
        F_tk *= tau
        u_tkPlut1 = u_tk + F_tk
        
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