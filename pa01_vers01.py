import numpy as np
import matplotlib.pyplot as plt

#%% implement as lamda functions

f = lambda x: (x[0] + x[1]) ** 2 + (np.sin(x[1]) * 3)
f_derivative = lambda x: np.array([(x[0] + x[1]) * 2, (x[0] + x[1]) * 2 + 3 * np.cos(x[1])])

#%%
#x0 = startvalue, H0 = initial Hessematrixapprox, a0 = initial stepwith, o = adaptive factor for a0, e = approx-closeness
def BFGS(x0, H0, a0, o, e):
    '''TODO: 
        - Implement BFSG from Lecture
        - use adaptive stepps for ak (can be resett on every step)
    '''
    """ Notes from Lecture:
        sk = xk+1 - xk
        yk = delta_f(xk+1) - delta_f(xk)
        Hk+1 * yk = sk                      --- ist vertauschtes mit dem was an der Tafel stand? (5)
        
    strategy:
        xk+1 = xk - Hk * f1_derivative
    """
    ak = a0
    xk = x0
    fk = np.inf # fk is always the f of the prefious xk
    fk_plus1 = f(x0) # the f call on the current xk
    
    #TODO: find out how to calculate this without xk+1:
    sk = 0
    yk = 0
    
    rohk = 1/(np.dot(yk,sk))
    Hk = H0
    
    res = []
    res.append(np.hstack((xk,fk)))

    while np.linalg.norm(fk_plus1 - fk) > e:
        #df = f_derivative(xk) # --- Do we even need this?
        
        #calculate Hk+1
        rys = rohk * yk * sk.T
        rsy = rohk * sk * yk.T
        Hk1 = Hk - (Hk * rys) - (rsy * Hk) + (rsy * Hk * rys) + (rohk * sk * sk.T)
        
        # adjust ak to take step
        delta_xk = ak*Hk1
        while (f(xk - delta_xk) - fk) > e: # --- Use norm here ?
            ak *= o
            delta_xk =  ak*Hk1
        
        #prepare for next step
        # TODO: adjust sk & yk
        sk = sk
        yk = yk
        
        xk = xk - delta_xk
        fk = np.copy(fk_plus1)
        fk_plus1 = f(xk)
        res.append(np.hstack((xk,fk_plus1)))

    return np.array(res)
    
    
#%%
'''TODO:
    - test BFSG with startingvalues of x0 = [-5,-5], a0 = 1, H0 = 1n
    
    PLOT:
        - pathing with quiver
        - in 'Konturenplot' 
        - presision: e-6'''
    
