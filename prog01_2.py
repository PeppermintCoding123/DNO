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
    # initialize iterations
    xk = x0
    Hk = H0
    fk = f(xk)
    fk_plus1 = np.inf
    '''xk_plus1 = xk + ak*(-Hk)*f_derivative(xk)
    fk = f(xk) # fk is always the f of the prefious xk
    fk_plus1 = f(xk_plus1) # the f call on the current xk
    '''
    res = []
    res.append(np.hstack((xk,fk)))
    '''res.append(np.hstack((xk_plus1,fk_plus1)))
    '''
    while np.linalg.norm(fk_plus1 - fk) > e:
        ak = a0
        fk = f(xk)
        # adjust ak to take step
        dfk = f_derivative(xk)
        
        #xk = xk_plus1
        #fk = fk_plus1
        xk_plus1 = xk + ak*(-Hk)*dfk
        fk_plus1 = f(xk_plus1)
        
        while fk_plus1 >= fk:
                ak *= o
                xk_plus1 = xk + ak*(-Hk)*dfk
                fk_plus1 = f(xk_plus1)
                
        res.append(np.hstack((xk_plus1,fk_plus1)))
        
        # calculate Hk+1
        sk = xk_plus1 - xk
        yk = f_derivative(xk_plus1) - f_derivative(xk)
        rohk = 1/(np.dot(yk,sk))
        
        rys = rohk * yk * sk.T
        rsy = rohk * sk * yk.T
        Hk = Hk - (Hk * rys) - (rsy * Hk) + (rsy * Hk * rys) + (rohk * sk * sk.T)
        
        # move for next iteration
        xk = np.copy(xk_plus1)
        
    return np.array(res)
    
    
#%%
'''TODO:
    - test BFSG with startingvalues of x0 = [-5,-5], a0 = 1, H0 = 1n
    o = 0.5
    
    PLOT:
        - pathing with quiver
        - in 'Konturenplot' 
        - presision: e-6
    Kommandozeile: 
        - number of iterations = print(res.T[0].size - 1)
        - x_sternchen (letztes xk_plus1)'''

    
    
