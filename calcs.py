import math
import numpy as np
import random
import scipy as sp

def correlation_path(T):
    t = 1/252
    rho = [0.7]
    for i in range(T-1):
        rho.append(rho[-1]+math.sqrt(t)*random.gauss(0,1))
    return rho
    
def jumps():
    rate, T = 0.2, 504
    n = sp.stats.poisson(rate*T).rvs()
    times = np.sort(np.random.uniform(0,T,size=n))
    jumptimes = [math.ceil(i) for i in times]
    return jumptimes

gamma, m, alpha, beta, rho = 1, 2, 0.01, 0.1, correlation_path()
alphaQ = alpha + beta*rho*gamma
jumptimes = jumps()

def jump(x,t):
    if t+1 not in jumptimes:
        return 0
    else:
        return J(x)

def next_step_P(x,t,v,rho):
    Js = jump(x,t)
    Jv = jump(v,t)
    W1,W2 = math.sqrt(1/252)*np.random.normal(size=2)
    Ws = W1
    Wv = rho[t]*W1 + math.sqrt(1-rho[t]*rho[t])*W2
    x_new = x + gamma*v*v*x*(1/252) + v*x*Ws + Js
    v_new = v + (m-alpha*v)*(1/252) + beta*Wv + Jv
    return [x_new, v_new]

def next_step_Q(x,t,v,rho):
    Js = jump(x,t)
    Jv = jump(v,t)
    W1,W2 = math.sqrt(1/252)*np.random.normal(size=2)
    Ws = W1
    Wv = rho[t]*W1 + math.sqrt(1-rho[t]*rho[t])*W2
    x_new = v*x*Ws + Js
    v_new = v + (m-alphaQ*v)*(1/252) + beta*Wv + Jv
    return [x_new, v_new]
    
def stock_sim_P(S,T,v):
    x = [[S,v]]
    for i in range(T):
        x.append(next_step_P(x[-1][0], i, x[-1][1]))
    return x

def stock_sim_Q(S,T,v):
    x = [[S,v]]
    for i in range(T):
        x.append(next_step_Q(x[-1][0], i, x[-1][1]))
    return x