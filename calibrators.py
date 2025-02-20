import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import Bounds, differential_evolution, minimize

### HESTON ###
# Calculates the characteristic function of log(S(t)) in parameter w
def CharFunc(S0, v0, vbar, kappa, zeta, r, rho, T, w):
  alpha = - ((w ** 2) / 2.0) - ((1j * w) / 2.0)
  beta = kappa - rho * zeta * 1j * w
  gamma = (zeta ** 2) / 2.0
  h = np.sqrt(beta ** 2 - 4 * alpha * gamma)
  rplus = (beta + h) / (zeta ** 2)
  rmin = (beta - h) / (zeta ** 2)
  g = rmin / rplus
  C = kappa * (rmin * T - (2/zeta**2)*np.log( (1 - g * np.exp(-h*T))/(1-g) ))
  D = rmin * ( (1-np.exp(-h*T))/(1-g*np.exp(-h*T)) )
  return np.exp(C * vbar + D * v0 + 1j * w * np.log(S0 * np.exp(r*T)))

# Calculates price of European vanilla call option
def H_EuroCall(S,K,T,v,kappa,theta,sigma,rho):
  S0, v0, kappa, vbar, zeta, r, rho = S, v, kappa, theta, sigma, 0, rho
  cf = lambda w: CharFunc(S0, v0, vbar, kappa, zeta, r, rho, T, w)
  i1 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w - 1j)) / (1j * w * cf(- 1j)))
  I1 = quad(i1, 0, np.inf)
  Pi1 = 0.5 + I1[0]/np.pi 
  i2 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w)) / (1j * w))
  I2 = quad(i2, 0, np.inf)
  Pi2 = 0.5 + I2[0]/np.pi
  return S0 * Pi1 - K * np.exp(-r*T) * Pi2

def calibrate_H(strikes, maturities, prices,S):
  K = strikes
  T = maturities
  P = prices
  def ObjectiveFunction(theta, S0, r, K=K, T=T, P=P):
    sum = 0
    for i in range(0,len(K)):
      for j in range(0,len(T)):
        p = H_EuroCall(S0, K[i], T[j], theta[3], theta[1], theta[0], theta[2], r, theta[4])
        sum += np.abs(P[i] - p)
    return sum #Sum of the difference between the model and sample prices to minimise

  S0 = S
  r = 0
  # Parameter domains and iteration limit
  imax = 100
  lb = [0.1,0,0.1,0,-1]
  ub = [10,1,10,1,0]

  # Minimising function yielding the 5 parameters
  result = differential_evolution(ObjectiveFunction, Bounds(lb,ub), args=(S0, r), disp=True, maxiter=imax)
  theta = result.x
  return theta


### BLACK SCHOLES ###
# Calculates price of European vanilla call option
def EuroCall(S0, K, T, r, sigma):
  d1 = (np.log(S0/K) + T * (r + (sigma ** 2)/2))/(sigma * np.sqrt(T))
  d2 = (np.log(S0/K) + T * (r - (sigma ** 2)/2))/(sigma * np.sqrt(T))
  return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Finds optimal value for sigma across 100 iterations
def bisection(S0, K, T, p, r=0, a=0, b=1, imax=100):
  x=0
  for _ in range(0, imax):
    x+=1
    mid = (a + b)/2
    BS = EuroCall(S0, K, T, r, mid)
    if (abs(BS - p) < 0.01) or x==imax:
      return mid
    elif (BS < p):
      a = mid
    elif (BS >= p):
      b = mid

def calibrate_BS(option, S): #option = [price, maturity, strike]
    vol = bisection(S, option[2], option[1]/252, option[0])
    return vol