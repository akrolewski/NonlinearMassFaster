import numpy as np
import matplotlib.pyplot as plt
import time
from nonlinear_mass_faster import *

# This script checks whether the repeated timing tests
# are screwed up by caching: are any of the timing results significantly
# different after the first iteration? By plotting the time per iteration
# for just the first few iterations, we can see that there is no such dependence
# of timing on iteration number and thus using %timeit is justified.

plt.ion()
plt.show()

dat = np.loadtxt('cfz0_4e3.txt')

Dzs = np.loadtxt('Dz.txt')

R = dat[:,0]
cflinr = dat[:,1]
interp_cf_lin = lambda x: np.interp(x, R, cflinr)

	

js = [0, 100, 200, 300, 400, 500, 600]

#for j in js:
j = 30
Dz = Dzs[j]
# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100)
dy = y[1]-y[0]
ysqkernel = y * y * kernel(y)
ts = np.zeros(1000)
for i in range(1000):
	t0 = time.time()
	Rnl_actual = op.brentq(lambda R: sigma(R,y,dy,ysqkernel,interp_cf_lin) - 1./Dz, 0.0, 8.0)
	ts[i] = time.time()-t0
#%timeit op.brentq(lambda R: sigma(R,y,dy,ysqkernel) - 1./Dz, 0.0, 8.0)
#Rnl_actual = op.brentq(lambda R: sigma(R,y,dy,ysqkernel) - 1./Dz, 0.0, 8.0)
# can we do better than brentq on the optimizer?

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = 71.38238397491413
sigma2_norm = -1.8 * norm/(np.pi * gamma(4.) * math.cos(-np.pi))
ts2 = np.zeros(1000)
for i in range(1000):
	t0 = time.time()
	rnl_analytic = op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 10.0)
	ts2[i] = time.time()-t0
#%timeit op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 10.0)

# Pre-load some matrices for polynomial coefficient fitting
Rsq = R * R
Y = Rsq * cflinr
xmatT_all = np.array([np.ones_like(R), R, R * R, Rsq * R])
ts3 = np.zeros(1000)
for i in range(1000):
	t0 = time.time()
	c0,c1,c2,c3 = get_poly_coeffs(2.*rnl_analytic, xmatT_all, Y)
	ts3[i] = time.time()-t0
#%timeit get_poly_coeffs(2.*rnl_analytic, xmatT_all, Y)

#%timeit getR(c0,c1,c2,c3,Dz)
ts4 = np.zeros(1000)
for i in range(1000):
	t0 = time.time()
	Rnl = getR(c0,c1,c2,c3,Dz)
	ts4[i] = time.time()-t0
	
print j, np.abs(Rnl-Rnl_actual)/Rnl_actual

plt.figure()
plt.plot(ts,color='b',label='Naive method')
plt.plot(ts2,color='r',label='Get rmax')
plt.plot(ts3,color='g',label='Find coefficients')
plt.plot(ts4,color='k',label='Get r')
plt.xlim(0,10)
plt.legend(frameon=False)
plt.xlabel('Iteration',size=20)
plt.ylabel('Time (s)',size=20)