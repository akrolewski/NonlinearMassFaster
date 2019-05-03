import numpy as np
import matplotlib.pyplot as plt
import time
from nonlinear_mass_faster import *

plt.ion()
plt.show()


dat = np.loadtxt('cosmo_files/default/cfz0_2e4.txt')

Dzs = np.loadtxt('cosmo_files/default/Dz.txt')

R = dat[:,0]
cflinr = dat[:,1]
interp_cf_lin = lambda x: np.interp(x, R, cflinr)

	

js = [0, 100, 200, 300, 400, 500, 600]

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) #make this more efficient!
dy = y[1]-y[0]
ysqkernel = y * y * kernel(y)

# Pre-load some matrices for polynomial coefficient fitting
Rsq = R * R
Y = Rsq * cflinr
xmatT_all = np.array([np.ones_like(R), R, R * R, Rsq * R])

# Need to add these constant factors to the timing test...

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = 71.38238397491413
sigma2_norm = -1.8 * norm/(np.pi * gamma(4.) * math.cos(-np.pi))


for j in js:
	Dz = Dzs[j]
	
	rnl_analytic = op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 8.0)
	%timeit op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 8.0)

	# This is a bit faster than the other optimize methods
	%timeit op.newton(lambda R: sigma(R,y,dy,ysqkernel,interp_cf_lin) - 1./Dz, rnl_analytic)
	Rnl_actual = op.newton(lambda R: sigma(R,y,dy,ysqkernel,interp_cf_lin) - 1./Dz, rnl_analytic)


	c0,c1,c2,c3 = get_poly_coeffs(2.*rnl_analytic, interp_cf_lin)
	print c0, c1, c2, c3
	%timeit get_poly_coeffs(2.*rnl_analytic, interp_cf_lin)

	%timeit getR(c0,c1,c2,c3,Dz)
	Rnl = getR(c0,c1,c2,c3,Dz)

	print j, np.abs(Rnl-Rnl_actual)/Rnl_actual