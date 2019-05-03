# source activate nbodykit-env on nersc
from nbodykit.cosmology import correlation
from nbodykit import cosmology
import numpy as np
from scipy.special import gamma
import scipy.optimize as op
import math
from scipy.integrate import romberg

# Checks accuracy of two approximations we make:
# 1. that the correlation function can be adequately approximated by a linear interpolation
# 2. that np.sum is accurate compared to romberg integration for the naive method


om0 = 0.3111
om0_err = 0.0056
redshift = 0

# I think the sigma8 here is a bit messed up...

c = cosmology.Cosmology(Omega0_cdm=om0-0.04897,h=0.6766,Omega0_b=0.04897,sigma8=0.8102,ns=0.9665,T0_cmb=2.7255)
Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
cf_lin = cosmology.CorrelationFunction(Plin)

R = np.linspace(0,20,2e4+1)
cflinr = cf_lin(R)
interp_cf_lin = lambda x: np.interp(x, R, cflinr)

Dzs = c.scale_independent_growth_factor(np.linspace(0,6,601))

def kernel(y):
	'''kernel that we integrate against'''
	return (3.-2.25 * y + 3.* y ** 3.0/16.)	

def sigma(R, y, dy, ysqkernel, interp_cf_lin):
	'''Directly do the integral for the "Daniel" method.
	Pre assign y and ysqkernel to save time in the optimization.
	We find that 100 sampling points give accuracy of 10^-3 as compared
	to Romberg'''
	return (dy * np.sum(interp_cf_lin(y*R) * ysqkernel))**0.5

def sigma_rom(R, interp_cf_lin):
	'''Romberg version of direct integral.  A factor of 50 slower
	than np.sum method, and not much more accurate'''
	return np.sqrt(romberg(lambda y: y * y * interp_cf_lin(y*R) * kernel(y), 0, 2))
	
j = 600
Dz = Dzs[j]

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100)
dy = y[1]-y[0]
ysqkernel = y * y * kernel(y)


# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = 71.38238397491413
sigma2_norm = -1.8 * norm/(np.pi * gamma(4.) * math.cos(-np.pi))

rnl_analytic = op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 8.0)
	
Rnl_interp = op.newton(lambda R: sigma(R,y,dy,ysqkernel,interp_cf_lin) - 1./Dz, rnl_analytic)
Rnl_actual_sum = op.newton(lambda R: sigma(R,y,dy,ysqkernel,cf_lin) - 1./Dz, rnl_analytic)
Rnl_actual_rom = op.newton(lambda R: sigma_rom(R,cf_lin) - 1./Dz, rnl_analytic)

print np.abs(Rnl_actual_rom-Rnl_actual_sum)/Rnl_actual_rom
print np.abs(Rnl_interp-Rnl_actual_sum)/Rnl_actual_sum