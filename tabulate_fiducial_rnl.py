import numpy as np
import matplotlib.pyplot as plt
import timeit
import nonlinear_mass_faster as nlm
import scipy.optimize as op
from scipy.special import gamma
import math

# Plot settings
plt.ion()
plt.show()

# Load the power spectrum
pk_fiducial = np.loadtxt('pklin/default.txt')


# Cosmology
om_fid = 0.3096
delta_c = 1.686

# Specify your r-range
rs = np.linspace(0,5,4e4+1) # experiment with the number of points...


# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Compute the correlation function
cflinr_fid = nlm.pk_to_xi(pk_fiducial[:,0], pk_fiducial[:,1], rs)
interp_cf_lin_fid = lambda x: np.interp(x, rs, cflinr_fid)


# Redshifts
zs = np.linspace(0,6,61)


# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk_fiducial[:,0], pk_fiducial[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)


f = open('fiducial_rnl.txt','w')
f.write('# z Rnl_fid\n')
for j,z in enumerate(zs):
	#j = 1
	#z = zs[j]
	Dz_fid = nlm.growth_factor(z, om_fid)
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz_fid, 0.01, 8.0)
	
	if z > 5 or rnl_analytic < 0:
		start = 0.01
	else:
		start = rnl_analytic
		
	# This is a bit faster than the other optimize methods
	#try:
	Rnl_fid = op.brentq(lambda R: nlm.sigma_Pk(R,pk_fiducial[:,0],np.log(pk_fiducial[:,0]),pk_fiducial[:,1]) - delta_c/Dz_fid, 0, 5)

	f.write('%.2f %15.10f\n' % (z, Rnl_fid))

f.close()	