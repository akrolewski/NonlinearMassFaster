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
#pk = np.loadtxt('planck18+bao_pklin_z0.txt')
pk =  np.loadtxt('pklin/default.txt')

# Cosmology
om = 0.3096

# Specify your r-range
rs = np.linspace(0,20,2e4+1) # experiment with the number of points...

# output file
f = open('timing_and_accuracy_19_09_16_py35_updated_numpy_scipy.txt','w')
f.write('z     PL time (s)      Fourier time (s)          Naive time (s)      Coeffs time (s)     Taylor time (s)        Cubic time (s)     Accuracy\n')

# Timing test stuff
ntest = 10000
#ntest = 10

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,50) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)

	def test():
		return op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - 1./Dz, 0.01, 8.0)
	t_analytic = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest
	rnl_analytic = test()


	# This is a bit faster than the other optimize methods
	def test():
		return nlm.newton(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - 1./Dz, rnl_analytic)
	t_naive_config_space = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest
	Rnl_actual = test()

	# This is a bit faster than the other optimize methods
	lnk = np.log(pk[:,0])
	def test():
		return nlm.newton(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - 1./Dz, rnl_analytic)
	Rnl_fourier_FULL = test()
	#t_naive_fourier_FULL = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest


	lnk = np.log(pk[::30,0])
	def test():
		return nlm.newton(lambda R: nlm.sigma_Pk(R,pk[::30,0],lnk,pk[::30,1]) - 1./Dz, rnl_analytic)
	Rnl_fourier30 = test()
	t_naive_fourier30 = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest


	lnk = np.log(pk[::120,0])
	def test():
		return nlm.newton(lambda R: nlm.sigma_Pk(R,pk[::120,0],lnk,pk[::120,1]) - 1./Dz, rnl_analytic)
	Rnl_fourier120 = test()
	#t_naive_fourier120 = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest


	def test():
		return nlm.get_poly_coeffs(2.*Rnl_actual, interp_cf_lin)
	c0,c1,c2,c3 = test()
	t_poly_coeffs = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest

	def test():
		return nlm.getR(c0,c1,c2,c3,Dz)
	Rnl = test()
	t_getR = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest

	print(c0, c1, c2, c3)
	def test():
		#return nlm.get_poly_coeffs_taylor(0.3096, 0.04897, 0.9665, z)
		return nlm.get_poly_coeffs_taylor(0.3196, 0.04997, 0.9765, z)
	c0,c1,c2,c3 = test()
	t_taylor = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest

	accuracy = np.abs(Rnl-Rnl_actual)/Rnl_actual
	f.write('%10.3f %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %12.10f %12.10f %12.10f %12.10f %12.10f\n' % (z, t_analytic, t_naive_fourier30, t_naive_config_space, t_poly_coeffs, t_taylor, t_getR, accuracy, Rnl, Rnl_actual, Rnl_fourier_FULL, Rnl_fourier30, Rnl_fourier120))
	f.flush()
	#print(5/0)
f.close()
	