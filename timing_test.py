import numpy as np
import nonlinear_mass_faster as nlm
import scipy.optimize as op
from scipy.special import gamma
import math
import timeit


# Load the power spectrum
pk =  np.loadtxt('pklin/default.txt')

# Cosmology
om = 0.3096

# Specify your r-range
rs = np.linspace(0,5 ,2e4+1) # experiment with the number of points...

# output file
f = open('timing.txt','w')
f.write('#z     PL time (s)      Fourier time (s)          Naive time (s)      Coeffs time (s)     Taylor time (s)        Cubic time (s)     Accuracy\n')

# NL scale defined in terms of deltac, threshold for nonlinear collapse
deltac = 1.686

# Timing test stuff
ntest = 1000

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
	#z = 0.0
	Dz = nlm.growth_factor(z, om)

	def test():
		return op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - deltac/Dz, 0.01, 8.0)
	t_analytic = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest
	rnl_analytic = test()


	# This is a bit faster than the other optimize methods
	def test():
		return op.brentq(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - deltac/Dz, 0.0, 5.0)
	a = test()
	
	t_naive_config_space = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest
	print(j, t_naive_config_space)
	Rnl_actual = test()
	

	# This is a bit faster than the other optimize methods
	lnk = np.log(pk[::30,0])
	def test():
		return op.brentq(lambda R: nlm.sigma_Pk(R,pk[::30,0],lnk,pk[::30,1]) - deltac/Dz, 0.0, 5.0)
	Rnl_fourier30 = test()
	t_naive_fourier30 = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest

	def test():
		return nlm.get_poly_coeffs(2.*Rnl_actual, interp_cf_lin)
	c0,c1,c2,c3 = test()
	t_poly_coeffs = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest

	def test():
		return nlm.getR(c0,c1,c2,c3,deltac,Dz)
	Rnl = test()
	t_getR = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest

	print(c0, c1, c2, c3)
	def test():
		#return nlm.get_poly_coeffs_taylor(0.3096, 0.04897, 0.9665, z)
		return nlm.get_poly_coeffs_taylor(0.3196, 0.04997, 0.9765, z)
	c0,c1,c2,c3 = test()
	t_taylor = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest

	#accuracy = np.abs(Rnl-Rnl_actual)/Rnl_actual
	f.write('%10.3f  %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e \n' % (z, t_analytic, t_naive_fourier30,  t_naive_config_space,t_poly_coeffs, t_taylor, t_getR))
	f.flush()
	#print(5/0)
	#break
f.close()
	