import numpy as np
import timeit
import nonlinear_mass_faster as nlm
import scipy.optimize as op
from scipy.special import gamma
import math


# Load the power spectrum
pk =  np.loadtxt('pklin/default.txt')

# Specify your r-range
rs = np.linspace(0,5,2e5+1) # experiment with the number of points...

# output file
f = open('accuracy.txt','w')
f.write('# z     Accuracy\n')

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

# Fiducial rnl
rnl_fid = np.loadtxt('fiducial_rnl.txt')

om = 0.3096
delta_c = 1.686
delta_c_sq = delta_c * delta_c

for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)

	Rnl_actual = op.brentq(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, 0, 2*rnl_analytic)

	lnk = np.log(pk[:,0])
	Rnl_fourier = op.brentq(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0, 2*rnl_analytic)

	c0,c1,c2,c3 = nlm.get_poly_coeffs_taylor(0.3096, 0.04897, 0.9665, z)

	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)

	accuracy_fourier = np.abs(Rnl-Rnl_fourier)/Rnl_fourier
	
	c0,c1,c2,c3 = nlm.get_poly_coeffs(1.9*rnl_fid[j,1], interp_cf_lin)
	
	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)
	
	accuracy_fourier_full_method = np.abs(Rnl-Rnl_fourier)/Rnl_fourier


	f.write('%10.3f %10.5e %10.5e %10.5f\n' % (z, accuracy_fourier, accuracy_fourier_full_method, Rnl_fourier))
	f.flush()
f.close()

	
# Load the power spectrum
pk =  np.loadtxt('pklin/sig8_075.txt')

# Specify your r-range
rs = np.linspace(0,5,2e5+1) # experiment with the number of points...

# output file
f = open('accuracy_sig8_075.txt','w')
f.write('# z     Accuracy\n')

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

om = 0.3096

for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)
	
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)

	Rnl_actual = op.brentq(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, 0, 2*rnl_analytic)

	lnk = np.log(pk[:,0])
	Rnl_fourier = op.brentq(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0, 2*rnl_analytic)

	c0,c1,c2,c3 = nlm.get_poly_coeffs_taylor(0.3096, 0.04897, 0.9665, z)
	
	c0 = c0 * (0.75/0.81)**2.
	c1 = c1 * (0.75/0.81)**2.
	c2 = c2 * (0.75/0.81)**2.
	c3 = c3 * (0.75/0.81)**2.

	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)

	accuracy_fourier = np.abs(Rnl-Rnl_fourier)/Rnl_fourier
	
	c0,c1,c2,c3 = nlm.get_poly_coeffs(1.9*rnl_fid[j,1], interp_cf_lin)

	
	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)
	
	accuracy_fourier_full_method = np.abs(Rnl-Rnl_fourier)/Rnl_fourier


	f.write('%10.3f %10.5e %10.5e %10.5f\n' % (z, accuracy_fourier, accuracy_fourier_full_method,Rnl_fourier))
	f.flush()
	#print 5/0
f.close()


# Load the power spectrum
pk =  np.loadtxt('pklin/sig8_087.txt')

# Specify your r-range
rs = np.linspace(0,5,2e5+1) # experiment with the number of points...

# output file
f = open('accuracy_sig8_087.txt','w')
f.write('# z     Accuracy\n')

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

om = 0.3096

for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)
	
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)

	Rnl_actual = op.brentq(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, 0, 2*rnl_analytic)

	lnk = np.log(pk[:,0])
	Rnl_fourier = op.brentq(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0, 2*rnl_analytic)

	c0,c1,c2,c3 = nlm.get_poly_coeffs_taylor(0.3096, 0.04897, 0.9665, z)
	
	c0 = c0 * (0.87/0.81)**2.
	c1 = c1 * (0.87/0.81)**2.
	c2 = c2 * (0.87/0.81)**2.
	c3 = c3 * (0.87/0.81)**2.

	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)

	accuracy_fourier = np.abs(Rnl-Rnl_fourier)/Rnl_fourier
	
	c0,c1,c2,c3 = nlm.get_poly_coeffs(1.9*rnl_fid[j,1], interp_cf_lin)
	
	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)
	
	accuracy_fourier_full_method = np.abs(Rnl-Rnl_fourier)/Rnl_fourier


	f.write('%10.3f %10.5e %10.5e %10.5f\n' % (z, accuracy_fourier, accuracy_fourier_full_method,Rnl_fourier))
	f.flush()
f.close()

#print 5/0


# Load the power spectrum
pk =  np.loadtxt('pklin/random1.txt')

# Specify your r-range
rs = np.linspace(0,5,2e5+1) # experiment with the number of points...

# output file
f = open('accuracy_random1.txt','w')
f.write('# z     Accuracy\n')

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

om = 0.3129

for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)
	
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)


	lnk = np.log(pk[:,0])
	Rnl_fourier = op.brentq(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0, 2*rnl_analytic)

	c0,c1,c2,c3 = nlm.get_poly_coeffs_taylor(0.3129, 0.0490, 0.9669, z)

	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)

	accuracy_fourier = np.abs(Rnl-Rnl_fourier)/Rnl_fourier
	
	c0,c1,c2,c3 = nlm.get_poly_coeffs(1.9*rnl_fid[j,1], interp_cf_lin)
		
	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)
	
	accuracy_fourier_full_method = np.abs(Rnl-Rnl_fourier)/Rnl_fourier


	f.write('%10.3f %10.5e %10.5e %10.5f\n' % (z, accuracy_fourier, accuracy_fourier_full_method,Rnl_fourier))
	f.flush()
f.close()


# Load the power spectrum
pk =  np.loadtxt('pklin/random2.txt')

# Specify your r-range
rs = np.linspace(0,5,2e5+1) # experiment with the number of points...

# output file
f = open('accuracy_random2.txt','w')
f.write('# z     Accuracy\n')

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

om = 0.3185

for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)
	
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)

	Rnl_actual = op.brentq(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, 0, 2*rnl_analytic)

	lnk = np.log(pk[:,0])
	Rnl_fourier = op.brentq(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0, 2*rnl_analytic)

	c0,c1,c2,c3 = nlm.get_poly_coeffs_taylor(0.3185, 0.0498, 0.9676, z)

	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)

	accuracy_fourier = np.abs(Rnl-Rnl_fourier)/Rnl_fourier
	
	c0,c1,c2,c3 = nlm.get_poly_coeffs(1.9*rnl_fid[j,1], interp_cf_lin)
		
	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)
	
	accuracy_fourier_full_method = np.abs(Rnl-Rnl_fourier)/Rnl_fourier


	f.write('%10.3f %10.5e %10.5e %10.5f\n' % (z, accuracy_fourier, accuracy_fourier_full_method,Rnl_fourier))
	f.flush()
f.close()




# Load the power spectrum
pk =  np.loadtxt('pklin/random3.txt')

# Specify your r-range
rs = np.linspace(0,5,2e5+1) # experiment with the number of points...

# output file
f = open('accuracy_random3.txt','w')
f.write('# z     Accuracy\n')

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

om = 0.3146

for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)
	
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)

	Rnl_actual = op.brentq(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, 0, 2*rnl_analytic)

	lnk = np.log(pk[:,0])
	Rnl_fourier = op.brentq(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0, 2*rnl_analytic)

	c0,c1,c2,c3 = nlm.get_poly_coeffs_taylor(0.3146, 0.0479, 0.9616, z)

	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)

	accuracy_fourier = np.abs(Rnl-Rnl_fourier)/Rnl_fourier
	
	c0,c1,c2,c3 = nlm.get_poly_coeffs(1.9*rnl_fid[j,1], interp_cf_lin)
		
	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)
	
	accuracy_fourier_full_method = np.abs(Rnl-Rnl_fourier)/Rnl_fourier


	f.write('%10.3f %10.5e %10.5e %10.5f\n' % (z, accuracy_fourier, accuracy_fourier_full_method,Rnl_fourier))
	f.flush()
f.close()




# Load the power spectrum
pk =  np.loadtxt('pklin/random4.txt')

# Specify your r-range
rs = np.linspace(0,5,2e5+1) # experiment with the number of points...

# output file
f = open('accuracy_random4.txt','w')
f.write('# z     Accuracy\n')

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

om = 0.3086

for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)
	
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)

	Rnl_actual = op.brentq(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, 0, 2*rnl_analytic)

	lnk = np.log(pk[:,0])

	Rnl_fourier = op.brentq(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0, 2*rnl_analytic)

	c0,c1,c2,c3 = nlm.get_poly_coeffs_taylor(0.3086, 0.0507, 0.9591, z)

	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)

	accuracy_fourier = np.abs(Rnl-Rnl_fourier)/Rnl_fourier
	
	c0,c1,c2,c3 = nlm.get_poly_coeffs(1.9*rnl_fid[j,1], interp_cf_lin)
		
	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)
	
	accuracy_fourier_full_method = np.abs(Rnl-Rnl_fourier)/Rnl_fourier


	f.write('%10.3f %10.5e %10.5e %10.5f\n' % (z, accuracy_fourier, accuracy_fourier_full_method, Rnl_fourier))
	f.flush()
f.close()





# Load the power spectrum
pk =  np.loadtxt('pklin/random5.txt')

# Specify your r-range
rs = np.linspace(0,5,2e5+1) # experiment with the number of points...

# output file
f = open('accuracy_random5.txt','w')
f.write('# z     Accuracy\n')

# Compute the correlation function
cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
interp_cf_lin = lambda x: np.interp(x, rs, cflinr)

# Redshifts
zs = np.linspace(0,6,61)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)

# Get R_NL for a power-law cosmology to set the fitting range
# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
sigma2_norm = nlm.sigma2_prefactor(-2, norm)

om = 0.3004



for j,z in enumerate(zs):
	Dz = nlm.growth_factor(z, om)
	
	
	rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)


	Rnl_actual = op.brentq(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, 0, 2*rnl_analytic)

	lnk = np.log(pk[:,0])

	Rnl_fourier = op.brentq(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0, 2*rnl_analytic)

	c0,c1,c2,c3 = nlm.get_poly_coeffs_taylor(0.3004, 0.0504, 0.9663, z)


	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)


	accuracy_fourier = np.abs(Rnl-Rnl_fourier)/Rnl_fourier
	
	c0,c1,c2,c3 = nlm.get_poly_coeffs(1.9*rnl_fid[j,1], interp_cf_lin)
		
	Rnl = nlm.getR(c0,c1,c2,c3,delta_c*delta_c,Dz)


	
	accuracy_fourier_full_method = np.abs(Rnl-Rnl_fourier)/Rnl_fourier


	f.write('%10.3f %10.5e %10.5e %10.5f\n' % (z, accuracy_fourier, accuracy_fourier_full_method,Rnl_fourier))
	f.flush()
f.close()