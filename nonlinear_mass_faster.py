import numpy as np
import math
from scipy import optimize as op
from scipy.special import gamma, spherical_jn
import time
from scipy.integrate import romberg
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import scipy as sp

# Load Taylor coefficients
all_file = np.fromfile('taylor_coeffs.bin').reshape((61,16))

def kernel(y):
	'''Kernel that we integrate against: K(y) (Eq. 8) in Krolewski & Slepian 2019'''
	return (3.-2.25 * y + 3.* y ** 3.0/16.)	

def solve(a, b, c, d):
	'''CUBIC ROOT SOLVER
	From : https://github.com/shril/CubicEquationSolver (Shril Kumar & Devojotyi Halder)
	plus accelerations and improvements due to Alex Krolewski & Zack Slepian.
	Solves a * x^3 + b * x^2 + c * x + d = 0 for x.'''
	
	#General principles: don't repeat calculations.
	#Dots after integers speed things up.
	#a * a is faster than a ** 2., but a **3. is faster than a * a * a [only true for scalars, for vectors a * a * a wins]
	#a ** 0.5 is better than math.sqrt(a) is better than np.sqrt(a)
	#-a is faster than -1 * a
	#np.array is very slow
	
	onethird = 1./3.
	a3 = 3.0 * a

	asq = a * a
	f = findF(a, b, c, asq)                          # Helper Temporary Variable
	g = findG(a, b, c, d, asq)                       # Helper Temporary Variable
	gsq = g * g
	h = findH(g, f, gsq)                             # Helper Temporary Variable

	if f == 0 and g == 0 and h == 0:            # All 3 Roots are Real and Equal
		if (d / a) >= 0:
			x = -((d / (a)) ** (onethird))
		else:
			x = (-d / (a)) ** (onethird)
		return x, x, x              # Returning Equal Roots as numpy array.

	elif h <= 0:                                # All 3 roots are Real
		i = ((0.25 * (gsq)) - h) ** 0.5  # Helper Temporary Variable
		j = i ** (onethird)                      # Helper Temporary Variable
		k = math.acos(-(g / (2. * i)))           # Helper Temporary Variable
		L = -j                              # Helper Temporary Variable
		M = math.cos(k / 3.0)
		N = (3. * (1. - M * M)) ** 0.5    # Helper Temporary Variable
		P = -b / a3                # Helper Temporary Variable

		LMP = L * M + P
		x1 = LMP - 3. * L * M
		LN = L * N
		x2 = LMP + LN
		x3 = LMP - LN

		return x1, x2, x3           # Returning Real Roots as numpy array.

	elif h > 0:                                 # One Real Root and two Complex Roots
		R = -(0.5 * g) + h ** 0.5           # Helper Temporary Variable
		if R >= 0:
			S = R ** (onethird)                  # Helper Temporary Variable
		else:
			S = -((-R) ** (onethird))          # Helper Temporary Variable
		T = -(0.5 * g) - h ** 0.5
		if T >= 0:
			U = (T ** (onethird))                # Helper Temporary Variable
		else:
			U = -(((-T) ** (onethird)))        # Helper Temporary Variable

		Spu = S + U
		Smu = S - U
		x1 = (Spu) - (b / (a3))
		#x2 = -(Spu) / 2. - (b / (a3)) + (Smu) * 3. ** 0.5 * 0.5j
		#x3 = -(Spu) / 2. - (b / (a3)) - (Smu) * 3. ** 0.5 * 0.5j

		#return x1, x2, x3           # only return the real root here
		return [x1]

def findF(a, b, c, asq):
	'''Helper function to return float value of f.'''
	return (c / a) - (((b * b) / (asq))) / 3.0

def findG(a, b, c, d, asq):
	'''Helper function to return float value of g.'''
	return ((2.0 * (b / a) ** 3.) - ((9.0 * b * c) / (asq)) + (27.0 * d / a)) /27.0

def findH(g, f, gsq):
	'''Helper function to return float value of h.'''
	return (0.25 * (gsq) + (f ** 3.0) / 27.0)

	
def get_poly_coeffs(Rmax, interp_cf_lin):
	'''Fits cubic polynomial coefficients to s^2 xi(s), where xi is the linear correlation function.
	Accepts as arguments Rmax (the maximum R out to which you want to fit)
	and interp_cf_lin, an interpolating function for s^2 xi(s).'''

	delta = Rmax / 999.
	r = np.arange(1000.) * delta
	rsq = r * r
	xmatT = np.zeros((4,1000))
	xmatT[0,:] = np.ones(1000)
	xmatT[1,:] = r
	xmatT[2,:] = rsq
	xmatT[3,:] = rsq * r
	xTy = np.dot(xmatT, rsq * interp_cf_lin(r))

	#use some low-level lapack routines to get a faster matrix inverse, taking
	#advantage of the fact that the matrix is symmetric. Source: 
	#https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
	return sp.linalg.lapack.dpotrs(sp.linalg.lapack.dpotrf(np.dot(xmatT,xmatT.T), False, False)[0],xTy)[0]

    
def nearest_neighbor(z, om0_file):
	''' Pulls out the coefficients from the nearest gridpoint (\Delta z= 0.1)
	with redshift less than or equal to input z.'''
	tenz = 10.*z
	lowind = int(tenz)
	#highind = lowind + 1
	om0_file_lowind = om0_file[lowind,:]
	return om0_file_lowind

def get_poly_coeffs_taylor(om0, ob0, ns, z):
	'''Gets polynomial coefficients in Taylor approximation, as a function of om0, ob0, and n_s,
	at some redshift z.
	Taylor expanded about the Planck18 cosmology, using the baryons + CDM power spectrum.'''
	om0_fid = 0.3096

	ob0_fid = 0.04897

	ns_fid = 0.9665
		
	taylor_all = nearest_neighbor(z, all_file)
    
	delta_om0 = om0 - om0_fid
	delta_ob0 = ob0 - ob0_fid
	delta_ns = ns - ns_fid
	
	c0 = (taylor_all[0] + taylor_all[1] * (delta_om0) + taylor_all[2] * (delta_ob0)
		 + taylor_all[3] * (delta_ns))
	c1 = (taylor_all[4] + taylor_all[5] * (delta_om0) + taylor_all[6] * (delta_ob0)
		 + taylor_all[7] * (delta_ns))
	c2 = (taylor_all[8] + taylor_all[9] * (delta_om0) + taylor_all[10] * (delta_ob0)
		 + taylor_all[11] * (delta_ns))
	c3 = (taylor_all[12] + taylor_all[13] * (delta_om0) + taylor_all[14] * (delta_ob0)
		 + taylor_all[15] * (delta_ns))		 
	
	return c0, c1, c2, c3
	

	 
def getR(c0,c1,c2,c3,deltac_sq,Dz):
	'''Given the polynomial coefficients for s^2 xi(s), solves
	for r at which \sigma_R D(z) = \delta_c, i.e. the nonlinear scale at redshift z.
	\delta_c is typically 1.686 but we allow the user to solve for a different scale
	(i.e. where \sigma = 1).'''
	a0 = 2.25 * c0
	a1 = 1.2 * c1
	a3 = (36./35.) * c3

	roots = solve(a3, c2 - deltac_sq/(Dz * Dz), a1, a0)
	#print roots
	# Pick the correct root.
	# roots only returns real roots--if there is only one, it's the right one, and move on
	# if not, check the first derivative at each root: if it's negative, then keep that root
	# and move on
	if len(roots) == 3:
		a02 = 2. * a0
		if  a3 - a1/(roots[0] * roots[0]) - a02 / (roots[0] ** 3.) < 0:
			return roots[0]
		elif a3 - a1/(roots[1] * roots[1]) - a02 / (roots[1] ** 3.) < 0:
			return roots[1]
		else:
			return roots[2]
	else:
		return roots

def fast_interpolate(z, om0_file):
	''' Fast interpolation of an array (om0_file) at redshift z.
	Spacing of the array is linear in z with \Delta z = 0.1 from z=0 to 6.'''
	tenz = 10.*z
	lowind = int(tenz)
	highind = lowind + 1
	om0_file_lowind = om0_file[lowind,1]
	if highind < np.shape(om0_file)[0]:
		return om0_file_lowind * (1. - (tenz - lowind))+ om0_file[highind,1] * (tenz - lowind)
	else:
		return om0_file_lowind

		
def get_rnl_fid(z):
	'''Gets fiducial RNL for setting the fitting range at redshift z.'''
	rnl_fid_file = np.loadtxt('fiducial_rnl.txt')
	return fast_interpolate(z, rnl_fid_file, 1)

def sigma_Pk(R, k, lnk, pklin, spacing='log10'):
	'''Computes sigma in Fourier space using the standard definition.
	Take as argument R, the radius that you desire, a grid of k,
	a grid of log k, and a grid of the linear power spectrum.'''
	if R <= 0:
		# Don't want nan values if R <= 0.
		# So set negative R to a small positive value
		# to ensure continuity
		R = 1e-6
	dlnk = lnk[1]-lnk[0]
	kr = k * R
	bess = (3. * spherical_jn(1, kr))/kr
	return (1./(2.* np.pi * np.pi) * np.trapz(k * k ** 2. * bess * bess * pklin, x=lnk)) ** 0.5
	
def sigma(R, y, dy, ysqkernel, interp_cf_lin):
	'''Directly do the integral for the Zehavi+05 method in real space.
	Pre assign y and ysqkernel to save time in the optimization.
	We find that 100 sampling points give accuracy of 10^-3 as compared
	to Romberg'''
	return (dy * np.sum(interp_cf_lin(y*R) * ysqkernel))**0.5


def pk_to_xi(k, pk, rs):
	'''Turns the power spectrum into a correlation function, given
	some user-supplied input power spectrum and k.'''
	dk = np.gradient(k)

	xi = np.zeros_like(rs)

	for i,r in enumerate(rs):
		#xi[i] = np.sum(pk * spherical_jn(0,k * r) * k * k * dk * np.exp(-( k/np.max(k))**2.))
		# Don't suppress with an exponential! Causes the fourier-space answers to be WRONG
		xi[i] = np.sum(pk * spherical_jn(0,k * r) * k * k * dk)
		
	return xi * 1./(2. * np.pi * np.pi)
	
def growth_factor(zz,om,**kwargs):
	"""Approximate the growth factor, normalized to 1 today.
	Use the form f = Omega_m^{\gamma}. By default, gamma = 0.55.  One can change
	this either by passing a different value for gamma, or by passing a value for w that is not -1,
	and then gamma = 3 (w-1)/(6w-5)."""
	if len(kwargs.keys()) > 1:
		raise TypeError("growth factor expected at most 3 arguments, redshift, OmegaM(z=0), and either w or gamma")
	elif len(kwargs.keys()) == 1:
		if 'gamma' not in kwargs.keys():
			gamma = 3 * (kwargs['w']-1) /(6 * kwargs['w']-5)
		else:
			gamma = kwargs['gamma']	
	elif len(kwargs.keys()) == 0:
		gamma = 0.55
	afid = 1.0/(1.0+zz)
	zval = 1/np.logspace(np.log10(afid),0.0,100)-1.0
	omz = om * (1+zval)**3. / (om * (1+zval)**3. + 1 - om)
	Dz   = np.exp(-np.trapz(omz**gamma,x=np.log(1/(1+zval))))
	return(Dz)
    
def sigma2_prefactor(n,norm):
	""" for power-law solution to sigma^2, from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb"""
	return norm*9*2**(-n - 1)/(np.pi**2*(n - 3)) * -(1+n)*np.pi/(2*gamma(2-n)*np.cos(np.pi*n/2))

def sigma_pl(sigma2_norm, n, R):
	""" power law solution to sigma^2"""
	return sigma2_norm ** 0.5 * R ** (0.5 * (-n-3))