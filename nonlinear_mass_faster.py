# source activate nbodykit-env on nersc
#from nbodykit.cosmology import correlation
#from nbodykit import cosmology
import numpy as np
import math
from scipy import optimize as op
from scipy.special import gamma
import time
from scipy.integrate import romberg
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import scipy as sp

dat = np.loadtxt('Desktop/cfz0_4e3.txt')
R = dat[:,0]
cflinr = dat[:,1]
interp_cf_lin = lambda x: np.interp(x, R, cflinr)

def solve(a, b, c, d):
	'''CUBIC ROOT SOLVER

	Date Created   :    24.05.2017
	Created by     :    Shril Kumar [(shril.iitdhn@gmail.com),(github.com/shril)] &
						Devojoyti Halder [(devjyoti.itachi@gmail.com),(github.com/devojoyti)]

	Project        :    Classified 
	Use Case       :    Instead of using standard numpy.roots() method for finding roots,
						we have implemented our own algorithm which is ~10x faster than
						in-built method.

	Algorithm Link :    www.1728.org/cubic2.htm

	This script (Cubic Equation Solver) is an indipendent program for computation of cubic equations. This script, however, has no
	relation with original project code or calculations. It is to be also made clear that no knowledge of it's original project 
	is included or used to device this script. This script is complete freeware developed by above signed users, and may furthur be
	used or modified for commercial or non - commercial purpose.
	From : https://github.com/shril/CubicEquationSolver
	plus accelerations and improvements due to AK and ZS.
	General principles: don't repeat calculations.
	Dots after integers speed things up.
	a * a is faster than a ** 2., but a **3. is faster than a * a * a [only true for scalars, for vectors a * a * a wins]
	a ** 0.5 is better than math.sqrt(a) is better than np.sqrt(a)
	-a is faster than -1 * a
	np.array is very slow'''
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
		return x1

def findF(a, b, c, asq):
	'''Helper function to return float value of f.'''
	return (c / a) - (((b * b) / (asq))) / 3.0

def findG(a, b, c, d, asq):
	'''Helper function to return float value of g.'''
	return ((2.0 * (b / a) ** 3.) - ((9.0 * b * c) / (asq)) + (27.0 * d / a)) /27.0

def findH(g, f, gsq):
	'''Helper function to return float value of h.'''
	return (0.25 * (gsq) + (f ** 3.0) / 27.0)

def kernel(y):
	'''kernel that we integrate against'''
	return (3.-2.25 * y + 3.* y ** 3.0/16.)	

def fastinv(mat):
	'''use some low-level lapack routines to get a faster matrix inverse, taking
	advantage of the fact that the matrix is symmetric. Source: 
	https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi'''
	step1, _ = sp.linalg.lapack.dpotri(sp.linalg.lapack.dpotrf(mat, False, False)[0])
	# lapack only returns the upper or lower triangular part
	# inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T
	# turns out that np.triu is ridiculously slow
	# so we just explicitly write out the matrix
	# leaving it as a list of lists is faster
	# than calling np.array() [array incurs a large overhead]
	# we also tried scipy.linalg.solve instead of inverting the matrix, but this was no help
	# and we tried explicitly writing down the determinant of the 4by4 matrix
	# (taking advantage of a couple additional symmetries beyond symmetric,
	# bringing down #coeffs to 7)
	# but again this wasn't any faster--seems that the vectorization/lapack magic
	# helps more than just cutting down on the raw# of calculations
	inv_M = [step1[0,:],
		[step1[0,1], step1[1,1], step1[1,2], step1[1,3]],
		[step1[0,2], step1[1,2], step1[2,2], step1[2,3]],
		step1[:,3]]
	return inv_M
	
def get_poly_coeffs(Rmax, xmatT_all, Y):
	'''gets polynomial coefficients given some rmax'''
	ind = int(Rmax * 200.) + 1
	# precompute xmatT to speed things up
	xmatT = xmatT_all[:,:ind]
	inv= fastinv(np.dot(xmatT, xmatT.T))
	xTy = np.dot(xmatT, Y[:ind])

	# a bit faster to do this because then I don't have to convert stuff to np.array
	return (np.dot(inv[0],xTy),
	 inv[1][0] * xTy[0] + inv[1][1] * xTy[1] + inv[1][2] * xTy[2] + inv[1][3] * xTy[3],
	 inv[2][0] * xTy[0] + inv[2][1] * xTy[1] + inv[2][2] * xTy[2] + inv[2][3] * xTy[3],
	 np.dot(inv[3],xTy))
	 
def getR(c0,c1,c2,c3,Dz):
	'''Finds R'''
	a0 = 2.25 * c0
	a1 = 1.2 * c1
	a3 = (36./35.) * c3

	roots = solve(a3, c2 - 1./(Dz * Dz), a1, a0)
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

	
def sigma(R,y,dy,ysqkernel):
	'''Directly do the integral for the "Daniel" method.
	Pre assign y and ysqkernel to save time in the optimization.
	We find that 100 sampling points give accuracy of 10^-3 as compared
	to Romberg'''
	return (dy * np.sum(interp_cf_lin(y*R) * ysqkernel))**0.5

def sigma_rom(R):
	'''Romberg version of direct integral.  A factor of 50 slower
	than np.sum method, and not much more accurate'''
	return np.sqrt(romberg(lambda y: y * y * interp_cf_lin(y*R) * kernel(y), 0, 2))
	
Dzs = np.loadtxt('Desktop/Dz.txt')

js = [0, 100, 200, 300, 400, 500, 600]

for j in js:
	Dz = Dzs[j]
	# Pre-assign y-kernel for the direct approach
	y = np.linspace(0,2,100)
	dy = y[1]-y[0]
	ysqkernel = y * y * kernel(y)
	%timeit op.brentq(lambda R: sigma(R,y,dy,ysqkernel) - 1./Dz, 0.0, 8.0)
	Rnl_actual = op.brentq(lambda R: sigma(R,y,dy,ysqkernel) - 1./Dz, 0.0, 8.0)
	# can we do better than brentq on the optimizer?

	# Get R_NL for a power-law cosmology to set the fitting range
	# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
	norm = 71.38238397491413
	sigma2_norm = -1.8 * norm/(np.pi * gamma(4.) * math.cos(-np.pi))
	rnl_analytic = op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 10.0)
	%timeit op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 10.0)

	# Pre-load some matrices for polynomial coefficient fitting
	Rsq = R * R
	Y = Rsq * cflinr
	xmatT_all = np.array([np.ones_like(R), R, R * R, Rsq * R])
	c0,c1,c2,c3 = get_poly_coeffs(2.*rnl_analytic, xmatT_all, Y)
	%timeit get_poly_coeffs(2.*rnl_analytic, xmatT_all, Y)

	%timeit getR(c0,c1,c2,c3,Dz)
	Rnl = getR(c0,c1,c2,c3,Dz)

	print j, np.abs(Rnl-Rnl_actual)/Rnl_actual