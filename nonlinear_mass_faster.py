import numpy as np
import math
from scipy import optimize as op
from scipy.special import gamma, spherical_jn
import time
from scipy.integrate import romberg
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import scipy as sp

# Load Taylor coefficients
#om0_file = np.fromfile('taylor_coeffs_kmax4/om0.bin').reshape((61,9))
#ob0_file = np.fromfile('taylor_coeffs_kmax4/ob0.bin').reshape((61,9))
#ns_file = np.fromfile('taylor_coeffs_kmax4/ns.bin').reshape((61,9))

all_file = np.fromfile('taylor_coeffs_camb_kmax4_rmax_fourier/all_coeffs.bin').reshape((61,16))


#c0_file = np.fromfile('taylor_coeffs_kmax4/c0.bin').reshape((61,4))
#c1_file = np.fromfile('taylor_coeffs_kmax4/c1.bin').reshape((61,4))
#c2_file = np.fromfile('taylor_coeffs_kmax4/c2.bin').reshape((61,4))
#c3_file = np.fromfile('taylor_coeffs_kmax4/c3.bin').reshape((61,4))



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
	
#def get_poly_coeffs(Rmax, xmatT_all, Y):
def get_poly_coeffs(Rmax, interp_cf_lin):
	'''gets polynomial coefficients given some rmax'''
	#ind = int(Rmax * 1000.) + 1
	# precompute xmatT to speed things up
	#xmatT = xmatT_all[:,:ind]
	delta = Rmax / 999.
	r = np.arange(1000.) * delta
	rsq = r * r
	xmatT = np.zeros((4,1000))
	xmatT[0,:] = np.ones(1000)
	xmatT[1,:] = r
	xmatT[2,:] = rsq
	xmatT[3,:] = rsq * r
	xTy = np.dot(xmatT, rsq * interp_cf_lin(r))

	# a bit faster to do this because then I don't have to convert stuff to np.array
	#return (np.dot(inv[0],xTy),
	# inv[1][0] * xTy[0] + inv[1][1] * xTy[1] + inv[1][2] * xTy[2] + inv[1][3] * xTy[3],
	# inv[2][0] * xTy[0] + inv[2][1] * xTy[1] + inv[2][2] * xTy[2] + inv[2][3] * xTy[3],
	# np.dot(inv[3],xTy))
	return sp.linalg.lapack.dpotrs(sp.linalg.lapack.dpotrf(np.dot(xmatT,xmatT.T), False, False)[0],xTy)[0]


def fast_interpolate(z, om0_file, index):
    tenz = 10.*z
    lowind = int(tenz)
    highind = lowind + 1
    om0_file_lowind = om0_file[lowind,index]
    if highind < np.shape(om0_file)[0]:
    	return om0_file_lowind * (1. - (tenz - lowind))+ om0_file[highind,index] * (tenz - lowind)
    else:
    	return om0_file_lowind
    
def nearest_neighbor(z, om0_file):
    tenz = 10.*z
    lowind = int(tenz)
    #highind = lowind + 1
    om0_file_lowind = om0_file[lowind,:]
    return om0_file_lowind

def get_poly_coeffs_taylor(om0, ob0, ns, z):
	'''gets polynomial coefficients in Taylor approximation'''
	om0_fid = 0.3096

	ob0_fid = 0.04897

	ns_fid = 0.9665
	
	#taylor_om0 = nearest_neighbor(z, om0_file)
	
	#taylor_ob0 = nearest_neighbor(z, ob0_file)
	
	#taylor_ns = nearest_neighbor(z, ns_file)
	
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
	

def get_poly_coeffs_taylor_vec(om0, ob0, ns, z):
	'''gets polynomial coefficients in Taylor approximation'''
	om0_fid = 0.3096

	ob0_fid = 0.04897

	ns_fid = 0.9665
	
	taylor_all = nearest_neighbor(z, all_file)
	
    
	delta_om0 = om0 - om0_fid
	delta_ob0 = ob0 - ob0_fid
	delta_ns = ns - ns_fid
	
	#c0_arr = np.array([0,1,0,0,0,delta_om0,0,0,0,delta_ob0,0,0,0,delta_ns,0,0,0])
	c0_arr = np.zeros(16)
	c0_arr[0] = 1.
	c0_arr[1] = delta_om0
	c0_arr[2] = delta_ob0
	c0_arr[3] = delta_ns
	c0 = np.dot(c0_arr,taylor_all)
	
	#c1_arr = np.array([0,0,1,0,0,0,delta_om0,0,0,0,delta_ob0,0,0,0,delta_ns,0,0])
	c1_arr = np.zeros(16)
	c1_arr[4] = 1.
	c1_arr[5] = delta_om0
	c1_arr[6] = delta_ob0
	c1_arr[7] = delta_ns
	c1 = np.dot(c1_arr,taylor_all)
	
	#c2_arr = np.array([0,0,0,1,0,0,0,delta_om0,0,0,0,delta_ob0,0,0,0,delta_ns,0])
	c2_arr = np.zeros(16)
	c2_arr[8] = 1.
	c2_arr[9] = delta_om0
	c2_arr[10] = delta_ob0
	c2_arr[11] = delta_ns
	c2 = np.dot(c2_arr,taylor_all)
	
	#c3_arr = np.array([0,0,0,0,1,0,0,0,delta_om0,0,0,0,delta_ob0,0,0,0,delta_ns])
	c3_arr = np.zeros(16)
	c3_arr[12] = 1.
	c3_arr[13] = delta_om0
	c3_arr[14] = delta_ob0
	c3_arr[15] = delta_ns
	c3 = np.dot(c3_arr,taylor_all)
	
	# this is 28.6 mus versus 13.5 mus for the other version
	# seems like all the time is going into np.array
	# and np.dot
	# so might be inefficient array formation?
	# are there faster alternatives to np.dot? 
		 
	
	return c0, c1, c2, c3
	

	
def get_poly_coeffs_taylor_quad(om0, ob0, ns, z):
	'''gets polynomial coefficients in Taylor approximation'''
	om0_fid = 0.3096

	ob0_fid = 0.04897

	ns_fid = 0.9665
	


	# VECTORIZE

	taylor_om0_1 = nearest_neighbor(z, om0_file_quad, 1)
	taylor_om0_2 = nearest_neighbor(z, om0_file_quad, 2)
	#taylor_om0_2 = 0.1
	taylor_om0_3 = nearest_neighbor(z, om0_file_quad, 3)
	taylor_om0_4 = nearest_neighbor(z, om0_file_quad, 4)
	#taylor_om0_4 = 0.1
	taylor_om0_5 = nearest_neighbor(z, om0_file_quad, 5)
	taylor_om0_6 = nearest_neighbor(z, om0_file_quad, 6)
	#taylor_om0_6 = 0.1
	taylor_om0_7 = nearest_neighbor(z, om0_file_quad, 7)
	taylor_om0_8 = nearest_neighbor(z, om0_file_quad, 8)
	taylor_om0_9 = nearest_neighbor(z, om0_file_quad, 9)
	taylor_om0_10 = nearest_neighbor(z, om0_file_quad, 10)
	taylor_om0_11 = nearest_neighbor(z, om0_file_quad, 11)
	taylor_om0_12 = nearest_neighbor(z, om0_file_quad, 12)
	#taylor_om0_8 = 0.1
	
	taylor_ob0_1 = nearest_neighbor(z, ob0_file_quad, 1)
	taylor_ob0_4 = nearest_neighbor(z, ob0_file_quad, 4)
	taylor_ob0_7 = nearest_neighbor(z, ob0_file_quad, 7)
	taylor_ob0_10 = nearest_neighbor(z, ob0_file_quad, 10)

	taylor_ob0_2 = nearest_neighbor(z, ob0_file_quad, 2)
	taylor_ob0_5 = nearest_neighbor(z, ob0_file_quad, 5)
	taylor_ob0_8 = nearest_neighbor(z, ob0_file_quad, 8)
	taylor_ob0_11 = nearest_neighbor(z, ob0_file_quad, 11)
	
	taylor_ns_1 = nearest_neighbor(z, ns_file_quad, 1)
	taylor_ns_4 = nearest_neighbor(z, ns_file_quad, 4)
	taylor_ns_7 = nearest_neighbor(z, ns_file_quad, 7)
	taylor_ns_10 = nearest_neighbor(z, ns_file_quad, 10)

	taylor_ns_2 = nearest_neighbor(z, ns_file_quad, 2)
	taylor_ns_5 = nearest_neighbor(z, ns_file_quad, 5)
	taylor_ns_8 = nearest_neighbor(z, ns_file_quad, 8)
	taylor_ns_11 = nearest_neighbor(z, ns_file_quad, 11)
	
    
	delta_om0 = om0 - om0_fid
	delta_ob0 = ob0 - ob0_fid
	delta_ns = ns - ns_fid
	
	delta_om0_sq = delta_om0 * delta_om0
	delta_ob0_sq = delta_ob0 * delta_ob0
	delta_ns_sq = delta_ns * delta_ns
	
	c0 = (taylor_om0_1 * (delta_om0) + taylor_om0_2 * (delta_om0_sq) + taylor_om0_3 + taylor_ob0_1 * (delta_ob0)
		 + taylor_ob0_2 * (delta_ob0_sq) + taylor_ns_1 * (delta_ns) + taylor_ns_2 * (delta_ns_sq))
	c1 = (taylor_om0_4 * (delta_om0) + taylor_om0_5 * (delta_om0_sq) + taylor_om0_6 + taylor_ob0_4 * (delta_ob0)
		 + taylor_ob0_5 * (delta_ob0_sq) + taylor_ns_4 * (delta_ns) + taylor_ns_5 * (delta_ns_sq))
	c2 = (taylor_om0_7 * (delta_om0) + taylor_om0_8 * (delta_om0_sq) + taylor_om0_9 + taylor_ob0_7 * (delta_ob0)
		 + taylor_ob0_8 * (delta_ob0_sq) + taylor_ns_7 * (delta_ns) + taylor_ns_8 * (delta_ns_sq))
	c3 = (taylor_om0_10 * (delta_om0) + taylor_om0_11 * (delta_om0_sq) + taylor_om0_12 + taylor_ob0_10 * (delta_ob0)
		 + taylor_ob0_11 * (delta_ob0_sq) + taylor_ns_10 * (delta_ns) + taylor_ns_11 * (delta_ns_sq))
		 
	
	return c0, c1, c2, c3

def get_poly_coeffs_taylor_second_order(om0, ob0, ns, z):
	'''gets polynomial coefficients in Taylor approximation'''
	om0_fid = 0.3096

	ob0_fid = 0.04897

	ns_fid = 0.9665
	


	# VECTORIZE

	taylor_intercept1 = nearest_neighbor(z, second_order, 1)
	taylor_intercept2 = nearest_neighbor(z, second_order, 2)
	taylor_intercept3 = nearest_neighbor(z, second_order, 3)
	taylor_intercept4 = nearest_neighbor(z, second_order, 4)
	
	taylor_slope_om0_1 = nearest_neighbor(z, second_order, 5)
	taylor_slope_om0_2 = nearest_neighbor(z, second_order, 6)
	taylor_slope_om0_3 = nearest_neighbor(z, second_order, 7)
	taylor_slope_om0_4 = nearest_neighbor(z, second_order, 8)
	
	taylor_slope_ob0_1 = nearest_neighbor(z, second_order, 9)
	taylor_slope_ob0_2 = nearest_neighbor(z, second_order, 10)
	taylor_slope_ob0_3 = nearest_neighbor(z, second_order, 11)
	taylor_slope_ob0_4 = nearest_neighbor(z, second_order, 12)
	
	taylor_slope_ns_1 = nearest_neighbor(z, second_order, 13)
	taylor_slope_ns_2 = nearest_neighbor(z, second_order, 14)
	taylor_slope_ns_3 = nearest_neighbor(z, second_order, 15)
	taylor_slope_ns_4 = nearest_neighbor(z, second_order, 16)

	taylor_quad_om0_1 = nearest_neighbor(z, second_order, 17)
	taylor_quad_om0_2 = nearest_neighbor(z, second_order, 18)
	taylor_quad_om0_3 = nearest_neighbor(z, second_order, 19)
	taylor_quad_om0_4 = nearest_neighbor(z, second_order, 20)
	
	taylor_quad_ob0_1 = nearest_neighbor(z, second_order, 21)
	taylor_quad_ob0_2 = nearest_neighbor(z, second_order, 22)
	taylor_quad_ob0_3 = nearest_neighbor(z, second_order, 23)
	taylor_quad_ob0_4 = nearest_neighbor(z, second_order, 24)
	
	taylor_quad_ns_1 = nearest_neighbor(z, second_order, 25)
	taylor_quad_ns_2 = nearest_neighbor(z, second_order, 26)
	taylor_quad_ns_3 = nearest_neighbor(z, second_order, 27)
	taylor_quad_ns_4 = nearest_neighbor(z, second_order, 28)

	taylor_mixed_om0_ob0_1 = nearest_neighbor(z, second_order, 29)
	taylor_mixed_om0_ob0_2 = nearest_neighbor(z, second_order, 30)
	taylor_mixed_om0_ob0_3 = nearest_neighbor(z, second_order, 31)
	taylor_mixed_om0_ob0_4 = nearest_neighbor(z, second_order, 32)

	taylor_mixed_om0_ns_1 = nearest_neighbor(z, second_order, 33)
	taylor_mixed_om0_ns_2 = nearest_neighbor(z, second_order, 34)
	taylor_mixed_om0_ns_3 = nearest_neighbor(z, second_order, 35)
	taylor_mixed_om0_ns_4 = nearest_neighbor(z, second_order, 36)

	taylor_mixed_ob0_ns_1 = nearest_neighbor(z, second_order, 37)
	taylor_mixed_ob0_ns_2 = nearest_neighbor(z, second_order, 38)
	taylor_mixed_ob0_ns_3 = nearest_neighbor(z, second_order, 39)
	taylor_mixed_ob0_ns_4 = nearest_neighbor(z, second_order, 40)
	
    
	delta_om0 = om0 - om0_fid
	delta_ob0 = ob0 - ob0_fid
	delta_ns = ns - ns_fid
	
	delta_om0_sq = delta_om0 * delta_om0
	delta_ob0_sq = delta_ob0 * delta_ob0
	delta_ns_sq = delta_ns * delta_ns
	
	
	c0 = (taylor_intercept1 + taylor_slope_om0_1 * delta_om0 
		+ taylor_slope_ob0_1 * delta_ob0
		+ taylor_slope_ns_1 * delta_ns
		+ taylor_quad_om0_1 * delta_om0_sq
		+ taylor_quad_ob0_1 * delta_ob0_sq
		+ taylor_quad_ns_1 * delta_ns_sq
		+ taylor_mixed_om0_ob0_1 * delta_om0 * delta_ob0
		+ taylor_mixed_om0_ns_1 * delta_om0 * delta_ns
		+ taylor_mixed_ob0_ns_1 * delta_ob0 * delta_ns)
	
	c1 = (taylor_intercept2 + taylor_slope_om0_2 * delta_om0 
		+ taylor_slope_ob0_2 * delta_ob0
		+ taylor_slope_ns_2 * delta_ns
		+ taylor_quad_om0_2 * delta_om0_sq
		+ taylor_quad_ob0_2 * delta_ob0_sq
		+ taylor_quad_ns_2 * delta_ns_sq
		+ taylor_mixed_om0_ob0_2 * delta_om0 * delta_ob0
		+ taylor_mixed_om0_ns_2 * delta_om0 * delta_ns
		+ taylor_mixed_ob0_ns_2 * delta_ob0 * delta_ns)
	
	c2 = (taylor_intercept3 + taylor_slope_om0_3 * delta_om0 
		+ taylor_slope_ob0_3 * delta_ob0
		+ taylor_slope_ns_3 * delta_ns
		+ taylor_quad_om0_3 * delta_om0_sq
		+ taylor_quad_ob0_3 * delta_ob0_sq
		+ taylor_quad_ns_3 * delta_ns_sq
		+ taylor_mixed_om0_ob0_3 * delta_om0 * delta_ob0
		+ taylor_mixed_om0_ns_3 * delta_om0 * delta_ns
		+ taylor_mixed_ob0_ns_3 * delta_ob0 * delta_ns)

	c3 = (taylor_intercept4 + taylor_slope_om0_4 * delta_om0 
		+ taylor_slope_ob0_4 * delta_ob0
		+ taylor_slope_ns_4 * delta_ns
		+ taylor_quad_om0_4 * delta_om0_sq
		+ taylor_quad_ob0_4 * delta_ob0_sq
		+ taylor_quad_ns_4 * delta_ns_sq
		+ taylor_mixed_om0_ob0_4 * delta_om0 * delta_ob0
		+ taylor_mixed_om0_ns_4 * delta_om0 * delta_ns
		+ taylor_mixed_ob0_ns_4 * delta_ob0 * delta_ns)	
	
	return c0, c1, c2, c3
	#return taylor_om0_2, taylor_om0_4, taylor_om0_6, taylor_om0_8

	 
def getR(c0,c1,c2,c3,deltac_sq,Dz):
	'''Finds R'''
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
		
def get_rnl_fid(z):
	'''Gets fiducial RNL for setting the fitting range.'''
	rnl_fid_file = np.loadtxt('fiducial_rnl_deltac_kmax4.txt')
	return fast_interpolate(z, rnl_fid_file, 1)
		
def rnl_faster(interp_cf_lin, z, fitting_range_mode='fiducial', pk=1.0):
	'''Full function to compute Rnl, given a linear power spectrum
	and a redshift.  Fitting_range_mode tells you how to set the maximum of the
	polynomial fitting range.  Fiducial = use 2*RNL for the true cosmology (accurate to 1e-4).
	power law = use 2 * RNL from a power-law cosmology matched to the amplitude
	of the input P(k) at k = 1.'''
	c0,c1,c2,c3 = nlm.get_poly_coeffs(2.*Rnl_actual, interp_cf_lin)
	print(c0, c1, c2, c3)
	def test():
		return nlm.get_poly_coeffs(2.*Rnl_actual, interp_cf_lin)
	t_poly_coeffs = timeit.timeit('test()',setup='from __main__ import test',number=ntest)/ntest

	Rnl = nlm.getR(c0,c1,c2,c3,Dz)

def sigma_Pk(R, k, lnk, pklin, spacing='log10'):
	'''Computes sigma in Fourier space using the standard definition.'''
	if R <= 0:
		# Don't want nan values if R <= 0.
		# So set negative R to a small positive value
		# to ensure continuity
		R = 1e-6
	#if spacing == 'linear':
	#	dk = k[1]-k[0]
	#	kr = k * R
	#	bess = (3 * spherical_jn(1, kr))/kr
	#	return dk / (2. * np.pi * np.pi) * np.sum(k * k * bess * bess * pklin)
	#elif spacing == 'log10':
	#lnk = np.log(k)
	dlnk = lnk[1]-lnk[0]
	kr = k * R
	bess = (3. * spherical_jn(1, kr))/kr
	#return (dlnk/ (2. * np.pi * np.pi) * np.sum(k * k ** 2. * bess * bess * pklin)) ** 0.5
	return (1./(2.* np.pi * np.pi) * np.trapz(k * k ** 2. * bess * bess * pklin, x=lnk)) ** 0.5
	#else:
	#	print('Spacing must be logarithmic base 10 (log10) or linear (linear).')
	#	return None
	
def sigma_Pk_smoothing(R, k, lnk, pklin, ksmooth, spacing='log10'):
	'''Computes sigma in Fourier space using the standard definition.'''
	if R <= 0:
		# Don't want nan values if R <= 0.
		# So set negative R to a small positive value
		# to ensure continuity
		R = 1e-6
	#if spacing == 'linear':
	#	dk = k[1]-k[0]
	#	kr = k * R
	#	bess = (3 * spherical_jn(1, kr))/kr
	#	return dk / (2. * np.pi * np.pi) * np.sum(k * k * bess * bess * pklin)
	#elif spacing == 'log10':
	#lnk = np.log(k)
	dlnk = lnk[1]-lnk[0]
	kr = k * R
	bess = (3. * spherical_jn(1, kr))/kr
	#return (dlnk/ (2. * np.pi * np.pi) * np.sum(k * k ** 2. * bess * bess * pklin)) ** 0.5
	return (1./(2.* np.pi * np.pi) * np.trapz(k * k ** 2. * bess * bess * pklin * np.exp(-(k/ksmooth)**2.), x=lnk)) ** 0.5
	#else:
	#	print('Spacing must be logarithmic base 10 (log10) or linear (linear).')
	#	return None
	
def sigma(R, y, dy, ysqkernel, interp_cf_lin):
	'''Directly do the integral for the Zehavi+05 method in real space.
	Pre assign y and ysqkernel to save time in the optimization.
	We find that 100 sampling points give accuracy of 10^-3 as compared
	to Romberg'''
	return (dy * np.sum(interp_cf_lin(y*R) * ysqkernel))**0.5

def sigma_rom(R, interp_cf_lin):
	'''Romberg version of direct integral.  A factor of 50 slower
	than np.sum method, and not much more accurate'''
	return (romberg(lambda y: y * y * interp_cf_lin(y*R) * kernel(y), 0, 2))**0.5
	
def pk_to_xi(k, pk, rs):
	'''Turns the power spectrum into a correlation function, given
	some user-supplied input power spectrum and k.'''
	dk = np.gradient(k)

	xi = np.zeros_like(rs)

	for i,r in enumerate(rs):
		#xi[i] = np.sum(pk * spherical_jn(0,k * r) * k * k * dk * np.exp(-( k/np.max(k))**2.))
		# Don't suppress with an exponential! Causes the fourier-space answers to be WRONG
		xi[i] = np.sum(pk * spherical_jn(0,k * r) * k * k * dk)
		#xi[i] = np.trapz(pk * spherical_jn(0,k * r) * k * k, x=k)
		
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
	
	
def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None, x1=None, rtol=0.0,
           full_output=False, disp=True):
    """
    A "higher-performance" version of scipy.optimize.newton.
    Replaces np.isclose with abs(p1-p0) < tol, which is 
    simpler and faster.
    Find a zero of a real or complex function using the Newton-Raphson
    (or secant or Halley's) method.

    Find a zero of the function `func` given a nearby starting point `x0`.
    The Newton-Raphson method is used if the derivative `fprime` of `func`
    is provided, otherwise the secant method is used.  If the second order
    derivative `fprime2` of `func` is also provided, then Halley's method is
    used.

    If `x0` is a sequence with more than one item, then `newton` returns an
    array, and `func` must be vectorized and return a sequence or array of the
    same shape as its first argument. If `fprime` or `fprime2` is given then
    its return must also have the same shape.

    Parameters
    ----------
    func : callable
        The function whose zero is wanted. It must be a function of a
        single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``
        are extra arguments that can be passed in the `args` parameter.
    x0 : float, sequence, or ndarray
        An initial estimate of the zero that should be somewhere near the
        actual zero. If not scalar, then `func` must be vectorized and return
        a sequence or array of the same shape as its first argument.
    fprime : callable, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value.  If `func` is complex-valued,
        a larger `tol` is recommended as both the real and imaginary parts
        of `x` contribute to ``|x - x0|``.
    maxiter : int, optional
        Maximum number of iterations.
    fprime2 : callable, optional
        The second order derivative of the function when available and
        convenient. If it is None (default), then the normal Newton-Raphson
        or the secant method is used. If it is not None, then Halley's method
        is used.
    x1 : float, optional
        Another estimate of the zero that should be somewhere near the
        actual zero.  Used if `fprime` is not provided.
    rtol : float, optional
        Tolerance (relative) for termination.
    full_output : bool, optional
        If `full_output` is False (default), the root is returned.
        If True and `x0` is scalar, the return value is ``(x, r)``, where ``x``
        is the root and ``r`` is a `RootResults` object.
        If True and `x0` is non-scalar, the return value is ``(x, converged,
        zero_der)`` (see Returns section for details).
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge, with
        the error message containing the number of iterations and current
        function value.  Otherwise the convergence status is recorded in a
        `RootResults` return object.
        Ignored if `x0` is not scalar.
        *Note: this has little to do with displaying, however
        the `disp` keyword cannot be renamed for backwards compatibility.*

    Returns
    -------
    root : float, sequence, or ndarray
        Estimated location where function is zero.
    r : `RootResults`, optional
        Present if ``full_output=True`` and `x0` is scalar.
        Object containing information about the convergence.  In particular,
        ``r.converged`` is True if the routine converged.
    converged : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements converged successfully.
    zero_der : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements had a zero derivative.

    See Also
    --------
    brentq, brenth, ridder, bisect
    fsolve : find zeros in n dimensions.

    Notes
    -----
    The convergence rate of the Newton-Raphson method is quadratic,
    the Halley method is cubic, and the secant method is
    sub-quadratic.  This means that if the function is well behaved
    the actual error in the estimated zero after the n-th iteration
    is approximately the square (cube for Halley) of the error
    after the (n-1)-th step.  However, the stopping criterion used
    here is the step size and there is no guarantee that a zero
    has been found. Consequently the result should be verified.
    Safer algorithms are brentq, brenth, ridder, and bisect,
    but they all require that the root first be bracketed in an
    interval where the function changes sign. The brentq algorithm
    is recommended for general use in one dimensional problems
    when such an interval has been found.

    When `newton` is used with arrays, it is best suited for the following
    types of problems:

    * The initial guesses, `x0`, are all relatively the same distance from
      the roots.
    * Some or all of the extra arguments, `args`, are also arrays so that a
      class of similar problems can be solved together.
    * The size of the initial guesses, `x0`, is larger than O(100) elements.
      Otherwise, a naive loop may perform as well or better than a vector.

    Examples
    --------
    >>> from scipy import optimize
    >>> import matplotlib.pyplot as plt

    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    ``fprime`` is not provided, use the secant method:

    >>> root = optimize.newton(f, 1.5)
    >>> root
    1.0000000000000016
    >>> root = optimize.newton(f, 1.5, fprime2=lambda x: 6 * x)
    >>> root
    1.0000000000000016

    Only ``fprime`` is provided, use the Newton-Raphson method:

    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)
    >>> root
    1.0

    Both ``fprime2`` and ``fprime`` are provided, use Halley's method:

    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,
    ...                        fprime2=lambda x: 6 * x)
    >>> root
    1.0

    When we want to find zeros for a set of related starting values and/or
    function parameters, we can provide both of those as an array of inputs:

    >>> f = lambda x, a: x**3 - a
    >>> fder = lambda x, a: 3 * x**2
    >>> np.random.seed(4321)
    >>> x = np.random.randn(100)
    >>> a = np.arange(-50, 50)
    >>> vec_res = optimize.newton(f, x, fprime=fder, args=(a, ))

    The above is the equivalent of solving for each value in ``(x, a)``
    separately in a for-loop, just faster:

    >>> loop_res = [optimize.newton(f, x0, fprime=fder, args=(a0,))
    ...             for x0, a0 in zip(x, a)]
    >>> np.allclose(vec_res, loop_res)
    True

    Plot the results found for all values of ``a``:

    >>> analytical_result = np.sign(a) * np.abs(a)**(1/3)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(a, analytical_result, 'o')
    >>> ax.plot(a, vec_res, '.')
    >>> ax.set_xlabel('$a$')
    >>> ax.set_ylabel('$x$ where $f(x, a)=0$')
    >>> plt.show()

    """
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if np.size(x0) > 1:
        return op._array_newton(func, x0, fprime, args, tol, maxiter, fprime2,
                             full_output)

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    if fprime is not None:
        # Newton-Raphson method
        for itr in range(maxiter):
            # first evaluate fval
            fval = func(p0, *args)
            funcalls += 1
            # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return op._results_select(
                    full_output, (p0, funcalls, itr, _ECONVERGED))
            fder = fprime(p0, *args)
            funcalls += 1
            if fder == 0:
                msg = "Derivative was zero."
                if disp:
                    msg += (
                        " Failed to converge after %d iterations, value is %s."
                        % (itr + 1, p0))
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning)
                return op._results_select(
                    full_output, (p0, funcalls, itr + 1, _ECONVERR))
            newton_step = fval / fder
            if fprime2:
                fder2 = fprime2(p0, *args)
                funcalls += 1
                # Halley's method:
                #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
                # Only do it if denominator stays close enough to 1
                # Rationale:  If 1-adj < 0, then Halley sends x in the
                # opposite direction to Newton.  Doesn't happen if x is close
                # enough to root.
                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p0 - newton_step
            if np.isclose(p, p0, rtol=rtol, atol=tol):
                return op._results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED))
            p0 = p
    else:
        # Secant method
        if x1 is not None:
            if x1 == x0:
                raise ValueError("x1 and x0 must be different")
            p1 = x1
        else:
            eps = 1e-4
            p1 = x0 * (1 + eps)
            p1 += (eps if p1 >= 0 else -eps)
        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        for itr in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = "Tolerance of %s reached." % (p1 - p0)
                    if disp:
                        msg += (
                            " Failed to converge after %d iterations, value is %s."
                            % (itr + 1, p1))
                        raise RuntimeError(msg)
                    warnings.warn(msg, RuntimeWarning)
                p = (p1 + p0) / 2.0
                return op._results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED))
            else:
                if abs(q1) > abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            #if np.isclose(p, p1, rtol=rtol, atol=tol):
            #    return _results_select(
            #        full_output, (p, funcalls, itr + 1, _ECONVERGED))
            if abs(p - p1) < tol:
                return p
            p0, q0 = p1, q1
            p1 = p
            q1 = func(p1, *args)
            funcalls += 1

    if disp:
        msg = ("Failed to converge after %d iterations, value is %s."
               % (itr + 1, p))
        raise RuntimeError(msg)

    return op._results_select(full_output, (p, funcalls, itr + 1, _ECONVERR))
