import numpy as np
import matplotlib.pyplot as plt
import time
from nonlinear_mass_faster import *
from scipy import optimize as op

plt.ion()
plt.show()

param = 'om0'

dir = 'cosmo_files/'
subdirs = ['%s_minus5/' % param,'%s_minus1/' % param,'default/','%s_plus1/' % param,'%s_plus5/' % param]

om0 = 0.3111
om0_err = 0.0056

ob0 = 0.04897
ob0_err = 0.001

ns = 0.9665
ns_err = 0.0038

om0s = [om0 - 5*om0_err, om0 - 1*om0_err, om0, om0 + 1 * om0_err, om0 + 5 * om0_err]
ob0s = [ob0 - 5*ob0_err, ob0 - 1*ob0_err, ob0, ob0 + 1 * ob0_err, ob0 + 5 * ob0_err]
nsss = [ns - 5*ns_err, ns - 1 * ns_err, ns, ns + 1 * ns_err, ns + 5 * ns_err]




Rnl_all_actuals = []

all_coeffs = np.zeros((5,4,7))
for i,subdir in enumerate(subdirs):
	dat = np.loadtxt(dir + subdir + 'cfz0_2e4.txt')

	Dzs = np.loadtxt(dir + subdir + 'Dz.txt')

	R = dat[:,0]
	cflinr = dat[:,1]
	interp_cf_lin = lambda x: np.interp(x, R, cflinr)

	

	js = [0, 100, 200, 300, 400, 500, 600]
	z = [0, 1, 2, 3, 4, 5, 6]

	# Pre-assign y-kernel for the direct approach
	y = np.linspace(0,2,100)
	dy = y[1]-y[0]
	ysqkernel = y * y * kernel(y)

	# Pre-load some matrices for polynomial coefficient fitting
	Rsq = R * R
	Y = Rsq * cflinr
	xmatT_all = np.array([np.ones_like(R), R, R * R, Rsq * R])

	# Get R_NL for a power-law cosmology to set the fitting range
	norm = float(np.loadtxt(dir + subdir + 'norm.txt'))
	sigma2_norm = -1.8 * norm/(np.pi * gamma(4.) * math.cos(-np.pi))

	coeffs = np.zeros((4,7))
	for k,j in enumerate(js):
		Dz = Dzs[j]
	
		rnl_analytic = op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 8.0)
		
		c0,c1,c2,c3 = get_poly_coeffs(2.*rnl_analytic, interp_cf_lin)
		coeffs[:,k] = np.array([c0,c1,c2,c3])
		
	all_coeffs[i,:,:] = coeffs

f = open('%s_1st_order_taylor.txt' % param,'w')	
f.write('# Slope and intercept for c0, c1, c2, c3\n')
for i in range(7):
	allps = []
	for j in range(4):
		intercept = all_coeffs[2,j,i]
		def lin(x,a):
			return a * x + intercept
		if param == 'ns':
			p = op.curve_fit(lin,np.array(nsss)-ns,all_coeffs[:,j,i])
		elif param == 'om0':
			p = op.curve_fit(lin,np.array(om0s)-om0,all_coeffs[:,j,i])
		elif param == 'ob0':
			p = op.curve_fit(lin,np.array(ob0s)-ob0,all_coeffs[:,j,i])
		allps.append(np.array([p[0][0],all_coeffs[2,j,i]]))
	
	flatallps = np.array(allps).flatten()
	str = ''
	str += '%.2f ' % z[i]
	for j in range(len(flatallps)):
		str += '%.10f ' % flatallps[j]
		
	str += '\n'
	f.write(str)
	
f.close()
	