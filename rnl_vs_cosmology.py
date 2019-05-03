import numpy as np
import matplotlib.pyplot as plt
import time
from nonlinear_mass_faster import *

plt.ion()
plt.show()

dir = 'cosmo_files/'
subdirs = ['ns_minus5/','ns_minus1/','default/','ns_plus1/','ns_plus5/']

Rnl_all_actuals = []

for subdir in subdirs:
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
	# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
	norm = float(np.loadtxt(dir + subdir + 'norm.txt'))
	sigma2_norm = -1.8 * norm/(np.pi * gamma(4.) * math.cos(-np.pi))

	Rnl_actuals = []
	for j in js:
		Dz = Dzs[j]
	
		rnl_analytic = op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 8.0)

		Rnl_actuals.append(op.newton(lambda R: sigma(R,y,dy,ysqkernel,interp_cf_lin) - 1./Dz, rnl_analytic))
		
	Rnl_all_actuals.append(Rnl_actuals)
	
Rnl_all_actuals = np.array(Rnl_all_actuals)

plt.figure()
for i in range(5):
	plt.plot(z,Rnl_all_actuals[i,:]/Rnl_all_actuals[2,:])
	
plt.xlabel('z',size=20)
plt.ylabel('Ratio to fiducial',size=20)
plt.savefig('rnl_ns.pdf')

	