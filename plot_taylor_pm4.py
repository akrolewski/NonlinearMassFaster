import numpy as np
import matplotlib.pyplot as plt
import time
from nonlinear_mass_faster import *

plt.ion()
plt.show()

om0 = 0.3111
om0_err = 0.0056

ob0 = 0.04897
ob0_err = 0.001

ns = 0.9665
ns_err = 0.0038

om0s = [om0 - 4*om0_err, om0, om0 + 4 * om0_err]
ob0s = [ob0 - 4*ob0_err, ob0, ob0 + 4 * ob0_err]
nsss = [ns - 4*ns_err, ns, ns + 4 * ns_err]

cnt = 54

err1 = np.zeros((27,7))
err2 = np.zeros((27,7))

for i,om0 in enumerate(om0s):
	for j,ob0 in enumerate(ob0s):
		for k,ns in enumerate(nsss):
				
			dat = np.loadtxt('cosmo_files/%i/cfz0_2e4.txt' % cnt)

			Dzs = np.loadtxt('cosmo_files/%i/Dz.txt' % cnt)

			R = dat[:,0]
			cflinr = dat[:,1]
			interp_cf_lin = lambda x: np.interp(x, R, cflinr)

			js = [0, 100, 200, 300, 400, 500, 600]
			z = [0, 1, 2, 3, 4, 5, 6]

			# Pre-assign y-kernel for the direct approach
			y = np.linspace(0,2,100)
			dy = y[1]-y[0]
			ysqkernel = y * y * kernel(y)
			
			# Get R_NL for a power-law cosmology to set the fitting range
			norm = float(np.loadtxt('cosmo_files/%i/norm.txt' % cnt))
			sigma2_norm = -1.8 * norm/(np.pi * gamma(4.) * math.cos(-np.pi))
			
			for jjj, jj in enumerate(js):
				Dz = Dzs[jj]
				rnl_analytic = op.brentq(lambda R: sigma2_norm ** 0.5 * R**-0.5 - 1./Dz, 0.01, 8.0)

				# This is a bit faster than the other optimize methods
				Rnl_actual = op.newton(lambda R: sigma(R,y,dy,ysqkernel,interp_cf_lin) - 1./Dz, rnl_analytic)

				c0,c1,c2,c3 = get_poly_coeffs(2.*rnl_analytic, interp_cf_lin)
				#print c0, c1, c2, c3

				Rnl = getR(c0,c1,c2,c3,Dz)

				print 'fitting', cnt, z[jjj], np.abs(Rnl-Rnl_actual)/Rnl_actual			
				err1[cnt-54,jjj] = np.abs(Rnl-Rnl_actual)/Rnl_actual
				
				c0, c1, c2, c3 = get_poly_coeffs_taylor(om0, ob0, ns, jjj)
				#print c0, c1, c2, c3

				Rnl = getR(c0,c1,c2,c3,Dz)

				print 'taylor series', cnt, z[jjj], np.abs(Rnl-Rnl_actual)/Rnl_actual
				err2[cnt-54,jjj] = np.abs(Rnl-Rnl_actual)/Rnl_actual
				
			cnt += 1								


for i in range(7):
	plt.figure()
	plt.plot(err1[:,i])
	plt.plot(err2[:,i])