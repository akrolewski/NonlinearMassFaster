import numpy as np
import time
import nonlinear_mass_faster as nlm
from scipy import optimize as op

# Fiducial rnl
rnl_fid = np.loadtxt('fiducial_rnl.txt')

delta_c = 1.686

om0 = 0.3096
om0_err = 0.0056

ob0 = 0.04897
ob0_err = 0.001

ns = 0.9665
ns_err = 0.0038

# Redshifts
zmin = 0
zmax = 6
Nz = 61
zs = np.linspace(zmin,zmax,Nz)

# Pre-assign y-kernel for the direct approach
y = np.linspace(0,2,100) 
dy = y[1]-y[0]
ysqkernel = y * y * nlm.kernel(y)


Nfiles = 5 # number of files per parameter
Ncoeffs = 4

om0s = [om0 - 5*om0_err, om0 - 1*om0_err, om0, om0 + 1 * om0_err, om0 + 5 * om0_err]
ob0s = [ob0 - 5*ob0_err, ob0 - 1*ob0_err, ob0, ob0 + 1 * ob0_err, ob0 + 5 * ob0_err]
nsss = [ns - 5*ns_err, ns - 1 * ns_err, ns, ns + 1 * ns_err, ns + 5 * ns_err]

params = ['om0','ob0','ns']

for k,param in enumerate(params):
	#k = 2
	#param = params[2]
	files = ['pklin/%s_minus5.txt' % param, 'pklin/%s_minus1.txt' % param, 'pklin/default.txt',
		'pklin/%s_plus1.txt' % param, 'pklin/%s_plus5.txt' % param]

	all_coeffs = np.zeros((Nfiles,Nz,Ncoeffs))
	for i,file in enumerate(files):
		pk = np.loadtxt(file)
	
		# Get R_NL for a power-law cosmology to set the fitting range
		# Formulae taken from https://gist.github.com/lgarrison/7e41ee280c57554e256b834ac5c3f753?short_path=2385d0c#file-scale_free_sigma8-ipynb
		norm = np.interp(1.0, pk[:,0], pk[:,1], left=np.nan, right=np.nan)
		sigma2_norm = nlm.sigma2_prefactor(-2, norm)
	
		if param == 'om0':
			om0_ind = om0s[i]
		else:
			om0_ind = om0

		# Specify your r-range
		rs = np.linspace(0,5,2e4+1) # experiment with the number of points...

		# Compute the correlation function
		cflinr = nlm.pk_to_xi(pk[:,0], pk[:,1], rs)
		interp_cf_lin = lambda x: np.interp(x, rs, cflinr)


		coeffs = np.zeros((Nz,Ncoeffs))
	

		for j,z in enumerate(zs):
			Dz = nlm.growth_factor(z, om0_ind)

			rnl_for_range = nlm.get_rnl_fid(z,rnl_fid)

			rnl_analytic = op.brentq(lambda R: nlm.sigma_pl(sigma2_norm, -2, R) - delta_c/Dz, 0.01, 8.0)

			try:
				Rnl_actual = op.newton(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, rnl_analytic)
			except RuntimeError:
				Rnl_actual = op.newton(lambda R: nlm.sigma(R,y,dy,ysqkernel,interp_cf_lin) - delta_c/Dz, 0.1)

			lnk = np.log(pk[:,0])
			try:
				Rnl_fourier = op.newton(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, rnl_analytic)
			except RuntimeError:
				Rnl_fourier = op.newton(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0.1)
			if Rnl_fourier < 0:
				Rnl_fourier = op.newton(lambda R: nlm.sigma_Pk(R,pk[:,0],lnk,pk[:,1]) - delta_c/Dz, 0.01)

	
			c0,c1,c2,c3 = nlm.get_poly_coeffs(2.*Rnl_fourier, interp_cf_lin)
			coeffs[j,:] = np.array([c0,c1,c2,c3])
	
		np.savetxt('pklin_coeffs/%s' % file.split('pklin/')[1],coeffs)
		all_coeffs[i,:,:] = coeffs

	#f = open('taylor_coeffs/%s.txt' % param,'w')
	#f.write('# Slope and intercept for c0, c1, c2, c3\n')
	all_to_save = []
	for i in range(Nz):
		allps = []
		for j in range(Ncoeffs):
			intercept = all_coeffs[2,i,j]
			def lin(x,a):
				return a * x + intercept
			def quad(x,a,b):
				return b * x **2. + a * x + intercept
			if param == 'ns':
				p = op.curve_fit(lin,np.array(nsss)-ns,all_coeffs[:,i,j])
			elif param == 'om0':
				p = op.curve_fit(lin,np.array(om0s)-om0,all_coeffs[:,i,j])
			elif param == 'ob0':
				p = op.curve_fit(lin,np.array(ob0s)-ob0,all_coeffs[:,i,j])
			allps.append(np.array([p[0][0],all_coeffs[2,i,j]]))
			#if i == Nz-1 and j == 2 and k == 2:
			#	print 5/0

		flatallps = np.array(allps).flatten()
		#print 5/0
		#str = ''
		#str += '%.2f ' % zs[i]
		#for j in range(len(flatallps)):
		#			str += '%.10f ' % flatallps[j]
	
		#str += '\n'
		#f.write(str)
		all_to_save.append(flatallps)
	all_to_save = np.array(all_to_save)
	all_to_save = np.concatenate((zs[:,np.newaxis],all_to_save),axis=1)
	#np.save('taylor_coeffs/%s.npy' % param,all_to_save)
	all_to_save.tofile('taylor_coeffs/%s.bin' % param)

	#f.close()
	
# Load Taylor coefficients
om0_file = np.fromfile('taylor_coeffs/om0.bin').reshape((61,9))
ob0_file = np.fromfile('taylor_coeffs/ob0.bin').reshape((61,9))
ns_file = np.fromfile('taylor_coeffs/ns.bin').reshape((61,9))

all_file = np.zeros((61,16))
#all_file[:,0] = om0_file[:,0]
all_file[:,0] = om0_file[:,2]
all_file[:,1] = om0_file[:,1]
all_file[:,2] = ob0_file[:,1]
all_file[:,3] = ns_file[:,1]

all_file[:,4] = om0_file[:,4]
all_file[:,5] = om0_file[:,3]
all_file[:,6] = ob0_file[:,3]
all_file[:,7] = ns_file[:,3]

all_file[:,8] = om0_file[:,6]
all_file[:,9] = om0_file[:,5]
all_file[:,10] = ob0_file[:,5]
all_file[:,11] = ns_file[:,5]

all_file[:,12] = om0_file[:,8]
all_file[:,13] = om0_file[:,7]
all_file[:,14] = ob0_file[:,7]
all_file[:,15] = ns_file[:,7]

all_file.tofile('taylor_coeffs.bin')
