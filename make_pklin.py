import camb
import numpy as np

z=0.

kmin=1e-3
kmax=1e4

hh = 0.6766

 
cp = camb.set_params(ns=0.9665, H0=67.66, ombh2=0.02242, omch2=0.11933, w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
                     mnu = 0.06, neutrino_hierarchy='degenerate',num_massive_neutrinos=1,YHe=0.2454,

                     WantTransfer=True, dark_energy_model='DarkEnergyPPF',

                     kmax=kmax, redshifts=[z],  Want_CMB=False, Want_CMB_lensing =False,WantCls=False)

r = camb.get_results(cp)

k, z, P = r.get_linear_matter_power_spectrum('delta_nonu','delta_nonu')

PK_NN = camb.get_matter_power_interpolator(cp, nonlinear=False, 
			hubble_units=True, k_hunit=True, kmax=kmax,
			 zmax=0.1,
			 var1 = 'delta_nonu', var2 = 'delta_nonu')
			 
z = 0

dk = 800 * 7.0

kk = np.logspace(np.log10(kmin),np.log10(kmax),dk)
pk = PK_NN.P(0,kk)

fn = "pklin/default.txt"
ff = open(fn,"w")
ff.write("# The (real space) linear power spectrum from CAMB.\n")
ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
for i in range(kk.size):
	ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
ff.close()


om0 = 0.3096
om0_err = 0.0056
redshift = 0

om0s = [om0 - 5*om0_err, om0 - om0_err, om0, om0 + om0_err, om0 + 5 * om0_err]
names = ['om0_minus5','om0_minus1','default','om0_plus1','om0_plus5']

for i,om0_ind in enumerate(om0s):
	cp = camb.set_params(ns=0.9665, H0=67.66, ombh2=0.02242, omch2=om0_ind*hh**2. - 0.02242, w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
						 mnu = 0.06, neutrino_hierarchy='degenerate',num_massive_neutrinos=1,YHe=0.2454,

						 WantTransfer=True, dark_energy_model='DarkEnergyPPF',

						 kmax=kmax, redshifts=[z],  Want_CMB=False, Want_CMB_lensing =False,WantCls=False)

	r = camb.get_results(cp)

	k, z, P = r.get_linear_matter_power_spectrum('delta_nonu','delta_nonu')

	PK_NN = camb.get_matter_power_interpolator(cp, nonlinear=False, 
				hubble_units=True, k_hunit=True, kmax=kmax,
				 zmax=0.1,
				 var1 = 'delta_nonu', var2 = 'delta_nonu')

	#
	pk = PK_NN.P(0,kk)

	fn = "pklin/%s.txt" % names[i]
	ff = open(fn,"w")
	ff.write("# The (real space) linear power spectrum from CAMB.\n")
	ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
	for i in range(kk.size):
		ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
	ff.close()
	
	
ob0 = 0.04897
ob0_err = 0.001
redshift = 0


ob0s = [ob0 - 5*ob0_err, ob0 - ob0_err, ob0, ob0 + ob0_err, ob0 + 5 * ob0_err]
names = ['ob0_minus5','ob0_minus1','default','ob0_plus1','ob0_plus5']

for i,ob0_ind in enumerate(ob0s):
	cp = camb.set_params(ns=0.9665, H0=67.66, ombh2=ob0_ind*hh**2., omch2=0.11933, w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
						 mnu = 0.06, neutrino_hierarchy='degenerate',num_massive_neutrinos=1,YHe=0.2454,

						 WantTransfer=True, dark_energy_model='DarkEnergyPPF',

						 kmax=kmax, redshifts=[z],  Want_CMB=False, Want_CMB_lensing =False,WantCls=False)

	r = camb.get_results(cp)

	k, z, P = r.get_linear_matter_power_spectrum('delta_nonu','delta_nonu')

	PK_NN = camb.get_matter_power_interpolator(cp, nonlinear=False, 
				hubble_units=True, k_hunit=True, kmax=kmax,
				 zmax=0.1,
				 var1 = 'delta_nonu', var2 = 'delta_nonu')

	#
	pk = PK_NN.P(0,kk)

	fn = "pklin/%s.txt" % names[i]
	ff = open(fn,"w")
	ff.write("# The (real space) linear power spectrum from CAMB.\n")
	ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
	for i in range(kk.size):
		ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
	ff.close()	


ns = 0.9665
ns_err = 0.0038
redshift = 0


nsss = [ns - 5 * ns_err, ns - ns_err, ns, ns + ns_err, ns + 5 * ns_err]
names = ['ns_minus5','ns_minus1','default','ns_plus1','ns_plus5']

for i,ns_ind in enumerate(nsss):
	cp = camb.set_params(ns=ns_ind, H0=67.66, ombh2=0.02242, omch2=0.11933, w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
						 mnu = 0.06, neutrino_hierarchy='degenerate',num_massive_neutrinos=1,YHe=0.2454,

						 WantTransfer=True, 

						 kmax=kmax, redshifts=[z],  Want_CMB=False, Want_CMB_lensing =False,WantCls=False)

	r = camb.get_results(cp)

	k, z, P = r.get_linear_matter_power_spectrum('delta_nonu','delta_nonu')

	PK_NN = camb.get_matter_power_interpolator(cp, nonlinear=False, 
				hubble_units=True, k_hunit=True, kmax=kmax,
				 zmax=0.1,
				 var1 = 'delta_nonu', var2 = 'delta_nonu')

	#
	pk = PK_NN.P(0,kk)

	fn = "pklin/%s.txt" % names[i]
	ff = open(fn,"w")
	ff.write("# The (real space) linear power spectrum from CAMB.\n")
	ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
	for i in range(kk.size):
		ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
	ff.close()