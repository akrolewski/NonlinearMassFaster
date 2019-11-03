#!/usr/bin/env python3
#
# Pre-computes the HaloFit tables used to model P(k).
#
import camb
import numpy as np

z=0.

kmin=1e-3
kmax=1e4

hh = 0.6766


# Set the cosmological parameters for our model.
cp = camb.set_params(ns=0.9665, H0=67.66, ombh2=0.02242, omch2=0.11933, w=-1.0, Alens=1.0, lmax=2000, As=(0.87/0.81)*np.exp(3.047)*10**-10,
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

#kmin = -3.0
#kmax = 4.0
dk = 800 * 7.0

kk = np.logspace(np.log10(kmin),np.log10(kmax),dk)
pk = PK_NN.P(0,kk)

fn = "pklin/sig8_087.txt"
ff = open(fn,"w")
ff.write("# The (real space) linear power spectrum from CAMB.\n")
ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
for i in range(kk.size):
	ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
ff.close()


cp = camb.set_params(ns=0.9665, H0=67.66, ombh2=0.02242, omch2=0.11933, w=-1.0, Alens=1.0, lmax=2000, As=(0.75/0.81)*np.exp(3.047)*10**-10,
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

#kmin = -3.0
#kmax = 4.0
dk = 800 * 7.0

kk = np.logspace(np.log10(kmin),np.log10(kmax),dk)
pk = PK_NN.P(0,kk)

fn = "pklin/sig8_075.txt"
ff = open(fn,"w")
ff.write("# The (real space) linear power spectrum from CAMB.\n")
ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
for i in range(kk.size):
	ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
ff.close()






cp = camb.set_params(ns=0.9669, H0=67.66, ombh2=0.0490* hh **2., omch2=(0.3129 -0.0490)* hh **2., w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
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

#kmin = -3.0
#kmax = 4.0
dk = 800 * 7.0

kk = np.logspace(np.log10(kmin),np.log10(kmax),dk)
pk = PK_NN.P(0,kk)

fn = "pklin/random1.txt"
ff = open(fn,"w")
ff.write("# The (real space) linear power spectrum from CAMB.\n")
ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
for i in range(kk.size):
	ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
ff.close()




cp = camb.set_params(ns=0.9676, H0=67.66, ombh2=0.0498* hh **2., omch2=(0.3185 -0.0498)* hh **2., w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
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

#kmin = -3.0
#kmax = 4.0
dk = 800 * 7.0

kk = np.logspace(np.log10(kmin),np.log10(kmax),dk)
pk = PK_NN.P(0,kk)

fn = "pklin/random2.txt"
ff = open(fn,"w")
ff.write("# The (real space) linear power spectrum from CAMB.\n")
ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
for i in range(kk.size):
	ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
ff.close()



cp = camb.set_params(ns=0.9616, H0=67.66, ombh2=0.0479* hh **2., omch2=(0.3145 -0.0479)* hh **2., w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
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

#kmin = -3.0
#kmax = 4.0
dk = 800 * 7.0

kk = np.logspace(np.log10(kmin),np.log10(kmax),dk)
pk = PK_NN.P(0,kk)

fn = "pklin/random3.txt"
ff = open(fn,"w")
ff.write("# The (real space) linear power spectrum from CAMB.\n")
ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
for i in range(kk.size):
	ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
ff.close()



cp = camb.set_params(ns=0.9591, H0=67.66, ombh2=0.0507* hh **2., omch2=(0.3086 -0.0507)* hh **2., w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
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

#kmin = -3.0
#kmax = 4.0
dk = 800 * 7.0

kk = np.logspace(np.log10(kmin),np.log10(kmax),dk)
pk = PK_NN.P(0,kk)

fn = "pklin/random4.txt"
ff = open(fn,"w")
ff.write("# The (real space) linear power spectrum from CAMB.\n")
ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
for i in range(kk.size):
	ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
ff.close()



cp = camb.set_params(ns=0.9663, H0=67.66, ombh2=0.0504* hh **2., omch2=(0.3004 -0.0504)* hh **2., w=-1.0, Alens=1.0, lmax=2000, As=np.exp(3.047)*10**-10,
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

#kmin = -3.0
#kmax = 4.0
dk = 800 * 7.0

kk = np.logspace(np.log10(kmin),np.log10(kmax),dk)
pk = PK_NN.P(0,kk)

fn = "pklin/random5.txt"
ff = open(fn,"w")
ff.write("# The (real space) linear power spectrum from CAMB.\n")
ff.write("# {:>13s} {:>15s}\n".format("k[h/Mpc]","P(k)"))
for i in range(kk.size):
	ff.write("{:15.5e} {:15.5e}\n".format(kk[i],pk[i]))
ff.close()
