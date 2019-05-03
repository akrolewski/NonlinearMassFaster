# source activate nbodykit-env on nersc
from nbodykit.cosmology import correlation
from nbodykit import cosmology
import numpy as np
import os

om0 = 0.3111
om0_err = 0.0056
redshift = 0

om0s = [om0 - 5*om0_err, om0 - om0_err, om0, om0 + om0_err, om0 + 5 * om0_err]
names = ['om0_minus5','om0_minus1','default','om0_plus1','om0_plus5']

for i,om0 in enumerate(om0s):
	c = cosmology.Cosmology(Omega0_cdm=om0-0.04897,h=0.6766,Omega0_b=0.04897,sigma8=0.8102,ns=0.9665,T0_cmb=2.7255)
	c = c.clone(n_s=0.9665)
	c = c.match(sigma8=0.8102)
	Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
	cf_lin = cosmology.CorrelationFunction(Plin)

	R = np.linspace(0,20,2e4+1)
	cflinr = cf_lin(R)
	interp_cf_lin = lambda x: np.interp(x, R, cflinr)
	
	os.system('mkdir cosmo_files/%s' % names[i])

	np.savetxt('cosmo_files/%s/cfz0_2e4.txt' % names[i],np.array([R,cflinr]).transpose())

	Dz = c.scale_independent_growth_factor(np.linspace(0,6,601))

	np.savetxt('cosmo_files/%s/Dz.txt' % names[i],Dz)

	norm = Plin(1.0)

	f = open('cosmo_files/%s/norm.txt' % names[i],'w')
	f.write('%.10f' % norm)
	f.close()
	
ob0 = 0.04897
ob0_err = 0.001
redshift = 0

ob0s = [ob0 - 5*ob0_err, ob0 - ob0_err, ob0, ob0 + ob0_err, ob0 + 5 * ob0_err]
names = ['ob0_minus5','ob0_minus1','default','ob0_plus1','ob0_plus5']

for i,ob0 in enumerate(ob0s):
	c = cosmology.Cosmology(Omega0_cdm=0.3111-ob0,h=0.6766,Omega0_b=ob0,sigma8=0.8102,ns=0.9665,T0_cmb=2.7255)
	c = c.clone(n_s=0.9665)
	c = c.match(sigma8=0.8102)
	Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
	cf_lin = cosmology.CorrelationFunction(Plin)

	R = np.linspace(0,20,2e4+1)
	cflinr = cf_lin(R)
	interp_cf_lin = lambda x: np.interp(x, R, cflinr)
	
	os.system('mkdir cosmo_files/%s' % names[i])

	np.savetxt('cosmo_files/%s/cfz0_2e4.txt' % names[i],np.array([R,cflinr]).transpose())

	Dz = c.scale_independent_growth_factor(np.linspace(0,6,601))

	np.savetxt('cosmo_files/%s/Dz.txt' % names[i],Dz)

	norm = Plin(1.0)

	f = open('cosmo_files/%s/norm.txt' % names[i],'w')
	f.write('%.10f' % norm)
	f.close()
	
ns = 0.9665
ns_err = 0.0038
redshift = 0

nsss = [ns - 5 * ns_err, ns - ns_err, ns, ns + ns_err, ns + 5 * ns_err]
names = ['ns_minus5','ns_minus1','default','ns_plus1','ns_plus5']

for i,ns in enumerate(nsss):
	c = cosmology.Cosmology(Omega0_cdm=0.3111-0.04897,h=0.6766,Omega0_b=0.04897,sigma8=0.8102,ns=ns,T0_cmb=2.7255)
	c = c.clone(n_s=ns)
	c = c.match(sigma8=0.8102)
	Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
	cf_lin = cosmology.CorrelationFunction(Plin)

	R = np.linspace(0,20,2e4+1)
	cflinr = cf_lin(R)
	interp_cf_lin = lambda x: np.interp(x, R, cflinr)
	
	os.system('mkdir cosmo_files/%s' % names[i])

	np.savetxt('cosmo_files/%s/cfz0_2e4.txt' % names[i],np.array([R,cflinr]).transpose())

	Dz = c.scale_independent_growth_factor(np.linspace(0,6,601))

	np.savetxt('cosmo_files/%s/Dz.txt' % names[i],Dz)

	norm = Plin(1.0)

	f = open('cosmo_files/%s/norm.txt' % names[i],'w')
	f.write('%.10f' % norm)
	f.close()
	
	
om0 = 0.3111
om0_err = 0.0056

ob0 = 0.04897
ob0_err = 0.001

ns = 0.9665
ns_err = 0.0038

om0s = [om0 - 2*om0_err, om0, om0 + 2 * om0_err]
ob0s = [ob0 - 2*ob0_err, ob0, ob0 + 2 * ob0_err]
nsss = [ns - 2*ns_err, ns, ns + 2 * ns_err]

cnt = 0

for i,om0 in enumerate(om0s):
	for j,ob0 in enumerate(ob0s):
		for k,ns in enumerate(nsss):
			c = cosmology.Cosmology(Omega0_cdm=om0-ob0,h=0.6766,Omega0_b=ob0,sigma8=0.8102,ns=0.9665,T0_cmb=2.7255)
			c = c.clone(n_s=ns)
			c = c.match(sigma8=0.8102)
			Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
			cf_lin = cosmology.CorrelationFunction(Plin)

			R = np.linspace(0,20,2e4+1)
			cflinr = cf_lin(R)
			interp_cf_lin = lambda x: np.interp(x, R, cflinr)
	
			os.system('mkdir cosmo_files/%s' % cnt)

			np.savetxt('cosmo_files/%s/cfz0_2e4.txt' % cnt,np.array([R,cflinr]).transpose())

			Dz = c.scale_independent_growth_factor(np.linspace(0,6,601))

			np.savetxt('cosmo_files/%s/Dz.txt' % cnt,Dz)

			norm = Plin(1.0)

			f = open('cosmo_files/%s/norm.txt' % cnt,'w')
			f.write('%.10f' % norm)
			f.close()
			
			cnt += 1
			

om0 = 0.3111
om0_err = 0.0056

ob0 = 0.04897
ob0_err = 0.001

ns = 0.9665
ns_err = 0.0038

om0s = [om0 + 0.1 * om0_err, om0 + 0.2 * om0_err, om0 + 0.5 * om0_err]
ob0s = [ob0 + 0.1 * ob0_err, ob0 + 0.2 * ob0_err, ob0 + 0.5 * ob0_err]
nsss = [ns + 0.1 * ns_err, ns + 0.2 * ns_err, ns + 0.5 * ns_err]

cnt = 27

for i,om0 in enumerate(om0s):
	for j,ob0 in enumerate(ob0s):
		for k,ns in enumerate(nsss):
			c = cosmology.Cosmology(Omega0_cdm=om0-ob0,h=0.6766,Omega0_b=ob0,sigma8=0.8102,ns=0.9665,T0_cmb=2.7255)
			c = c.clone(n_s=ns)
			c = c.match(sigma8=0.8102)
			Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
			cf_lin = cosmology.CorrelationFunction(Plin)

			R = np.linspace(0,20,2e4+1)
			cflinr = cf_lin(R)
			interp_cf_lin = lambda x: np.interp(x, R, cflinr)
	
			os.system('mkdir cosmo_files/%s' % cnt)

			np.savetxt('cosmo_files/%s/cfz0_2e4.txt' % cnt,np.array([R,cflinr]).transpose())

			Dz = c.scale_independent_growth_factor(np.linspace(0,6,601))

			np.savetxt('cosmo_files/%s/Dz.txt' % cnt,Dz)

			norm = Plin(1.0)

			f = open('cosmo_files/%s/norm.txt' % cnt,'w')
			f.write('%.10f' % norm)
			f.close()
			
			cnt += 1
			
			
om0 = 0.3111
om0_err = 0.0056

ob0 = 0.04897
ob0_err = 0.001

ns = 0.9665
ns_err = 0.0038

om0s = [om0 + 0.1 * om0_err, om0 + 0.2 * om0_err, om0 + 0.5 * om0_err]
ob0s = [ob0 + 0.1 * ob0_err, ob0 + 0.2 * ob0_err, ob0 + 0.5 * ob0_err]
nsss = [ns + 0.1 * ns_err, ns + 0.2 * ns_err, ns + 0.5 * ns_err]

cnt = 27

for i,om0 in enumerate(om0s):
	#for j,ob0 in enumerate(ob0s):
	#	for k,ns in enumerate(nsss):
	c = cosmology.Cosmology(Omega0_cdm=om0-ob0,h=0.6766,Omega0_b=ob0,sigma8=0.8102,ns=0.9665,T0_cmb=2.7255)
	c = c.clone(n_s=ns)
	c = c.match(sigma8=0.8102)
	Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
	cf_lin = cosmology.CorrelationFunction(Plin)

	R = np.linspace(0,20,2e4+1)
	cflinr = cf_lin(R)
	interp_cf_lin = lambda x: np.interp(x, R, cflinr)

	os.system('mkdir cosmo_files/test')

	np.savetxt('cosmo_files/test/cfz0_2e4.txt',np.array([R,cflinr]).transpose())

	Dz = c.scale_independent_growth_factor(np.linspace(0,6,601))

	np.savetxt('cosmo_files/test/Dz.txt',Dz)

	norm = Plin(1.0)

	f = open('cosmo_files/test/norm.txt','w')
	f.write('%.10f' % norm)
	f.close()
	
	cnt += 1
	
	
	
om0 = 0.3111
om0_err = 0.0056

ob0 = 0.04897
ob0_err = 0.001

ns = 0.9665
ns_err = 0.0038

om0s = [om0 + 0.1 * om0_err, om0 + 0.2 * om0_err, om0 + 0.5 * om0_err]
#ob0s = [ob0 + 0.1 * ob0_err, ob0 + 0.2 * ob0_err, ob0 + 0.5 * ob0_err]
#nsss = [ns + 0.1 * ns_err, ns + 0.2 * ns_err, ns + 0.5 * ns_err]

cnt = 27

for i,om0 in enumerate(om0s):
	#for j,ob0 in enumerate(ob0s):
	#	for k,ns in enumerate(nsss):
	c = cosmology.Cosmology(Omega0_cdm=om0-ob0,h=0.6766,Omega0_b=ob0,sigma8=0.8102,ns=0.9665,T0_cmb=2.7255)
	c = c.clone(n_s=ns)
	c = c.match(sigma8=0.8102)
	Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
	cf_lin = cosmology.CorrelationFunction(Plin)

	R = np.linspace(0,20,2e4+1)
	cflinr = cf_lin(R)
	interp_cf_lin = lambda x: np.interp(x, R, cflinr)

	os.system('mkdir cosmo_files/test%i' % i)

	np.savetxt('cosmo_files/test%i/cfz0_2e4.txt' % i,np.array([R,cflinr]).transpose())

	Dz = c.scale_independent_growth_factor(np.linspace(0,6,601))

	np.savetxt('cosmo_files/test%i/Dz.txt' % i,Dz)

	norm = Plin(1.0)

	f = open('cosmo_files/test%i/norm.txt' % i,'w')
	f.write('%.10f' % norm)
	f.close()
	
	cnt += 1
	
	
om0 = 0.3111
om0_err = 0.0056

ob0 = 0.04897
ob0_err = 0.001

ns = 0.9665
ns_err = 0.0038

om0s = [om0 - 4.0 * om0_err, om0 , om0 + 4.0 * om0_err]
ob0s = [ob0 - 4.0 * ob0_err, ob0, ob0 + 4.0 * ob0_err]
nsss = [ns - 4.0 * ns_err, ns, ns + 4.0 * ns_err]

cnt = 54

for i,om0 in enumerate(om0s):
	for j,ob0 in enumerate(ob0s):
		for k,ns in enumerate(nsss):
			c = cosmology.Cosmology(Omega0_cdm=om0-ob0,h=0.6766,Omega0_b=ob0,sigma8=0.8102,ns=0.9665,T0_cmb=2.7255)
			c = c.clone(n_s=ns)
			c = c.match(sigma8=0.8102)
			Plin = cosmology.LinearPower(c,redshift=redshift,transfer='CLASS')
			cf_lin = cosmology.CorrelationFunction(Plin)

			R = np.linspace(0,20,2e4+1)
			cflinr = cf_lin(R)
			interp_cf_lin = lambda x: np.interp(x, R, cflinr)
	
			os.system('mkdir cosmo_files/%s' % cnt)

			np.savetxt('cosmo_files/%s/cfz0_2e4.txt' % cnt,np.array([R,cflinr]).transpose())

			Dz = c.scale_independent_growth_factor(np.linspace(0,6,601))

			np.savetxt('cosmo_files/%s/Dz.txt' % cnt,Dz)

			norm = Plin(1.0)

			f = open('cosmo_files/%s/norm.txt' % cnt,'w')
			f.write('%.10f' % norm)
			f.close()
			
			cnt += 1