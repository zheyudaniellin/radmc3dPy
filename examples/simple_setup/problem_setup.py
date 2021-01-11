# problem_setup.py
# prepare files necessary for creating images
import numpy as np
import matplotlib.pyplot as plt
import radmc3dPy as rmc
au = rmc.natconst.au
rad = np.pi / 180

# ==== prepare spatial grid ====
grid = rmc.reggrid.radmc3dGrid()
grid.makeSpatialGrid(
    crd_sys='sph', act_dim=[1,1,0],
    xbound=[1*au, 120*au], nx=[128],
    ybound=[60*rad, np.pi/2.], ny=[64])

grid.writeSpatialGrid()

# ==== prepare wavelength grid ====
grid.makeWavelengthGrid(wbound=[100, 400., 1e4], nw=[10, 100])
grid.writeWavelengthGrid()

# create the spatial arrays that'll be needed for calculations 
# radius, theta, phi
rr, tt, pp = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
zz = rr * np.cos(tt)

# 
# ==== density distribution ====
# 
# parameters
# 
R0 = 100 * au
sig0 = 0.1
H0 = 10 * au
p = 1.
sigma = sig0 * (rr / R0)**(-p)
height = H0 * (rr / R0)**(1.25)

# calculate the gas distribution 
rhogas = sigma / np.sqrt(2*np.pi) / height * np.exp(-0.5 * (zz / height)**2)

# the dust distribution depends on the gas
ngsize = 1
mass_fraction = np.array([1])
gas_to_dust = 100.
rhod = np.zeros([grid.nx, grid.ny, grid.nz, ngsize])
for ii in range(ngsize):
    rhod[:,:,:,ii] = rhogas * mass_fraction[ii] / gas_to_dust

# ==== temperature distribution ====
#
# parameters
# 
T0 = 10.
q = 0.5
temp = np.zeros([grid.nx, grid.ny, grid.nz, ngsize])
for ii in range(ngsize):
    temp[:,:,:,ii] = T0 * (rr / R0)**(-q)

# ==== output structure information ====
dat = rmc.data.radmc3dData()
dat.grid = grid
dat.rhodust = rhod
dat.dusttemp = temp

dat.writeDustDens()
dat.writeDustTemp()

# ==== radmc3d settings ====
par = {
    'nphot_scat': '1000000',
    'istar_sphere': '0',
#    'setthreads': '4',
    'mc_scat_maxtauabs': '10', 
    'scattering_mode_max': 1
      }
fname = 'radmc3d.inp'
with open(fname, 'w') as wfile:
    for ikey in par.keys():
        wfile.write('%s = %s\n'%(ikey, par[ikey]))
    wfile.close()

# ==== Write the stars.inp file ====
# some star parameters 
rstar = rmc.natconst.rs
mstar = rmc.natconst.ms
tstar = 5000.
pstar = [0, 0, 0]

# now write the file
with open('stars.inp','w+') as f:
    f.write('2\n')
    f.write('1 %d\n\n'%(grid.nwav))
    f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(rstar,mstar,pstar[0],pstar[1],pstar[2]))
    for value in grid.wav:
        f.write('%13.6e\n'%(value))
    f.write('\n%13.6e\n'%(-tstar))


