import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import colors
import radmc3dPy as rmc
rad = np.pi / 180
au = rmc.natconst.au

#
# Read the data
#
dat   = rmc.analyze.readData(dtemp=True, ddens=True, binary=False)
rr,tt = np.meshgrid(dat.grid.x,dat.grid.y,indexing='ij')
zzr = np.pi/2 - tt
zz   = rr * np.cos(tt)
rhod  = dat.rhodust[:,:,0,0]
temp  = dat.dusttemp[:,:,0,0]

# 
# Plot the 2D data in cartesian coordinates
#
fig, axes = plt.subplots(1, 2, squeeze=True)
ax = axes[0]
pc = ax.pcolormesh(rr/au, zz/au, rhod, norm=colors.LogNorm(vmin=1e-20, vmax=1e-16))
cbar = plt.colorbar(pc, ax=ax)
ax.set_title('Density')
ax.set_aspect('equal')

ax = axes[1]
pc = ax.pcolormesh(rr/au, zz/au, temp, norm=colors.LogNorm(), vmin=5)
cbar = plt.colorbar(pc, ax=ax)
ax.set_title('Temperature')
ax.set_aspect('equal')

fig.tight_layout()
plt.show()

#
# Plot the vertical density structure at different radii
#
irr = [0,10,20,30]
plt.figure()
for ir in irr:
    r    = dat.grid.x[ir]
    rstr = '{0:4.0f}'.format(r/au)
    rstr = 'r = '+rstr.strip()+' au'
    plt.semilogy(zzr[ir,:],rhod[ir,:],label=rstr)
plt.ylim((1e-25,1e-15))
plt.xlabel(r'$\pi/2-\theta\simeq z/r$')
plt.ylabel(r'$\rho_{\mathrm{dust}}\;[\mathrm{g}/\mathrm{cm}^3]$')
plt.legend()

#
# Plot the vertical temperature structure at different radii
#
irr = [0,10,20,30]
plt.figure()
for ir in irr:
    r    = dat.grid.x[ir]
    rstr = '{0:4.0f}'.format(r/au)
    rstr = 'r = '+rstr.strip()+' au'
    plt.plot(zzr[ir,:],temp[ir,:],label=rstr)
plt.xlabel(r'$\pi/2-\theta\simeq z/r$')
plt.ylabel(r'$T_{\mathrm{dust}}\;[\mathrm{K}]$')
plt.legend()

plt.show()
