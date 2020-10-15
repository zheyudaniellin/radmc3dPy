import numpy as np
from matplotlib import pyplot as plt
import radmc3dPy as rmc

#
# Plot the opacity table
#
op = rmc.dustopac.radmc3dDustOpac()
mop = op.readMasterOpac()
op.readOpac(ext=mop['ext'])

plt.figure()
plt.loglog(op.wav[0],op.kabs[0],label=r'$\kappa_\nu^{\mathrm{abs}}$ (absorption)')
plt.loglog(op.wav[0],op.ksca[0],':',label=r'$\kappa_\nu^{\mathrm{scat}}$ (scattering)')
plt.ylim((1e-2,1e5))
plt.xlabel(r'$\lambda\;[\mu\mathrm{m}]$')
plt.ylabel(r'$\kappa_\nu\;[\mathrm{cm}^2/\mathrm{g}]$')
plt.title(r'Dust opacity')
plt.legend()
plt.show()
