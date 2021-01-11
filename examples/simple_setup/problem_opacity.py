# problem_opacity.py
# calculate the opacity of the grain 
import numpy as np
import matplotlib.pyplot as plt
import radmc3dPy as rmc
import dsharp_opac as dso

# ==== load the opacity ====
d      = np.load(dso.get_datafile('default_opacities_smooth.npz'))
agrid      = d['a']
wgrid    = d['lam']
k_abs  = d['k_abs']
k_sca  = d['k_sca']
gsca   = d['g']

# ==== pick out the desired sizes ====
a = [1e-3]
name = ['small']
ngsize = len(a)

# organize it into radmc3dPy format 
opac = rmc.dustopac.radmc3dDustOpac()
for ii in range(ngsize):
    opac.wav.append(wgrid)
    opac.nwav.append(len(opac.wav[ii]))
    inx = np.argmin(abs(a[ii] - agrid))
    opac.kabs.append(k_abs[inx,:])
    opac.ksca.append(k_sca[inx,:])
    opac.phase_g.append(gsca[inx,:])

# write out the file
for ii in range(ngsize):
    opac.writeOpac(ext=name[ii], idust=ii)
opac.writeMasterOpac(ext=name, scattering_mode_max=2)
opac.writeDustInfo(ext=name, matdens=[1]*ngsize, gsize=a, dweights=np.ones(ngsize)/ngsize)


# ==== plotting ====
plt.loglog(wgrid, k_abs[20, :])
plt.loglog(wgrid, k_sca[20, :], '--')
plt.show()


