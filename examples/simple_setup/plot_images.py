import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import radmc3dPy as rmc

au = rmc.natconst.au

# distance in pc
dpc = 140.     # Distance in parsec (for conversion to Jy/pixel in 1.3 mm map)

#
# Make and plot image of full disk at 1.3 mm: thermal dust emission
#
imwav = np.array([850, 1300])
rmc.image.writeCameraWavelength(camwav=imwav)
rmc.image.makeImage(npix=200,incl=60.,phi=0.,sizeau=200,loadlambda=True)   # This calls radmc3d

# read the image once the calculation is done
im = rmc.image.readImage()

# ==== now plot the image ====
fig = plt.figure()
ax = plt.gca()

wavinx = 0
# the image is a function of x, y, and wavelength
pc = ax.pcolormesh(im.x/au, im.y/au, im.image[:,:,wavinx].T, norm=colors.LogNorm(), cmap=plt.cm.gist_heat)
plt.colorbar(pc, ax=ax)
ax.set_xlabel('x [au]')
ax.set_ylabel('y [au]')
plt.show()


