# plot_images.py
# this file produces the image using radmc3d and plots it
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import radmc3dPy as rmc
au = rmc.natconst.au

# distance in pc
dpc = 140.     # Distance in parsec (for conversion to Jy/pixel in 1.3 mm map)

# Make and plot image of full disk at different wavelengths 
imwav = np.array([850, 1300]) # [micron]
rmc.image.writeCameraWavelength(camwav=imwav)

# ==== creating the image ====
# this line uses radmc3dPy to calculate image
#rmc.image.makeImage(npix=200,incl=60.,phi=0.,sizeau=200,loadlambda=True)

# in cases when you don't want to use that, directly use the command line
cmd = './radmc3d.exe image npix 200 incl 60 sizeau 200 loadlambda secondorder'
os.system(cmd)

# read the image once the calculation is done
im = rmc.image.readImage()

# ==== now plot the image ====
fig = plt.figure()
ax = plt.gca()

# choose a wavelength index 
wavinx = 0

# the image is a function of x, y, and wavelength
# - you can access the image values by the im.image attribute
# - the image intensity units are in cgs units: erg / cm2 / s / ster / Hz
pc = ax.pcolormesh(im.x/au, im.y/au, im.image[:,:,wavinx].T, 
    norm=colors.LogNorm(), cmap=plt.cm.gist_heat)
plt.colorbar(pc, ax=ax)
ax.set_xlabel('x [au]')
ax.set_ylabel('y [au]')
ax.set_aspect('equal')
plt.show()


