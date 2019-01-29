""" simALMA.py

collection of functions to simulate alma observations

assume in casa environment

"""

from datetime import datetime
import numpy as np
import os, glob
import shutil

os.getenv('CASAPATH').split(' ')[0] + '/data/alma/simmos/'
sim_start = str(datetime.now()

def getSimobserve(pref, fitsname, 
        indirection = 'J2000 23h59m59.96s -34d59m59.50s', 
        incell='0.06arcsec', incenter='330.0GHz', inwidth='1875MHz', inbright='0.1', 
        antennalist='alma.cycle5.4.cfg', obsmode='int', totaltime='360s', pwv=0.6, 
        mapsize='5arcsec'):
    default(simobserve)

    print '>>> simobserve underway'

    project = pref		# root prefix for output file names

    overwrite = True

    # set up the sky
    skymode = fitsname		# the input sky image you are using for simulation

    indirection = indirection	# set to the southern sky
    incell = incell		# rescale pixel size
    inbright = inbright		# peak brightness to 0.1 Jy/pixel
    incenter = incenter		# set image central frequency
    inwidth = inwidth		# set spectral window width

    # set up the obs
    antennalist = antennalist	
    obsmode = obsmode		# 'int' for interferometer, 'sd' for single dish
    totaltime = totaltime	#set the 12m array observing time
    pwv = pwv			# will add noise to the data based on this setting
    mapsize = mapsize		# primary beamsize
    simobserve()
    print('simobserve complete')

def getSimanalye(pref, ):
    default(simanalyze)
    project = pref
    overwrite = True
    # set up image
    vis = vis
    imsize = imsize
    cell = cell
    niter = niter
    threshold = 
