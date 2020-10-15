# panel to use radmc3dPy in using radmc3d
import os
import numpy as np
import matplotlib.pyplot as plt
import fneq
import fntools
from radmc3dPy import *

# check out all the available models
models.getModelNames()

# look at description of disk_kwon2011
models.getModelDesc('disk_zyl8')

analyze.writeDefaultParfile(model='disk_zyl8')

op = dustopac.radmc3dDustOpac()
mopac = op.readMasterOpac()
opac_align = mopac['align'][0]
if opac_align: 
    opac_align = '1'
else:
    opac_align = '0'

# setup parameters
model = setup.radmc3dModel(binary=False,
	model='disk_zyl8', 
        # code, parameters
        nphot_scat='1e4',nphot='5e4', scattering_mode_max=1,
        mc_scat_maxtauabs='15',
        setthreads='16', alignment_mode=opac_align, modified_random_walk='1',
        incl_lines=0, lines_mode=1,
        # coordinates
        nx='[20,30,20]',xbound='[0.5*au, 30*au, 120*au, 250*au]',
        nz='[0]',
        # star
        mstar='[0.25*ms]', tstar='[5000]', rstar='[4.*rs]',
        # envelope
        dMenv='5e-5', Rc='70*au', 
        Rcav='1*au', Hcav='0.0025*au', qcav='2', Hoff='20*au', delHcav='2*au', 
        # disk parameters
        mdisk='0.1*ms',
        Rsig='70.*au', sigp='0.8', sigma_type='1',
        Rinner='0.1*au', Rt='35.*au', Router='70.*au',
        H0='8*au', qheight='1.25', zqratio='4',
        T0mid='120', qmid='-0.5', 
        T0atm='150', qatm='-0.5',
        g2d='0.01', altype="'z'"
        )
# a100 to a50 is about 6.8 times at 870 micron
domctherm = 1
dohydroeq = 0 
itermax = 1

# read parameter and print to visualize
par = analyze.readParams()
#par.printPar()

model.writeRadmc3dInp()
model.makeGrid(writeToFile=True)
model.makeRadSources(writeToFile=True)
model.makeVar(ddens=True, writeToFile=True)
#model.makeVar(alvec=True, writeToFile=True)
#model.makeVar(qvis=True, writeToFile=True)

if domctherm is 0:
    model.makeVar(dtemp=True, writeToFile=True)
else:
    zylutils.set_dtemp.pipeline(dohydroeq=dohydroeq, itermax=itermax, scatmode='1')

#-- line radiative transfer related

#model.writeLinesInp()
#model.makeVar(gdens=True, writeToFile=True)
#model.makeVar(gtemp=True, writeToFile=True)
#model.makeVar(gvel=True, writeToFile=True)
#model.makeVar(vturb=True, writeToFile=True)
#image.makeImage(npix=300, incl=45, phi=0, sizeau=300, stokes=True, widthkms=5., linenlam=20, imolspec=1, iline=2) 

