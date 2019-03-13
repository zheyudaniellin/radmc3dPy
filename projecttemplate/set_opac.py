# set_opac.py
# sets up opacity

import numpy as np
import matplotlib.pyplot as plt
import fneq 
import fntools
from radmc3dPy import *

# --------------
# set up opacity
# --------------
dustspec = zylutils.set_dustspec.dustspec()
dustspec.getSpec(specs=['Sil_Draine', 'WarrenWater', 'Henning_Organics', 'Henning_Troilite'])
mixabun = np.array([0.20, 0.33, 0.07,0.40], dtype=np.float64)

#dustspec.getSpec(specs=['Sil_Draine', 'Henning_Troilite'])
#mixabun = np.array(dustspec.mabun) / sum(dustspec.mabun)
#mixabun = np.array([0.20, 0.4, 0.40], dtype=np.float64)

## set up ppar
ppar = {'nw':[25, 50, 60], 'wbound':[0.1, 7.0, 25., 1.5e4],
        'lnk_fname':dustspec.lnkfiles,
        'nscatang':181, 'chopforward':0,
        'scattering_mode_max':1,
        'na':20, 'logawidth':0.05,
        'gsmin':10., 'gsmax':100., 'ngs':1, 
        'gsdist_powex':-3.5, 'mixgsize':0,
        'gdens':dustspec.swgt, 'mixabun':mixabun,
        'alignment_mode':0,
        'miescat_verbose':False,
        'wfact':3.0, 'extrapolate':True, 'errtol':0.01, 
        'kpara0':0.05
        }
# nscatang = 2881 is good

grid = reggrid.radmc3dGrid()
grid.makeWavelengthGrid(ppar=ppar)
wav = grid.wav

op = dustopac.radmc3dDustOpac()
#op.makeOpac(ppar=ppar, ksca0=1.)
op.makeBeckOpac(wav=wav, beck0=10, beta=0.6)

masteropac = op.readMasterOpac()
ext = masteropac['ext']
for idust in range(len(ext)):
    op.readOpac(ext=ext[idust], scatmat=masteropac['scatmat'][idust])

if ppar['alignment_mode'] is 1:
    op.makeDustAlignFact(ppar=ppar, wav=wav)

