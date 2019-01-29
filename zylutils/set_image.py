# create image
from radmc3dPy import *
import time
import fntools
import numpy as np
import os

#-------------------- make image ------------------------
def makemcImage(inc=[0., 45., 75., 90.], posang=None, 
                camwav=[434., 870., 2600., 9098.4],
                npix=300, sizeau=200., phi=0., 
                dotausurf=False, dooptdepth=False):
    inc = np.array(inc, dtype=np.float64)
    ninc= len(inc)

    image.writeCameraWavelength(camwav=camwav)

    # write an inp.imageinc 
    dum = fntools.zylwritevec(inc, 'inp.imageinc')

    clock_begin = time.time()

    # continuum
    for ii in range(ninc):
        fname = 'myimage.i%d.out'%(inc[ii])
        image.makeImage(npix=npix, loadlambda=1, incl=inc[ii], phi=phi, 
            sizeau=sizeau, stokes=True, secondorder=0, nostar=True, 
            posang=posang, 
            fname=fname)

    # --------- make the tau surface or optical depth imag---------
        if (dotausurf or dooptdepth):
            # cannot do aligned grains with other modes than imaging
            op = dustopac.radmc3dDustOpac()
            mopac = op.readMasterOpac()
            original_align = mopac['align'][0]
            if original_align:
                op.writeMasterOpac(ext=mopac['ext'], therm=mopac['therm'], 
                    scattering_mode_max=5, alignment_mode=0)
                par = analyze.readParams()
                par.ppar['alignment_mode'] = 0
                setup.writeRadmc3dInp(modpar=par)

        if dotausurf:
            fname = 'mytausurf1.i%d.out'%(inc[ii])
            image.makeImage(npix=npix, loadlambda=1, incl=inc[ii], phi=phi, 
                sizeau=sizeau,stokes=True, secondorder=1, nostar=True, 
                posang=posang, 
                fname=fname,tausurf=1.)
            if os.path.isfile('tausurface_3d.out'):
                fname = 'mytau3d.i%d.out'%(inc[ii])
                os.system('mv tausurface_3d.out '+fname)

        if dooptdepth:
            # create optical depth image
            fname = 'myoptdepth.i%d.out'%(inc[ii])
            image.makeImage(npix=npix, loadlambda=1, incl=inc[ii], phi=phi, 
                sizeau=sizeau, stokes=True, secondorder=1,nostar=True, 
                posang=posang, 
                fname=fname,tracetau=True)

        if (dotausurf or dooptdepth):
            if original_align:
                # redo the settings
                op.writeMasterOpac(ext=mopac['ext'],therm=mopac['therm'],
                    scattering_mode_max=5, alignment_mode=1)
                par = analyze.readParams()
                par.ppar['alignment_mode'] = 1
                setup.writeRadmc3dInp(modpar=par)

    #-----------------------------
    clock_stop = time.time()

    print 'Making Image total elapsed time = '
    print clock_stop - clock_begin
    #-----------------------------

#--------------------------- line ------------------------
#image.makeImage(npix=300, incl=45, phi=0, sizeau=300, stokes=True, widthkms=5., linenlam=20, imolspec=1, iline=2) 

