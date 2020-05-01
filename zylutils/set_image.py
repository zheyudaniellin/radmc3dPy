# create image
from radmc3dPy import image, dustopac, setup
import time
import fntools
import numpy as np
import os

#-------------------- make image ------------------------
def makemcImage(inc=[0., 45., 75., 90.], posang=None, 
                camwav=[434., 870., 2600., 9098.4], wavfname=None, 
                npix=300, sizeau=None, phi=0., 
                dotausurf=False, dooptdepth=False, dostokes=True, 
                secondorder=False):
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
            sizeau=sizeau, stokes=dostokes, secondorder=secondorder, nostar=True, 
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
                # change radmc3d.inp
                setup.modifyRadmc3dInp({'alignment_mode':'0'})
#                par = analyze.readParams()
#                par.ppar['alignment_mode'] = 0
#                setup.writeRadmc3dInp(modpar=par)

        if dotausurf:
            fname = 'mytausurf1.i%d.out'%(inc[ii])
            image.makeImage(npix=npix, loadlambda=1, incl=inc[ii], phi=phi, 
                sizeau=sizeau,stokes=True, secondorder=1, nostar=True, 
                posang=posang, nphot_scat=1e3, 
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
                nphot_scat=1e3,# hard set nphot_scat, any low value is enough
                fname=fname,tracetau=True)

        if (dotausurf or dooptdepth):
            if original_align:
                # redo the settings
                op.writeMasterOpac(ext=mopac['ext'],therm=mopac['therm'],
                    scattering_mode_max=5, alignment_mode=1)
                setup.modifyRadmc3dInp({'alignment_mode':'1'})
#                par = analyze.readParams()
#                par.ppar['alignment_mode'] = 1
#                setup.writeRadmc3dInp(modpar=par)

    # change the 'camera_wavelength_micron.inp' to wavfname
    if wavfname is not None:
        os.system('mv camera_wavelength_micron.inp '+wavfname)

    #-----------------------------
    clock_stop = time.time()

    print('Making Image total elapsed time = ')
    print(clock_stop - clock_begin)
    #-----------------------------

def makemcSpectrum(inc=[0.], posang=None,
                camwav=[0.1, 1., 10., 100., 1e3, 1e4],
                npix=300, sizeau=None, phi=0., 
                nostar=False, noscat=False, secondorder=True):
    """ creates spectrum
    """

    inc = np.array(inc, dtype=np.float64)
    ninc= len(inc)

    image.writeCameraWavelength(camwav=camwav)

    # write an inp.spectruminc 
    dum = fntools.zylwritevec(inc, 'inp.spectruminc')

    clock_begin = time.time()

    for ii in range(ninc):
        fname = 'myspectrum.i%d.out'%(inc[ii])

        image.makeSpectrum(npix=npix, incl=inc[ii], sizeau=sizeau,
              phi=phi, posang=posang, nostar=nostar, noscat=noscat, 
              secondorder=secondorder,
              loadlambda=True,
              fname=fname)

    clock_stop = time.time()
    print('Making spectrum total elapsed time = ')
    print(clock_stop - clock_begin)


#--------------------------- line ------------------------
#image.makeImage(npix=300, incl=45, phi=0, sizeau=300, stokes=True, widthkms=5., linenlam=20, imolspec=1, iline=2) 

