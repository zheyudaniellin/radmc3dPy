"""This module contains classes/functions to create and read images with RADMC-3D and to calculate
interferometric visibilities and write fits files
For help on the syntax or functionality of each function see the help of the individual functions

"""
from __future__ import absolute_import
from __future__ import print_function
import traceback
import copy
import subprocess as sp
import os

import pdb

try:
    import numpy as np
except ImportError:
    np = None
    print(' Numpy cannot be imported ')
    print(' To use the python module of RADMC-3D you need to install Numpy')
    print(traceback.format_exc())

try:
    import scipy.special as spc
except ImportError:
    spc = None
    print('scipy.special cannot be imported ')
    print('This module is required to be able to calculate Airy-PSFs. Now PSF calculation is limited to Gaussian.')
    print(traceback.format_exc())

try:
    from astropy.io import fits as pf
except ImportError:
    print('astropy.io.fits cannot be imported trying pyfits')
    try:
        import pyfits as pf
    except ImportError:
        pf = None
        print('pyfits cannot be imported. Either of these modules is needed to write RADMC-3D images '
              + 'to FITS format. The rest of radmc3dPy can be used but fits output is now disabled.')
        print(traceback.format_exc())

try:
    import matplotlib.pylab as plt
except ImportError:
    plt = None
    print('Warning')
    print('matplotlib.pyplot cannot be imported')
    print('Without matplotlib you can use the python module to set up a model but you will not be able to plot things')
    print('or display images')

try:
    from matplotlib.patches import Ellipse
except ImportError:
    Ellipse = None
    print('Warning')
    print('Ellipse cannot be imported from matplotlib.patches')
    print('Without this you cannot plot ellipses onto images')

from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import natconst as nc

class baseImage(object):
    """
    the parent of all images. This will keep the intensity, tau surface, optical depth information, etc

    Attributes
    ----------
    freq : 1d array
        frequency in Hz
    wav : 1d array
        wavelength in micron 
    image : ndarray
        image in [x, y, stokes, w]. typically the intensity (in cgs), but it can be optical depth, or
        the tau=1 location 

    imageJyppix : ndarray, optional
        The image with pixel units of Jy/pixel

    lpi : ndarray, optional
        linear polarized intensity

    lpol : ndarray, optional 
        linear polarization fraction 

    lpa : ndarray, optional
        linear polarization angle

    cpol : ndarray, optional 
        circular polarization fraction 

    pol : ndarray, optional 
        polarization fraction 
    """
    def __init__(self):
        self.image = None
        self.stokes = None

    def isStokes(self, iformat):
        """
        determines if the image is a 4D image based on iformat
        """
        if iformat == 3:
            self.stokes = True
        else:
            self.stokes = False

    def getFreq(self):
        """ get frequency after wavelength is known 
        """
        self.freq = nc.cc * 1e4 / self.wav

    def getPolarization(self):
        """ calculate polarization related properties
        """
        if self.stokes is False:
            return

        if len(self.image.shape) == 3:
            dummyI = self.image[...,0]
            # linear polarized intensity 
            self.lpi = np.sqrt(np.sum(self.image[...,1:3]**2, axis=2))

            # total polarized intensity 
            self.pi = np.sqrt(np.sum(self.image[...,1:4]**2, axis=2))

            # polarization angle
            self.lpa = calcPolAng(self.image[:,:,1], self.image[:,:,2])
        else:
            dummyI = self.image[...,0,:]
            # linear polarized intensity 
            self.lpi = np.sqrt( np.sum(self.image[...,1:3,:]**2, axis=2) )

            # total polarized intensity 
            self.pi = np.sqrt( np.sum(self.image[...,1:4,:]**2, axis=2) )

            # polarization angle
            self.lpa = calcPolAng(self.image[:,:,1,:], self.image[:,:,2,:])

        # linear polarization fraction 
        self.lpol = self.lpi / (dummyI + 1e-90)

        # total polarization fraction 
        self.pol = self.pi / (dummyI + 1e-90)

    def getTotalFlux(self):
        if hasattr(self, 'pixel_area') is False:
            self.getPixelArea()

        self.totalflux = np.sum(self.image * self.pixel_area, axis=(0,1))

class radmc3dImage(baseImage):
    """
    RADMC-3D rectangular image class

    Attributes
    ----------

    image       : ndarray
                  The image as calculated by radmc3d (the values are intensities in erg/s/cm^2/Hz/ster)

    imageJyppix : ndarray
                  The image with pixel units of Jy/pixel

    tausurf     : ndarray
                  The tausurface iamge calculated by radmc3d. values are in cm. assume the settings are
                  the same as when making the image

    x           : ndarray
                  x coordinate of the image [cm]

    y           : ndarray
                  y coordinate of the image [cm]

    nx          : int
                  Number of pixels in the horizontal direction

    ny          : int   
                  Number of pixels in the vertical direction

    sizepix_x   : float
                  Pixel size in the horizontal direction [cm]

    sizepix_y   : float
                  Pixel size in the vertical direction [cm]

    nfreq       : int
                  Number of frequencies in the image cube

    freq        : ndarray
                  Frequency grid in the image cube

    nwav        : int
                  Number of wavelengths in the image cube (same as nfreq)

    wav         : ndarray
                  Wavelength grid in the image cube

    Optional
    lpol : ndarray
        linear polarization fraction 
    lpa : ndarray
        polarization angle
    lpi : ndarray 
        linear polarized intensity 
    pol : ndarray
        total polarization fraction 
    pi : ndarray
        total polarizated intensity
    """

    def __init__(self):
        self.image = None
        self.tausurf = None #try not to use this
        self.totflux = None
        self.x = None
        self.y = None
        self.nx = 0
        self.ny = 0
        self.sizepix_x = 0
        self.sizepix_y = 0
        self.nfreq = 0
        self.freq = None
        self.nwav = 0
        self.wav = None
        self.stokes = False
        self.psf = {}
        self.fwhm = []
        self.pa = 0
        self.dpc = 0
        self.rms = None
        self.filename = 'image.out'

    def getPixelArea(self):
        """ calculate pixel area 
        Assigns
        -------
        pixel_area
        """
        self.pixel_area = self.sizepix_x * self.sizepix_y

    def getJyppix(self, dpc):
        """ calculate jyppix image
        Assigns
        -------
        imageJyppix
        dpc
        """
        # calculate the pixel area
        if hasattr(self, 'pixel_area') is False:
            self.getPixelArea()

        # Conversion from erg/s/cm/cm/Hz/ster to Jy/pixel
        self.imageJyppix = self.image * self.pixel_area / (dpc * nc.pc)**2 * 1e23
        self.dpc = dpc

    def getClosurePhase(self, bl=None, pa=None, dpc=None):
        """Calculates clusure phases for a given model image for any arbitrary baseline triplet.

        Parameters
        ----------

        bl  : list/ndarray
              A list or ndrray containing the length of projected baselines in meter.

        pa  : list/ndarray
              A list or Numpy array containing the position angles of projected baselines in degree.

        dpc : distance of the source in parsec


        Returns
        -------
        Returns a dictionary with the following keys:

            * bl     : projected baseline in meter
            * pa     : position angle of the projected baseline in degree
            * nbl    : number of baselines
            * u      : spatial frequency along the x axis of the image
            * v      : spatial frequency along the v axis of the image
            * vis    : complex visibility at points (u,v)
            * amp    : correlation amplitude 
            * phase  : Fourier phase
            * cp     : closure phase
            * wav    : wavelength 
            * nwav   : number of wavelengths

        Notes
        -----

        bl and pa should either be an array with dimension [N,3] or if they are lists each element of
        the list should be a list of length 3, since closure phases are calculated only for closed triangles
        """

        ntri = len(bl)
        res = {'bl': np.array(bl, dtype=np.float64),
               'pa': np.array(pa, dtype=np.float64),
               'ntri': ntri,
               'nbl': 3,
               'nwav': self.nwav,
               'wav': self.wav,
               'u': np.zeros([ntri, 3, self.nwav], dtype=np.float64),
               'v': np.zeros([ntri, 3, self.nwav], dtype=np.float64),
               'vis': np.zeros([ntri, 3, self.nwav], dtype=np.complex64),
               'amp': np.zeros([ntri, 3, self.nwav], dtype=np.float64),
               'phase': np.zeros([ntri, 3, self.nwav], dtype=np.float64),
               'cp': np.zeros([ntri, self.nwav], dtype=np.float64)}

        # l = self.x / nc.au / dpc / 3600. / 180. * np.pi
        # m = self.y / nc.au / dpc / 3600. / 180. * np.pi
        # dl = l[1] - l[0]
        # dm = m[1] - m[0]

        for itri in range(ntri):
            print('Calculating baseline triangle # : ', itri)

            dum = self.getVisibility(bl=res['bl'][itri, :], pa=res['pa'][itri, :], dpc=dpc)
            res['u'][itri, :, :] = dum['u']
            res['v'][itri, :, :] = dum['v']
            res['vis'][itri, :, :] = dum['vis']
            res['amp'][itri, :, :] = dum['amp']
            res['phase'][itri, :, :] = dum['phase']
            res['cp'][itri, :] = (dum['phase'].sum(0) / np.pi * 180.) % 360.
            ii = res['cp'][itri, :] > 180.
            if (res['cp'][itri, ii]).shape[0] > 0:
                res['cp'][itri, ii] = res['cp'][itri, ii] - 360.

        return res

    # --------------------------------------------------------------------------------------------------
    def getVisibility(self, bl=None, pa=None, dpc=None):
        """Calculates visibilities for a given set of projected baselines and position angles
        with Discrete Fourier Transform.

        Parameters
        ----------

        bl  : list/ndarray
              A list or ndrray containing the length of projected baselines in meter.

        pa  : list/ndarray
              A list or Numpy array containing the position angles of projected baselines in degree.

        dpc : distance of the source in parsec

        Returns
        -------
        Returns a dictionary with the following keys:

            * bl     : projected baseline in meter
            * pa     : position angle of the projected baseline in degree
            * nbl    : number of baselines
            * u      : spatial frequency along the x axis of the image
            * v      : spatial frequency along the v axis of the image
            * vis    : complex visibility at points (u,v)
            * amp    : correlation amplitude 
            * phase  : Fourier phase
            * wav    : wavelength 
            * nwav   : number of wavelengths
        """

        nbl = len(bl)
        res = {'bl': np.array(bl, dtype=np.float64),
               'pa': np.array(pa, dtype=np.float64),
               'nbl': nbl,
               'nwav': self.nwav,
               'wav': self.wav,
               'u': np.zeros([nbl, self.nwav], dtype=np.float64),
               'v': np.zeros([nbl, self.nwav], dtype=np.float64),
               'stokes':self.stokes}
        if res['stokes']:
            res.update({'vis': np.zeros([nbl, 4, self.nwav], dtype=np.complex64),
                        'amp': np.zeros([nbl, 4, self.nwav], dtype=np.float64),
                        'phase': np.zeros([nbl, 4, self.nwav], dtype=np.float64)})
        else:
            res.update({'vis': np.zeros([nbl, self.nwav], dtype=np.complex64),
                        'amp': np.zeros([nbl, self.nwav], dtype=np.float64),
                        'phase': np.zeros([nbl, self.nwav], dtype=np.float64)})

        # res = {}
        # res['bl'] = np.array(bl, dtype=float)
        # res['pa'] = np.array(pa, dtype=float)
        # res['nbl'] = res['bl'].shape[0]
        #
        # res['wav'] = np.array(self.wav)
        # res['nwav'] = self.nwav
        #
        # res['u'] = np.zeros([res['nbl'], self.nwav], dtype=float)
        # res['v'] = np.zeros([res['nbl'], self.nwav], dtype=float)
        #
        # res['vis'] = np.zeros([res['nbl'], self.nwav], dtype=np.complex64)
        # res['amp'] = np.zeros([res['nbl'], self.nwav], dtype=np.float64)
        # res['phase'] = np.zeros([res['nbl'], self.nwav], dtype=np.float64)

        l = self.x / nc.au / dpc / 3600. / 180. * np.pi
        m = self.y / nc.au / dpc / 3600. / 180. * np.pi
        dl = l[1] - l[0]
        dm = m[1] - m[0]

        if res['stokes']:
            for istokes in range(4):
                for iwav in range(res['nwav']):
    
                    # Calculate spatial frequencies
                    res['u'][:, iwav] = res['bl'] * np.cos(res['pa'] * np.pi / 180.) * 1e6 / self.wav[iwav]
                    res['v'][:, iwav] = res['bl'] * np.sin(res['pa'] * np.pi / 180.) * 1e6 / self.wav[iwav]

                    for ibl in range(res['nbl']):
                        dum = complex(0.)
                        imu = complex(0., 1.)

                        for il in range(len(l)):
                            phase = 2. * np.pi * (res['u'][ibl, iwav] * l[il] + res['v'][ibl, iwav] * m)
                            cterm = np.cos(phase)
                            sterm = -np.sin(phase)
                            dum = dum + (self.image[il, :, istokes, iwav] * (cterm + imu * sterm)).sum() * dl * dm

                        res['vis'][ibl, istokes, iwav] = dum
                        res['amp'][ibl, istokes, iwav] = np.sqrt(abs(dum * np.conj(dum)))
                        res['phase'][ibl, istokes, iwav] = np.arccos(np.real(dum) / res['amp'][ibl, istokes, iwav])
                        if np.imag(dum) < 0.:
                            res['phase'][ibl, istokes, iwav] = 2. * np.pi - res['phase'][ibl, istokes, iwav]

#                        print('Calculating baseline # : ', ibl, ' stokes # : ', istokes, ' wavelength # : ', iwav)
        else:
            for iwav in range(res['nwav']):

                # Calculate spatial frequencies
                res['u'][:, iwav] = res['bl'] * np.cos(res['pa'] * np.pi / 180.) * 1e6 / self.wav[iwav]
                res['v'][:, iwav] = res['bl'] * np.sin(res['pa'] * np.pi / 180.) * 1e6 / self.wav[iwav]

                for ibl in range(res['nbl']):
                    dum = complex(0.)
                    imu = complex(0., 1.)

                    for il in range(len(l)):
                        phase = 2. * np.pi * (res['u'][ibl, iwav] * l[il] + res['v'][ibl, iwav] * m)
                        cterm = np.cos(phase)
                        sterm = -np.sin(phase)
                        dum = dum + (self.image[il, :, iwav] * (cterm + imu * sterm)).sum() * dl * dm

                    res['vis'][ibl, iwav] = dum
                    res['amp'][ibl, iwav] = np.sqrt(abs(dum * np.conj(dum)))
                    res['phase'][ibl, iwav] = np.arccos(np.real(dum) / res['amp'][ibl, iwav])
                    if np.imag(dum) < 0.:
                        res['phase'][ibl, iwav] = 2. * np.pi - res['phase'][ibl, iwav]

#                    print('Calculating baseline # : ', ibl, ' wavelength # : ', iwav)

        return res

    # --------------------------------------------------------------------------------------------------
    def writeFits(self, fname='', dpc=1., coord='03h10m05s -10d05m30s', bandwidthmhz=2000.0,
                  casa=False, nu0=0., stokes='I', fitsheadkeys=[], ifreq=None, fdir=None, overwrite=False):
        """Writes out a RADMC-3D image data in fits format. 

        Parameters
        ----------

        fname        : str
                        File name of the radmc3d output image (if omitted 'image.fits' is used)

        dpc          : float
                        Distance of the source in pc
                        
        coord        : str
                        Image center coordinates

        bandwidthmhz : float
                        Bandwidth of the image in MHz (equivalent of the CDELT keyword in the fits header)

        casa         : bool 
                        If set to True a CASA compatible four dimensional image cube will be written

        nu0          : float
                        Rest frequency of the line (for channel maps)

        stokes       : {'I', 'Q', 'U', 'V', 'PI'}
                       Stokes parameter to be written if the image contains Stokes IQUV (possible 
                       choices: 'I', 'Q', 'U', 'V', 'PI' -Latter being the polarized intensity)

        fitsheadkeys : dictionary
                        Dictionary containing all (extra) keywords to be added to the fits header. If 
                        the keyword is already in the fits header (e.g. CDELT1) it will be updated/changed
                        to the value in fitsheadkeys, if the keyword is not present the keyword is added to 
                        the fits header. 
                       
        ifreq        : int
                       Frequency index of the image array to write. If set only this frequency of a multi-frequency
                       array will be written to file.

        overwrite    : boolean
                       If True, overwrite the new fits file. If False, will ask whether or not to overwrite.  Default is False
        """
        # --------------------------------------------------------------------------------------------------
        istokes = 0


        if self.stokes:
            if fname == '':
                fname = 'image_stokes_' + stokes.strip().upper() + '.fits'

        else:
            if fname == '':
                fname = 'image.fits'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        # Decode the image center cooridnates
        # Check first whether the format is OK
        dum = coord

        ra = []
        delim = ['h', 'm', 's']
        for i in delim:
            ind = dum.find(i)
            if ind <= 0:
                msg = 'coord keyword has a wrong format. The format should be coord="0h10m05s -10d05m30s"'
                raise ValueError(msg)
            ra.append(float(dum[:ind]))
            dum = dum[ind + 1:]

        dec = []
        delim = ['d', 'm', 's']
        for i in delim:
            ind = dum.find(i)
            if ind <= 0:
                msg = 'coord keyword has a wrong format. The format should be coord="0h10m05s -10d05m30s"'
                raise ValueError(msg)
            dec.append(float(dum[:ind]))
            dum = dum[ind + 1:]

        target_ra = (ra[0] + ra[1] / 60. + ra[2] / 3600.) * 15.
        if dec[0] >= 0:
            target_dec = (dec[0] + dec[1] / 60. + dec[2] / 3600.)
        else:
            target_dec = (dec[0] - dec[1] / 60. - dec[2] / 3600.)

        if len(self.fwhm) == 0:
            # Conversion from erg/s/cm/cm/ster to Jy/pixel
            conv = self.sizepix_x * self.sizepix_y / (dpc * nc.pc)**2. * 1e23
        else:
            # If the image has already been convolved with a gaussian psf then self.image has
            # already the unit of erg/s/cm/cm/beam, so we need to multiply it by 10^23 to get
            # to Jy/beam
            conv = 1e23

        # Create the data to be written
        if casa:
            # Put the stokes axis to the 4th dimension
            # data = np.zeros([1, self.nfreq, self.ny, self.nx], dtype=float)

            if stokes.strip().upper() == 'IQUV':
                data = np.zeros([4, self.nfreq, self.ny, self.nx], dtype=float)
            else:
                data = np.zeros([1, self.nfreq, self.ny, self.nx], dtype=float)

            if self.stokes:
                if stokes.strip().upper() == 'IQUV':
                    if self.nfreq == 1:
                        for istokes in range(4):
                            data[istokes,0,:,:] = self.image[:,:,istokes, 0] * conv
                    else:
                        for istokes in range(4):
                            for inu in range(self.nfreq):
                                data[istokes, inu, :, :] = self.image[:,:,istokes, inu] * conv
                else:
                    if stokes.strip().upper() == 'I':
                        istokes = 0
                    if stokes.strip().upper() == 'Q':
                        istokes = 1
                    if stokes.strip().upper() == 'U':
                        istokes = 2
                    if stokes.strip().upper() == 'V':
                        istokes = 3

                    if self.nfreq == 1:
                        data[0,0,:,:] = self.image[:,:,istokes, 0] * conv
                    else:
                        for inu in range(self.nfreq):
                            data[0, inu, :, :] = self.image[:,:,istokes, inu] * conv
            else:
                if self.nfreq == 1:
                    data[0, 0, :, :] = self.image[:, :, 0] * conv
                else:
                    for inu in range(self.nfreq):
                        data[0, inu, :, :] = self.image[:, :, inu] * conv
        else: # if not casa
            if stokes.strip().upper == 'IQUV':
                raise ValueError('4D image with Stokes is not available if not casa')

            if stokes.strip.upper() == 'I':
                istokes = 0
            if stokes.strip().upper() == 'Q':
                istokes = 1
            if stokes.strip().upper() == 'U':
                istokes = 2
            if stokes.strip().upper() == 'V':
                istokes = 3

            data = np.zeros([self.nfreq, self.ny, self.nx], dtype=float)
            if self.stokes:
                if stokes.strip().upper() != 'PI':
                    if self.nfreq == 1:
                        data[0, :, :] = self.image[:, :, istokes, 0] * conv

                    else:
                        for inu in range(self.nfreq):
                            data[inu, :, :] = self.image[:, :, istokes, inu] * conv
                else:
                    if self.nfreq == 1:
                        data[0, :, :] = np.sqrt(self.image[:, :, 1, 0]**2 + self.image[:, :, 2, 0]**2) * conv

                    else:
                        for inu in range(self.nfreq):
                            data[inu, :, :] = np.sqrt(
                                self.image[:, :, 1, inu]**2 + self.image[:, :, 2, inu]**2) * conv

            else:
                if self.nfreq == 1:
                    data[0, :, :] = self.image[:, :, 0] * conv

                else:
                    for inu in range(self.nfreq):
                        data[inu, :, :] = self.image[:, :, inu] * conv

        if ifreq is not None:
            if len(data.shape) == 3:
                data = data[ifreq, :, :]

        naxis = len(data.shape)
        hdu = pf.PrimaryHDU(data.swapaxes(naxis - 1, naxis - 2))
        hdulist = pf.HDUList([hdu])

        hdulist[0].header.set('CRPIX1', (self.nx + 1.) / 2., ' ')
        hdulist[0].header.set('CDELT1', -self.sizepix_x / nc.au / dpc / 3600., '')
        # hdulist[0].header.set('CRVAL1', self.sizepix_x/1.496e13/dpc/3600.*0.5+target_ra, '')
        hdulist[0].header.set('CRVAL1', target_ra, '')
        hdulist[0].header.set('CUNIT1', '     DEG', '')
        hdulist[0].header.set('CTYPE1', 'RA---SIN', '')
        hdulist[0].header.set('NAXIS1', self.nx, '')

        hdulist[0].header.set('CRPIX2', (self.ny + 1.) / 2., '')
        hdulist[0].header.set('CDELT2', self.sizepix_y / nc.au / dpc / 3600., '')
        # hdulist[0].header.set('CRVAL2', self.sizepix_y/1.496e13/dpc/3600.*0.5+target_dec, '')
        hdulist[0].header.set('CRVAL2', target_dec, '')
        hdulist[0].header.set('CUNIT2', '     DEG', '')
        hdulist[0].header.set('CTYPE2', 'DEC--SIN', '')
        hdulist[0].header.set('NAXIS2', self.ny, '')

        # For ARTIST compatibility put the stokes axis to the 4th dimension
        if casa:
            hdulist[0].header.set('CRPIX4', 1., '')
            hdulist[0].header.set('CDELT4', 1., '')
            hdulist[0].header.set('CRVAL4', 1., '')
            hdulist[0].header.set('CUNIT4', '        ', '')
            hdulist[0].header.set('CTYPE4', 'STOKES  ', '')
            hdulist[0].header.set('NAXIS4', 1, '')

            if self.nwav == 1:
                hdulist[0].header.set('CRPIX3', 1.0, '')
                hdulist[0].header.set('CDELT3', bandwidthmhz * 1e6, '')
                hdulist[0].header.set('CRVAL3', self.freq[0], '')
                hdulist[0].header.set('CUNIT3', '      HZ', '')
                hdulist[0].header.set('CTYPE3', 'FREQ-LSR', '')
                hdulist[0].header.set('NAXIS3', 1, '')

            else:
                if ifreq is None:
                    hdulist[0].header.set('CRPIX3', 1.0, '')
                    hdulist[0].header.set('CDELT3', (self.freq[1] - self.freq[0]), '')
                    hdulist[0].header.set('CRVAL3', self.freq[0], '')
                    hdulist[0].header.set('CUNIT3', '      HZ', '')
                    hdulist[0].header.set('CTYPE3', 'FREQ-LSR', '')
                    hdulist[0].header.set('NAXIS3', self.nwav, '') #if ifreq is None, it should be a channel map
                else:
                    hdulist[0].header.set('CRPIX3', 1.0, '')
                    hdulist[0].header.set('CDELT3', bandwidthmhz * 1e6, '')
                    hdulist[0].header.set('CRVAL3', self.freq[ifreq], '')
                    hdulist[0].header.set('CUNIT3', '      HZ', '')
                    hdulist[0].header.set('CTYPE3', 'FREQ-LSR', '')
                    hdulist[0].header.set('NAXIS3', 1, '')

        else:
            if self.nwav == 1:
                hdulist[0].header.set('CRPIX3', 1.0, '')
                hdulist[0].header.set('CDELT3', bandwidthmhz * 1e6, '')
                hdulist[0].header.set('CRVAL3', self.freq[0], '')
                hdulist[0].header.set('CUNIT3', '      HZ', '')
                hdulist[0].header.set('CTYPE3', 'FREQ-LSR', '')
                hdulist[0].header.set('NAXIS3', 1, '')
            else:
                if ifreq is None:
                    hdulist[0].header.set('CRPIX3', 1.0, '')
                    hdulist[0].header.set('CDELT3', (self.freq[1] - self.freq[0]), '')
                    hdulist[0].header.set('CRVAL3', self.freq[0], '')
                    hdulist[0].header.set('CUNIT3', '      HZ', '')
                    hdulist[0].header.set('CTYPE3', 'FREQ-LSR', '')
                    hdulist[0].header.set('NAXIS3', self.nfreq, '')
                else:
                    hdulist[0].header.set('CRPIX3', 1.0, '')
                    hdulist[0].header.set('CDELT3', bandwidthmhz * 1e6, '')
                    hdulist[0].header.set('CRVAL3', self.freq[ifreq], '')
                    hdulist[0].header.set('CUNIT3', '      HZ', '')
                    hdulist[0].header.set('CTYPE3', 'FREQ-LSR', '')
                    hdulist[0].header.set('NAXIS3', 1, '')

        if nu0 > 0:
            hdulist[0].header.set('RESTFRQ', nu0, '')
        else:
            if self.nwav == 1:
                hdulist[0].header.set('RESTFRQ', self.freq[0], '')

        if len(self.fwhm) == 0:
            hdulist[0].header.set('BUNIT', 'JY/PIXEL', '')
        else:
            if ifreq is None:
                hdulist[0].header.set('BUNIT', 'JY/BEAM', '')
                hdulist[0].header.set('BMAJ', self.fwhm[0][0] / 3600., '')
                hdulist[0].header.set('BMIN', self.fwhm[0][1] / 3600., '')
                hdulist[0].header.set('BPA', -self.pa[0], '')
            elif (len(self.fwhm) == 1) & (len(self.pa) == 1):
                hdulist[0].header.set('BUNIT', 'JY/BEAM', '')
                hdulist[0].header.set('BMAJ', self.fwhm[0][0] / 3600., '')
                hdulist[0].header.set('BMIN', self.fwhm[0][1] / 3600., '')
                hdulist[0].header.set('BPA', -self.pa[0], '')
            elif (len(self.fwhm) == 1) & (len(self.pa) != 1):
                hdulist[0].header.set('BUNIT', 'JY/BEAM', '')
                hdulist[0].header.set('BMAJ', self.fwhm[0][0] / 3600., '')
                hdulist[0].header.set('BMIN', self.fwhm[0][1] / 3600., '')
                hdulist[0].header.set('BPA', -self.pa[ifreq], '')
            elif (len(self.fwhm) != 1) & (len(self.pa) == 1):
                hdulist[0].header.set('BUNIT', 'JY/BEAM', '')
                hdulist[0].header.set('BMAJ', self.fwhm[ifreq][0] / 3600., '')
                hdulist[0].header.set('BMIN', self.fwhm[ifreq][1] / 3600., '')
                hdulist[0].header.set('BPA', -self.pa[0], '')
            else:
                hdulist[0].header.set('BUNIT', 'JY/BEAM', '')
                hdulist[0].header.set('BMAJ', self.fwhm[ifreq][0] / 3600., '')
                hdulist[0].header.set('BMIN', self.fwhm[ifreq][1] / 3600., '')
                hdulist[0].header.set('BPA', -self.pa[ifreq], '')

        hdulist[0].header.set('BTYPE', 'INTENSITY', '')
        hdulist[0].header.set('BZERO', 0.0, '')
        hdulist[0].header.set('BSCALE', 1.0, '')

        hdulist[0].header.set('EPOCH', 2000.0, '')
        hdulist[0].header.set('LONPOLE', 180.0, '')

        if fitsheadkeys:
            if len(fitsheadkeys.keys()) > 0:
                for ikey in fitsheadkeys.keys():
                    # hdulist[0].header.update(ikey, fitsheadkeys[ikey], '')
                    hdulist[0].header.set(ikey, fitsheadkeys[ikey], '')

        if os.path.exists(fname):
            print(fname + ' already exists')
            if overwrite:
                print('overwrite = True')
                os.remove(fname)
                hdu.writeto(fname)
            else:
                dum = input('Do you want to overwrite it ("yes"/"no")?')
                if (dum.strip()[0] == 'y') | (dum.strip()[0] == 'Y'):
                    os.remove(fname)
                    hdu.writeto(fname)
                else:
                    print('No image has been written')
        else:
            hdu.writeto(fname)
            # --------------------------------------------------------------------------------------------------

    def plotMomentMap(self, moment=0, nu0=None, wav0=None, dpc=1., au=False, arcsec=False, cmap=None, vclip=None):
        """Plots moment maps

        Parameters
        ----------

        moment : int
                 Moment of the channel maps to be calculated 

        nu0    : float
                 Rest frequency of the line in Hz

        wav0   : float
                 Rest wavelength of the line in micron

        dpc    : float
                 Distance of the source in pc

        au     : bool
                 If True displays the image with AU as the spatial axis unit

        arcsec : bool
                 If True displays the image with arcsec as the spatial axis unit (dpc should also be set!)

        cmap   : matplotlib colormap
                 Color map to be used to display the moment map

        vclip  : list/ndarray
                 Two element list / Numpy array containin the lower and upper limits for the values in the moment
                  map to be displayed

        """

        # I/O error handling
        if nu0 is None:
            if wav0 is None:
                msg = 'Unknown nu0 and wav0. Neither the rest frequency (nu0) nor the rest wavelength (wav0)'\
                      + ' of the line is specified.'
                raise ValueError(msg)
            else:
                nu0 = nc.cc / wav0 * 1e4

        if len(self.image.shape) != 3:
            msg = 'Wrong image shape. Channel map calculation requires a three dimensional array with '\
                  + '[Nx,  Ny,  Nnu] dimensions. The current image has the shape of ' + str(len(self.image.shape))
            raise ValueError(msg)
        mmap = self.getMomentMap(moment=moment, nu0=nu0, wav0=wav0)

        if moment > 0:
            mmap0 = self.getMomentMap(moment=0, nu0=nu0, wav0=wav0)
            mmap = mmap / mmap0

        # Select the coordinates of the data
        if au:
            x = self.x / nc.au
            y = self.y / nc.au
            xlab = 'X [AU]'
            ylab = 'Y [AU]'
        elif arcsec:
            x = self.x / nc.au / dpc
            y = self.y / nc.au / dpc
            xlab = 'RA offset ["]'
            ylab = 'DEC offset ["]'
        else:
            x = self.x
            y = self.y
            xlab = 'X [cm]'
            ylab = 'Y [cm]'

        ext = (x[0], x[self.nx - 1], y[0], y[self.ny - 1])

        cb_label = ''
        if moment == 0:
            mmap = mmap / (dpc * dpc)
            cb_label = 'I' + r'$_\nu$' + ' [erg/s/cm/cm/Hz/ster*km/s]'
        if moment == 1:
            mmap = mmap / (dpc * dpc)
            cb_label = 'v [km/s]'
        if moment > 1:
            mmap = mmap / (dpc * dpc)
            powex = str(moment)
            cb_label = r'v$^' + powex + '$ [(km/s)$^' + powex + '$]'

        if vclip is not None:
            if len(vclip) != 2:
                msg = 'Wrong shape in vclip. vclip should be a two element list with (clipmin, clipmax)'
                raise ValueError(msg)
            else:
                mmap = mmap.clip(vclip[0], vclip[1])

        implot = plt.imshow(mmap, extent=ext, cmap=cmap)
        cbar = plt.colorbar(implot)
        cbar.set_label(cb_label)
        plt.xlabel(xlab)
        plt.ylabel(ylab)

    def getMomentMap(self, moment=0, nu0=None, wav0=None):
        """Calculates moment maps.

        Parameters
        ----------

        moment : int
                 Moment of the channel maps to be calculated 

        nu0    : float
                 Rest frequency of the line in Hz

        wav0   : float
                 Rest wavelength of the line in micron

        Returns
        -------
        Ndarray with the same dimension as the individual channel maps
        """

        # I/O error handling
        if nu0 is None:
            if wav0 is None:
                msg = 'Unknown nu0 and wav0. Neither the rest frequency (nu0) nor the rest wavelength (wav0)' \
                      + ' of the line is specified.'
                raise ValueError(msg)
            else:
                nu0 = nc.cc / wav0 * 1e4

        if len(self.image.shape) != 3:
            msg = 'Wrong image shape. Channel map calculation requires a three dimensional array with '\
                  + '[Nx,  Ny,  Nnu] dimensions. The current image has the shape of ' + str(len(self.image.shape))
            raise ValueError(msg)

        # First calculate the velocity field
        v_kms = nc.cc * (nu0 - self.freq) / nu0 / 1e5

        vmap = np.zeros([self.nx, self.ny, self.nfreq], dtype=np.float64)
        for ifreq in range(self.nfreq):
            vmap[:, :, ifreq] = v_kms[ifreq]

        # Now calculate the moment map
        y = self.image * (vmap**moment)

        dum = (vmap[:, :, 1:] - vmap[:, :, :-1]) * (y[:, :, 1:] + y[:, :, :-1]) * 0.5

        return dum.sum(2)

    def readImage(self, fname=None, binary=False, old=False):
        """Reads a rectangular image calculated by RADMC-3D 

        Parameters
        ----------

        fname   : str, optional
                 File name of the radmc3d output image (if omitted 'image.out' is used)

        old     : bool
                 If set to True it reads old radmc-2d style image        

        binary  : bool, optional
                 False - the image format is formatted ASCII if True - C-compliant binary (omitted if old=True)

        """
        if old:
            if fname is None:
                fname = 'image.dat'

            self.filename = fname
            print('Reading ' + fname)


            with open(fname, 'r') as rfile:

                dum = rfile.readline().split()
                self.nx = int(dum[0])
                self.ny = int(dum[1])
                self.nfreq = int(dum[2])
                self.nwav = self.nfreq

                dum = rfile.readline().split()
                self.sizepix_x = float(dum[0])
                self.sizepix_y = float(dum[1])
                self.wav = np.zeros(self.nwav, dtype=float) - 1.
                self.freq = np.zeros(self.nwav, dtype=float) - 1.

                self.stokes = False
                self.image = np.zeros([self.nx, self.ny, self.nwav], dtype=np.float64)
                for iwav in range(self.nwav):
                    dum = rfile.readline()
                    for iy in range(self.nx):
                        for ix in range(self.ny):
                            self.image[ix, iy, iwav] = float(rfile.readline())

        else:
            if binary:
                if fname is None:
                    fname = 'image.bout'

                self.filename = fname

                dum = np.fromfile(fname, count=4, dtype=int)
                iformat = dum[0]
                self.nx = dum[1]
                self.ny = dum[2]
                self.nfreq = dum[3]
                self.nwav = self.nfreq
                dum = np.fromfile(fname, count=-1, dtype=np.float64)

                self.sizepix_x = dum[4]
                self.sizepix_y = dum[5]
                self.wav = dum[6:6 + self.nfreq]
                self.freq = nc.cc / self.wav * 1e4

                if iformat == 1:
                    self.stokes = False
                    self.image = np.reshape(dum[6 + self.nfreq:], [self.nfreq, self.ny, self.nx])
                    self.image = np.swapaxes(self.image, 0, 2)
                elif iformat == 3:
                    self.stokes = True
                    self.image = np.reshape(dum[6 + self.nfreq:], [self.nfreq, 4, self.ny, self.nx])
                    self.image = np.swapaxes(self.image, 0, 3)
                    self.image = np.swapaxes(self.image, 1, 2)

            else:

                # Look for the image file

                if fname is None:
                    fname = 'image.out'

                if os.path.isfile(fname) is False:
                    raise ValueError('file for image does not exist: %s'%fname)

                print('Reading '+ fname)

                self.filename = fname
                with open(fname, 'r') as rfile:

                    dum = ''

                    # Format number
                    iformat = int(rfile.readline())

                    # Nr of pixels
                    dum = rfile.readline()
                    dum = dum.split()
                    self.nx = int(dum[0])
                    self.ny = int(dum[1])
                    # Nr of frequencies
                    self.nfreq = int(rfile.readline())
                    self.nwav = self.nfreq
                    # Pixel sizes
                    dum = rfile.readline()
                    dum = dum.split()
                    self.sizepix_x = float(dum[0])
                    self.sizepix_y = float(dum[1])
                    # Wavelength of the image
                    self.wav = np.zeros(self.nwav, dtype=np.float64)
                    for iwav in range(self.nwav):
                        self.wav[iwav] = float(rfile.readline())
                    self.wav = np.array(self.wav)
                    self.freq = nc.cc / self.wav * 1e4

                    # If we have a normal total intensity image
                    if iformat == 1:
                        self.stokes = False

                        self.image = np.zeros([self.nx, self.ny, self.nwav], dtype=np.float64)
                        for iwav in range(self.nwav):
                            # Blank line
                            dum = rfile.readline()
                            for iy in range(self.nx):
                                for ix in range(self.ny):
                                    self.image[ix, iy, iwav] = float(rfile.readline())

                    # If we have the full stokes image
                    elif iformat == 3:
                        self.stokes = True
                        self.image = np.zeros([self.nx, self.ny, 4, self.nwav], dtype=np.float64)
                        for iwav in range(self.nwav):
                            # Blank line
                            dum = rfile.readline()
                            for iy in range(self.nx):
                                for ix in range(self.ny):
                                    dum = rfile.readline().split()
                                    imstokes = [float(i) for i in dum]
                                    self.image[ix, iy, 0, iwav] = float(dum[0])
                                    self.image[ix, iy, 1, iwav] = float(dum[1])
                                    self.image[ix, iy, 2, iwav] = float(dum[2])
                                    self.image[ix, iy, 3, iwav] = float(dum[3])

        self.x = ((np.arange(self.nx, dtype=np.float64) + 0.5) - self.nx / 2) * self.sizepix_x
        self.y = ((np.arange(self.ny, dtype=np.float64) + 0.5) - self.ny / 2) * self.sizepix_y

    def writeImage(self, fname=None,  binary=False, old=False ):
        """ writes the radmc3d image file """
        if old:
            raise ValueError('not available')
        else:
            if binary:
                raise ValueError('not available')
            else:

                print('writing '+ fname)

                if self.stokes:
                    iformat = 3
                else:
                    iformat = 1

                with open(fname, 'w') as wfile:

                    # Format number
                    wfile.write('%d\n'%iformat)

                    # Nr of pixels
                    wfile.write('%d %d \n'%(self.nx, self.ny))

                    # Nr of frequencies
                    wfile.write('%d\n'%self.nwav)

                    # Pixel sizes
                    wfile.write('%.3f %.3f\n'%(self.sizepix_x, self.sizepix_y))

                    # Wavelength of the image
                    for iwav in range(self.nwav):
                        wfile.write('%e\n'%self.wav[iwav])

                    # write image
                    if iformat == 1:
                        for iwav in range(self.nwav):
                            # blank line
                            wfile.write('\n')
                            for ix in range(self.nx):
                                for iy in range(self.ny):
                                    wfile.write('%e\n'%self.image[ix,iy,iwav])

                    elif iformat == 3:
                        for iwav in range(self.nwav):
                            # blank line
                            wfile.write('\n')
                            for iy in range(self.ny):
                                for ix in range(self.nx):
                                    wfile.write('%e \t %e \t %e \t %e\n'%(self.image[ix,iy,0,iwav], 
                                                               self.image[ix,iy,1,iwav],
                                                               self.image[ix,iy,2,iwav],
                                                               self.image[ix,iy,3,iwav]))


    def readTausurf(self, fname=None, binary=False, old=False):
        """Reads a tau surface  image calculated by RADMC-3D

        Parameters
        ----------

        fname   : str, optional
                 File name of the radmc3d output image (if omitted 'tausurface.out' is used)

        old     : bool
                 If set to True it reads old radmc-2d style image

        binary  : bool, optional
                 False - the image format is formatted ASCII if True - C-compliant binary (omitted if old=True)
        """
        if old:
            raise ValueError('reading old files of tau surface is not supported yet')

            if fname is None:
                fname = 'image.dat'

            self.filename = fname
            print('Reading ' + fname)


            with open(fname, 'r') as rfile:

                dum = rfile.readline().split()
                self.nx = int(dum[0])
                self.ny = int(dum[1])
                self.nfreq = int(dum[2])
                self.nwav = self.nfreq

                dum = rfile.readline().split()
                self.sizepix_x = float(dum[0])
                self.sizepix_y = float(dum[1])
                self.wav = np.zeros(self.nwav, dtype=float) - 1.
                self.freq = np.zeros(self.nwav, dtype=float) - 1.

                self.stokes = False
                self.image = np.zeros([self.nx, self.ny, self.nwav], dtype=np.float64)
                for iwav in range(self.nwav):
                    dum = rfile.readline()
                    for iy in range(self.nx):
                        for ix in range(self.ny):
                            self.image[ix, iy, iwav] = float(rfile.readline())
        else:
            if binary:
                raise ValueError('binary for reading tausurface images are not supported yet')

                if fname is None:
                    fname = 'image.bout'

                self.filename = fname

                dum = np.fromfile(fname, count=4, dtype=int)
                iformat = dum[0]
                self.nx = dum[1]
                self.ny = dum[2]
                self.nfreq = dum[3]
                self.nwav = self.nfreq
                dum = np.fromfile(fname, count=-1, dtype=np.float64)

                self.sizepix_x = dum[4]
                self.sizepix_y = dum[5]
                self.wav = dum[6:6 + self.nfreq]
                self.freq = nc.cc / self.wav * 1e4

                if iformat == 1:
                    self.stokes = False
                    self.image = np.reshape(dum[6 + self.nfreq:], [self.nfreq, self.ny, self.nx])
                    self.image = np.swapaxes(self.image, 0, 2)
                elif iformat == 3:
                    self.stokes = True
                    self.image = np.reshape(dum[6 + self.nfreq:], [self.nfreq, 4, self.ny, self.nx])
                    self.image = np.swapaxes(self.image, 0, 3)
                    self.image = np.swapaxes(self.image, 1, 2)
            else:

                # Look for the image file

                if fname is None:
                    fname = 'tausurface.out'

                print('Reading '+ fname)

                self.filename = fname
                with open(fname, 'r') as rfile:

                    dum = ''

                    # Format number
                    iformat = int(rfile.readline())

                    # Nr of pixels
                    dum = rfile.readline()
                    dum = dum.split()
                    self.nx = int(dum[0])
                    self.ny = int(dum[1])
                    # Nr of frequencies
                    self.nfreq = int(rfile.readline())
                    self.nwav = self.nfreq
                    # Pixel sizes
                    dum = rfile.readline()
                    dum = dum.split()
                    self.sizepix_x = float(dum[0])
                    self.sizepix_y = float(dum[1])
                    # Wavelength of the image
                    self.wav = np.zeros(self.nwav, dtype=np.float64)
                    for iwav in range(self.nwav):
                        self.wav[iwav] = float(rfile.readline())
                    self.wav = np.array(self.wav)
                    self.freq = nc.cc / self.wav * 1e4

                    # If we have a normal total intensity image
                    if iformat == 1:

                        self.tausurf = np.zeros([self.nx, self.ny, self.nwav], dtype=np.float64)
                        for iwav in range(self.nwav):
                            # Blank line
                            dum = rfile.readline()
                            for iy in range(self.ny):
                                for ix in range(self.nx):
                                    self.tausurf[ix, iy, iwav] = float(rfile.readline())

                    # If we have the full stokes image
                    elif iformat == 3:
                        self.tausurf = np.zeros([self.nx, self.ny, 4, self.nwav], dtype=np.float64)
                        for iwav in range(self.nwav):
                            # Blank line
                            dum = rfile.readline()
                            for iy in range(self.ny):
                                for ix in range(self.nx):
                                    dum = rfile.readline().split()
                                    imstokes = [float(i) for i in dum]
                                    self.tausurf[ix, iy, 0, iwav] = float(dum[0])
                                    self.tausurf[ix, iy, 1, iwav] = float(dum[1])
                                    self.tausurf[ix, iy, 2, iwav] = float(dum[2])
                                    self.tausurf[ix, iy, 3, iwav] = float(dum[3])

        # x and y axis
        self.x = ((np.arange(self.nx, dtype=np.float64) + 0.5) - self.nx / 2) * self.sizepix_x
        self.y = ((np.arange(self.ny, dtype=np.float64) + 0.5) - self.ny / 2) * self.sizepix_y


    def readSpectrum(self, fname=None, binary=False, dpc=1.):
        """
        reads the spectrum.out file for the spectrum. note the data structure is different from ordinary images, but this will still use the image object
        
        Parameters
        ----------

        fname   : str, optional
                 File name of the radmc3d output image (if omitted 'spectrum.out' is used)

        binary  : bool, optional
                 False - the image format is formatted ASCII if True - C-compliant binary (omitted if old=True)

        dpc     : float. Default 1
                  Distance to source in pc for calculating flux per pixel. 

        """
        if binary:
            if fname is None:
                fname = 'spectrum.bout'
            raise ValueError('binary file for spectrum is not done yet')

        else:
            # Look for the image file

            if fname is None:
                fname = 'spectrum.out'

            print('Reading '+ fname)

            self.filename = fname
            with open(fname, 'r') as rfile:

                dum = ''

                # Format number
                iformat = int(rfile.readline())

                if iformat == 1:
                    self.stokes = False

                # number of wavelengths
                nwav = int(rfile.readline())
                self.nwav = nwav
                self.wav = np.zeros([self.nwav], dtype=np.float64)

                self.image = np.zeros([1, 1, self.nwav], dtype=np.float64)

                # Blank line
                dum = rfile.readline()

                for iwav in range(self.nwav):
                    # actual data
                    dum = rfile.readline()
                    dum = dum.split()        
                    self.wav[iwav] = float(dum[0])
                    self.image[0, 0, iwav] = float(dum[1])

                # calculate frequency
                self.nfreq = nwav
                self.freq = nc.cc * 1e4 / self.wav

                # dimensions
                self.nx = 1
                self.ny = 1
                self.sizepix_x = 0.
                self.sizepix_y = 0.
                self.x = 0.
                self.y = 0.

                # distance
                self.dpc = dpc

                # stokes
                self.stokes = False

                # total flux
                self.totflux = np.squeeze(self.image)

                # image in jansky
                self.imageJyppix = self.image * nc.jy


    def imConv(self, dpc=1., psfType='gauss', fwhm=None, pa=None, tdiam_prim=8.2, tdiam_sec=0.94):
        """Convolves a RADMC-3D image with a two dimensional Gaussian psf. The output images will have the same
        brightness units as the input images.

        Parameters
        ----------
        dpc         : float
                      Distance of the source in pc.

        psfType     : {'gauss', 'airy'}
                      Shape of the PSF. If psfType='gauss', fwhm and pa should also be given. If psfType='airy', the 
                      tdiam_prim, tdiam_sec and wav parameters should also be specified.

        fwhm        : list, optional
                      Let one element be a list of 2 numbers to represent FWHM of two dimensional psf along two
                      principal axes. If the list contains one element, this will be used for all wavelengths. If
                      this contains the same number as the wavelengths (self.nfreq), it will represent the FWHM 
                      per wavelength. The unit is assumed to be arcsec. (should only be set if psfType='gauss')

        pa          : list, optional
                      Position angle of the psf ellipse (counts from North counterclockwise, should only be set 
                      if psfType='gauss'). If only one element, pa will be the position angle for all wavelenghts.
                      If contains the same number of elements as wavelength, then it will be used per wavelength

        tdiam_prim  : float, optional
                      Diameter of the primary aperture of the telescope in meter. (should be set only if psfType='airy')

        tdiam_sec   : float, optional
                      Diameter of the secondary mirror (central obscuration), if there is any, in meter. If no 
                      secondary mirror/obscuration is present, this parameter should be set to zero. 
                      (should only be set if psfType='airy')

        Returns
        -------

        Returns a radmc3dImage 
        """

        dx = self.sizepix_x / nc.au / dpc # in arcseconds
        dy = self.sizepix_y / nc.au / dpc
        nfreq = self.nfreq
        psf = None
        cimage = None

        nfwhm = len(fwhm)
        if isinstance(fwhm, list):
            fwhm = np.array(fwhm, dtype=np.float64)
        if (len(fwhm) != nfreq) & (len(fwhm) != 1):
            raise ValueError('number of elements for fwhm must be 1 or equal to nfreq')

        npa = len(pa)
        if isinstance(pa, list):
            pa = np.array(pa, dtype=np.float64)
        if (len(pa) != nfreq) & (len(pa) != 1):
            raise ValueError('number of elements for pa must be 1 or equal to nfreq')

        if self.stokes:
            if self.nfreq == 1:
                # Generate the  psf
                dum = getPSF(nx=self.nx, ny=self.ny, pscale=[dx, dy], psfType=psfType, fwhm=fwhm[0], pa=pa[0],
                             tdiam_prim=tdiam_prim, tdiam_sec=tdiam_sec, wav=self.wav[0])
                psf = dum['psf']
                f_psf = np.fft.fft2(psf)

                cimage = np.zeros([self.nx, self.ny, 4, 1], dtype=np.float64)
                for istokes in range(4):
                    imag = self.image[:, :, istokes, 0]
#                    f_imag = np.fft.fft2(imag)
#                    f_cimag = f_psf * f_imag
#                    cimage[:, :, istokes, 0] = np.abs(np.fft.ifftshift(np.fft.ifft2(f_cimag)))
#                    cimage[:, :, istokes, 0] = np.real(np.fft.ifftshift(np.fft.ifft2(f_cimag)))

                    cimage[:,:,istokes,0] = getConvolve(imag, psf)
            else:
                # If we have a simple Gaussian PSF it will be wavelength independent so we can take it out from the
                # frequency loop
                if psfType.lower().strip() == 'gauss':
                    cimage = np.zeros([self.nx, self.ny, 4, self.nfreq], dtype=np.float64)
                    # Generate the gaussian psf
                    for ifreq in range(nfreq):
                        if nfwhm == 1:
                            ifwhm = 0
                        else:
                            ifwhm = ifreq
                        if npa == 1:
                            ipa = 0
                        else:
                            ipa = ifreq

                        dum = getPSF(nx=self.nx, ny=self.ny, pscale=[dx, dy], psfType=psfType, fwhm=fwhm[ifwhm], 
                                 pa=pa[ipa],tdiam_prim=tdiam_prim, tdiam_sec=tdiam_sec, wav=self.wav[ifreq])
                        psf = dum['psf']
                        f_psf = np.fft.fft2(psf)

                        for istokes in range(4):
                            imag = self.image[:, :, istokes, ifreq]

                            cimage[:,:,istokes,ifreq] = getConvolve(imag, psf)

                # If we have an Airy-PSF calculated from the aperture size(s) and wavelenght, the PSF will depend
                # on the frequency so it has to be re-calculated for each wavelength
                elif psfType.lower().strip() == 'airy':
                    cimage = np.zeros([self.nx, self.ny, 4, self.nfreq], dtype=np.float64)
                    for ifreq in range(nfreq):
                        if nfwhm == 1:
                            ifwhm = 0
                        else:
                            ifwhm = ifreq
                        if npa == 1:
                            ipa = 0
                        else:
                            ipa = ifreq

                        # Generate the wavelength-dependent airy-psf
                        dum = getPSF(nx=self.nx, ny=self.ny, pscale=[dx, dy], psfType=psfType, fwhm=fwhm[ifwhm], 
                                     pa=pa[ipa], tdiam_prim=tdiam_prim, tdiam_sec=tdiam_sec, wav=self.wav[ifreq])
                        psf = dum['psf']
                        f_psf = np.fft.fft2(psf)

                        for istokes in range(4):
                            imag = self.image[:, :, istokes, ifreq]
                            cimage[:,:,istokes,ifreq] = getConvolve(imag, psf)

        else:
            # If we have a simple Gaussian PSF it will be wavelength independent so we can take it out from the
            # frequency loop
            if psfType.lower().strip() == 'gauss':
                cimage = np.zeros([self.nx, self.ny, self.nfreq], dtype=np.float64)
                for ifreq in range(nfreq):
                    if nfwhm == 1:
                        ifwhm = 0
                    else:
                        ifwhm = ifreq
                    if npa == 1:
                        ipa = 0
                    else:
                        ipa = ifreq

                    dum = getPSF(nx=self.nx, ny=self.ny, pscale=[dx, dy], psfType=psfType, fwhm=fwhm[ifwhm],
                            pa=pa[ipa],tdiam_prim=tdiam_prim, tdiam_sec=tdiam_sec, wav=self.wav[ifreq])

                    psf = dum['psf']

                    cimage[:,:,ifreq] = getConvolve(self.image[:,:,ifreq], psf)

            # If we have an Airy-PSF calculated from the aperture size(s) and wavelenght, the PSF will depend
            # on the frequency so it has to be re-calculated for each wavelength
            elif psfType.lower().strip() == 'airy':
                cimage = np.zeros([self.nx, self.ny, self.nfreq], dtype=np.float64)
                for ifreq in range(nfreq):
                    # Generate the wavelength-dependent airy-psf
                    if nfwhm == 1:
                        ifwhm = 0
                    else:
                        ifwhm = ifreq
                    if npa == 1:
                        ipa = 0
                    else:
                        ipa = ifreq

                    dum = getPSF(nx=self.nx, ny=self.ny, pscale=[dx, dy], psfType=psfType, fwhm=fwhm[ifwhm], 
                            pa=pa[ipa], tdiam_prim=tdiam_prim, tdiam_sec=tdiam_sec, wav=self.wav[ifreq])
                    psf = dum['psf']

                    cimage[:,:,ifreq] = getConvolve(self.image[:,:,ifreq], psf)

        # cimage = squeeze(cimage)

        # Return the convolved image (copy the image class and replace the image attribute to the convolved image)
        res = copy.deepcopy(self)
        res.image = cimage
        conv = self.sizepix_x * self.sizepix_y / nc.pc**2. * 1e23
        res.imageJyppix = res.image * conv

        res.psf = psf
        res.fwhm = fwhm
        res.pa = pa
        res.dpc = dpc

        return res

    def getTotalFlux(self, threshold=1.):
        if self.stokes:
            if self.nfreq == 1:
                totflux = np.zeros([4], dtype=np.float64)
                for istokes in range(4):
                    imag = self.image[:,:,istokes, 0].copy()
                    if self.rms != None:
                        reg = imag < (threshold * self.rms[istokes,0])
                        imag[reg] = 0.0

                    totflux[istokes] = np.sum(np.sum(imag))
            else:
                totflux = np.zeros([4, self.nfreq], dtype=np.float64)
                for ifreq in range(self.nfreq):
                    for istokes in range(4):
                        imag = self.image[:,:,istokes,ifreq].copy()
                        if self.rms != None:
                            reg = imag < (threshold * self.rms[istokes, ifreq])
                            imag[reg] = 0.0
                        totflux[istokes,ifreq] = np.sum(np.sum(imag))
        else:
            totflux = np.zeros([self.nfreq], dtype=np.float64)
            for ifreq in range(self.nfreq):
                imag = self.image[:,:,ifreq].copy()
                if self.rms != None:
                    reg = imag < (threshold * self.rms[ifreq])
                    imag[reg] = 0.0
                totflux[ifreq] = np.sum(np.sum(imag))

        # convert to Jy
        totflux = totflux * self.sizepix_x * self.sizepix_y / (self.dpc * nc.pc)**2 / nc.jy

        self.totflux = totflux

def getPSF(nx=None, ny=None, psfType='gauss', pscale=None, fwhm=None, pa=None, tdiam_prim=8.2, tdiam_sec=0.94,
           wav=None):
    """Calculates a two dimensional Gaussian PSF.

    Parameters
    ----------
    nx          : int
                  Image size in the first dimension

    ny          : int
                  Image size in the second dimension

    psfType     : {'gauss', 'airy'}
                  Shape of the PSF. If psfType='gauss', fwhm and pa should also be given. If psfType='airy', the 
                  tdiam_prim, tdiam_sec and wav parameters should also be specified.

    pscale      : list
                  Pixelscale of the image, if set fwhm should be in the same unit, if not set unit of fwhm is pixels

    fwhm        : list, optional
                  Full width at half maximum of the psf along the two axis (should be set only if psfType='gauss')

    pa          : float, optional
                  Position angle of the gaussian if the gaussian is not symmetric. In degrees
                  (should be set only if psfType='gauss')

    tdiam_prim  : float, optional
                  Diameter of the primary aperture of the telescope in meter. (should be set only if psfType='airy')

    tdiam_sec   : float, optional
                  Diameter of the secondary mirror (central obscuration), if there is any, in meter. If no secondary
                  mirror/obscuration is present, this parameter should be set to zero. 
                  (should be set only if psfType='airy')

    wav         : float, optional
                  Wavelength of observation in micrometer (should be set only if psfType='airy')


    Returns
    -------

    Returns a dictionary with the following keys:

        * psf : ndarray
                The two dimensional psf
        * x   : ndarray
                The x-axis of the psf 
        * y   : ndarray
                The y-axis of the psf 
    """
    # --------------------------------------------------------------------------------------------------

    # Create the two axes

    if pscale is not None:
        dx, dy = pscale[0], pscale[1]
    else:
        dx, dy = 1., 1.

    x = (np.arange(nx, dtype=np.float64) - nx / 2) * dx
    y = (np.arange(ny, dtype=np.float64) - ny / 2) * dy

    # Create the Gaussian PSF
    psf = None
    if psfType.strip().lower() == 'gauss':

        # Calculate the standard deviation of the Gaussians
        sigmax = fwhm[0] / (2.0 * np.sqrt(2.0 * np.log(2.)))
        sigmay = fwhm[1] / (2.0 * np.sqrt(2.0 * np.log(2.)))
        norm = (2. * np.pi * sigmax * sigmay) / dx / dy
        # Pre-compute sin and cos angles

        sin_pa = np.sin(pa / 180. * np.pi - np.pi / 2.)
        cos_pa = np.cos(pa / 180. * np.pi - np.pi / 2.)

        # Define the psf
        psf = np.zeros([nx, ny], dtype=np.float64)
        cos_pa_x = cos_pa * x
        cos_pa_y = cos_pa * y
        sin_pa_x = sin_pa * x
        sin_pa_y = sin_pa * y
        for ix in range(nx):
            for iy in range(ny):
                # xx = cos_pa * x[ix] - sin_pa * y[iy]
                # yy = sin_pa * x[ix] + cos_pa * y[iy]
                xx = cos_pa_x[ix] - sin_pa_y[iy]
                yy = sin_pa_x[ix] + cos_pa_y[iy]

                psf[ix, iy] = np.exp(-0.5 * xx * xx / sigmax / sigmax - 0.5 * yy * yy / sigmay / sigmay)

        psf /= norm

    elif psfType.strip().lower() == 'airy':

        # Check whether scipy was successfully imported
        if not spc:
            msg = 'scipy.special was not imported. PSF calculation is limited to Gaussian only.'
            raise ImportError(msg)

        # Unit conversion
        x_rad = x / 3600. / 180. * np.pi
        y_rad = y / 3600. / 180. * np.pi
        x2 = x_rad**2
        y2 = y_rad**2
        wav_m = wav * 1e-6
        psf = np.zeros([nx, ny], dtype=np.float64)
        if tdiam_sec == 0.:
            for ix in range(nx):
                r = np.sqrt(x2[ix] + y2)
                u = np.pi / wav_m * tdiam_prim * r

                if 0. in u:
                    ii = (u == 0.)
                    u[ii] = 1e-5
                psf[ix, :] = (2.0 * spc.j1(u) / u)**2
        else:
            for ix in range(nx):
                r = np.sqrt(x2[ix] + y2)
                u = np.pi / wav_m * tdiam_prim * r
                eps = tdiam_sec / tdiam_prim
                if 0. in u:
                    ii = (u == 0.)
                    u[ii] = 1e-5
                psf[ix, :] = 1.0 / (1.0 - eps**2)**2 * ((2.0 * spc.j1(u) / u)
                                                        - (eps**2 * 2.0 * spc.j1(eps * u) / (eps * u)))**2

        dum = 0.44 * (wav * 1e-6 / tdiam_prim / np.pi * 180. * 3600.) * 2. * np.sqrt(2. * np.log(2.))
        fwhm = [dum, dum]
        norm = fwhm[0] * fwhm[1] * np.pi / (4. * np.log(2.)) / dx / dy
        psf /= norm

    res = {'psf': psf, 'x': x, 'y': y}

    return res

def getConvolve(imag, psf, method='2'):
    """
    reads two images and convolves
    method 1: numpy
    method 2: scipy.signal.fftconvolve
    method 3: scipy.signal.convolve(method='direct')
    """
    if method is '1':
        # numpy fft method
        f_psf = np.fft.fft2(psf)
        f_imag = np.fft.fft2(imag)
        f_cimag = f_psf * f_imag
#        conved = np.abs(np.fft.ifftshift(np.fft.ifft2(f_cimag)))
        conved = np.real(np.fft.ifftshift(np.fft.ifft2(f_cimag)))

    elif method is '2':
        from scipy import signal
        # scipy.signal method
        conved = signal.fftconvolve(imag, psf, mode='same')

    elif method is '3':
        from scipy import signal
        conved = signal.convolve(imag, psf, mode='same', method='direct')

    else:
        raise ValueError('Choose an established method for convolving')

    return conved

def readImage(fname=None, binary=False, old=False):
    """Reads an image calculated by RADMC-3D.
       This function is an interface to radmc3dImage.readImage().

    Parameters
    ----------
        fname   : str, optional
                 File name of the radmc3d output image (if omitted 'image.out' is used)

        old     : bool
                 If set to True it reads old radmc-2d style image        

        binary  : bool, optional
                 False - the image format is formatted ASCII if True - C-compliant binary (omitted if old=True)
    """

    dum = radmc3dImage()
    dum.readImage(fname=fname, binary=binary, old=old)
    return dum

def readFitsToImage(fname=None, dpc=None, wav=None, rms=None, recen=None, padnan=None):
    """Reads a fits image and puts the information into a radmc3dImage object.
    Parameters
    ----------
        fname   : str
                  Name of the fits file

        dpc     : float
                  The distance to source in pc. This is required to convert to radmc3dImage

        wav     : float or ndarray, optional
                  The wavelength of this image. If this is given, this will be preferred over the fits file
                  The number of elements must be the same as the number of elements of the third axis of 
                  the fits file
        recen   : list of 2 floats
                  The new center relative to the original, in arcseconds
        padnan  : float
                  value to give for locations with nan
    Returns
    -------
        res     : radmc3dImage object
    """
    if fname is None:
        raise ValueError('fname must be given')
    if dpc is None:
        raise ValueError('dpc must be given')
    if os.path.exists(fname) is False:
        raise ValueError('fits file does not exist: '+fname)
    if dpc is None:
        raise ValueError('dpc must be given')

    rad = np.pi / 180.

    hdul = pf.open(fname)
    hdr = hdul[0].header

    # data
    fitsdat = hdul[0].data

    if padnan is not None:
        reg = np.isnan(fitsdat)
        fitsdat[reg] = padnan

    cdelt1 = hdr['cdelt1'] * 3600. #in arcsec
    naxis1 = hdr['naxis1']
    ctype1 = hdr['ctype1']
    crpix1 = hdr['crpix1'] - 1
    crval1 = hdr['crval1']
#    cunit1 = hdr['cunit1']

    cdelt2 = hdr['cdelt2'] * 3600.
    naxis2 = hdr['naxis2']
    ctype2 = hdr['ctype2']
    crpix2 = hdr['crpix2'] - 1
    crval2 = hdr['crval2']
#    cunit2 = hdr['cunit2']

    if 'cdelt3' in hdr:
        cdelt3 = hdr['cdelt3']
        naxis3 = hdr['naxis3']
        ctype3 = hdr['ctype3']
        crpix3 = hdr['crpix3'] - 1
        crval3 = hdr['crval3']
#        cunit3 = hdr['cunit3']
    else:
        ctype3 = None
        naxis3 = 1

    if 'cdelt4' in hdr:
        cdelt4 = hdr['cdelt4']
        naxis4 = hdr['naxis4']
        ctype4 = hdr['ctype4']
        crpix4 = hdr['crpix4'] - 1
        crval4 = hdr['crval4']
#        cunit4 = hdr['cunit4']
    else:
        ctype4 = None
        naxis4 = 1

    # find the stokes and wavelength info
    nstokes = 1
    nfreq = 1
    crvalz = None
    
    if ctype3 is not None:
        if 'stokes' in ctype3.lower():
            nstokes = naxis3
        elif 'freq' in ctype3.lower():
            nfreq = naxis3
            cdeltz = cdelt3
            crpixz = crpix3
            crvalz = crval3

    if ctype4 is not None:
        if 'stokes' in ctype4.lower():
            nstokes = naxis4
        elif 'freq' in ctype4.lower():
            nfreq = naxis4  
            cdeltz = cdelt4
            crpixz = crpix4
            crvalz = crval4

    if nstokes > 1:
        isStokes = True
    else:
        isStokes = False

    # there seems to be a transpose difference
    ndim = fitsdat.shape
    if ndim[0] != naxis1:
        fitsdat = fitsdat.T

    # check the beam info
    if 'bmaj' in hdr:
        bmaj = hdr['bmaj'] * 3600. # in arcsec
    else:
        bmaj = None

    if 'bmin' in hdr:
        bmin = hdr['bmin'] * 3600.
    else:
        bmin = None

    if 'bpa' in hdr:
        bpa = hdr['bpa']
    else:
        bpa = None

    if (bmaj is not None) & (bmin is not None) & (bpa is not None):
        isJypBeam = True
    else:
        isJypBeam = False

    # recalculate center
    if recen is None:
        xcen, ycen = 0, 0
    else:
        xcen, ycen = recen[0], recen[1]

    x = (np.array(range(naxis1)) - crpix1) * cdelt1 - xcen #this is in arcsec
    x = x * dpc * nc.au 	#converted to cm
    y = (np.array(range(naxis2)) - crpix2) * cdelt2 - ycen
    y = y * dpc * nc.au
    if crvalz is not None:
        z = (np.array(range(nfreq)) - crpixz) * cdeltz + crvalz
    else:
        z = np.array([1])

    sizepix_x = abs(cdelt1 * dpc * nc.au)
    sizepix_y = abs(cdelt2 * dpc * nc.au)

    if isJypBeam:
        imJypBeam = fitsdat
        del_beam = np.pi * (bmaj /3600. * rad) * (bmin / 3600. * rad) / 4. / np.log(2.0)
        del_px = abs(cdelt1 / 3600. * rad) * abs(cdelt2 / 3600. * rad)
        imJyppix = imJypBeam * dpc**2 / del_beam * del_px
        if rms is not None:
            rms_ppix = rms * dpc**2 / del_beam * del_px
    else:
        imJyppix = fitsdat * dpc**2 	#imJyppix is Jy/pixel Image at 1 parsec
        if rms is not None:
            rms_ppix = rms * dpc**2

    conv = sizepix_x * sizepix_y / nc.pc**2. * 1e23
    im_cgs = imJyppix / conv
    if rms is not None:
        rms_cgs = rms_ppix / conv

    res = radmc3dImage()
    res.image = im_cgs
    res.imageJyppix = imJyppix
    if rms is not None:
        res.rms = [rms_cgs]
    res.sizepix_x = sizepix_x
    res.sizepix_y = sizepix_y
    res.x = x
    res.y = y
    res.nx = naxis1
    res.ny = naxis2
    res.nfreq = nfreq
    res.nwav = nfreq
    if wav is None:
        res.freq = abs(z)
        res.wav = nc.cc * 1e4 / res.freq
    else:
        if type(wav) == float or type(wav) == int:
            wav = np.array([wav], dtype=np.float64)
        
        res.freq = nc.cc * 1e4 / np.array(wav)
        res.wav = wav
    res.stokes = isStokes

    if (bmaj is not None) & (bmin is not None):
        res.fwhm = [[bmaj, bmin]]
    if bpa is not None:
        res.pa = [bpa]
    res.dpc = dpc
    res.filename = fname

    res.getTotalFlux()

    hdul.close()
    del fitsdat

    return res

def calcPolAng(Q, U):
    """ function to calculate polarization angle
    Parameters 
    ----------
    Q : ndarray
    U : ndarray

    Returns
    -------
    ang : ndarray
        the polarization angle in radians and in between 0, 2pi
    """
    ang = np.arctan2(U, Q) / 2.
    reg = ang < 0.
    ang[reg] = ang[reg] + 2*np.pi
    reg = ang > np.pi
    ang[reg] = ang[reg] - np.pi
    return ang

def plotPolDir(image=None, arcsec=False, au=False, dpc=None, ifreq=0, cmask_rad=None, color='w', nx=20, ny=20, 
    turn90=False, polunitlen=-1, quivwidth=0.005, textcolor='k', textxy=None, ax=None):
    """
    Function to plot the polarisation direction for full stokes images

    Parameters
    ----------

    image         : radmc3dImage
                    A radmc3dImage class returned by readimage   

    arcsec        : bool
                    If True image axis will have the unit arcsec (NOTE: dpc keyword should also be set!)

    au            : bool
                    If True image axis will have the unit AU

    dpc           : float
                    Distance to the source in parsec (This keywords should be set if arcsec=True, or bunit!='norm')

    ifreq         : int
                    If the image file/array consists of multiple frequencies/wavelengths ifreq denotes the index
                    of the frequency/wavelength in the image array to be plotted

    cmask_rad     : float
                    Simulates coronographyic mask : sets the image values to zero within this radius of the image center
                    The unit is the same as the image axis (au, arcsec, cm)
                    NOTE: this works only on the plot, the image array is not changed (for that used the cmask() 
                    function)
                    
    color         : str
                    Color for the polarisation direction plot

    nx            : int
                    Number of grid points along the horizontal axis at which the direction should be displayed

    ny            : int
                    Number of grid points along the vertical axis at which the direction should be displayed

    turn90        : bool, optional
                    if True, then turn the angle by 90 degrees. This would follow B field when optically thin

    polunitlen    : float
                    The percent value as a multiple of unit length. The given value will be the value for unit length.
                    Input -1 to use median value of polarization fraction
                    Input -2 to turn this option off, and let all vectors lengths to unity
    textcolor     : str
                    Color for any overplotted text
    textxy        : list
                    two element list for x,y coordinate
    """

    #
    # First check if the images is a full stokes image
    #

    if not image.stokes:
        msg = 'The image is not a full stokes image. Polarisation direction can only be displayed if '\
              + 'the full stokes vector is present at every pixel of the image'
        raise ValueError(msg)

    if cmask_rad is not None:
        dum_image = cmask(image, rad=cmask_rad, au=au, arcsec=arcsec, dpc=dpc)
    else:
        dum_image = copy.deepcopy(image)

    # Select the coordinates of the data
    if au:
        x = image.x / nc.au
        y = image.y / nc.au
        xlab = 'X [AU]'
        ylab = 'Y [AU]'
    elif arcsec:
        x = image.x / nc.au / dpc
        y = image.y / nc.au / dpc
        xlab = 'RA offset ["]'
        ylab = 'DEC offset ["]'
    else:
        x = image.x
        y = image.y
        xlab = 'X [cm]'
        ylab = 'Y [cm]'

    # ext = (x[0], x[image.nx-1], y[0], y[image.ny-1])

    # indices of polarization points defined by nx,ny
    iix = [int(np.floor(i)) for i in np.arange(nx) * float(x.shape[0]) / nx]
    iiy = [int(np.floor(i)) for i in np.arange(ny) * float(x.shape[0]) / ny]

    # space coordinates of those indices
    xr = x[iix]
    yr = y[iiy]

    # convert to meshgrid
    xxr, yyr = np.meshgrid(xr, yr, indexing='ij')

    # stokes Q / stokes I
    qqr = (np.squeeze(dum_image.image[:, :, 1, ifreq])
           / np.squeeze(dum_image.image[:, :, 0, ifreq]).clip(1e-60))[np.ix_(iix, iiy)]
    # stokes U / stokes I
    uur = (np.squeeze(dum_image.image[:, :, 2, ifreq])
           / np.squeeze(dum_image.image[:, :, 0, ifreq]).clip(1e-60))[np.ix_(iix, iiy)]
    # fraction of linear polarization = sqrt(Q^2 + U^2) / I
    lpol = np.sqrt(qqr**2 + uur**2).clip(1e-60) 

    qqr /= lpol # Q / sqrt(Q**2 + U**2)
    uur /= lpol # U / sqrt(Q**2 + U**2)
    
    # determine angles
#    ang = np.arccos(qqr) / 2.0
#    ii = (uur < 0)
#    if True in ii:
#        ang[ii] = np.pi - ang[ii]
    ang = calcPolAng(qqr, uur)

    if turn90:
        ang = ang + np.pi/2.0

    # length of line
    if polunitlen == -1:
#        vlen = 1.0
#        pltlen = 'not to scale'
        polunitlen = np.median(lpol*100.)
        if polunitlen > 1.:
            polunitlen = int(polunitlen)
        vlen = lpol * 100. / polunitlen
        pltlen = str(polunitlen) + '%'
    elif polunitlen == -2:
        vlen = 1.0
#        pltlen = 'not to scale'
        pltlen = ''
    else:
        vlen = lpol * 100. / polunitlen  #polunitlen is in percent
        pltlen = str(polunitlen) + '%'

    if quivwidth is None:
        quivwidth = 0.005
         
    vx = vlen * np.cos(ang)
    vy = vlen * np.sin(ang)

    ii = (lpol < 1e-6)
    vx[ii] = 0.001
    vy[ii] = 0.001

    # set up axes object
    if ax is None:
        ax = plt.gca()

    ax.quiver(xxr, yyr, vx, vy, color=color, 
        pivot='mid', scale=1.25 * np.max([nx, ny]),
        width=quivwidth,  
        headwidth=1e-10, headlength=1e-10, headaxislength=1e-10)

    # set up the text 
    if textxy is None:
        textx = 0.75 * xr.max()
        texty = 0.75 * yr.min()
    else:
        textx = textxy[0]
        texty = textxy[1]

    if pltlen is not '':
        ax.quiver(textx, texty, np.cos(np.pi), np.sin(np.pi), scale=1.25*np.max([nx,ny]), 
               color=textcolor, pivot='mid', headwidth=1e-10, headlength=1e-10, headaxislength=1e-10)
    
    ax.text(textx, texty*1.2, pltlen, color=textcolor, va='center', ha='center')

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    return


def plotImage(image=None, arcsec=False, au=False, log=False, dpc=None, maxlog=None, saturate=None, bunit='norm',
              recenter=None, 
              ifreq=0, cmask_rad=None, cmask_thres=None, interpolation='bilinear', cmap=plt.cm.gist_gray, stokes='I',
              oplotbeam=None, beamxy=None, 
              textcolor='w', nocolorbar=False, titleplt=None, nolabel=False, 
              clevs=None, cllabel=True, cllabel_fontsize=10, cllabel_fmt="%.1f", clcol='k', 
              fig=None, ax=None, projection='polar', deg=True, rmax=None, rlog=True, **kwargs):
    """Plots a radmc3d image.


    Parameters
    ----------
    image         : radmc3dImage
                    A radmc3dImage class returned by readimage   

    arcsec        : bool
                    If True image axis will have the unit arcsec (NOTE: dpc keyword should also be set!)

    au            : bool
                    If True image axis will have the unit AU

    log           : bool
                    If True image scale will be logarithmic, otherwise linear

    dpc           : float
                    Distance to the source in parsec (This keywords should be set if arcsec=True, or bunit!='norm')

    maxlog        : float
                    Logarithm of the lowest pixel value to be plotted, lower pixel values will be clippde

    saturate      : str
                    Highest pixel values to be plotted. 
                    If given a number, then means the peak value, higher pixel values will be clipped
                    If given like '##percent', where ## is a number, then it is in percent of highest pixel value

    bunit         : {'norm', 'inu', 'snu', 'jy/beam', 'jy/pixel', 'tb', 'percent', 'optdepth', 'surface', 'deg'}
                    Unit of the image, ('norm' - Inu/max(Inu), 'inu' - Inu, 'snu' - Jy/pixel, 'jy/pixel' - Jy/pixel,
                    'jy/beam' - Jy/beam), default is 'norm'. The 'snu' keyword value is kept for backward compatibility
                    as it is fully equivalent with the 'jy/pixel' keyword value.
                    added 'tb' - brightness temperature
                    added 'percent' - use this only for stoke='P' or 'PL' to present data in percentage
                    added 'optdepth' - use this only for plotting optical depth images
                    added 'surface' - use for plotting tau surface. units will be same as the x,y axis
                    added 'deg' - for polarization angle

    recenter      : 2 element array 
                    center coordinate relative to original coordinates, to plot as (0,0). In units of AU

    ifreq         : int
                    If the image file/array consists of multiple frequencies/wavelengths ifreq denotes the index
                    of the frequency/wavelength in the image array to be plotted

    cmask_rad     : float
                    Simulates coronographyic mask : sets the image values to zero within this radius of the image center
                    The unit is the same as the image axis (au, arcsec, cm)
                    NOTE: this works only on the plot, the image array is not changed (for that used the cmask() 
                    function)

    cmask_thres     : float
                      masks the intensity in terms of factor of maximum intensity when plotting any type of stokes. 
                      Ex: 1e-5, 0.1

    interpolation : str
                    interpolation keyword for imshow (e.g. 'nearest', 'bilinear', 'bicubic')

    cmap          : matplotlib color map

    stokes        : {'I', 'Q', 'U', 'V', 'PI', 'P', 'PIL', 'PL', 'ANG'}
                   What to plot for full stokes images, Stokes I/Q/U/V,
                   PI  - polarised intensity (PI = sqrt(Q^2 + U^2 + V^2))
                   P   - polarisation fraction (i.e. sqrt(Q^2 + U^2 + V^2) / I)
                   PIL - linear polarised intensity (PI = sqrt(Q^2 + U^2))
                   PL  - fraction of linear polarisation (i.e. sqrt(Q^2 + U^2) / I)
                   ANG - polarization angle ( ang=arctan2(U, Q) / 2 )

    oplotbeam     : str, optional
                    Input a string for its color, then will overplot an ellipse for the fwhm of psf

    beamxy        : array, optional
                    Input a two element array for the center coordinate of plotting the beam

    textcolor     : str, optional
                    The color for any string annotation on the image

    fig           : matplotlig.figure.Figure, optional
                   An instance of a matplotlib Figure. If not provided a new Figure will be generated. If provided 
                   plotImage will add a single Axes to the Figure, using add_subplots() with the appropriate projection.
                   If the desired plot is to be made for a multi-panel plot, the appropriate Axes instance can be 
                   passed to the ax keyword. This keyword is only used for circular images.

    ax            : matplotlib.axes.Axes, optional
                   An instance of a matplotlib Axes to draw the plot on. Note, that the projection of the axes should 
                   be the same as the projection keyword passed to plotImage. 

    projection    : {'polar', 'cartesian'}
                   Projection of the plot. For cartesian plots a rectangular plot will be drawn, with the horizontal 
                   axis being the azimuth angle, and the vertical axis the radius. Only for circular images

    deg           : bool
                   If True the unit of the azimuthal coordinates will degree, if False it will be radian. Used only for
                   circular images and for cartesian projection. 

    rmax          : float
                   Maximum value of the radial coordinate for polar projection. Used only for circular images.

    rlog          : bool
                   If True the radial coordiante axis will be set to logarithmic for cartesian projection. Used only 
                   for circular images.


    Example
    -------

    result = plotImage(image='image.out', arcsec=True, au=False, log=True, dpc=140, maxlog=-6., 
             saturate=0.1, bunit='Jy')
    """

    if isinstance(image, radmc3dImage):

        if ifreq is None:
            ifreq = 0

        # Check whether or not we need to mask the image

        # data as [ra, dec, stokes, wav]
        dum_image = copy.deepcopy(image)
        if dum_image.stokes:
            if stokes.strip().upper() == 'I':
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 0]
                else:
                    dum_image.image = image.image[:, :, 0, :]
                istokes = 0
                dototflux = 1
                title_stokes = 'Stokes I'

            if stokes.strip().upper() == 'Q':
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 1]
                else:
                    dum_image.image = image.image[:, :, 1, :]
                istokes = 1
                dototflux = 1
                title_stokes= 'Stokes Q'

            if stokes.strip().upper() == 'U':
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 2]
                else:
                    dum_image.image = image.image[:, :, 2, :]
                istokes = 2
                dototflux = 1
                title_stokes = 'Stokes U'

            if stokes.strip().upper() == 'V':
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 3]
                else:
                    dum_image.image = image.image[:, :, 3, :]
                istokes = 3
                dototflux = 1
                title_stokes = 'Stokes V'

            if stokes.strip().upper() == 'PI':
                if dum_image.nwav == 1:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2 + image.image[:, :, 3]**2)
                else:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2 + image.image[:, :, 3, :]**2)
                dototflux = 0
                title_stokes = 'Polarized Intensity'

            if stokes.strip().upper() == 'P':
                if dum_image.nwav == 1:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2 + image.image[:, :, 3]**2) / \
                                      (image.image[:, :, 0] + 1e-90)
                else:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2 + image.image[:, :, 3, :]**2) / \
                                      (image.image[:, :, 0, :] + 1e-90)
                dototflux = 0
                title_stokes = 'Polarization Fraction'

            if stokes.strip().upper() == 'PIL':
                if dum_image.nwav == 1:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2)
                else:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2)
                dototflux = 0
                title_stokes = 'Linear Polarization Intensity'

            if stokes.strip().upper() == 'PL':
                if dum_image.nwav == 1:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2) / (image.image[:, :, 0] + 1e-90)
#                    reg = image.image[:,:,0] <= 1e-60
#                    dum_image.image[reg] = 0.

                else:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2) / (image.image[:, :, 0, :] + 1e-90)
                     
                dototflux = 0
                title_stokes = 'Linear Polarization Fraction'
            if stokes.strip().upper() == 'ANG':
                if dum_image.nwav == 1:
                    dum_image.image = calcPolAng(image.image[:,:,1], image.image[:,:,2])
                else:
                    dum_image.image = calcPolAng(image.image[:,:,1,:], image.image[:,:,2,:])
                dototflux = 0
                title_stokes = 'Polarization Angle'
        else:
            if dum_image.nwav == 1:
                dum_image.image = image.image[:, :, :]
            else:
                dum_image.image = image.image[:, :, :]
            dototflux = 1
            title_stokes = 'Stokes I'

        if cmask_rad is not None:
            dum_image = cmask(dum_image, rad=cmask_rad, au=au, arcsec=arcsec, dpc=dpc)
        else:
            dum_image = dum_image

# mask based on amplitude
        if cmask_thres is not None:
            if image.nwav == 1:
                if image.stokes:
                    stokesImax = image.image[:,:,0].max()
                    reg = image.image[:,:,0]<=(stokesImax*cmask_thres)
                else:
                    stokesImax = image.image[:,:,:].max()
                    reg = image.image[:,:,:]<=(stokesImax*cmask_thres)
                dum_image.image[reg] = 0.
            else:
                for iwav in range(image.nwav):
                    if image.stokes:
                        stokesImax = image.image[:,:,0,iwav].max()
                        reg = image.image[:,:,0,iwav]<=(stokesImax*cmask_thres)
                        dumdum = np.squeeze(dum_image.image[:,:,0,iwav].copy())
                    else:
                        stokesImax = image.image[:,:,iwav].max()
                        reg = image.image[:,:,iwav]<=(stokesImax*cmask_thres)
                        dumdum= np.squeeze(image.image[:,:,iwav].copy())
                    dumdum = np.squeeze(dum_image.image[:,:,iwav].copy())
                    dumdum[reg] = 0.
                    dum_image.image[:,:, iwav] = dumdum

#        data = np.squeeze(dum_image.image[:, ::-1, ifreq].T) #flips y for imshow
        data = np.squeeze(dum_image.image[:,:,ifreq].T.copy()) #doesn't flip y for pcolormesh

        if bunit == 'norm':
            norm = data.max()
            if abs(-data.min()) > norm:
                norm = abs(-data.min())
            data = data / norm

        clipnorm = data.max()
        # Check if the data should be plotted on a log scale even considering the negative
        # negative after logged
        if log:
            imgpos = data >0
            imgeq0 = data == 0
            imgneg = data < 0

            if True in imgpos:
                data[imgpos] = np.log10(data[imgpos])
            if True in imgeq0:
                data[imgeq0] = data[imgpos].min() 
            if True in imgneg:
                data[imgneg] = np.nan
#                raise ValueError('data is negative but log is on. not acceptable now')

            # clipmin is the minimum after logged
            clipmin = np.nanmin(data)

            # Clipping the data
            if maxlog is not None:
                clipmin = maxlog
        else:
            clipmin = data.min()

        if saturate is not None:
            if 'percent' in saturate:
                saturatenum = float(saturate.replace('percent', ''))
                if saturatenum > 100.:
                    saturatenum = 100.
                if log:
                    clipmax = np.log10(saturatenum/100.) + np.log10(clipnorm)
                else:
                    clipmax = clipnorm * saturatenum/100.
            else: 
                saturatenum = float(saturate)
                if log:
                    clipmax = np.log10(saturatenum)
                else:
                    clipmax = saturatenum
        else:
            clipmax = clipnorm

        data = data.clip(clipmin, clipmax)

        # mask the data
        data_original = data
        data = np.ma.masked_where(data == clipmin, data)
        cmap.set_bad(color='k', alpha=0)


#        if image.dpc is not 0:
#            dpc = image.dpc

        # Select the unit of the data

        title_bunit = ''
        if bunit.lower() == 'norm':
            if log:
                cb_label = r'$log_{10}(I$' + r'$_\nu$' + '/max(I' + r'$_\nu$' + '))'
            else:
                cb_label = 'I' + r'$_\nu$' + '/max(I' + r'$_\nu$' + ')'
        elif bunit.lower() == 'inu':
            if log:
                cb_label = r'log$_{10}$(I' + r'$_\nu$' + ' [erg/s/cm/cm/Hz/ster])'
            else:
                cb_label = 'I' + r'$_\nu$' + ' [erg/s/cm/cm/Hz/ster]'

        elif (bunit.lower() == 'snu') | (bunit.lower() == 'jy/pixel'):
            if dpc is None:
                msg = 'Unknown dpc. If Jy/pixel is selected for the image unit the dpc keyword should also be set'
                raise ValueError(msg)
            else:
                if log:
                    data = data + np.log10(image.sizepix_x * image.sizepix_y / (dpc * nc.pc)**2. * 1e23)
                    cb_label = 'log(S' + r'$_\nu$' + '[Jy/pixel])'
                else:
                    data = data * (image.sizepix_x * image.sizepix_y / (dpc * nc.pc)**2. * 1e23)
                    cb_label = 'S' + r'$_\nu$' + ' [Jy/pixel]'

        elif bunit.lower() == 'jy/beam':
            if len(image.fwhm) == 0:
                msg = 'The image does not appear to be convolved with a Gaussain (fwhm data attribute is empty). ' \
                      'The intensity unit can only be converted to Jy/beam if the convolving beam size is known'
                raise ValueError(msg)

            #pixel_area = (image.sizepix_x * image.sizepix_y)/(dpc * nc.pc)**2 * (180./np.pi*3600.)**2
            #beam_area = image.fwhm[0] * image.fwhm[1] * np.pi / 4. / np.log(2.0)
            pixel_area = (image.sizepix_x * image.sizepix_y)/(dpc * nc.pc)**2   # in radians**2
            # fwhm should be in arcseconds. then beam_area in radians**2
            beam_area = (image.fwhm[ifreq][0]/3600.0*np.pi/180.) * (image.fwhm[ifreq][1]/3600.*np.pi/180.) * np.pi / 4. / np.log(2.0)

            if log:
                # Convert data to Jy/pixel
                data += np.log10((image.sizepix_x * image.sizepix_y / (dpc * nc.pc)**2. * 1e23))
                # Convert data to Jy/beam
                data += np.log10(beam_area / pixel_area)

                cb_label = r'log$_{10}$(S' + r'$_\nu$' + '[Jy/beam])'
            else:
                # Convert data to Jy/pixel
                #data *= (image.sizepix_x * image.sizepix_y / (dpc * nc.pc)**2. * 1e23) #this is also correct
                data *= pixel_area * 1e23
                # Convert data to Jy/beam
                data *= beam_area / pixel_area
                cb_label = 'S' + r'$_\nu$' + ' [Jy/beam]'
        elif bunit.lower() == 'mjy/beam':
            if len(image.fwhm) == 0:
                msg = 'The image does not appear to be convolved with a Gaussain (fwhm data attribute is empty). ' \
                      'The intensity unit can only be converted to Jy/beam if the convolving beam size is known'
                raise ValueError(msg)
            beam_area = (image.fwhm[ifreq][0]/3600.0*np.pi/180.) * (image.fwhm[ifreq][1]/3600.*np.pi/180.) * np.pi / 4. / np.log(2.0)
            if log:
                # convert to mJy/beam
                data += np.log10(beam_area * 1e3 * 1e23)
                cb_label = r'log$_{10}$(S' + r'$_\nu$' + '[mJy/beam])'
            else:
                data *= beam_area * 1e3 * 1e23
                cb_label = 'S' + r'$_\nu$' + ' [mJy/beam]'
        elif bunit.lower() == 'tb':
            if dpc is None:
                msg = 'Unknown dpc. If tb is selected for the image unit the dpc keyword should also be set'
                raise ValueError(msg)

            if len(image.fwhm) == 0:
                # using pixel area
                norm_area = (image.sizepix_x * image.sizepix_y)/(nc.pc * dpc)**2 
            else:
                # using beam area
                if len(image.fwhm) == 1:
                    norm_area = (image.fwhm[0][0]/3600.*np.pi/180.) * (image.fwhm[0][1]/3600.*np.pi/180.) * np.pi / 4. / np.log(2.0)
                elif len(image.fwhm) == image.nfreq:
                    norm_area = (image.fwhm[ifreq][0]/3600.*np.pi/180.) * (image.fwhm[ifreq][1]/3600.*np.pi/180.) * np.pi / 4. / np.log(2.0)
                else:
                    raise ValueError('wrong number of elements for fwhm')
            ## though it seems like radmc3dImage are not in jy/pixel, but in intensity

            ld2 = (image.wav[ifreq] * 1e-4 )** 2.
            hmu = nc.hh * image.freq[ifreq]
            hmu3_c2 = nc.hh * image.freq[ifreq]**3 / nc.cc**2

            if log:
#                data = data + np.log10(ld2 / 2. / nc.kk ) #Rayleigh-Jeans limit
                data = np.log10(hmu / nc.kk / np.log(2.*hmu3_c2/(data+1e-90) + 1.)) #Planck
                cb_label = 'log(Tb) [K]'
            else:
#                data = data * ld2 / 2. / nc.kk #Rayleigh-Jeans limit
                # negative values
                reg = data <= 0.
                data[reg] = data[reg] * ld2 / 2. / nc.kk
                # positive values
                reg = ~ reg
                data[reg] = hmu / nc.kk / np.log(2.*hmu3_c2/(data[reg]+1e-90)+1.) #Planck
                cb_label = 'Tb [K]'
        elif bunit.lower() == 'percent':
            if log:
                data = data + np.log10(100.)
                cb_label = 'log(percent)'
            else:
                data = data * 100.
                #cb_label = 'Percent [%]'
                cb_label = '[%]'
        elif bunit.lower() == 'optdepth':
            if log:
                cb_label = 'log(Optical Depth)'
            else:
                #cb_label = 'Optical Depth'
                cb_label = ''
            dototflux = 0
            title_bunit = 'Optical Depth'

        elif bunit.lower() == 'length':
            if au:
                data = data / nc.au
                cb_label_unit = '[AU]'
            elif arcsec:
                data = data / nc.au / dpc
                cb_label_unit = '[arcsec]'
            else:
                cb_label_unit = '[cm]'

            if log:
                data = np.log10(data)
                cb_label = 'log(Distance) '+cb_label_unit
            else:
                cb_label = 'Distance '+cb_label_unit
            dototflux = 0
            title_bunit = 'Tau Surface'
        elif bunit.lower() == 'deg':
            data = data / nc.rad
            cb_label = r'Degrees [$^{\circ}$]'
        else:
            msg = 'Unknown bunit: ' + bunit + ' Allowed values are "norm", "inu", "snu", "tb", "percent", "surface", "optdepth"'
            raise ValueError(msg)

        # Select the coordinates of the data
        x = image.x
        y = image.y
        if recenter is not None:
            x = x - np.array(recenter[0]) * nc.au
            y = y - np.array(recenter[1]) * nc.au

        if au:
            x = x / nc.au
            y = y / nc.au
            xlab = 'X [AU]'
            ylab = 'Y [AU]'
        elif arcsec:
            x = x / nc.au / dpc
            y = y / nc.au / dpc
            xlab = 'RA offset ["]'
            ylab = 'DEC offset ["]'
        else:
            x = x
            y = y
            xlab = 'X [cm]'
            ylab = 'Y [cm]'

        ext = (x[0], x[image.nx - 1], y[0], y[image.ny - 1])

        # Now finally put everything together and plot the data
#        plt.delaxes()
#        plt.delaxes()

#        implot = plt.imshow(data, extent=ext, cmap=cmap, interpolation=interpolation, **kwargs)
        if ax is None:
            ax = plt.gca()

        implot = ax.pcolormesh(x, y, data, cmap=cmap, **kwargs)
        ax.set_aspect(abs((ext[1]-ext[0])/(ext[3]-ext[2])) / 1.)

        if clevs is not None:
            cc = ax.contour(data, clevs, extent=ext, colors=clcol)
            if cllabel:
                ax.clabel(cc, inline=1, fontsize=cllabel_fontsize, fmt=cllabel_fmt)

        # x,y labels
        if nolabel is False:
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)

        # title
        if titleplt is None:
            ax.set_title(title_stokes + ' ' + title_bunit + ' at ' + r'$\lambda$=' + 
                   ("%.2f" % image.wav[ifreq]) + r'$\mu$m')
        else:
            ax.set_title(titleplt)

        # colorbar
        if nocolorbar is True:
            cbar = None
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(implot, cax=cax)
            cbar.set_label(cb_label)

        # over plot the normalization values
        if bunit is 'norm':
            plottxt = 'Norm=%.2e'%(norm)
            ax.text(x.max()*0.9,y.min()*0.9, plottxt, color=textcolor, va='bottom', ha='right') 

        # overplot the total flux in Jy
        if dototflux:
            if image.totflux is None:
                image.getTotalFlux()
            if image.stokes:
                if image.nwav == 1:
                    plottotflux = image.totflux[istokes]
                else:
                    plottotflux = image.totflux[istokes, ifreq]
            else: 
                    plottotflux = image.totflux[ifreq]
            plottotflux = '%.2e Jy' % plottotflux
            #plt.text(x.max()*0.7, y.max()*0.9, plottotflux, color=textcolor, va='bottom', ha='right')

        # overplotting beam
        if (oplotbeam is not None) & (len(image.fwhm) != 0):
            if image.nfreq == len(image.fwhm):
                ifwhm = ifreq
            else:
                ifwhm = 0
            if image.nfreq == len(image.pa):
                ipa = ifreq
            else:
                ipa = 0

            if au:
                if dpc is None:
                    raise ValueError('dpc is needed to overplot beam size')
                ewidth = image.fwhm[ifwhm][1] * dpc
                eheight = image.fwhm[ifwhm][0] * dpc
            elif arcsec:
                ewidth = image.fwhm[ifwhm][1]
                eheight = image.fwhm[ifwhm][0]
            else:
                ewidth = image.fwhm[ifwhm][1] * dpc * nc.au
                eheight = image.fwhm[ifwhm][0] * dpc * nc.au

            # coordinate of the beam
            if beamxy is None:
                ellx = x.min()*0.75
                elly = y.min()*0.55
            else:
                ellx = beamxy[0]
                elly = beamxy[1]

            # angle
            beamang = 180. - image.pa[ipa]

            ells = Ellipse(xy=(ellx, elly), width=ewidth,
                               height=eheight, angle=beamang)
            ells.set_facecolor(oplotbeam)
            ells.set_fill(True)
            implot.axes.add_patch(ells)
#        plt.show()

    # ------------------------------------------------------#
    elif isinstance(image, radmc3dCircimage):

        # Check whether or not we need to mask the image

        dum_image = copy.deepcopy(image)
        if dum_image.stokes:
            if stokes.strip().upper() == 'I':
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 0]
                else:
                    dum_image.image = image.image[:, :, 0, :]

            if stokes.strip().upper() == 'Q':
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 1]
                else:
                    dum_image.image = image.image[:, :, 1, :]

            if stokes.strip().upper() == 'U':
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 2]
                else:
                    dum_image.image = image.image[:, :, 2, :]

            if stokes.strip().upper() == 'V':
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 3]
                else:
                    dum_image.image = image.image[:, :, 3, :]

            if stokes.strip().upper() == 'PI':
                if dum_image.nwav == 1:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2 + image.image[:, :, 3]**2)
                else:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2 + image.image[:, :, 3, :]**2)

            if stokes.strip().upper() == 'P':
                if dum_image.nwav == 1:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2 + image.image[:, :, 3]**2) / \
                                      image.image[:, :, 0]

                else:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2 + image.image[:, :, 3, :]**2) / \
                                      image.image[:, :, 0, :]

            if stokes.strip().upper() == 'PIL':
                if dum_image.nwav == 1:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2)
                else:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2)

            if stokes.strip().upper() == 'PL':
                if dum_image.nwav == 1:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2) / image.image[:, :, 0]

                else:
                    dum_image.image = np.sqrt(
                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2) / image.image[:, :, 0, :]
        else:
            dum_image.image = image.image[:, :, 0, :]

        if cmask_rad is not None:
            cmask_rad_cm = cmask_rad
            if au:
                cmask_rad_cm = cmask_rad * 1.496e13
            if arcsec:
                cmask_rad_cm = cmask_rad * dpc * nc.pc

            ii = (dum_image.ri <= cmask_rad_cm)

            if len(dum_image.image.shape) == 3:
                dum_image.image[ii, :, :] = 0.

            elif len(dum_image.image.shape) == 4:
                dum_image.image[ii, :, :, :] = 0.

        if ifreq is None:
            ifreq = 0

        if bunit == 'snu':
            if dpc is None:
                msg = 'Unknown bunit: ' + bunit + ' Allowed values are "norm", "inu", "snu"'
                raise ValueError(msg)
            else:
                pixel_area = image.getPixelSize()
                conv = pixel_area / (dpc * nc.pc**2) * 1e23
                data = dum_image.image[:, :, ifreq] * conv
        else:
            data = dum_image.image[:, :, ifreq]

        norm = data.max()
        if abs(-data.min()) > norm:
            norm = abs(-data.min())
        if bunit == 'norm':
            data = data / norm

        clipnorm = data.max()
        # Check if the data should be plotted on a log scale
        if log:
            clipmin = np.log10(data[data > 0.].min())
            data = np.log10(data.clip(1e-90))

            # Clipping the data
            if maxlog is not None:
                clipmin = -maxlog + np.log10(clipnorm)
        else:
            clipmin = data.min()

        if saturate is not None:
            if saturate > 1.:
                saturate = 1.0
            if log:
                clipmax = np.log10(saturate) + np.log10(clipnorm)
            else:
                clipmax = clipnorm * saturate
        else:
            clipmax = clipnorm

        # Select the unit of the data

        if bunit == 'norm':
            if log:
                cb_label = 'log(I' + r'$_\nu$' + '/max(I' + r'$_\nu$' + '))'
            else:
                cb_label = 'I' + r'$_\nu$' + '/max(I' + r'$_\nu$' + ')'
        elif bunit == 'inu':
            if log:
                cb_label = 'log(I' + r'$_\nu$' + ' [erg/s/cm/cm/Hz/ster])'
            else:
                cb_label = 'I' + r'$_\nu$' + ' [erg/s/cm/cm/Hz/ster]'
        elif bunit == 'snu':
            if log:
                cb_label = 'log(S' + r'$_\nu$' + '[Jy/pixel])'
            else:
                cb_label = 'S' + r'$_\nu$' + ' [Jy/pixel]'

        else:
            msg = 'Unknown bunit: ' + bunit + ' Allowed values are "norm", "inu", "snu"'
            raise ValueError(msg)

        # Select the coordinates of the data
        x = image.phii
        xlab = 'Azimuth angle [rad]'
        if projection == 'cartesian':
            if deg:
                x = image.phii / np.pi * 180.
                xlab = 'Azimuth angle [deg]'

        if au:
            y = image.rc / 1.496e13
            ylab = 'R [AU]'
        elif arcsec:
            y = image.rc / 1.496e13 / dpc
            ylab = 'R [arcsec]'
        else:
            y = image.rc
            ylab = 'R [cm]'

        if rmax is None:
            rmax = x.max()

        if fig is None:
            fig = plt.figure()

        if projection == 'polar':
            ax = fig.add_subplot(111, projection='polar')
            implot = plt.pcolormesh(x, y, data, **kwargs)
            ax.set_rmax(rmax)

        elif projection == 'cartesian':
            if ax is not None:
                plt.sca(ax)

            implot = plt.pcolormesh(x, y, data, **kwargs)
            plt.xlim(x[0], x[-1])
            plt.ylim(y[0], y[-1])
            if rlog:
                plt.yscale('log')
        else:
            msg = 'Unknown projection. Accepted values for projection keyword are "polar" or "cartesian".'
            raise ValueError(msg)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        cbar = plt.colorbar()
        cbar.set_label(cb_label)
        plt.show()

    return {'implot': implot, 'cbar': cbar, 'ax':ax}

def plotChannel(image=None, wavinx=None, chnfig=None, chngrid=None,
    restfreq=None, **kwargs):
    """
    Plots series of radmc3d Image using plotImage, iterating in wavelength
    Parameters
    ----------
    image : radmc3dImage
             
    wavinx : list of int
             index for ifreq to be plotted. Default is to plot all wavelengths
    chnfig : plt.figure()
             the figure to be plotted on
    **kwargs : other keywords for plotImage
    """
    if wavinx is None:
        wavinx = list(np.arange(image.nwav))
        nwavinx = image.nwav
    nwavinx = len(wavinx)

    nrows = np.floor(np.sqrt(nwavinx))
    ncols = np.ceil(nwavinx / nrows)
    nrows, ncols = int(nrows), int(ncols)

    from mpl_toolkits.axes_grid1 import ImageGrid
    if chnfig is None:
        chnfig = plt.figure(figsize=(4*ncols, 3*nrows))
    if chngrid is None:
        chngrid = ImageGrid(chnfig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.1,
            share_all=True, cbar_location='right', cbar_mode='single',
            cbar_size='5%', cbar_pad=0.15)

    # calculate velocity information
    if restfreq is not None:
        vel = nc.cc * (1. - image.freq / restfreq) * (-1)
        veltxt = ['%.2f km/s'%(ivel/1e5) for ivel in vel]

    for ii in range(nwavinx):
        # plotting axes
        axii = chngrid[ii]
        dum = plotImage(image=image, ifreq=wavinx[ii], ax=axii, 
            nocolorbar=True, titleplt='', **kwargs)
        if restfreq is not None:
            axii.text(0.98, 0.95, veltxt[ii], ha='right', va='top', 
                transform=axii.transAxes, color='w')

    cbar = axii.cax.colorbar(dum['implot'])

    return chngrid

def plotTausurf(image=None, arcsec=False, au=False, log=False, dpc=None, maxlog=None, saturate=None, bunit='norm',
              ifreq=0, cmask_rad=None, interpolation='nearest', cmap=plt.cm.gist_gray, stokes='I',
              fig=None, ax=None, projection='polar', deg=True, rmax=None, rlog=True, **kwargs):
    """Plots a radmc3d image.


    Parameters
    ----------
    image         : radmc3dImage
                    A radmc3dImage class returned by readimage that read a tau surface image

    arcsec        : bool
                    If True image axis will have the unit arcsec (NOTE: dpc keyword should also be set!)

    au            : bool
                    If True image axis will have the unit AU

    log           : bool
                    If True image scale will be logarithmic, otherwise linear

    dpc           : float
                    Distance to the source in parsec (This keywords should be set if arcsec=True, or bunit!='norm')

    maxlog        : float
                    Logarithm of the lowest pixel value to be plotted, lower pixel values will be clippde

    saturate      : str
                    Highest pixel values to be plotted.
                    If given a number, then means the peak value, higher pixel values will be clipped
                    If given like '##percent', where ## is a number, then it is in percent of highest pixel value

    bunit         : {'norm', 'length'}
                    Unit of the image, ('norm' - tausurf / max(tausurf), 'length' - values are in units of coordinates

    ifreq         : int
                    If the image file/array consists of multiple frequencies/wavelengths ifreq denotes the index
                    of the frequency/wavelength in the image array to be plotted

    cmask_rad     : float
                    Simulates coronographyic mask : sets the image values to zero within this radius of the image center
                    The unit is the same as the image axis (au, arcsec, cm)
                    NOTE: this works only on the plot, the image array is not changed (for that used the cmask()
                    function)

    interpolation : str
                    interpolation keyword for imshow (e.g. 'nearest', 'bilinear', 'bicubic')

    cmap          : matplotlib color map

    stokes        : {'I', 'Q', 'U', 'V', 'PI', 'P', 'PIL', 'PL'}
                   What to plot for full stokes images, Stokes I/Q/U/V,
                   PI  - polarised intensity (PI = sqrt(Q^2 + U^2 + V^2))
                   P   - polarisation fraction (i.e. sqrt(Q^2 + U^2 + V^2) / I)
                   PIL - linear polarised intensity (PI = sqrt(Q^2 + U^2))
                   PL  - fraction of linear polarisation (i.e. sqrt(Q^2 + U^2) / I)
                   Basically, tausurf is only for 'I', but just leave these as place holders

    fig           : matplotlig.figure.Figure, optional
                   An instance of a matplotlib Figure. If not provided a new Figure will be generated. If provided
                   plotImage will add a single Axes to the Figure, using add_subplots() with the appropriate projection.
                   If the desired plot is to be made for a multi-panel plot, the appropriate Axes instance can be
                   passed to the ax keyword. This keyword is only used for circular images.

    ax            : matplotlib.axes.Axes, optional
                   An instance of a matplotlib Axes to draw the plot on. Note, that the projection of the axes should
                   be the same as the projection keyword passed to plotImage. This keyword is only used for circular
                   images.

    projection    : {'polar', 'cartesian'}
                   Projection of the plot. For cartesian plots a rectangular plot will be drawn, with the horizontal
                   axis being the azimuth angle, and the vertical axis the radius. Only for circular images

    deg           : bool
                   If True the unit of the azimuthal coordinates will degree, if False it will be radian. Used only for
                   circular images and for cartesian projection.

    rmax          : float
                   Maximum value of the radial coordinate for polar projection. Used only for circular images.

    rlog          : bool
                   If True the radial coordiante axis will be set to logarithmic for cartesian projection. Used only
                   for circular images.


    Example
    -------

    result = plotImage(image='tausurf.out', arcsec=True, au=False, log=True, dpc=140, maxlog=-6.,
             saturate=0.1, bunit='Jy')
    """

    if isinstance(image, radmc3dImage):

        # Check whether or not we need to mask the image

        dum_image = copy.deepcopy(image)
        if dum_image.stokes:
            if stokes.strip().upper() == 'I' or stokes.strip().upper() != 'I': #force to use this only
                if dum_image.nwav == 1:
                    dum_image.image = image.image[:, :, 0]
                else:
                    dum_image.image = image.image[:, :, 0, :]

#            if stokes.strip().upper() == 'Q':
#                if dum_image.nwav == 1:
#                    dum_image.image = image.image[:, :, 1]
#                else:
#                    dum_image.image = image.image[:, :, 1, :]
#
#            if stokes.strip().upper() == 'U':
#                if dum_image.nwav == 1:
#                    dum_image.image = image.image[:, :, 2]
#                else:
#                    dum_image.image = image.image[:, :, 2, :]
#
#            if stokes.strip().upper() == 'V':
#                if dum_image.nwav == 1:
#                    dum_image.image = image.image[:, :, 3]
#                else:
#                    dum_image.image = image.image[:, :, 3, :]
#            if stokes.strip().upper() == 'PI':
#                if dum_image.nwav == 1:
#                    dum_image.image = np.sqrt(
#                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2 + image.image[:, :, 3]**2)
#                else:
#                    dum_image.image = np.sqrt(
#                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2 + image.image[:, :, 3, :]**2)
#
#            if stokes.strip().upper() == 'P':
#                if dum_image.nwav == 1:
#                    dum_image.image = np.sqrt(
#                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2 + image.image[:, :, 3]**2) / \
#                                      image.image[:, :, 0]
#
#                else:
#                    dum_image.image = np.sqrt(
#                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2 + image.image[:, :, 3, :]**2) / \
#                                      image.image[:, :, 0, :]
#
#            if stokes.strip().upper() == 'PIL':
#                if dum_image.nwav == 1:
#                    dum_image.image = np.sqrt(
#                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2)
#                else:
#                    dum_image.image = np.sqrt(
#                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2)
#
#            if stokes.strip().upper() == 'PL':
#                if dum_image.nwav == 1:
#                    dum_image.image = np.sqrt(
#                        image.image[:, :, 1]**2 + image.image[:, :, 2]**2) / image.image[:, :, 0]
#
#                else:
#                    dum_image.image = np.sqrt(
#                        image.image[:, :, 1, :]**2 + image.image[:, :, 2, :]**2) / image.image[:, :, 0, :]
#
        if cmask_rad is not None:
            dum_image = cmask(dum_image, rad=cmask_rad, au=au, arcsec=arcsec, dpc=dpc)
        else:
            dum_image = dum_image

        if ifreq is None:
            ifreq = 0
#        data = np.squeeze(dum_image.image[:, ::-1, ifreq].T) #flips y for imshow
        data = np.squeeze(dum_image.image[:, :, ifreq].T) #doesn't flip y for pcolormesh

        if bunit == 'norm':
            norm = data.max()
            data = data / norm

        clipnorm = data.max()
       
        # Check if the data should be plotted on a log scale
        if log:
            taupos = data >= 0
            tauneg = data < 0

            clipmin = - np.log10(abs(data.min()))
            data[taupos] = np.log10(data[taupos].clip(1e-90))
            data[tauneg] = -np.log10(abs(data[tauneg]))

            # Clipping the data
            if maxlog is not None:
                clipmin = -maxlog + np.log10(clipnorm)
        else:
            clipmin = data.min()

        if saturate is not None:
            if 'percent' in saturate:
                saturatenum = float(saturate.replace('percent', ''))
                if saturatenum > 100.:
                    saturatenum = 100.
                if log:
                    clipmax = np.log10(saturatenum/100.) + np.log10(clipnorm)
                else:
                    clipmax = clipnorm * saturatenum/100.
            else:
                saturatenum = float(saturate)
                if log:
                    clipmax = np.log10(saturatenum)
                else:
                    clipmax = saturatenum
        else:
            clipmax = clipnorm

        data = data.clip(clipmin, clipmax)

        # mask the data
        data_original = data
        data = np.ma.masked_where(data == clipmin)

        # Select the unit of the data

        if bunit.lower() == 'norm':
            if log:
                cb_label = 'log(Tausurf/max(Tausurf))'
            else:
                cb_label = 'Tausurf / max(Tausurf)'
        elif bunit.lower() == 'length':
            if au:
                data = data / nc.au
                cb_label_unit = '[AU]'
            elif arcsec:
                data = data / nc.au / dpc
                cb_label_unit = '[arcsec]'
            else:
                cb_label_unit = '[cm]'

            if log:
                data = np.log10(data)
                cb_label = 'log(Distance) '+cb_label_unit
            else:
                cb_label = 'Distance '+cb_label_unit
        else:
            msg = 'Unknown bunit: ' + bunit + ' Allowed values are "norm", "length"'
            raise ValueError(msg)

        # Select the coordinates of the data
        if au:
            x = image.x / nc.au
            y = image.y / nc.au
            xlab = 'X [AU]'
            ylab = 'Y [AU]'
        elif arcsec:
            x = image.x / nc.au / dpc
            y = image.y / nc.au / dpc
            xlab = 'RA offset ["]'
            ylab = 'DEC offset ["]'
        else:
            x = image.x
            y = image.y
            xlab = 'X [cm]'
            ylab = 'Y [cm]'

        ext = (x[0], x[image.nx - 1], y[0], y[image.ny - 1])
        # Now finally put everything together and plot the data
#        plt.delaxes()
#        plt.delaxes()

# this doesn't seem to work...
#        implot = plt.imshow(data, extent=ext, cmap=cmap, interpolation=interpolation, **kwargs)
        ax = plt.gca()
        p = ax.pcolormesh(x, y, data, cmap=cmap, **kwargs)
        ax.set_aspect(abs((ext[1]-ext[0])/(ext[3]-ext[2])) / 1.)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(r'Tau surface at $\lambda$=' + ("%.5f" % image.wav[ifreq]) + r'$\mu$m')
#        cbar = plt.colorbar(implot)
        cbar = plt.colorbar(p)
        cbar.set_label(cb_label)
#        plt.show()

    elif isinstance(image, radmc3dCircimage):
        raise ValueError('this is not implemented. need to copy and paste stuff in plotImage')

def plotSpectrum(image=None, ax=None, pltx='wav', plty='fnu', pltyunit='cgs', 
        **kwargs):
    """ plots the spectrum 
    Parameters
    ----------
    image : radmc3dImage object
    pltx : string
            'wav': wavelength [micron]
            'freq': frequency [GHz]
            'ev': electronvolt
    plty : string
            'fnu': plot spectrum as flux at a certain distance
            'nufnu': plot flux x nu
            'lnu':  plot spectrum as luminosity
    pltyunit : string
            for fluxes (fnu, nufnu)
                'cgs': usual cgs [default]
                'jy' : jansky when plotting in flux
            for luminosities (lnu, nulnu)
                'cgs': in [default]
                'lsun' : solar luminosity for luminosity

    ax : matplotlib axes object (optional)
    """
    if ax is None:
        ax = plt.gca()

    if pltx is 'wav':
        xaxis = image.wav
        xlabel = r'log(Wavelength [$\mu m]$)'
    elif pltx is 'freq':
        xaxis = image.freq / 1e9
        xlabel = r'log(Frequency [GHz])'
    elif pltx is 'ev':
        xaxis = 4.13566553853599E-15 * image.freq
        xlabel = 'log(Frequency [eV])'
    else:
        raise ValueError('option for pltx is unapplicable')

    spec = np.squeeze(image.image)

    if plty is 'fnu':
        ydata = spec / image.dpc**2
        if pltyunit is 'cgs':
            ylabel = r'$log(F_{\nu} [erg cm^{-2} s^{-1} Hz^{-1}])$'
        elif pltyunit is 'jy':
            ydata = ydata / nc.jy
            ylabel = r'$log(F_{\nu} [jy])$'
        else:
            raise ValueError('pltyunit not applicable')
    elif plty is 'nufnu':
        ydata = spec * image.freq
        if pltyunit is 'cgs':
            ylabel = r'$log(\nu F_{\nu} [Hz x erg cm^{-2} s^{-1} Hz^{-1}])$'
        elif pltyunit is 'jy':
            ydata = ydata / nc.jy
            ylabel = r'$log(\nu F_{\nu} [jy Hz])$'
        else:
            raise ValueError('pltyunit not applicable')
    elif plty is 'lnu':
        ydata = spec * image.dpc**2 * 4. * np.pi * nc.pc**2
        if pltyunit is 'cgs':
            ylabel = r'$log(L_{\nu} [erg cm^{-2} s^{-1} Hz^{-1}])$'
        elif pltyunit is 'lsun':
            ydata = ydata / nc.ls
            ylabel = r'$log(L_{\nu} [Lsun Hz^{-1}])$'
        else:
            raise ValueError('pltyunit not applicable')
    elif plty is 'nulnu':
        ydata = spec * image.dpc**2 * 4. * np.pi * nc.pc**2 * image.freq
        if pltyunit is 'cgs':
            ylabel = r'$log(\nu L_{\nu} [Hz x erg cm^{-2} s^{-1} Hz^{-1}])$'
        elif pltyunit is 'lsun':
            ydata = ydata / nc.ls
            ylabel = r'$log(\nu L_{\nu} [Lsun])$'
        else: 
            raise ValueError('pltyunit not applicable')

    # plotting
    ax.plot(xaxis, ydata, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return {'specplot': ax}

    
def makeImage(circ=False, npix=None, incl=None, wav=None, sizeau=None, phi=None, posang=None, pointau=None,
              fluxcons=True, nostar=False, noscat=False, secondorder=1,
              widthkms=None, linenlam=None, vkms=None, imolspec=None, iline=None,
              lambdarange=None, nlam=None, loadlambda=False, stokes=False, doppcatch=False,
              maxnrscat=None, nphot_scat=None, 
              binary=False, fname='', 
              tausurf=-1, tracetau=False, tracecolumn=False):
    """Calculates a rectangular image with RADMC-3D 

    Parameters
    ----------

    npix        : int
                  Number of pixels on the rectangular images

    sizeau      : float
                  Diameter of the image in au

    incl        : float
                  Inclination angle of the source

    wav         : float
                  Wavelength of the image in micron

    phi         : float, optional
                  Azimuthal rotation angle of the source in the model space

    posang      : float, optional
                  Position angle in degrees of the source in the image plane

    pointau     : Float, optional
                  Three elements list of the cartesian coordinates of the image center

    widthkms    : float, optional
                  Width of the frequency axis of the channel maps

    linenlam    : int, optional
                  Number of wavelengths to calculate images at. (for channel maps)

    vkms        : float, optional
                  A single velocity value at which a channel map should be calculated. (channel map)

    imolspec    : int, optional
                  which molecule to use if there are multiple gas species. (channel maps).
                  Default is 1 (begins with 1) 

    iline       : int, optional
                  Line transition index. iline=1 means j=1-0, iline=2 means j=2-1 

    lambdarange : list, optional
                  Two element list with the wavelenght boundaries between which
                  multiwavelength images should be calculated. (continuum)

    nlam        : int, optional
                  Number of wavelengths to be calculated in lambdarange (continuum)

    loadlambda  : bool, optional
                  If set to True, radmc3d will search for camera_wavelength_micron.inp to create images

    fluxcons    : bool, optional
                  This should not even be a keyword argument, it ensures flux conservation 
                  (adaptive subpixeling) in the rectangular images

    nostar      : bool, optional
                  If True the calculated images will not contain stellar emission

    noscat      : bool, optional
                  If True, scattered emission will be neglected in the source function, however, 
                   extinction will contain scattering if kappa_scat is not zero.  

    secondorder : bool, optional
                  If True, integration for rendering images will be done in second order. usually 
                   recommended to use this.

    stokes      : bool, optional
                  If True, images in all four stokes parameters (IQUV) will be calculated, if
                  False only the intensity will be calculated

    doppcatch   : bool, optional
                  If True, images will use doppler catching option in radmc3d image makeing

    binary      : bool, optional
                  If True the output image will be written in a C-style binary format, if False
                  the image format will be ASCII

    fname       : '', optional
                  The name for the output file. Default is to leave 'image.out' as is

    tausurf     : float, optional. Default=-1
                  Set this to greater than 0 for imaging the tausurface. For clarity, the default name
                  of this file will be renamed from "image.out" to "tausurface.out" or will use fname. 
                  Note there will also be a tausurface_3d.out

    tracetau    : bool, optional. Default=False
                  Set this to calculate optical depth?

    tracecolumn : bool, optional. Default=False
                  Set this to calculate column density in g/cm^2

    Example
    -------

    makeImage(npix=100, incl=60.0, wav=10.0, sizeau=300., phi=0., posang=15., 
        pointau=[0., 0.,0.], fluxcons=True, nostar=False, noscat=False)

    """
    #
    # The basic keywords that should be set
    #
    if npix is None:
        msg = 'Unkonwn npix. Number of pixels must be set.'
        raise ValueError(msg)

    if incl is None:
        msg = 'Unkonwn incl. Inclination angle must be set.'
        raise ValueError(msg)

    if wav is None:
        if (lambdarange is None) & (nlam is None) & (loadlambda is False):
            if vkms is None:
                if (widthkms is None) & (linenlam is None):
                    msg = 'Neither wavelength nor velocity is specified at which the image should be calculated'
                    raise ValueError(msg)
                else:
                    if iline is None:
                        msg = 'Unknown iline. widthkms, linenlam keywords are set indicating that a line '\
                              + 'channel map should be calculated, but the iline keyword is not specified'
                        raise ValueError(msg)
            else:
                if iline is None:
                    msg = 'Unknown iline. vkms keyword is set indicating that a line channel map should be'\
                          + 'calculated, but the iline keyword is not specified'
                    raise ValueError(msg)
    else:
        if lambdarange is not None:
            msg = 'Either lambdarange or wav should be set but not both'
            raise ValueError(msg)

    if lambdarange is not None:
        if len(lambdarange) != 2:
            msg = 'lambdarange must have two and only two elements'
            raise ValueError(msg)

    #
    # Kees' fix for the case when a locally compiled radmc3d exists in the current directory
    #
    com = ''
    if os.path.isfile('radmc3d'):
        com = com + './'

#    com = com + 'radmc3d image'
    com = com + 'radmc3d'
    if tausurf > 1e-3:
        com = com + ' tausurf ' + ("%.3f" % tausurf)
    else:
        com = com + ' image'

        if circ:
            com += ' circ'

        if tracetau:
            com = com + ' tracetau'
        elif tracecolumn:
            com = com + ' tracecolumn'

    com = com + ' npix ' + str(int(npix))
    com = com + ' incl ' + str(incl)

    if sizeau is not None:
        com = com + ' sizeau ' + str(sizeau)

    if wav is not None:
        com = com + ' lambda ' + str(wav)
    elif (lambdarange is not None) & (nlam is not None):
        com = com + ' lambdarange ' + str(lambdarange[0]) + ' ' + str(lambdarange[1]) + ' nlam ' + str(int(nlam))
    elif vkms is not None:
        com = com + ' vkms ' + str(vkms)
    elif (widthkms is not None) & (linenlam is not None):
        com = com + ' widthkms ' + str(widthkms) + ' linenlam ' + str(linenlam)
    elif loadlambda is not False:
        com = com + ' loadlambda'

    #
    # Now add additional optional keywords/arguments
    #
    if phi is not None:
        com = com + ' phi ' + str(phi)

    if posang is not None:
        com = com + ' posang ' + str(posang)

    if pointau is not None:
        if len(pointau) != 3:
            msg = ' pointau should be a list of 3 elements corresponding to the  cartesian coordinates of the ' \
                  + 'image center'
            raise ValueError(msg)
        else:
            com = com + ' pointau ' + str(pointau[0]) + ' ' + str(pointau[1]) + ' ' + str(pointau[2])
    else:
        com = com + ' pointau 0.0  0.0  0.0'

    if fluxcons:
        com = com + ' fluxcons'

    if iline:
        if imolspec is None:
            com = com + ' imolspec 1'
        else:
            com = com + ' imolspec '+ ("%d" % imolspec)

        com = com + ' iline ' + ("%d" % iline)

    if stokes:
        com = com + ' stokes'

    if doppcatch:
        com = com + ' doppcatch'

    if binary:
        com = com + ' imageunform'

    if nostar:
        com = com + ' nostar'

    if noscat:
        com = com + ' noscat'

    if secondorder:
        com = com + ' secondorder'

    if maxnrscat is not None:
        com = com + ' maxnrscat %d'%maxnrscat

    if nphot_scat is not None:
        com = com + ' nphot_scat %d'%nphot_scat

    print('executing command: '+com)

    #
    # Now finally run radmc3d and calculate the image
    #
    # dum = sp.Popen([com], stdout=sp.PIPE, shell=True).wait()
    dum = sp.Popen([com], shell=True).wait()

    possible_imname = ['image.out', 'circimage.out']

    detect_image = False
    for iname in possible_imname:
        detect_image = detect_image | os.path.isfile(iname)

    if detect_image is False:
        msg = 'Did not succeed in making image. \n'
        msg = msg + 'Failed command: '+com
        raise ValueError(msg)

    if fname != '':
        for iname in possible_imname:
            if os.path.isfile(iname):
                os.system('mv %s %s'%(iname, fname))

    if tausurf > 1e-3 and fname == '':
        fname = 'tausurface.out'

    if tracetau and fname == '':
        fname = 'optdepth.out'

    if tracecolumn and fname == '':
        fname = 'columndens.out'

    print('Ran command: '+com)
    print('Resulting file: '+fname)

    return 0

def cmask(im=None, rad=0.0, au=False, arcsec=False, dpc=None):
    """Simulates a coronographic mask.
        Sets the image values to zero within circle of a given radius around the
        image center.

    Parameters
    ----------
    im     : radmc3dImage
            A radmc3dImage class containing the image

    rad    : float
            The raadius of the mask. positive is to mask the inside. negative will
            mask outsid of the radius

    au     : bool
            If true the radius is taken to have a unit of AU

    arcsec : bool
            If true the radius is taken to have a unit of arcsec (dpc
            should also be set)

    dpc    : float
            Distance of the source (required if arcsec = True)

    NOTE if arcsec=False and au=False rad is taken to have a unit of pixel

    Returns
    -------

    Returns a radmc3dImage class containing the masked image
    """

    if au:
        if arcsec:
            msg = ' Either au or arcsec should be set, but not both of them'
            raise ValueError(msg)

        crad = rad * nc.au
    else:
        if arcsec:
            crad = rad * nc.au * dpc
        else:
            crad = rad * im.sizepix_x

    res = copy.deepcopy(im)
    if im.nfreq != 1:
        for ix in range(im.nx):
            r = np.sqrt(im.y**2 + im.x[ix]**2)
            if crad > 0:
                ii = r <= crad
            else:
                ii = r >= abs(crad)
            res.image[ix, ii, :] = 0.0
    else:
        for ix in range(im.nx):
            r = np.sqrt(im.y**2 + im.x[ix]**2)
            if crad > 0:
                ii = r <= crad
            else:
                ii = r >= abs(crad)
            res.image[ix, ii] = 0.0

    return res

def writeMovieInp():
    return True

def makeMovie():
    return 0

def plotMovie():
    return True

def makeSpectrum(npix=None, incl=None, wav=None, sizeau=None, 
              phi=None, posang=None, pointau=None,
              fluxcons=True, nostar=False, noscat=False, secondorder=True,
              lambdarange=None, nlam=None, loadlambda=False,
              binary=False, fname=''):
    """ creates spectrum based on radmc3d
    Parameters
    ----------
    
    """
    #
    # The basic keywords that should be set
    #
    if npix is None:
        msg = 'Unkonwn npix. Number of pixels must be set.'
        raise ValueError(msg)

    if incl is None:
        msg = 'Unkonwn incl. Inclination angle must be set.'
        raise ValueError(msg)

    if wav is None:
        if (lambdarange is None) & (nlam is None) & (loadlambda is False):
            msg = 'should specify wavelength, either by wav, lambdarange, nlam, or loadlambda'
            raise ValueError(msg)
    else:
        if lambdarange is not None:
            msg = 'Either lambdarange or wav should be set but not both'
            raise ValueError(msg)

    if lambdarange is not None:
        if len(lambdarange) != 2:
            msg = 'lambdarange must have two and only two elements'
            raise ValueError(msg)

    if sizeau is None:
        msg = 'Unknown sizeau.'
        raise ValueError(msg)

    com = ''
    com = com + 'radmc3d spectrum'

    com = com + ' npix ' + str(int(npix))
    com = com + ' incl ' + str(incl)
    com = com + ' sizeau ' + str(sizeau)

    if wav is not None:
        com = com + ' lambda ' + str(wav)
    elif (lambdarange is not None) & (nlam is not None):
        com = com + ' lambdarange ' + str(lambdarange[0]) + ' ' + str(lambdarange[1]) + ' nlam ' + str(int(nlam))
    elif loadlambda is not False:
        com = com + ' loadlambda'

    #
    # Now add additional optional keywords/arguments
    #
    if phi is not None:
        com = com + ' phi ' + str(phi)

    if posang is not None:
        com = com + ' posang ' + str(posang)

    if pointau is not None:
        if len(pointau) != 3:
            msg = ' pointau should be a list of 3 elements corresponding to the  cartesian coordinates of the ' \
                  + 'image center'
            raise ValueError(msg)
        else:
            com = com + ' pointau ' + str(pointau[0]) + ' ' + str(pointau[1]) + ' ' + str(pointau[2])
    else:
        com = com + ' pointau 0.0  0.0  0.0'

    if fluxcons:
        com = com + ' fluxcons'

    if binary:
        com = com + ' imageunform'

    if nostar:
        com = com + ' nostar'

    if noscat:
        com = com + ' noscat'

    if secondorder:
        com = com + ' secondorder'

    print('executing command: '+com)

    #
    # Now finally run radmc3d and calculate the image
    #
    # dum = sp.Popen([com], stdout=sp.PIPE, shell=True).wait()
    dum = sp.Popen([com], shell=True).wait()

    if os.path.isfile('spectrum.out') is False:
        msg = 'Did not succeed in making spectrum. \n'
        msg = msg + 'Failed command: '+com
        raise ValueError(msg)

    if fname != '':
        os.system('mv spectrum.out '+fname)

    print('Ran command: '+com)
    print('Resulting file: '+fname)

    return 0

def writeCameraWavelength(camwav=None, fdir=None, fname='camera_wavelength_micron.inp'):
    """ writes camera_wavelength_micron.inp for this list of wavelengths
    Parameters
    ----------
    camwav = list
             the wavelengths in micron you want to write
    Returns
    -------
    bool
    """
    if camwav is None:
        raise ValueError('must provide wavelengths to create camera_wavelength_micron.inp')

    ncamwav = len(camwav)

    if fdir is not None:
        if fdir[-1] is '/':
            fname = fdir + fname
        else:
            fname = fdir + '/' + fname

    with open(fname, 'w') as wfile:
        wfile.write("%d\n" % (ncamwav))
        for ii in range(ncamwav):
            wfile.write("%.7e\n" % camwav[ii])
    print('wrote file '+fname)

def readCameraWavelength(fname=''):
    """ reads camera_wavelength_micron.inp
    Parameters
    ----------
    fname = string
            the filename to read. default is 'camera_wavelength_micron.inp'
    Returns
    -------
    camwav = list
    """
    if fname =='':
        fname = 'camera_wavelength_micron.inp'
    data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
    ncamwav = np.int(data[0])
    camwav = data[1:]
    return camwav

def readTauSurf3D(fname='tausurface_3d.out'):
    """ reads tau surface in x,y,z coordinates 
    Parameters
    ----------
    fname = string
            the filename to read. default is 'tausurface_3d.out'
    Returns
    -------
    {nx:nx, ny:ny, nwav:nwav, wav:wav, x:x, y:y, z:z}
    """

    with open(fname, 'r') as rfile:

        dum = ''
        # Format number
        iformat = int(rfile.readline())

        # Nr of pixels
        dum = rfile.readline()
        dum = dum.split()
        nx = int(dum[0])
        ny = int(dum[1])

        # Nr of frequencies
        nwav = int(rfile.readline())

        # Wavelength of the image
        wav = np.zeros(nwav, dtype=np.float64)
        for iwav in range(nwav):
            wav[iwav] = float(rfile.readline())

        # blank space
        dum = rfile.readline()

        x = np.zeros([nx, ny, nwav], dtype=np.float64)
        y = np.zeros([nx, ny, nwav], dtype=np.float64)
        z = np.zeros([nx, ny, nwav], dtype=np.float64)
        for iwav in range(nwav):
            for iy in range(ny):
                for ix in range(nx):
                    dum = rfile.readline().split()
                    x[ix, iy, iwav] = float(dum[0])
                    y[ix, iy, iwav] = float(dum[1])
                    z[ix, iy, iwav] = float(dum[2])

    return {'nx':nx, 'ny':ny, 'nwav':nwav, 'wav':wav, 
            'x':x, 'y':y,'z':z}

def getTb(im, wav=None, freq=None, bunit='inu', rj=False, 
        dpc=None, dxy=None, 
        fwhm=None):
    """
    calculate brightness temperature based on an image

    Paramters
    ---------

        im : ndarray, in units of bunit
        bunit : unit of given ndarray
            'inu' : intensity
            'jy/beam': needs fwhm
            'jy/pix': needs dpc, dxy
        wav : float
            wavelength in micron
        freq : float
            frequency in Hz. use either wav or freq
        dpc : float
            distance in pc
        dxy : list, [dx, dy]
            pixel size in cm. Assume uniform step size in x,y
        fwhm : list
            beam fwhm in arcsec
    """
    if wav is None and freq is None:
        raise ValueError('wavelength or frequency must be given')
    if wav is not None and freq is not None:
        raise ValueError('only one of wavelength or frequency can be given')

    if freq is None:
        freq = nc.cc * 1e4 / wav
    if wav is None:
        wav = nc.cc  * 1e4 / freq

    # determine from bunit
    if bunit.lower() == 'inu':
        inten = im

    elif bunit.lower() == 'jy/pixel':
        beam_area = dxy[0] * dxy[1] / (dpc * nc.pc)**2
        inten = im / beam_area / 1e23

    elif bunit.lower() == 'jy/beam':
        beam_area = (fwhm[0]/3600.*np.pi/180) * (fwhm[1]/3600.*np.pi/180) * np.pi / 4. / np.log(2.0)
        inten = im / beam_area / 1e23

    else:
        raise ValueError('bunit does not exist: %s'%bunit)

    # convert from intensity to temperature
    ld2 = wav**2
    hmu = nc.hh * freq
    hmu3_c2 = nc.hh * freq**3 / nc.cc**2
    if rj:
        tb = wav / 2. / nc.kk * inten
    else:
        tb = 0 * inten
        reg = (inten > 0) & ~np.isnan(inten)
        tb[reg] = hmu / nc.kk / np.log(2. * hmu3_c2 / (inten[reg] + 1e-90) + 1.)

    return tb

# ----------------------------------------------------------------------
# ------------------ Circular image ------------------------------------
# ----------------------------------------------------------------------
class radmc3dCircimage(baseImage):
    """
    RADMC-3D circular image class

    Attributes
    ----------

    image       : ndarray
                  The image as calculated by radmc3d (the values are intensities in erg/s/cm^2/Hz/ster)

    r          : ndarray
                  Radial cell center coordinate of the image [cm]

    ri          : ndarray
                  Radial cell interface coordinate of the image [cm]

    phi        : ndarray
                  Azimuthal cell center coordinate of the image [rad]

    phii        : ndarray
                  Azimuthal cell interface coordinate of the image [rad]

    nr          : int
                  Number of pixels in the radial direction

    nphi        : int   
                  Number of pixels in the azimuthal direction

    nfreq       : int
                  Number of frequencies in the image cube

    freq        : ndarray
                  Frequency grid in the image cube

    nwav        : int
                  Number of wavelengths in the image cube (same as nfreq)

    wav         : ndarray
                  Wavelength grid in the image cube

    filename    : str
                  Name of the file the image data was read from

    stokes      : bool
                  If True the image data contain the full stokes vector (I,Q,U,V)

    """

    def __init__(self):
        baseImage.__init__(self)

        self.ri = np.zeros(0, dtype=np.float64)
        self.phii = np.zeros(0, dtype=np.float64)
        self.r = np.zeros(0, dtype=np.float64)
        self.phi = np.zeros(0, dtype=np.float64)
        self.freq = np.zeros(0, dtype=np.float64)
        self.wav = np.zeros(0, dtype=np.float64)
        self.image = np.zeros((0, 0), dtype=np.float64)
        self.filename = 'circimage.out'
        self.nphi = 0
        self.nwav = 0
        self.nfreq = 0
        self.stokes = False
        self.npol = 1

        self.nr = 0
        self.nphi = 0

    def getPixelSize(self):
        """
        Calculates the pixel size

        Returns
        -------
        The pixel size in cm^2
        """

        nx, ny = self.image.shape[:2]
        pixel_area = np.zeros([nx, ny], dtype=np.float64)

        x2 = np.pi * (self.ri[1:]**2 - self.ri[:-1]**2)
        dy = self.phii[1:] - self.phii[:-1]

        for ix in range(nx):
            pixel_area[ix, :] = x2[ix] * dy / (2.0 * np.pi)

        self.pixel_area = pixel_area

    def readImage(self, fname='circimage.out', old=False):
        """
        Reads a circular image

        Parameters
        ----------

        filename        : str
                          Name of the file to be read.

        old             : bool
                          If True the image format of the old 2D code (radmc) will be used. If False (default) the 
                          RADMC-3D format is used.
        """

        self.filename = fname

        if old:
            self.stokes = False
            self.npol = 1

            with open(self.fname, 'r') as f:

                self.nfreq = int(f.readline())
                self.nwav = self.nfreq

                s = f.readline()
                self.freq = np.zeros(self.nfreq, dtype=np.float64)
                self.wav = np.zeros(self.nfreq, dtype=np.float64)
                for inu in range(self.nfreq):
                    s = f.readline()
                    self.freq[inu] = float(s)
                    self.wav[inu] = nc.cc / self.freq[inu]

                s = f.readline()
                s = f.readline().split()
                self.nr = int(s[0])
                self.nphi = int(s[1])
                self.nfreq = int(s[2])

                s = f.readline()
                self.r = np.zeros(self.nr, dtype=np.float64)
                for ir in range(self.nr):
                    s = f.readline()
                    self.r[ir] = float(s)

                s = f.readline()
                self.ri = np.zeros(self.nr + 1, dtype=np.float64)
                for ir in range(self.nr + 1):
                    s = f.readline()
                    self.ri[ir] = float(s)

                s = f.readline()
                self.phi = np.zeros(self.nphi, dtype=np.float64)
                for ip in range(self.nphi):
                    s = f.readline()
                    self.phi[ip] = float(s)

                s = f.readline()
                self.phii = np.zeros(self.nphi + 1, dtype=np.float64)
                for ip in range(self.nphi + 1):
                    s = f.readline()
                    self.phii[ip] = float(s)

                s = f.readline()
                self.image = np.zeros((self.nr, self.nphi, self.npol, self.nfreq), dtype=np.float64)

                for inu in range(self.nfreq):
                    for ir in range(self.nr):
                        s = f.readline().split()
                        for iphi in range(self.nphi):
                            self.image[ir, :, 0, inu] = float(s[iphi])

        else:

            with open(self.filename, 'r') as f:

                iformat = int(f.readline())
                if iformat == 1:
                    self.stokes = False
                    self.npol = 1
                elif iformat == 3:
                    self.stokes = True
                    self.npol = 4

                s = f.readline().split()
                self.nr = int(s[0])
                self.nphi = int(s[1])
                self.nfreq = int(f.readline())
                self.nwav = self.nfreq

                s = f.readline()
                self.ri = np.zeros(self.nr + 2, dtype=np.float64)
                for ir in range(self.nr + 2):
                    s = f.readline()
                    self.ri[ir] = float(s)

                s = f.readline()
                self.r = np.zeros(self.nr + 1, dtype=np.float64)
                for ir in range(self.nr + 1):
                    s = f.readline()
                    self.r[ir] = float(s)

                s = f.readline()
                self.phii = np.zeros(self.nphi + 1, dtype=np.float64)
                for ip in range(self.nphi + 1):
                    s = f.readline()
                    self.phii[ip] = float(s)

                s = f.readline()
                self.phi = np.zeros(self.nphi, dtype=np.float64)
                for ip in range(self.nphi):
                    s = f.readline()
                    self.phi[ip] = float(s)

                s = f.readline()
                self.freq = np.zeros(self.nfreq, dtype=np.float64)
                self.wav = np.zeros(self.nfreq, dtype=np.float64)
                for inu in range(self.nfreq):
                    s = f.readline()
                    self.wav[inu] = float(s)
                    self.freq[inu] = nc.cc * 1e4 / self.wav[inu]

                s = f.readline()
                self.image = np.zeros((self.nr + 1, self.nphi, self.npol, self.nfreq), dtype=np.float64)

                for inu in range(self.nfreq):
                    for iphi in range(self.nphi):
                        for ir in range(self.nr + 1):
                            s = f.readline().split()
                            self.image[ir, iphi, 0, inu] = float(s[0])
                            if self.npol > 1:
                                self.image[ir, iphi, 1, inu] = float(s[1])
                                self.image[ir, iphi, 2, inu] = float(s[2])
                                self.image[ir, iphi, 3, inu] = float(s[3])

                    dum = f.readline().split() # empty line

    def getJyppix(self, dpc):
        """
        calculate the jansky per pixel. 
        """
        # calculate the pixel area
        if hasattr(self, 'pixel_area') is False:
            self.getPixelArea()

        # Conversion from erg/s/cm/cm/Hz/ster to Jy/pixel
        self.imageJyppix = self.image * self.pixel_area[:,:,None,None] / (dpc * nc.pc)**2 * 1e23
        self.dpc = dpc

def readcircImage(fname='circimage.out', old=False):
    """
    A convenience function to read circular images

    Parameters
    ----------

    filename        : str
                      Name of the file to be read.

    old             : bool
                      If True the image format of the old 2D code (radmc) will be used. If False (default) the 
                      RADMC-3D format is used.
    """
    dum = radmc3dCircimage()
    dum.readImage(fname=fname, old=old)
    return dum


