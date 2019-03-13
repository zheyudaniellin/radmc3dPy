"""This module contains classes for handling dust opacities

some bugs I found:

- ~ line 603: array for gsize is defined incorrectly
- last outputs of computeDustOpacMie(), opac.wav had incorrect cm-micron conversion

updates I did:
- set up a ppar['mixgsize'] option to average the dust sizes
- added ksca_from_z11 attribute. changed: computeDustOpacMie, readOpac
- changed mixOpac to include mixing scattering matrix
- added readDustInfo and writeDustInfo to read and write dustinfo.zyl
- added alignment_mode option in writeMasterOpac to consider dust alignment
- added kscat0 to set all kscat, scattering matrix to 0

"""
from __future__ import absolute_import
from __future__ import print_function
import traceback
import subprocess as sp
import os

try:
    import numpy as np
except ImportError:
    np = None
    print(' Numpy cannot be imported ')
    print(' To use the python module of RADMC-3D you need to install Numpy')
    print(traceback.format_exc())


from . import natconst as nc
from . import miescat
from . reggrid import *
import warnings
from scipy.interpolate import interp1d

import fneq
import pdb

class radmc3dDustOpac(object):
    """
    Class to handle dust opacities.


    Attributes
    ----------

    wav     : list
                Each element of the list contains an ndarray with the wavelength grid

    freq    : list
                Each element of the list contains an ndarray with the frequency grid

    nwav    : list
                Each element of the list contains an integer with the number of wavelengths

    kabs    : list
                Each element of the list contains an ndarray with the absorption coefficient per unit mass

    ksca    : list
                Each element of the list contains an ndarray with the scattering coefficient per unit mass

    phase_g : list
                Each element of the list contains an ndarray with the hase function

    ext     : list
                Each element of the list contains a string wht the file name extension of the duskappa_ext.Kappa file

    therm   : list
                Each element of the list contains a bool, if it is set to False the dust grains are quantum-heated
                (default: True)

    idust   : lisintt
                Each element of the list contains an integer with the index of the dust species in the dust density
                distribution array

    scatmat : list
                Each element is a boolean indicating whether the dust opacity table includes (True) the full scattering
                matrix or not (False)

    nang    : list
                Each element is a string, containing the number of scattering angles in the scattering matrix if its
                given

    scatang : list
                Each element is a numpy ndarray containing the scattering angles in the scattering matrix if its given

    z11     : list
                Each element is a numpy ndarray containing the (1,1) element of the scattering angles in the scattering
                matrix if its given

    z12     : list
                Each element is a numpy ndarray containing the (1,2) element of the scattering angles in the scattering
                matrix if its given

    z22     : list
                Each element is a numpy ndarray containing the (2,2) element of the scattering angles in the scattering
                matrix if its given

    z33     : list
                Each element is a numpy ndarray containing the (3,3) element of the scattering angles in the scattering
                matrix if its given

    z34     : list
                Each element is a numpy ndarray containing the (3,4) element of the scattering angles in the scattering
                matrix if its given

    z44     : list
                Each element is a numpy ndarray containing the (4,4) element of the scattering angles in the scattering
                matrix if its given
    ksca_from_z11 : list
                Each element is a numpy ndarray containing the ksca calculated from z11. This should be close to ksca.
                Calculated like computeDustOpacMie

    kpara   : list
              Each element is a numpy ndarray containing the para (vertical) factor for opacity of oblate grains

    korth   : list
              Each element is a numpy ndarray containing the orth (horizontal) factor for opacity of oblate grains

    alignang : list
               Each element is a numpy ndarray containing the angle (degrees) grid of dustkapalignfact_*.inp. can be different from the angle grid in kapscatmat_*.inp. Note: the wavelength grid must be the same as self.wav

    """

    # --------------------------------------------------------------------------------------------------
    def __init__(self):

        self.wav = []
        self.freq = []
        self.nwav = []
        self.nfreq = []
        self.kabs = []
        self.ksca = []
        self.phase_g = []
        self.ext = []
        self.idust = []
        self.therm = []
        self.scatmat = []
        self.z11 = []
        self.z12 = []
        self.z22 = []
        self.z33 = []
        self.z34 = []
        self.z44 = []
        self.scatang = []
        self.nang = []
        self.ksca_from_z11 = []
        self.kpara = [] #dust alignment factor: vertical
        self.korth = [] #horizontal
        self.alignang = [] 

    def writeOpac(self, fname=None, ext=None, idust=None, scatmat=False, fdir=None):
        """
        Writes dust opacities to file

        Parameters
        ----------
        fname       : str
                      Name of the file to write the dust opacties into

        ext         : str
                      If fname is not specified, the output file name will be generated as dustkappa_EXT.inp or
                      dustkapscatmat_EXT.inp depending on the file format

        idust       : int
                      Dust species index whose opacities should be written to file

        scatmat     : bool
                      If True the full scattering matrix will be written to file on top of the opacities (i.e.
                      the file name should be dustkapscatmat_EXT.inp). If False only the dust opacities and the
                      asymmetry parameter (if present) will be written to file (dustkappa_EXT.inp type files)

        fdir        : string, optional
                      if set, will write the file into fdir directory, only if fname is None

        """

        if fname is None:
            if ext is None:
                msg = 'Neither fname nor ext is specified. Filename cannot be generated '
                raise ValueError(msg)
            else:
                if idust is None:
                    msg = 'idust is not specified. If output file name should be generated both ext and idust should ' \
                          'be set'
                    raise ValueError(msg)
                else:
                    if scatmat == True:
                        fname = 'dustkapscatmat_' + ext + '.inp'
                    else:
                        fname = 'dustkappa_' + ext + '.inp'
            fnameout = fname
            if fdir is not None:
                if fdir[-1] is '/':
                    fnameout = fdir + fnameout
                else:
                    fnameout = fdir + '/' + fnameout
        else:
            fnameout = fname

        with open(fnameout, 'w') as wfile:
            if scatmat == True:
                wfile.write('1\n')  # Format number
                wfile.write('%d\n' % self.nwav[idust])
                wfile.write('%d\n' % self.nang[idust])
                wfile.write('\n')
                for i in range(self.nwav[idust]):
                    wfile.write('%13.6e %13.6e %13.6e %13.6e\n' % (self.wav[idust][i],
                                                                   self.kabs[idust][i],
                                                                   self.ksca[idust][i],
                                                                   self.phase_g[idust][i]))
                wfile.write('\n')
                for j in range(self.nang[idust]):
                    wfile.write('%13.6e\n' % (self.scatang[idust][j]))
                wfile.write('\n')
                for i in range(self.nwav[idust]):
                    for j in range(self.nang[idust]):
                        wfile.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n' % (self.z11[idust][i, j],
                                                                                     self.z12[idust][i, j],
                                                                                     self.z22[idust][i, j],
                                                                                     self.z33[idust][i, j],
                                                                                     self.z34[idust][i, j],
                                                                                     self.z44[idust][i, j]))
                wfile.write('\n')
            else:
                if self.ksca[idust].mean() != -999.:
                    if self.phase_g[idust].mean() != -999.:
                        wfile.write('3\n')  # Format number
                    else:
                        wfile.write('2\n')  # Format number
                else:
                    wfile.write('1\n')  # Format number

                wfile.write('%d\n' % self.nwav[idust]) # Nr of wavelengths

                if self.ksca[idust].mean() != -999.:
                    if self.phase_g[idust].mean() != -999.:
                        for i in range(self.nwav[idust]):
                            wfile.write('%13.6e %13.6e %13.6e %13.6e\n' % (self.wav[idust][i],
                                                                           self.kabs[idust][i],
                                                                           self.ksca[idust][i],
                                                                           self.phase_g[idust][i]))
                    else:
                        for i in range(self.nwav[idust]):
                            wfile.write('%13.6e %13.6e %13.6e\n' % (self.wav[idust][i], self.kabs[idust][i],
                                                                           self.ksca[idust][i]))
                else:
                    for i in range(self.nwav[idust]):
                        wfile.write('%13.6e %13.6e \n' % (self.wav[idust][i], self.kabs[idust][i]))

                wfile.write('\n')


    def readOpac(self, ext=None, idust=None, scatmat=None, old=False, 
                 alignfact=False, fdir=None):
        """Reads the dust opacity files.

        Parameters
        ----------

        ext  : list
                File name extension (file names should look like 'dustkappa_ext.inp')

        idust: list
                Indices of the dust species in the master opacity file (dustopac.inp') - starts at 0

        scatmat: list
                If specified, its elements should be booleans indicating whether the opacity file
                contains also the full scattering matrix (True) or only dust opacities (False)

        old   : bool, optional
                If set to True the file format of the previous, 2D version of radmc will be used

        alignfact : bool, optional
                   If set to True, will read dustkapalignfact_ext.inp' if present
        fdir  : string
                the directory that contains the files
        """

        # Check the input keywords and if single strings are given convert them to lists
        # This assumes, though, that there is a single dust opacity file or dust species, though!!
        if ext is None:
            if idust is None:
                msg = 'Unknown ext and idust. File name extension must be given to be able to read the opacity ' \
                      'from file.'
                raise ValueError(msg)
            else:
                if isinstance(idust, int):
                    idust = [idust]
        else:
            if isinstance(ext, str):
                ext = [ext]

            if (len(ext) == 1) & (ext[0] != ''):
                if idust is not None:
                    msg = 'Either idust or ext should be specified, but not both'
                    raise ValueError(msg)

        if scatmat is None:
            # If the scatmat keyword is not given (i.e. if it is None) then assume that
            # it is False for all dust species
            scatmat = []
            if idust is None:
                for i in range(len(ext)):
                    scatmat.append(False)

            else:
                for i in range(len(idust)):
                    scatmat.append(False)
        else:
            if isinstance(scatmat, bool):
                scatmat = [scatmat]

        for i in range(len(scatmat)):
            self.scatmat.append(scatmat[i])

        # Find the file name extensions in the master opacity file if idust is specified instead of ext
        if idust:
            # Read the master dust opacity file to get the dust indices and dustkappa file name extensions
            mopac = self.readMasterOpac()

            ext = []
            for ispec in idust:
                if (ispec + 1) > len(mopac['ext']):
                    msg = 'No dust species found at index ' + ("%d" % ispec)
                    raise ValueError(msg)
                else:
                    ext.append(mopac['ext'][ispec])

        # If only the extension is specified look for the master opacity file and find the index of this dust species
        #  or set the index to -1 if no such dust species is present in the master opacity file
        else:
            # # Read the master dust opacity file to get the dust indices and dustkappa file name extensions
            idust = [i for i in range(len(ext))]

        # Now read all dust opacities
        for i in range(len(ext)):
            if scatmat[i]:
                fname = 'dustkapscatmat_' + ext[i] + '.inp'
                if fdir is not None:
                    if fdir[-1] is '/':
                        fname = fdir + fname
                    else:
                        fname = fdir + '/' + fname
                print('Reading ' + fname)

                # Check the file format
                iformat = np.fromfile(fname, count=1, sep=" ", dtype=np.int)
                iformat = iformat[0]
                if iformat != 1:
                    msg = 'Format number of the file dustkapscatmat_' + ext[i] + '.inp (iformat=' + ("%d" % iformat) + \
                          ') is unkown'
                    raise ValueError(msg)

                data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
                hdr = np.array(data[:3], dtype=np.int)
                data = data[3:]

                self.nwav.append(hdr[1])
                self.nfreq.append(hdr[1])
                self.nang.append(hdr[2])
                self.ext.append(ext[i])
                self.idust.append(idust[i])

                # Get the opacities
                data_opac = np.reshape(data[:hdr[1]*4], [hdr[1], 4])
                data = data[hdr[1]*4:]
                self.wav.append(data_opac[:, 0])
                self.freq.append(nc.cc / data_opac[:, 0] * 1e4)
                self.kabs.append(data_opac[:, 1])
                self.ksca.append(data_opac[:, 2])
                self.phase_g.append(data_opac[:, 3])

                # Get the angular grid
                anggrid = data[:hdr[2]]
                nang = hdr[2]
                self.scatang.append(data[:hdr[2]])
                data = data[hdr[2]:]

                # Now get the scattering matrix
                data = np.reshape(data, [hdr[1], hdr[2], 6])
                self.z11.append(data[:, :, 0])
                self.z12.append(data[:, :, 1])
                self.z22.append(data[:, :, 2])
                self.z33.append(data[:, :, 3])
                self.z34.append(data[:, :, 4])
                self.z44.append(data[:, :, 5])

                # calculate the ksca from scatmat
                if anggrid is not None:
                    z11dum = data[:,:,0]
                    mu = np.cos(anggrid * np.pi / 180.)
                    dmu = np.abs(mu[1:nang] - mu[0:nang-1])
                    kscat_from_z11 = np.zeros([hdr[1]])
                    for ii in range(hdr[1]):
                        zav = 0.5 * (z11dum[ii, 1:nang] + z11dum[ii,0:nang-1])
                        dum = 0.5 * zav * dmu
                        kscat_from_z11[ii] = dum.sum() * 4 * np.pi
                self.ksca_from_z11.append(kscat_from_z11)

            else:
                if not old:
                    fname = 'dustkappa_' + ext[i] + '.inp'
                    if fdir is not None:
                        if fdir[-1] is '/':
                            fname = fdir + fname
                        else:
                            fname = fdir + '/' + fname

                    print('Reading '+fname)

                    # Check the file format
                    iformat = np.fromfile(fname, count=1, sep=" ", dtype=np.int)
                    iformat = iformat[0]
                    if (iformat < 1) | (iformat > 3):
                        msg = 'Unknown file format in the dust opacity file ' + fname
                        raise ValueError(msg)

                    data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
                    hdr = np.array(data[:2], dtype=np.int)
                    data = data[2:]

                    self.ext.append(ext[i])
                    self.idust.append(idust[i])
                    self.nwav.append(hdr[1])
                    self.nfreq.append(hdr[1])

                    # If only the absorption coefficients are specified
                    if hdr[0] == 1:
                        data = np.reshape(data, [hdr[1], 2])
                        self.wav.append(data[:, 0])
                        self.freq.append(nc.cc / data[:, 0] * 1e4)
                        self.kabs.append(data[:, 1])
                        self.ksca.append([-999.])
                        self.phase_g.append([-999.])

                    # If the absorption and scattering coefficients are specified
                    elif hdr[0] == 2:
                        data = np.reshape(data, [hdr[1], 3])
                        self.wav.append(data[:, 0])
                        self.freq.append(nc.cc / data[:, 0] * 1e4)
                        self.kabs.append(data[:, 1])
                        self.ksca.append(data[:, 2])
                        self.phase_g.append([-999.])

                    # If the absorption and scattering coefficients and also the scattering phase
                    # function are specified
                    elif hdr[0] == 3:
                        data = np.reshape(data, [hdr[1], 4])
                        self.wav.append(data[:, 0])
                        self.freq.append(nc.cc / data[:, 0] * 1e4)
                        self.kabs.append(data[:, 1])
                        self.ksca.append(data[:, 2])
                        self.phase_g.append(data[:, 3])

                else:
                    fname = 'dustopac_' + ext[i] + '.inp'
                    if fdir is not None:
                        if fdir[-1] is '/':
                            fname = fdir + fname
                        else:
                            fname = fdir + '/' + fname

                    print('Reading '+fname)
                    freq = np.fromfile('frequency.inp', count=-1, sep=" ", dtype=np.float64)
                    nfreq = int(freq[0])
                    freq = freq[1:]
                    self.ext.append(ext[i])
                    self.idust.append(idust[i])

                    data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
                    hdr = np.array(data[:2], dtype=np.int)
                    data = data[2:]
                    if hdr[0] != nfreq:
                        msg = fname + ' contains a different number of frequencies than frequency.inp'
                        raise ValueError(msg)

                    wav = nc.cc / freq * 1e4
                    kabs = data[:nfreq]
                    ksca = data[nfreq:]

                    self.wav.append(wav[::-1])
                    self.freq.append(freq[::-1])
                    self.kabs.append(kabs[::-1])
                    self.ksca.append(ksca[::-1])
                    self.phase_g.append([-1])

            # read dustkapalignfact_ext.inp
            if alignfact:
                fname = 'dustkapalignfact_'+ext[i]+'.inp'
                if fdir is not None:
                    if fdir[-1] is '/':
                        fname = fdir + fname
                    else:
                        fname = fdir + '/' + fname

                print('Reading '+fname)

                # Check the file format
                iformat = np.fromfile(fname, count=1, sep=" ", dtype=np.int)
                iformat = iformat[0]
                if iformat != 1:
                    msg = 'Format number of the file dustkapscatmat_' + ext[i] + '.inp (iformat=' + ("%d" % iformat) + \
                          ') is unkown'
                    raise ValueError(msg)

                data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

                hdr = np.array(data[:3], dtype=np.int)
                data = data[3:]

                # get the wavelength grid
                dum_wav = data[:hdr[1]]
                data = data[hdr[1]:]

                # Get the angular grid
                anggrid = data[:hdr[2]]
                nang = hdr[2]
                self.alignang.append(anggrid)
                data = data[hdr[2]:]

                # Now get the kpara and korth
                data = np.reshape(data,[hdr[1],hdr[2], 2])
                self.korth.append(data[:,:,0])
                self.kpara.append(data[:,:,1])

        return 0

    def makeOpac(self, ppar=None, wav=None, old=False, code='python',
                 theta=None, logawidth=None, wfact=3.0, na=20, chopforward=0., errtol=0.01,
                 verbose=False, extrapolate=False, fdir=None,
                 ksca0=None, albedo0=None):
        """Createst the dust opacities using a Mie code distributed with RADMC-3D.

        Parameters
        ----------

        ppar        : dictionary
                      Parameters of the simulations

        wav         : ndarray, optional
                      Wavelength grid on which the mass absorption coefficients should be calculated

        code        : {'python', 'fortran'}
                      Version of the mie scattering code BHMIE to be used. 'fortran' - use the original fortran77
                      code of Bruce Drain (should be downloaded separately, compiled and its path added to the PATH
                      environment variable), 'python' a python version of BHMIE by Kees Dullemond (radmc3dPy.miescat).

        theta       : ndarray, optional
                      Angular grid (a numpy array) between 0 and 180
                      which are the scattering angle sampling points at
                      which the scattering phase function is computed.

        logawidth   : float, optional
                     If set, the size agrain will instead be a
                     sample of sizes around agrain. This helps to smooth out
                     the strong wiggles in the phase function and opacity
                     of spheres at an exact size. Since in Nature it rarely
                     happens that grains all have exactly the same size, this
                     is quite natural. The value of logawidth sets the width
                     of the Gauss in ln(agrain), so for logawidth<<1 this
                     give a real width of logawidth*agraincm.

        wfact       : float
                      Grid width of na sampling points in units
                      of logawidth. The Gauss distribution of grain sizes is
                      cut off at agrain * exp(wfact*logawidth) and
                      agrain * exp(-wfact*logawidth). Default = 3


        na          : int
                      Number of size sampling points (if logawidth set, default=20)

        chopforward : float
                      If >0 this gives the angle (in degrees from forward)
                      within which the scattering phase function should be
                      kept constant, essentially removing the strongly peaked
                      forward scattering. This is useful for large grains
                      (large ratio 2*pi*agraincm/lamcm) where the forward
                      scattering peak is extremely strong, yet extremely
                      narrow. If we are not interested in very forward-peaked
                      scattering (e.g. only relevant when modeling e.g. the
                      halo around the moon on a cold winter night), this will
                      remove this component and allow a lower angular grid
                      resolution for the theta grid.


        errtol      : float
                      Tolerance of the relative difference between kscat
                      and the integral over the zscat Z11 element over angle.
                      If this tolerance is exceeded, a warning is given.

        verbose     : bool
                      If set to True, the code will give some feedback so
                      that one knows what it is doing if it becomes slow.

        extrapolate : bool
                      If set to True, then if the wavelength grid lamcm goes
                      out of the range of the wavelength grid of the
                      optical constants file, then it will make a suitable
                      extrapolation: keeping the optical constants constant
                      for lamcm < minimum, and extrapolating log-log for
                      lamcm > maximum.


        old         : bool, optional
                      If set to True the file format of the previous, 2D version of radmc will be used

        fdir        : string, optional
                      the directory to store the files
        # some brute force settings to opacity
        ksca0      : float or list/1d array, optional
                      If set to a float value, multiplies all ksca related by ksca0.
                      If list, then means grain species dependent and should have 
                      same number of elements as grain species. 
                      Not sure why radmc3d cannot accept scattering_mode_max!=5 
                      for aligned grains, so use this brute force way to turn 
                      off scattering

        albedo0    : float or list, optional
                     Brute force way to control albedo at all wavelengths. 
                     The extinction opacity is kept the same, but alter kabs and ksca. 
                     Float for all grain species, or list to be species dependent. 
                     This is done after ksca0 argument. 
        """

        #
        # Create the wavelength grid if it is not specified
        #
        if wav is None:
            grid = radmc3dGrid()
            grid.makeWavelengthGrid(ppar=ppar)
            wav = grid.wav
            #
            # Do we need to mix the opacities?
            #
        if ppar is None:
            msg = 'Unknown ppar. The parameter dictionary is required to get the lnk file names.'
            raise ValueError(msg)

        if isinstance(ppar['lnk_fname'], str):
            ppar['lnk_fname'] = [ppar['lnk_fname']]

        if 'mixgsize' in ppar:
            mixgsize = ppar['mixgsize']
        else:
            mixgsize = 0
        # Get the grain sizes in micrometer
        if ppar['ngs'] > 1:
            gsize = ppar['gsmin'] * (ppar['gsmax'] / ppar['gsmin'])**(
                np.arange(ppar['ngs'], dtype=np.float64) / (float(ppar['ngs']) - 1.))
        else:
            gsize = [ppar['gsmin']]


        if len(ppar['lnk_fname']) > 1:
            ext = []
            gsizeinfo = []
            matdensinfo = []
            dweightsinfo = []
            for idust in range(len(ppar['lnk_fname'])):
                print('Reading: %s'%ppar['lnk_fname'][idust])

                # makedust needs the lnk file to be sorted in wavelength so create a dummy file
                # which contains the sorted optical constants
                with open(ppar['lnk_fname'][idust], 'r') as rfile:
                    w = []
                    n = []
                    k = []
                    dum = rfile.readline()
                    while len(dum) > 0:
                        dum = dum.split()
                        w.append(dum[0])
                        n.append(dum[1])
                        k.append(dum[2])
                        dum = rfile.readline()

                w = np.array(w, dtype=float)
                n = np.array(n, dtype=float)
                k = np.array(k, dtype=float)

                if float(w[0]) > float(w[w.shape[0] - 1]):
                    w = w[::-1]
                    n = n[::-1]
                    k = k[::-1]

                # Write out the dummy file containing the sorted optical constants
                with open('opt_const.dat', 'w') as wfile:
                    for iwav in range(w.shape[0]):
                        wfile.write("%s %s %s \n" % (w[iwav], n[iwav], k[iwav]))

                if code.lower().strip() == 'fortran':
                    # Run makedust
                    self.runMakedust(freq=nc.cc / wav * 1e4, gmin=ppar['gsmin'], gmax=ppar['gsmax'], ngs=ppar['ngs'],
                                     lnk_fname='opt_const.dat', gdens=ppar['gdens'][idust])

                    # Change the name of makedust's output
                    for igs in range(ppar['ngs']):
                        dum = sp.Popen('mv dustkappa_' + str(igs + 1) + '.inp dustkappa_idust_' + str(idust + 1)
                                       + '_igsize_' + str(igs + 1) + '.inp', shell=True).wait()
                        ext.append('idust_' + str(idust + 1) + '_igsize_' + str(igs + 1))
                        matdensinfo.append(ppar['gdens'][idust])
                        gdensinfo.append(gsize[igs])

                elif code.lower().strip() == 'python':

                    if 'nscatang' in ppar:
                        nang = ppar['nscatang']
                    else:
                        nang = 180
                    theta = 180. * np.arange(nang, dtype=np.float) / np.float(nang - 1)

                    if 'logawidth' in ppar:
                        logawidth = ppar['logawidth']
                    else:
                        logawidth = None

                    if 'wfact' in ppar:
                        wfact = ppar['wfact']
                    else:
                        wfact = 3.0

                    if 'chopforward' in ppar:
                        if ppar['chopforward'] > 0.:
                            chopforward = ppar['chopforward']
                        else:
                            chopforward = None
                    else:
                        chopforward = 0.0

                    if 'errtol' in ppar:
                        errtol = ppar['errtol']
                    else:
                        errtol = 0.01

                    if 'miescat_verbose' in ppar:
                        verbose = ppar['miescat_verbose']
                    else:
                        verbose = False

                    if 'extrapolate' in ppar:
                        extrapolate = ppar['extrapolate']
                    else:
                        extrapolate = False

                    swgt = ppar['gdens'][idust]
                    dweights = fneq.eq_dustweights(gsize, swgt, ppar['gsdist_powex'])

                    for igs in range(ppar['ngs']):
                        o = computeDustOpacMie(fname=ppar['lnk_fname'][idust], matdens=ppar['gdens'][idust],
                                                 agraincm=gsize[igs] * 1e-4, lamcm=wav * 1e-4, theta=theta,
                                                 logawidth=logawidth, wfact=wfact, na=na, chopforward=chopforward,
                                                 errtol=errtol, verbose=verbose, extrapolate=extrapolate, return_type=1)

                        # ksca0 setting
                        if type(ksca0) is float:
                            o.forceKsca(ksca0)
                        elif (type(ksca0) is list) or (type(ksca0) is np.ndarray):
                            o.forceKsca(ksca0[igs])

                        # albedo setting
                        if type(albedo0) is float:
                            o.forceAlbedo(albedo0)
                        elif (type(albedo0) is list) or (type(albedo0) is np.ndarray):
                            o.forceAlbedo(albedo0[igs])

                        # writing opacity
                        if ppar['scattering_mode_max'] <= 2:
                            o.writeOpac(ext='idust_' + (str(idust + 1)) + '_igsize_' + str(igs + 1), idust=0,
                                        scatmat=False, fdir=fdir)
                        else:
                            o.writeOpac(ext='idust_' + (str(idust + 1)) + '_igsize_' + str(igs + 1), idust=0, 
                                        scatmat=True, fdir=fdir)

                        ext.append('idust_' + (str(idust + 1)) + '_igsize_' + str(igs + 1))
                        matdensinfo.append(ppar['gdens'][idust])
                        gsizeinfo.append(gsize[igs])
                        dweightsinfo.append(dweights[igs] * swgt / sum(ppar['gdens']))

                        # if ppar['scattering_mode_max'] <= 2:
                        #     miescat.write_radmc3d_kappa_file(package=o, name='idust_1_igsize_' + str(igs + 1))
                        # else:
                        #     miescat.write_radmc3d_scatmat_file(package=o, name='idust_1_igsize_' + str(igs + 1))

                os.remove('opt_const.dat')
            # endfor idust

            # Mix the opacity of different dust species or grain size
	    # prior checks
            if 'mixabun' in ppar:
                if len(ppar['mixabun']) != len(ppar['lnk_fname']):
                    msg = 'ppar["mixabun"] or ppar["lnk_fname"] has the wrong shape. They both should have '\
                          + 'the same number of elements, but the number of elements are different.'
                    raise ValueError(msg)

            if mixgsize == 1:
		if 'gdens' not in ppar:
		    msg = 'gdens does not exist in ppar'
		    raise ValueError(msg)
		if 'gsdist_powex' not in ppar:
		    msg = 'gsdist_powex does not exist in ppar'
		    raise ValueError(msg)

            if 'mixabun' in ppar and mixgsize == 0:
                ext = []
                matdensinfo = []
                gsizeinfo = []
                dweightsinfo = []
                avgmatdens = 1. / np.sum( np.array(ppar['mixabun']) / np.array(ppar['gdens']) )
                dweights = fneq.eq_dustweights(gsize, avgmatdens, ppar['gsdist_powex'])
                for igs in range(ppar['ngs']):
                    if ppar['scattering_mode_max'] <= 2:
                        kaphead = 'dustkappa'
                        mixscatmat = False
                    else:
                        kaphead = 'dustkapscatmat'
                        mixscatmat = True

                    # new names
                    mixnames = [kaphead+'_igsize_' + str(igs + 1) + '.inp']
                    # the files you want to mix
                    mixspecs = [[kaphead+'_idust_' + str(idust + 1) + '_igsize_' + str(igs + 1) + '.inp'
                                 for idust in range(len(ppar['lnk_fname']))]]

                    self.mixOpac(mixnames=mixnames, mixspecs=mixspecs, mixabun=[ppar['mixabun']], scatmat=mixscatmat, fdir=fdir)
                    ext.append('igsize_' + str(igs + 1))
                    # the average specific weight: 1 / avg = sum( abundance / matdens )
                    matdensinfo.append(avgmatdens)
                    gsizeinfo.append(gsize[igs])
                    dweightsinfo.append(dweights[igs])
                    
            elif 'mixabun' in ppar and mixgsize ==1:
		# mix sizes first, then mix abundance
                for idust in range(len(ppar['lnk_fname'])):
                    swgt = ppar['gdens'][idust]
                    dweights = fneq.eq_dustweights(gsize, swgt, ppar['gsdist_powex'])
                    if ppar['scattering_mode_max'] <= 2:
                        kaphead = 'dustkappa'
                        mixscatmat = False
                    else:
                        kaphead = 'dustkapscatmat'
                        mixscatmat = True

		    # new names
                    mixnames = [kaphead+'_idust_' + str(idust + 1) + '.inp']
		    # the existing files you want to mix
                    mixspecs = [[kaphead+'_idust_' + str(idust+1) + '_igsize_' +str(igs+1) + '.inp'
                                for igs in range(ppar['ngs'])]]
                    self.mixOpac(mixnames=mixnames, mixspecs=mixspecs, mixabun=[dweights], scatmat=mixscatmat, fdir=fdir)
                #endfor idust
                mixnames = [kaphead+'_avg.inp']	#since we summed all the species and sizes
                mixspecs = [[kaphead+'_idust_'+str(idust + 1) + '.inp' 
			    for idust in range(len(ppar['lnk_fname']))]]
                self.mixOpac(mixnames=mixnames, mixspecs=mixspecs, mixabun=[ppar['mixabun']], scatmat=mixscatmat, fdir=fdir)
                ext = ['avg']

                avgmatdens = 1. / np.sum( np.array(ppar['mixabun']) / np.array(ppar['gdens']) )
                matdensinfo = [avgmatdens]
                
                avggsize = (1. / np.sum( np.array(dweights) / np.array(gsize)**3))**(1./3.)

                gsizeinfo = [avggsize]

                dweightsinfo = [1.]

            elif 'mixabun' not in ppar and mixgsize == 0:
		# simply not do anything
		msg = 'not doing anything'
            elif 'mixabun' not in ppar and mixgsize == 1:
                msg = 'cannot use the case for mixabun=0 and mixgsize=1'
                raise ValueError(msg)
            else:
                msg = 'there is something wrong for determining mixabun and mixgsize'
                raise ValueError(msg)
	    # Finished section for mixing opacities

            therm = [True for i in range(len(ext))]

            if 'alignment_mode' in ppar:
                alignment_mode = ppar['alignment_mode']
            else:
                alignment_mode = 0

            self.writeMasterOpac(ext=ext, therm=therm, scattering_mode_max=ppar['scattering_mode_max'], old=old, alignment_mode=alignment_mode)

            self.writeDustInfo(ext=ext, matdens=matdensinfo, gsize=gsizeinfo, dweights=dweightsinfo, fdir=fdir)

            if old:
                self.makeopacRadmc2D(ext=ext)

        else: #case which len(lnkfnames) < 0
            # makedust needs the lnk file to be sorted in wavelength so create a dummy file
            # which contains the sorted optical constants
            with open(ppar['lnk_fname'][0], 'r') as rfile:
                w = []
                n = []
                k = []
                dum = rfile.readline()
                while len(dum) > 0:
                    dum = dum.split()
                    w.append(dum[0])
                    n.append(dum[1])
                    k.append(dum[2])
                    dum = rfile.readline()

            w = np.array(w, dtype=float)
            n = np.array(n, dtype=float)
            k = np.array(k, dtype=float)

            if float(w[0]) > float(w[w.shape[0] - 1]):
                w = w[::-1]
                n = n[::-1]
                k = k[::-1]

            # Write out the dummy file containing the sorted optical constants
            with open('opt_const.dat', 'w') as wfile:
                for iwav in range(w.shape[0]):
                    wfile.write("%s %s %s \n" % (w[iwav], n[iwav], k[iwav]))

            if code.lower().strip() == 'fortran':
                # Run makedust
                self.runMakedust(freq=nc.cc / wav * 1e4, gmin=ppar['gsmin'], gmax=ppar['gsmax'], ngs=ppar['ngs'],
                                 lnk_fname='opt_const.dat', gdens=ppar['gdens'][0])

                # Change the name of makedust's output
                ext = []
                therm = []
                for igs in range(ppar['ngs']):
                    dum = sp.Popen('mv dustkappa_' + str(igs + 1) + '.inp dustkappa_idust_1_igsize_' + str(igs + 1)
                                   + '.inp', shell=True).wait()
                    ext.append('idust_1_igsize_' + str(igs + 1))
                    therm.append(True)

            elif code.lower().strip() == 'python':

                if 'nscatang' in ppar:
                    nang = ppar['nscatang']
                else:
                    nang = 180
                theta = 180. * np.arange(nang, dtype=np.float) / np.float(nang - 1)

                if 'logawidth' in ppar:
                    logawidth = ppar['logawidth']
                else:
                    logawidth = None

                if 'wfact' in ppar:
                    wfact = ppar['wfact']
                else:
                    wfact = 3.0

                if 'chopforward' in ppar:
                    if ppar['chopforward'] > 0.:
                        chopforward = ppar['chopforward']
                    else:
                        chopforward = None
                else:
                    chopforward = 0.0

                if 'errtol' in ppar:
                    errtol = ppar['errtol']
                else:
                    errtol = 0.01

                if 'miescat_verbose' in ppar:
                    verbose = ppar['miescat_verbose']
                else:
                    verbose = False

                if 'extrapolate' in ppar:
                    extrapolate = ppar['extrapolate']
                else:
                    extrapolate = False

                ext = []
                therm = []
                matdensinfo = []
                gsizeinfo = []
                dweightsinfo = []
                swgt = ppar['gdens'][0]
                dweights = fneq.eq_dustweights(gsize, swgt, ppar['gsdist_powex'])
                for igs in range(ppar['ngs']):
                    o = computeDustOpacMie(fname='opt_const.dat', matdens=ppar['gdens'][0],
                                             agraincm=gsize[igs] * 1e-4, lamcm=wav * 1e-4, theta=theta,
                                             logawidth=logawidth, wfact=wfact, na=na, chopforward=chopforward,
                                             errtol=errtol, verbose=verbose, extrapolate=extrapolate, return_type=1)

                    # brute force settings
                    # ksca0 setting
                    if type(ksca0) is float:
                        o.forceKsca(ksca0)
                    elif (type(ksca0) is list) or (type(ksca0) is np.ndarray):
                        o.forceKsca(ksca0[igs])

                    # albedo setting
                    if type(albedo0) is float:
                        o.forceAlbedo(albedo0)
                    elif (type(albedo0) is list) or (type(albedo0) is np.ndarray):
                        o.forceAlbedo(albedo0[igs])

                    # writing
                    if ppar['scattering_mode_max'] <= 2:
                        o.writeOpac(ext='idust_1_igsize_' + str(igs + 1), idust=0, scatmat=False, fdir=fdir)
                    else: 
                        o.writeOpac(ext='idust_1_igsize_' + str(igs + 1), idust=0, scatmat=True, fdir=fdir)

                    ext.append('idust_1_igsize_' + str(igs + 1))
                    therm.append(True)
                # if ppar['scattering_mode_max'] <= 2:
                #     miescat.write_radmc3d_kappa_file(package=o, name='idust_1_igsize_1')
                # else:
                #     miescat.write_radmc3d_scatmat_file(package=o, name='idust_1_igsize_1')
                    matdensinfo.append(ppar['gdens'][0])
                    gsizeinfo.append(gsize[igs])
                    dweightsinfo.append(dweights[igs])

                if mixgsize == 1:
                    swgt = ppar['gdens'][0]
                    dweights = fneq.eq_dustweights(gsize, swgt, ppar['gsdist_powex'])
                    # determining names
                    if ppar['scattering_mode_max'] <= 2:
                        kaphead = 'dustkappa'
                        mixscatmat = False
                    else: 
                        kaphead = 'dustkapscatmat'
                        mixscatmat = True

                    # new names
                    mixnames = [kaphead+'_avg.inp']
                    # existing files to mix
                    mixspecs = [[kaphead+'_idust_1_igsize_'+str(igs+1)+'.inp' for igs in range(ppar['ngs'])]]

                    self.mixOpac(mixnames=mixnames, mixspecs=mixspecs, mixabun=[dweights], scatmat=mixscatmat, fdir=fdir)
                    ext = ['avg']
                    therm = [True]
                    matdensinfo = [swgt]
                    avggsize = (1. / np.sum( np.array(dweights) / np.array(gsize)**3 ))**(1./3.)
                    gsizeinfo = [avggsize]
                    dweightsinfo = [1.0]
            else:
                msg = 'Unknown mie scattering code version ' + code
                raise ValueError(msg)

            os.remove('opt_const.dat')

            if 'alignment_mode' in ppar:
                alignment_mode = ppar['alignment_mode']
            else:
                alignment_mode = 0

            self.writeMasterOpac(ext=ext, therm=therm, scattering_mode_max=ppar['scattering_mode_max'], old=old, alignment_mode=alignment_mode)

            self.writeDustInfo(ext=ext, matdens=matdensinfo, gsize=gsizeinfo, dweights=dweightsinfo, fdir=fdir)

            if old:
                self.makeopacRadmc2D(ext=ext)
        #endif. case for len(lnkfnames) < 0

        # Clean up and remove dust.inp and frequency.inp
        if code.lower().strip() == 'fortran':
            os.remove('dust.inp')
            if not old:
                os.remove('frequency.inp')

    def forceKsca(self, ksca_fac):
        """
        direct manipulation for scattering opacity
        Parameters
        ----------
        ksca_fac : float or 1d list/ndarray
        """
        ngs = len(self.ksca)
        if (type(ksca_fac) is list) or (type(ksca_fac) is np.ndarray):
            if len(ksca_fac) != ngs:
                raise ValueError('number of input ksca_fac should be the same as number of grain species')

        for ig in range(ngs):
            if type(ksca_fac) is float:
                fac = ksca_fac
            elif (type(ksca_fac) is list) or (type(ksca_fac) is np.ndarray):
                fac = ksca_fac[ig]
            else:
                raise ValueError('input ksca_fac has wrong data type')

            self.ksca[ig] = self.ksca[ig] * fac
            self.z11[ig] = self.z11[ig] * fac
            self.z12[ig] = self.z12[ig] * fac
            self.z22[ig] = self.z22[ig] * fac
            self.z33[ig] = self.z33[ig] * fac
            self.z34[ig] = self.z34[ig] * fac
            self.z44[ig] = self.z44[ig] * fac
            self.ksca_from_z11[ig] = self.ksca_from_z11[ig] * fac

    def forceAlbedo(self, albedo):
        """
        Direct manipulation for albedo while extinction opacity is controlled. 
        Currently not wavelength dependent
        """
        ngs = len(self.ksca)
        if (type(albedo) is list) or (type(albedo) is np.ndarray):
            if len(albedo) != ngs:
                raise ValueError('number of input albedo should be the same as number of grain species')

        for ig in range(ngs):
            # original albedo, wavelength dependent
            albig = self.ksca[ig] / (self.kabs[ig] + self.ksca[ig])

            # determine factor
            if type(albedo) is float:
                alb = albedo
            elif (type(albedo) is list) or (type(albedo) is np.ndarray):
                alb = albedo[ig]
            else:
                raise ValueError('input albedo has wrong data type')

            # check value
            if (alb > 1) or (alb < 0):
                raise ValueError('input albedo must be within 0,1')

            # calculate
            self.kabs[ig] = self.kabs[ig] * (1. - alb) / (1. - albig)
            self.ksca[ig] = self.ksca[ig] * alb / albig

            # other scattering matrix by scaling. careful when albig is zero
            for iwav in range(len(albig)):
                self.z11[ig][iwav,:] = self.z11[ig][iwav,:] * alb / albig[iwav]
                self.z12[ig][iwav,:] = self.z12[ig][iwav,:] * alb / albig[iwav]
                self.z22[ig][iwav,:] = self.z22[ig][iwav,:] * alb / albig[iwav]
                self.z33[ig][iwav,:] = self.z33[ig][iwav,:] * alb / albig[iwav]
                self.z34[ig][iwav,:] = self.z34[ig][iwav,:] * alb / albig[iwav]
                self.z44[ig][iwav,:] = self.z44[ig][iwav,:] * alb / albig[iwav]

            self.ksca_from_z11[ig] = self.ksca_from_z11[ig] * alb / albig

    def writeDustAlignFact(self, fname=None, ext=None, idust=None):
        """
        Writes dust alignment factor into file

        Parameters
        ----------
        fname       : str
                      Name of the file to write the dust opacties into

        ext         : str
                      If fname is not specified, the output file name will be generated as dustkappa_EXT.inp or
                      dustkapscatmat_EXT.inp depending on the file format

        idust       : int
                      Dust species index whose opacities should be written to file

        """
        if fname is None:
            if ext is None:
                msg = 'Neither fname nor ext is specified. Filename cannot be generated '
                raise ValueError(msg)
            else:
                if idust is None:
                    msg = 'idust is not specified. If output file name should be generated both ext and idust should ' \
                          'be set'
                    raise ValueError(msg)
                else:
                    fname = 'dustkapalignfact_' + ext + '.inp'

        print('writing '+fname)
        with open(fname, 'w') as wfile:
            wfile.write('1\n')
            wfile.write('%d\n'%(self.nwav[idust]))
            wfile.write('%d\n'%(len(self.alignang[idust])))
            wfile.write('\n')
            for i in range(self.nwav[idust]):
                wfile.write('%13.6e\n'%(self.wav[idust][i]))
            wfile.write('\n')
            for i in range(len(self.alignang[idust])):
                wfile.write('%13.6e\n'%(self.alignang[idust][i]))
            wfile.write('\n')
            for inu in range(self.nwav[idust]):
                for imu in range(len(self.alignang[idust])):
                    wfile.write('%13.6e %13.6e\n'%(self.korth[idust][inu,imu], 
                                                   self.kpara[idust][inu,imu]))
                wfile.write('\n')

    def makeDustAlignFact(self, ppar=None, wav=None, theta=None):
        """ calculates dust alignment factor and outputs dustkapalignfact_ext.inp
        Parameters
        ----------

        ppar        : dictionary
                      Parameters of the simulations. 
                      Needs 'kpara0'

        wav         : ndarray, optional
                      Wavelength grid on which the alignment factor should be calculated

        theta       : ndarray, optional
                      Angular grid (a numpy array) between 0 and 90. Default is 0 to 90.  

        """

        #
        # Create the wavelength grid if it is not specified
        #
        if wav is None:
            grid = radmc3dGrid()
            grid.makeWavelengthGrid(ppar=ppar)
            wav = grid.wav
        nwav = len(wav)

        if theta is None:
            ntheta = 91
            theta = 90. * np.arange(ntheta, dtype=np.float) / np.float(ntheta-1)

        if ppar is None:
            msg = 'Unknown ppar. The parameter dictionary is required to get the lnk file names.'
            raise ValueError(msg)

        # read masteropac
        mopac = self.readMasterOpac()
        ext = mopac['ext']

        # calculate alingment factors
        if 'kpara0' in ppar:
            kpara0 = ppar['kpara0']
        else:
            raise ValueError('Unknown kpara0 in ppar')


        for idust in range(len(ext)):
            muang = np.cos(theta * np.pi / 180.)
            kpara = np.zeros([nwav,ntheta], dtype=np.float64)
            korth = np.zeros([nwav,ntheta], dtype=np.float64)
            for inu in range(nwav):
                # simple model
#                kpara[inu,:]  = ( 1.e0 - ppar['kpara0']*np.cos(muang*np.pi) ) / ( 1.e0 + ppar['kpara0'])
                # Lee & Draine 1985
                kpara[inu,:] = (1. + ppar['kpara0']*np.cos(2.*theta*np.pi/180.)) / (1. + ppar['kpara0'])
                korth[inu,:] = 1.0
            self.kpara.append(kpara)
            self.korth.append(korth)
            self.alignang.append(theta)

        for idust in range(len(ext)):
            self.writeDustAlignFact(ext=ext[idust], idust=idust)
            
    @staticmethod
    def makeBeckOpac(wav=None, beck0=10., beta=1., matdens=1.0, gsize=0.01, fdir=None):
        """creates opacity in the form of opacity = beck0 * (freq / 1e12)**beta like Beckwith opacity

        Parameters
        ----------
        wav      :  ndarray
                    Wavelength grid for opacities in micron
        beck0    : float
                   The value of opacity at 1e12 frequency
        beta     : float
                   The power-law index for opacity
        matdens  : float
                   specific weight of material. No use here, just for the formalism in the output files
        gsize    : float
                   size of grain. No use here, just her for the formalism in the output files
        """

        if wav is None:
            msg = 'Unknown wav.'
            raise ValueError(msg)

        nwav = len(wav)
        freq = nc.cc * 1e4 / wav
        kabs = beck0 * (freq / 1e12)**beta
        ksca = np.zeros([nwav], dtype=np.float64)
        phase_g = np.zeros([nwav], dtype=np.float64)

        opac = radmc3dDustOpac()
        opac.nwav = [nwav]
        opac.nfreq = [nwav]
#        opac.nang = [0]
        opac.wav = [wav]
#        opac.scatang = [0.]
        opac.freq = [freq]
        opac.kabs = [kabs]
        opac.ksca = [ksca]
        opac.phase_g = [phase_g]
#        opac.z11 = [zscat[:, :, 0]]
#        opac.z12 = [zscat[:, :, 1]]
#        opac.z22 = [zscat[:, :, 2]]
#        opac.z33 = [zscat[:, :, 3]]
#        opac.z34 = [zscat[:, :, 4]]
#        opac.z44 = [zscat[:, :, 5]]
        opac.therm = [True]
        opac.scatmat = [False]
#        opac.ksca_from_z11 = [kscat_from_z11]

        opac.writeOpac(ext='beck', idust=0, scatmat=False, fdir=fdir)
        ext = ['beck']
        matdensinfo = [matdens]
        gsizeinfo = [gsize]
        dweightsinfo = [1.]
        opac.writeMasterOpac(ext='beck', therm=True, scattering_mode_max=0, old=False, fdir=fdir)
        opac.writeDustInfo(ext=['beck'], matdens=matdensinfo, gsize=gsizeinfo, dweights=dweightsinfo, fdir=fdir)

    @staticmethod
    def mixOpac(ppar=None, mixnames=None, mixspecs=None, mixabun=None, writefile=True, scatmat=False, fdir=None):
        """Mixes dust opacities.


        Parameters
        -----------
        ppar      : dictionary, optional
                    All parameters of the actual model setup.

        mixnames  : list, optional
                    Names of the files into which the mixed dust opacities will be written
                    (not needed if writefile=False)

        mixspecs  : list, optional
                    Names of the files from which the dust opacities are read (not needed if readfile=False)

        mixabun   : list, optional
                    Abundances of different dust species

        writefile : bool
                    If False the mixed opacities will not be written out to files given in mixnames.
        scatmat   : bool 
                    If False, it will read 'dustkappa_**.inp' and mix those opacities. 
                    If True, it will read 'dustkapscatmat_**.inp' and mix opacities and scattering matrix.
                    Not sure how physical this is though. I will call radmc3dDustOpac() for reading, etc

        NOTE, either ppar or  mixname, mixspecs, and mixabun should be set.

        """

        if writefile:
            if mixnames is None:
                if ppar is None:
                    msg = 'Neither ppar nor mixnames are set in mixOpac'
                    raise ValueError(msg)
                else:
                    mixnames = ppar['mixnames']

        if mixspecs is None:
            if ppar is None:
                msg = ' Neither ppar nor mixspecs are set in mixOpac '
                raise ValueError(msg)
            else:
                mixspecs = ppar['mixspecs']

        if mixabun is None:
            if ppar is None:
                msg = ' Neither ppar nor mixabun are set in mixOpac '
                raise ValueError(msg)
            else:
                mixabun = ppar['mixabun']
        if scatmat is False: 
            for i in range(len(mixnames)):
                i#
                # Read the dust opacities to be mixed for composite dust species #1
                #
                ocabs = []
                ocsca = []
                ogsym = []
                oform = 0
                for j in range(len(mixspecs[i])):
                    with open(mixspecs[i][j], 'r') as rfile:
                        form = int(rfile.readline())
                        nwav = int(rfile.readline())
                        dw = np.zeros(nwav, dtype=float)
                        dcabs = np.zeros(nwav, dtype=float)
                        dcsca = np.zeros(nwav, dtype=float)
                        gsym = np.zeros(nwav, dtype=float)
                        if form == 1:
                            if (oform == 0) | (oform == 1):
                                oform = 1
                            else:
                                print(' ')
                                print('WARNING')
                                print(' You are trying to mix opacity tables with different formats. Some of the tables \n'
                                  + ' contain scattering coefficients while (format>=2) while others do not '
                                  + ' (format=1).\n'
                                  + ' If you wish to continue mixing will only be done for the absorption and the \n'
                                  + 'output opacity table will have a format number of 1.')
                                dum = input('Do you wish to continue (1-yes, 0-no) ?')
                                if dum.strip() != '1':
                                    return

                            for iwav in range(nwav):
                                dum = rfile.readline().split()
                                dw[iwav], dcabs[iwav] = float(dum[0]), float(dum[1])
                        if form == 2:
                            if (oform == 0) | (oform == 2):
                                oform = 2
                            else:
                                print(' ')
                                print('WARNING')
                                print(' You are trying to mix opacity tables with different formats. Some of the tables \n'
                                  + ' contain scattering coefficients while (format>=2) while other do not '
                                  + '(format=1). \n'
                                  + ' If you wish to continue mixing will only be done for the absorption and the \n'
                                  + 'output opacity table will have a format number of 1.')

                                dum = input('Do you wish to continue (1-yes, 0-no) ?')
                                if dum.strip() != '1':
                                    return
                            for iwav in range(nwav):
                                dum = rfile.readline().split()
                                dw[iwav], dcabs[iwav], dcsca[iwav] = float(dum[0]), float(dum[1]), float(dum[2])
                        if form == 3:
                            if (oform == 0) | (oform == 3):
                                oform = 3
                            else:
                                print(' ')
                                print('WARNING')
                                print(' You are trying to mix opacity tables with different formats. Some of the tables \n'
                                  + ' contain scattering coefficients while (format>=2) while other do not '
                                  + '(format=1) \n'
                                  + ' If you wish to continue mixing will only be done for the absorption and the '
                                  + 'output opacity table will have a format number of 1.')
                                dum = input('Do you wish to continue (1-yes, 0-no) ?')
                                if dum.strip() != '1':
                                    return
                            for iwav in range(nwav):
                                dum = rfile.readline().split()
                                dw[iwav], dcabs[iwav], dcsca[iwav], gsym[iwav] = float(dum[0]), float(dum[1]), float(
                                    dum[2]), float(dum[3])
                        if form > 3:
                            msg = ' Unsupported dust opacity table format (format number: ' + ("%d" % form) + ')' \
                              + ' Currently only format number 1 and 2 are supported'
                            raise ValueError(msg)

                        if dw[1] < dw[0]:
                            print(' Dust opacity table seems to be sorted in frequency instead of wavelength')
                            print(' Reversing the arrays')
                            dw = dw[::-1]
                            dcabs = dcabs[::-1]
                            dcsca = dcsca[::-1]

                    if j == 0:
                        ocabs = np.array(dcabs) * mixabun[i][j]
                        ocsca = np.array(dcsca) * mixabun[i][j]
                        ogsym = np.array(gsym) * mixabun[i][j]
                        nwav0 = dw.shape[0]
                        owav = np.array(dw)
                    else:
                        #
                        # Interpolate dust opacities to the wavelength grid of the first dust species
                        #
                        ii = ((owav >= dw[0]) & (owav <= dw[nwav - 1]))
                        il = (owav < dw[0])  ## index that are lower than minimum of first dust species
                        ih = (owav > dw[nwav - 1])  ## indicies that are higher than maximum
                        dum = np.zeros(nwav0, dtype=float)
                        dum[ii] = 10. ** np.interp(np.log10(owav[ii]), np.log10(dw), np.log10(dcabs))

                        # Edwtrapolate the absorption coefficients using linear fit in log-log space
                        # (i.e. fitting a polinomial) for short wavelengths
                        # der = np.log10(dcabs[1] / dcabs[0]) / np.log10(dw[1] / dw[0])
                        dum[il] = 10. ** (np.log10(dcabs[0]) + np.log10(dw[0] / owav[il]))
 
                        # Edwtrapolate the absorption coefficients using linear fit in log-log space
                        # (i.e. fitting a polinomial) for long wavelengths
                        # der = np.log10(dcabs[nwav - 1] / dcabs[nwav - 2]) / np.log10(dw[nwav - 1] / dw[nwav - 2])
                        dum[ih] = 10. ** (np.log10(dcabs[nwav - 1]) + np.log10(owav[il] / dw[nwav - 1]))
  
                        ocabs = ocabs + np.array(dum) * mixabun[i][j]

                        if oform == 2:
                            # Do the inter-/extrapolation of for the scattering coefficients
                            dum = np.zeros(nwav0, dtype=float)
                            dum[ii] = 10. ** np.interp(np.log10(owav[ii]), np.log10(dw), np.log10(dcsca))
    
                            # der = np.log10(dcsca[1] / dcsca[0]) / np.log10(dw[1] / dw[0])
                            dum[il] = 10. ** (np.log10(dcsca[0]) + np.log10(dw[0] / owav[il]))

                            # der = np.log10(dcsca[nwav - 1] / dcsca[nwav - 2]) / np.log10(dw[nwav - 1] / dw[nwav - 2])
                            dum[ih] = 10. ** (np.log10(dcsca[nwav - 1]) + np.log10(owav[il] / dw[nwav - 1]))

                            ocsca = ocsca + np.array(dum) * mixabun[i][j]

                        if oform == 3:
                            # Do the inter-/extrapolation of for the scattering phase function
                            dum = np.zeros(nwav0, dtype=float)
                            #dum[ii] = 10. ** np.interp(np.log10(owav[ii]), np.log10(dw), log10(gsym)) #sometimes gsym may be negative
                            dum[ii] = np.interp(np.log10(owav[ii]), np.log10(dw), gsym)

                            # der = np.log10(gsym[1] / gsym[0]) / np.log10(dw[1] / dw[0])
                            dum[il] = 10. ** (np.log10(gsym[0]) + np.log10(dw[0] / owav[il]))

                            # der = np.log10(gsym[nwav - 1] / gsym[nwav - 2]) / np.log10(dw[nwav - 1] / dw[nwav - 2])
                            if gsym[nwav-1] >= 0:
                                dum[ih] = 10. ** (np.log10(gsym[nwav - 1]) + np.log10(owav[il] / dw[nwav - 1]))
                            else:
                                dum[ih] = - 10.**(np.log10(-gsym[nwav-1]) + np.log10(owav[il] / dw[nwav-1]))

                            ogsym = ogsym + np.array(dum) * mixabun[i][j]

                #
                # Write out the mixed dust opacities
                #
                with open(mixnames[i], 'w') as wfile:
                    wfile.write("%d\n" % oform)
                    wfile.write("%d\n" % owav.shape[0])
                    if oform == 1:
                        for iwav in range(owav.shape[0]):
                            wfile.write("%.9e %.9e\n" % (owav[iwav], ocabs[iwav]))
                    if oform == 2:
                        for iwav in range(owav.shape[0]):
                            wfile.write("%.9e %.9e %.9e\n" % (owav[iwav], ocabs[iwav], ocsca[iwav]))
                    if oform == 3:
                        for iwav in range(owav.shape[0]):
                            wfile.write("%.9e %.9e %.9e %.9e\n" % (owav[iwav], ocabs[iwav], ocsca[iwav], ogsym[iwav]))
        else: #if scatmat is True
            idust_seq = [i for i in range(len(mixnames))]
            extnames = [mixnames[i].replace('dustkapscatmat_','') for i in range(len(mixnames))]
            extnames = [extnames[i].replace('.inp', '') for i in range(len(mixnames))]
            for i in range(len(mixnames)):
                if 'dustkapscatmat_' not in mixnames[i]:
                    pdb.set_trace()
                extspecs = [mixspecs[i][j].replace('dustkapscatmat_','') for j in range(len(mixspecs[i]))]
                extspecs = [extspecs[j].replace('.inp', '') for j in range(len(mixspecs[i]))]

                dumop = radmc3dDustOpac()	#this is to read each individual opacities
                dumop.readOpac(ext=extspecs, scatmat=[True for j in range(len(mixspecs[i]))], old=False)

                outop = radmc3dDustOpac() 	#this is for the summed data and for writing output
                #just assume that it is all in the same wavlength... so troublesome to intra/extrapolate to wavelength
                #and assume it is all in the same angular grid too...

                outop.wav.append(dumop.wav[0])
                outop.freq.append(dumop.freq[0])
                outop.nwav.append(dumop.nwav[0])
                outop.nfreq.append(dumop.nfreq[0])
                
                # kabs
                dumsum = np.zeros([dumop.nwav[0]], dtype=np.float64)
                for j in range(len(dumop.idust)):
                   dumsum = dumsum + dumop.kabs[j] * mixabun[i][j]
                outop.kabs.append(dumsum)

                # ksca
                dumsum = np.zeros([dumop.nwav[0]], dtype=np.float64)
                for j in range(len(dumop.idust)):
                   dumsum = dumsum + dumop.ksca[j] * mixabun[i][j]
                outop.ksca.append(dumsum)

                # phase_g
                dumsum = np.zeros([dumop.nwav[0]], dtype=np.float64)
                for j in range(len(dumop.idust)):
                   dumsum = dumsum + dumop.phase_g[j] * mixabun[i][j]
                outop.phase_g.append(dumsum)

                # ext
                outop.ext.append(extnames[i])

                # idust
                outop.idust.append(idust_seq[i])
 
                # therm
#                outop.therm.append(dumop.therm[0]) # original reading doesn't seem to include this

                # scatmat
                outop.scatmat.append(True)
                
                # z11
                dumsum = dumop.z11[0] * mixabun[i][0]
                for j in range(len(dumop.idust)-1):
                    dumsum = dumsum + dumop.z11[j+1] * mixabun[i][j+1]
                outop.z11.append(dumsum)

                # z12
                dumsum = dumop.z12[0] * mixabun[i][0]
                for j in range(len(dumop.idust)-1):
                    dumsum = dumsum + dumop.z12[j+1] * mixabun[i][j+1]
                outop.z12.append(dumsum)

                # z22
                dumsum = dumop.z22[0] * mixabun[i][0]
                for j in range(len(dumop.idust)-1):
                    dumsum = dumsum + dumop.z22[j+1] * mixabun[i][j+1]
                outop.z22.append(dumsum)

                # z33
                dumsum = dumop.z33[0] * mixabun[i][0]
                for j in range(len(dumop.idust)-1):
                    dumsum = dumsum + dumop.z33[j+1] * mixabun[i][j+1]
                outop.z33.append(dumsum)

                # z34
                dumsum = dumop.z34[0] * mixabun[i][0]
                for j in range(len(dumop.idust)-1):
                    dumsum = dumsum + dumop.z34[j+1] * mixabun[i][j+1]
                outop.z34.append(dumsum)

                # z44
                dumsum = dumop.z44[0] * mixabun[i][0]
                for j in range(len(dumop.idust)-1):
                    dumsum = dumsum + dumop.z44[j+1] * mixabun[i][j+1]
                outop.z44.append(dumsum)

                # scatang
                outop.scatang.append(dumop.scatang[0])

                # nang
                outop.nang.append(dumop.nang[0])

                # ksca_from_z11
                dumsum = dumop.ksca_from_z11[0] * mixabun[i][0]
                for j in range(len(dumop.idust)-1):
                    dumsum = dumsum + dumop.ksca_from_z11[j+1] * mixabun[i][j+1]
                outop.ksca_from_z11.append(dumsum)

                # write to file
                outop.writeOpac(ext=extnames[i], idust=0, scatmat=True, fdir=fdir)
           
        return

    @staticmethod
    def readDustInfo(fdir=None):
        """Reads the dust information file: 'dustinfo.zyl'
        This is not a standard file for radmc3d, but simply read the recorded information of dust species
        format:
            material density of the dust
            grain size in micron. 
        This will be in the order of dustopac.inp

        Parameters
        ----------
        fname = the name of the file to read. default is dustinfo.zyl in the current directory

        Returns
        -------
        Returns a dictionary with the following keys:
            matdens  : list of specific weights of dust

            gsize    : list of grain size in micron
        """
        fname = 'dustinfo.zyl'
        if fdir is not None:
            if fdir[-1] is '/':
                fname = fdir + fname
            else:
                fname = fdir + '/' + fname

        with open(fname, 'r') as rfile:
            # file format
            dum = rfile.readline()
            # nr of dust species 
            ndust = int(rfile.readline().split()[0])
            # Comment line
            dum = rfile.readline()

            matdens = []
            gsize = []
            dweights = []
            ext = []
            for idust in range(ndust):
                # read specific weight
                dum = rfile.readline().split()[0]
                matdens.append(float(dum))

                # read dust grain size in micron
                dum = rfile.readline().split()[0]
                gsize.append(float(dum))

                # read the weighting of the dust in fraction
                dum = rfile.readline().split()[0]
                dweights.append(float(dum))

                # Dustkappa filename extension
                dum = rfile.readline().split()[0]
                ext.append(dum)

                # Comment line
                dum = rfile.readline()

        return {'matdens': matdens, 'gsize': gsize, 'dweights':dweights}

    @staticmethod
    def writeDustInfo(fname=None, ext=None, matdens=None, gsize=None, dweights=None, fdir=None):
        """ Writes some dust information to dustinfo.zyl
        including specific weight, grain size

        Parameters
        ----------
        fname   : optional
                  name of the file
        ext     : list, string
                  the extension names of the opacity written in dustopac.inp. This will have no use, but 
                  just as a consistency check
        matdens : list
                  specific weight of the dust
        gsize   : list
                  grain size in micron
        """
        if fdir is not None:
            if fdir[-1] is '/':
                fname = fdir + 'dustinfo.zyl'
            else:
                fname = fdir + '/dustinfo.zyl'
        if fname is None:
            fname = 'dustinfo.zyl'
        if ext is None:
            raise ValueError('ext must be given when writing to dustinfo.zyl')
        if matdens is None:
            raise ValueError('matdens must be given when writing to dustinfo.zyl')
        if gsize is None:
            raise ValueError('gsize must be given when writing to dustinfo.zyl')

        ndust = len(matdens)
        if ndust != len(gsize):
            raise ValueError('matdens and gsize must be in the same length')

        with open('dustinfo.zyl', 'w') as wfile:
            # file format
            wfile.write('%-15s %s\n' % ('2', 'Format number of this file'))

            # number of dust species
            wfile.write('%-15s %s\n' % (str(len(ext)), 'Nr of dust species'))

            # Seperator
            wfile.write('%s\n' % '============================================================================')

            for idust in range(ndust):
                # specific weight
                wfile.write('%.5f %s %s\n' % (matdens[idust],'    ', 'Specific weight in cgs'))

                # grain size in micron
                wfile.write('%.5f %s %s\n' % (gsize[idust],'    ', 'grain size in micron'))

                # mass weighting of this type of dust
                wfile.write('%.5f %s %s\n' % (dweights[idust], '     ', 'dust weighting'))

                # extension names
                wfile.write('%s %s %s\n' % (ext[idust], '    ', 'Extension of name of dustkappa_***.inp file'))

                # Seperator
                wfile.write('%s\n' % '----------------------------------------------------------------------------')

    @staticmethod
    def readMasterOpac(fdir=None):
        """Reads the master opacity file 'dustopac.inp'.
        It reads the dustkappa filename extensions (dustkappa_ext.inp) corresponding to dust species indices

        Returns
        -------

        Returns a dictionary with the following keys:

            *ext   : list of dustkappa file name extensions

            *therm : a list of integers specifying whether the dust grain is thermal or quantum heated
            (0 - thermal, 1 - quantum heated)
        """
        fname = 'dustopac.inp'
        if fdir is not None:
            if fdir[-1] is '/':
                fname = fdir + fname
            else:
                fname = fdir + '/' + fname

        with open(fname, 'r') as rfile:

            # file format
            dum = rfile.readline()
            # nr of dust species
            ndust = int(rfile.readline().split()[0])
            # Comment line
            dum = rfile.readline()

            ext = []
            therm = []
            scatmat = []
            align = []
            for idust in range(ndust):
                # Check if we have dust opacities also for the full scattering matrix
                dum = rfile.readline().split()
                if int(dum[0]) == 1:
                    scatmat.append(False)
                    align.append(False)
                elif int(dum[0]) == 10:
                    scatmat.append(True)
                    align.append(False)
                elif int(dum[0]) == 20:
                    scatmat.append(True)
                    align.append(True)

                # Check if the dust grain is thermal or quantum heated
                dum = int(rfile.readline().split()[0])
                if dum == 0:
                    therm.append(True)
                else:
                    therm.append(False)
                # Dustkappa filename extension
                dum = rfile.readline().split()[0]
                ext.append(dum)
                # Comment line
                dum = rfile.readline()

        return {'ext': ext, 'therm': therm, 'scatmat': scatmat, 'align':align}

    @staticmethod
    def writeMasterOpac(ext=None, therm=None, scattering_mode_max=1, 
            alignment_mode=0, old=False, fdir=None):
        """Writes the master opacity file 'dustopac.inp'.

        Parameters
        ----------

        ext                 : list
                              List of dustkappa file name extensions

        therm               : list
                              List of integers specifying whether the dust grain is thermal or quantum heated
                              (0-thermal, 1-quantum)

        scattering_mode_max : int
                              Scattering mode code in radmc3d : 0 - no scattering, 1 - isotropic scattering,
                              2 - anisotropic scattering with Henyei-Greenstein phase function, 5 - anisotropic
                              scattering using the full scattering matrix and stokes vectors.

        alignment_mode      : int, optional
                              Currently, if set to 0, then no alignment

        old                 : bool, optional
                              If set to True the file format of the previous, 2D version of radmc will be used

        fdir                : string, optional
                              The name of directory to create this file
        """
        if fdir is None:
            fname = 'dustopac.inp'
        else:
            if fdir[-1] is '/':
                fname = fdir + 'dustopac.inp'
            else:
                fname = fdir + '/dustopac.inp'
        print('Writing '+fname)

        if not ext:
            msg = 'Unknown ext. No file name extension is specified. Without it dustopac.inp cannot be written'
            raise ValueError(msg)
        else:
            if isinstance(ext, str):
                ext = [ext]

        if therm is None:
            # If therm is not specified it is assumed that all grains are thermal, no quantum heating
            therm = [True for i in range(len(ext))]
        else:
            if isinstance(therm, int):
                therm = [therm]
            if len(ext) != len(therm):
                msg = ' The number of dust species in ext and in therm are different'
                raise ValueError(msg)

        with open(fname, 'w') as wfile:

            # File format
            wfile.write('%-15s %s\n' % ('2', 'Format number of this file'))
            # Number of dust species
            wfile.write('%-15s %s\n' % (str(len(ext)), 'Nr of dust species'))
            # Separator
            wfile.write('%s\n' % '============================================================================')

            if not old:
                for idust in range(len(ext)):
                    # Dust opacity will be read from a file
                    if scattering_mode_max < 5:
                        wfile.write('%-15s %s\n' % ('1', 'Way in which this dust species is read'))
                    else:
                        if alignment_mode is 0: 
                            wfile.write('%-15s %s\n' % ('10', 'Way in which this dust species is read'))
                        else:
                            wfile.write('%-15s %s\n' % ('20', 'Way in which this dust species is read'))

                    # Check if the dust grain is thermal or quantum heated
                    if therm:
                        if therm[idust]:
                            wfile.write('%-15s %s\n' % ('0', '0=Thermal grain, 1=Quantum heated'))
                    else:
                        wfile.write('%-15s %s\n' % ('1', '0=Thermal grain, 1=Quantum heated'))

                    # Dustkappa filename extension
                    wfile.write('%s %s %s\n' % (ext[idust], '    ', 'Extension of name of dustkappa_***.inp file'))
                    # Separator
                    wfile.write('%s\n' % '----------------------------------------------------------------------------')
            else:
                for idust in range(len(ext)):
                    # Dust opacity will be read from a file
                    wfile.write('%-15s %s\n' % ('-1', 'Way in which this dust species is read (-1=File)'))

                    # Check if the dust grain is thermal or quantum heated
                    wfile.write('%-15s %s\n' % ('0', '0=Thermal grain, 1=Quantum heated'))
                    # Dustkappa filename extension
                    wfile.write('%d %s %s\n' % ((idust + 1), '    ', 'Extension of name of dustopac_***.inp file'))
                    # Separator
                    wfile.write('%s\n' % '----------------------------------------------------------------------------')

    def makeopacRadmc2D(self, ext=None):
        """
        Creates dust opacities (dustopac_*.inp files) for the previous 2D version of radmc
        It takes the input dust opacity files and interpolates them onto the used frequency grid

        Parameters
        ----------

            ext : list
                  List of dustkappa file name extensions, i.e. the input file name has to be named
                  as dustkappa_ext[i].inp

        """

        if ext is None:
            msg = 'Unknown ext. Dust opacity file name extensions are mandatory.'
            raise ValueError(msg)

        else:
            if isinstance(ext, str):
                ext = [ext]

        self.readOpac(ext=ext, old=False)
        #
        # Read the frequency.inp file
        #
        freq = np.fromfile('frequency.inp', count=-1, sep="\n", dtype=float)
        nfreq = int(freq[0])
        freq = freq[1:]
        freq = freq[::-1]
        wav = nc.cc / freq * 1e4
        #
        # Check if the frequency grid is ordered in frequency or in wavelength
        #
        worder = False
        if freq[-1] < freq[0]:
            worder = True

        for i in range(len(ext)):
            kabs = np.zeros(nfreq, dtype=float)
            ksca = np.zeros(nfreq, dtype=float)
            ish = (wav < self.wav[i][0])
            ilo = (wav > self.wav[i][-1])
            ii = ((wav >= self.wav[i][0]) & (wav <= self.wav[i][-1]))

            #
            # Do logarithmic interpolation for the overlapping wavelenght domain
            #
            kabs[ii] = 10. ** np.interp(np.log10(wav[ii]), np.log10(self.wav[i]), np.log10(self.kabs[i]))
            if len(self.ksca[i]) > 1:
                ksca[ii] = 10. ** np.interp(np.log10(wav[ii]), np.log10(self.wav[i]), np.log10(self.ksca[i]))

            #
            # Do the long wavelength part
            #
            if True in ilo:
                x1 = np.log10(self.wav[i][-1])
                x0 = np.log10(self.wav[i][-2])

                y1 = np.log10(self.kabs[i][-1])
                y0 = np.log10(self.kabs[i][-2])
                der = (y1 - y0) / (x1 - x0)
                kabs[ilo] = 10. ** (y1 + der * (np.log10(wav[ilo]) - x1))

                y1 = np.log10(self.ksca[i][-1])
                y0 = np.log10(self.ksca[i][-2])
                der = (y1 - y0) / (x1 - x0)
                ksca[ilo] = 10. ** (y1 + der * (np.log10(wav[ilo]) - x1))

            #
            # Do the shorter wavelength
            #
            if True in ish:
                kabs[ish] = self.kabs[0][0]
                ksca[ish] = self.ksca[0][0]

            #
            # Now write the results to file
            #
            fname = 'dustopac_' + ("%d" % (i + 1)) + '.inp'
            with open(fname, 'w') as wfile:
                print('Writing ' + fname)
                wfile.write("%d 1\n" % nfreq)
                wfile.write(" \n")
                #
                # Reverse the order of kabs,ksca as they are ordered in frequency in radmc
                #
                if worder:
                    x = kabs[::-1]
                else:
                    x = kabs
                for ilam in range(nfreq):
                    wfile.write("%.7e\n" % x[ilam])

                wfile.write(" \n")
                if worder:
                    x = ksca[::-1]
                else:
                    x = ksca
                for ilam in range(nfreq):
                    wfile.write("%.7e\n" % x[ilam])

    @staticmethod
    def runMakedust(freq=None, gmin=None, gmax=None, ngs=None, lnk_fname=None, gdens=None):
        """Interface function to the F77 code makedust to calculate mass absorption coefficients.

        Parameters
        ----------
        freq       : ndarray
                    Contains the frequency grid on which the opacities should be calculated

        gmin       : float
                    Minimum grain size

        gmax       : float
                    Maximum grain size

        ngs        : int
                    Number of grain sizes

        gdens      : float
                    Density of the dust grain in g/cm^3

        lnk_fname  : str
                    Name of the file in which the optical constants are stored

        Returns
        -------

        Returns an ndarray with [nfreq,ngs] dimensions containing the resulting opacities
        """

        #
        # Calculate the grain sizes
        #
        if ngs > 1:
            gsize = gmin * (gmax / gmin) ** (np.arange(ngs, dtype=np.float64) / (float(ngs) - 1.))
        else:
            gsize = [gmin]

        #
        # Write the frequency.inp file
        #
        with open('frequency.inp', 'w') as wfile:
            wfile.write("%d\n" % freq.shape[0])
            wfile.write("  \n")
            for i in range(freq.shape[0]):
                wfile.write("%.10e\n" % freq[i])

        #
        # Write the dust.inp file (makedust main control file)
        #
        with open('dust.inp', 'w') as wfile:
            for igs in range(ngs):
                wfile.write("%s\n" % lnk_fname)
                wfile.write("%s\n" % "MIE")
                wfile.write("%d %f %f %f %d %f %f %f\n" %
                            (1, 0.0, np.log10(gsize[igs]), np.log10(gsize[igs]), 1., -3.5, gdens, -2.0))

        #
        # Run the Mie-code
        #
        dum = sp.Popen('makedust', shell=True).wait()



def computeDustOpacMie(fname='', matdens=None, agraincm=None, lamcm=None,
                     theta=None, logawidth=None, wfact=3.0, na=20,
                     chopforward=0.0, errtol=0.01, verbose=False,
                     extrapolate=False, return_type=1):
    """
    Compute dust opacity with Mie theory based on the optical constants
    in the optconst_file. Optionally also the scattering phase function
    in terms of the Mueller matrix elements can be computed. To smear out
    the resonances that appear due to the perfect sphere shape, you can
    optionally smear out the grain size distribution a bit with setting
    the width of a Gaussian grain size distribution.

    Parameters
    ----------
    fname       : str
                  File name of the optical constants file. This file
                  should contain three columns: first the wavelength
                  in micron, then the n-coefficient and then the
                  k-coefficient. See Jena optical constants database:
                  http://www.astro.uni-jena.de/Laboratory/Database/databases.html

    matdens     : float
                  Material density in g/cm^3

    agraincm    : float
                  Grain radius in cm

    lamcm       : ndarray
                  Wavelength grid in cm

    theta       : ndarray, optional
                  Angular grid (a numpy array) between 0 and 180
                  which are the scattering angle sampling points at
                  which the scattering phase function is computed.

    logawidth   : float, optional
                 If set, the size agrain will instead be a
                 sample of sizes around agrain. This helps to smooth out
                 the strong wiggles in the phase function and opacity
                 of spheres at an exact size. Since in Nature it rarely
                 happens that grains all have exactly the same size, this
                 is quite natural. The value of logawidth sets the width
                 of the Gauss in ln(agrain), so for logawidth<<1 this
                 give a real width of logawidth*agraincm.

    wfact       : float
                  Grid width of na sampling points in units
                  of logawidth. The Gauss distribution of grain sizes is
                  cut off at agrain * exp(wfact*logawidth) and
                  agrain * exp(-wfact*logawidth). Default = 3


    na          : int
                  Number of size sampling points (if logawidth set, default=20)

    chopforward : float
                  If >0 this gives the angle (in degrees from forward)
                  within which the scattering phase function should be
                  kept constant, essentially removing the strongly peaked
                  forward scattering. This is useful for large grains
                  (large ratio 2*pi*agraincm/lamcm) where the forward
                  scattering peak is extremely strong, yet extremely
                  narrow. If we are not interested in very forward-peaked
                  scattering (e.g. only relevant when modeling e.g. the
                  halo around the moon on a cold winter night), this will
                  remove this component and allow a lower angular grid
                  resolution for the theta grid.


    errtol      : float
                  Tolerance of the relative difference between kscat
                  and the integral over the zscat Z11 element over angle.
                  If this tolerance is exceeded, a warning is given.

    verbose     : bool
                  If set to True, the code will give some feedback so
                  that one knows what it is doing if it becomes slow.

    extrapolate : bool
                  If set to True, then if the wavelength grid lamcm goes
                  out of the range of the wavelength grid of the
                  optical constants file, then it will make a suitable
                  extrapolation: keeping the optical constants constant
                  for lamcm < minimum, and extrapolating log-log for
                  lamcm > maximum.

    return_type : {0, 1}
                  If 0 a dictionary is returned (original return type)
                  if 1 an instance of radmc3dDustOpac will be returned

    Returns
    -------
    A dictionary with the following keys:

        * kabs          : ndarray
                          Absorption opacity kappa_abs_nu (a numpy array) in
                          units of cm^2/gram

        * ksca          : ndarray
                          Scattering opacity kappa_abs_nu (a numpy array) in
                          units of cm^2/gram

        * gsca          : ndarray
                          The <cos(theta)> g-factor of scattering

        * theta         : ndarray (optional, only if theta is given at input)
                          The theta grid itself (just a copy of what was given)

        * zscat         : ndarray (optional, only if theta is given at input)
                          The components of the scattering Mueller matrix
                          Z_ij for each wavelength and each scattering angel.
                          The normalization of Z is such that kscat can be
                          reproduced (as can be checked) by the integral:
                          2*pi*int_{-1}^{+1}Z11(mu)dmu=kappa_scat.
                          For symmetry reasons only 6 elements of the Z
                          matrix are returned: Z11, Z12, Z22, Z33, Z34, Z44.
                          Note that Z21 = Z12 and Z43 = -Z34.
                          The scattering matrix is normalized such that
                          if a plane wave with Stokes flux
                             Fin = (Fin_I,Fin_Q,Fin_U,Fin_V)
                          hits a dust grain (which has mass mgrain), then
                          the scattered flux
                             Fout = (Fout_I,Fout_Q,Fout_U,Fout_V)
                          at distance r from the grain at angle theta
                          is given by
                             Fout(theta) = (mgrain/r^2) * Zscat . Fin
                          where . is the matrix-vector multiplication.
                          Note that the Stokes components must be such
                          that the horizontal axis in the "image" is
                          pointing in the scattering plane. This means
                          that radiation with Fin_Q < 0 is scattered well,
                          because it is vertically polarized (along the
                          scattering angle axis), while radiation with
                          Fin_Q > 0 is scatterd less well because it
                          is horizontally polarized (along the scattering
                          plane).

        * kscat_from_z11 : ndarray  (optional, only if theta is given at input)
                           The kscat computed from the (above mentioned)
                           integral of Z11 over all angles. This should be
                           nearly identical to kscat if the angular grid
                           is sufficiently fine. If there are strong
                           differences, this is an indication that the
                           angular gridding (the theta grid) is not fine
                           enough. But you should have then automatically
                           gotten a warning message as well (see errtol).

        * wavmic        : ndarray (optional, only if extrapolate is set to True)
                          The original wavelength grid from the optical constants file,
                          with possibly an added extrapolated

        * ncoef         : ndarray (optional, only if extrapolate is set to True)
                          The optical constant n at that grid

        * kcoef         : ndarray (optional, only if extrapolate is set to True)
                          The optical constant k at that grid

        * agr           : ndarray (optional, only if logawidth is not None)
                          Grain sizes

        * wgt           : ndarray (optional, only if logawidth is not None)
                          The averaging weights of these grain (not the masses!)
                          The sum of wgt.sum() must be 1.

        * zscat_nochop  : ndarray (optional, only if chopforward > 0)
                          The zscat before the forward scattering was chopped off

        * kscat_nochop  : ndarray (optional, only if chopforward > 0)
                          The kscat originally from the bhmie code
    """
    #
    # Load the optical constants
    #
    if matdens is None:
        msg = "Unknown material density matdens"
        raise ValueError(msg)

    if agraincm is None:
        msg = "Unknown grain size agraincm"
        raise ValueError(msg)

    if lamcm is None:
        msg = "Unknown wavelength grid lamcm"
        raise ValueError(msg)

    if theta is None:
        angles = np.array([0., 90., 180.])  # Minimalistic angular s
        if chopforward != 0.:
            warnings.warn("Chopping disabled. Chopping is only possible if theta grid is given. ", RuntimeWarning)
    else:
        angles = theta

    #
    # Check that the theta array goes from 0 to 180 or
    # 180 to 0, and store which is 0 and which is 180
    #
    if angles[0] != 0:
        msg = "First element of the angular grid array is not 0. Scattering angle grid must extend from 0 to 180 " \
              "degrees."
        raise ValueError(msg)
    if angles[-1] != 180:
        msg = "Last element of the angular grid array is not 180. Scattering angle grid must extend from 0 to 180 " \
              "degrees."
        raise ValueError(msg)

    nang = angles.shape[0]

    #
    # Load the optical constants
    #
    data = np.loadtxt(fname)
    wavmic, ncoef, kcoef = data.T

    if wavmic.size <= 1:
        msg = "Optical constants file must have at least two rows with two different wavelengths"
        raise ValueError(msg)

    if wavmic[1] == wavmic[0]:
        msg = "Optical constants file must have at least two rows with two different wavelengths"
        raise ValueError(msg)

    #
    # Check range, and if needed and requested, extrapolate the
    # optical constants to longer or shorter wavelengths
    #
    if extrapolate:
        wmin = np.min(lamcm)*1e4 * 0.999
        wmax = np.max(lamcm)*1e4 * 1.001
        if wmin < np.min(wavmic):
            if wavmic[0] < wavmic[1]:
                ncoef = np.append([ncoef[0]], ncoef)
                kcoef = np.append([kcoef[0]], kcoef)
                wavmic = np.append([wmin], wavmic)
            else:
                ncoef = np.append(ncoef, [ncoef[-1]])
                kcoef = np.append(kcoef, [kcoef[-1]])
                wavmic = np.append(wavmic, [wmin])
        if wmax > np.max(wavmic):
            if wavmic[0] < wavmic[1]:
                ncoef = np.append(ncoef, [ncoef[-1] * np.exp((np.log(wmax) - np.log(wavmic[-1])) *
                                                             (np.log(ncoef[-1]) - np.log(ncoef[-2])) /
                                                             (np.log(wavmic[-1]) - np.log(wavmic[-2])))])
                kcoef = np.append(kcoef, [kcoef[-1]*np.exp((np.log(wmax) - np.log(wavmic[-1])) *
                                                           (np.log(kcoef[-1]) - np.log(kcoef[-2])) /
                                                           (np.log(wavmic[-1]) - np.log(wavmic[-2])))])
                wavmic = np.append(wavmic, [wmax])
            else:
                ncoef = np.append(ncoef, [ncoef[0]*np.exp((np.log(wmax)-np.log(wavmic[0])) *
                                                          (np.log(ncoef[0]) - np.log(ncoef[1])) /
                                                          (np.log(wavmic[0]) - np.log(wavmic[1])))])
                kcoef = np.append(kcoef, [kcoef[0]*np.exp((np.log(wmax) - np.log(wavmic[0])) *
                                                          (np.log(kcoef[0]) - np.log(kcoef[1])) /
                                                          (np.log(wavmic[0]) - np.log(wavmic[1])))])
                wavmic = np.append([wmax], wavmic)
    else:
        if lamcm.min() <= wavmic.min()*1e4:
            raise ValueError("Wavelength range out of range of the optical constants file")

        if lamcm.max() >= wavmic.max()*1e-4:
            raise ValueError("Wavelength range out of range of the optical constants file")

    # Interpolate
    # Note: Must be within range, otherwise stop
    #
    f = interp1d(np.log(wavmic*1e-4), np.log(ncoef))
    ncoefi = np.exp(f(np.log(lamcm)))
    f = interp1d(np.log(wavmic*1e-4), np.log(kcoef))
    kcoefi = np.exp(f(np.log(lamcm)))
    #
    # Make the complex index of refraction
    #
    refidx = ncoefi + kcoefi*1j
    #
    # Make a size distribution for the grains
    # If width is not set, then take just one size
    #
    if logawidth is None:
        agr = np.array([agraincm])
        wgt = np.array([1.0])
    else:
        if logawidth != 0.0:
            agr = np.exp(np.linspace(np.log(agraincm) - wfact * logawidth, np.log(agraincm) + wfact * logawidth, na))
            wgt = np.exp(-0.5*((np.log(agr / agraincm)) / logawidth)**2)
            wgt = wgt / wgt.sum()
        else:
            agr = np.array([agraincm])
            wgt = np.array([1.0])
    #
    # Get the true number of grain sizes
    #
    nagr = agr.size
    #
    # Compute the geometric cross sections
    #
    siggeom = np.pi*agr*agr
    #
    # Compute the mass of the grain
    #
    mgrain = (4*np.pi/3.0)*matdens*agr*agr*agr
    #
    # Now prepare arrays
    #
    nlam = lamcm.size
    kabs = np.zeros(nlam)
    kscat = np.zeros(nlam)
    gscat = np.zeros(nlam)
    if theta is not None:
        zscat = np.zeros((nlam, nang, 6))
        S11 = np.zeros(nang)
        S12 = np.zeros(nang)
        S33 = np.zeros(nang)
        S34 = np.zeros(nang)
        if chopforward > 0:
            zscat_nochop = np.zeros((nlam, nang, 6))
            kscat_nochop = np.zeros(nlam)

    #
    # Set error flag to False
    #
    error = False
    errmax = 0.0
    kscat_from_z11 = np.zeros(nlam)
    #
    # Loop over wavelengths
    #
    for i in range(nlam):
        #
        # Message
        #
        if verbose:
            print("Doing wavelength %13.6e cm" % lamcm[i])
        #
        # Now loop over the grain sizes
        #
        for l in range(nagr):
            #
            # Message
            #
            if verbose and nagr > 1:
                print("...Doing grain size %13.6e cm" % agr[l])
            #
            # Compute x
            #
            x = 2*np.pi*agr[l]/lamcm[i]
            #
            # Call the bhmie code
            #
            S1, S2, Qext, Qabs, Qsca, Qback, gsca = miescat.bhmie(x, refidx[i], angles)
            #
            # Add results to the averaging over the size distribution
            #
            kabs[i] += wgt[l] * Qabs*siggeom[l] / mgrain[l]
            kscat[i] += wgt[l] * Qsca*siggeom[l] / mgrain[l]
            gscat[i] += wgt[l] * gsca
            #
            # If angles were set, then also compute the Z matrix elements
            #
            if theta is not None:
                #
                # Compute conversion factor from the Sxx matrix elements
                # from the Bohren & Huffman code to the Zxx matrix elements we
                # use (such that 2*pi*int_{-1}^{+1}Z11(mu)dmu=kappa_scat).
                # This includes the factor k^2 (wavenumber squared) to get
                # the actual cross section in units of cm^2 / ster, and there
                # is the mass of the grain to get the cross section per gram.
                #
                factor = (lamcm[i]/(2*np.pi))**2/mgrain[l]
                #
                # Compute the scattering Mueller matrix elements at each angle
                #
                S11[:] = 0.5 * (np.abs(S2[:])**2 + np.abs(S1[:])**2)
                S12[:] = 0.5 * (np.abs(S2[:])**2 - np.abs(S1[:])**2)
                S33[:] = np.real(S2[:] * np.conj(S1[:]))
                S34[:] = np.imag(S2[:] * np.conj(S1[:]))
                zscat[i, :, 0] += wgt[l] * S11[:] * factor
                zscat[i, :, 1] += wgt[l] * S12[:] * factor
                zscat[i, :, 2] += wgt[l] * S11[:] * factor
                zscat[i, :, 3] += wgt[l] * S33[:] * factor
                zscat[i, :, 4] += wgt[l] * S34[:] * factor
                zscat[i, :, 5] += wgt[l] * S33[:] * factor
        #
        # If possible, do a check if the integral over zscat is consistent
        # with kscat
        #
        if theta is not None:
            mu = np.cos(angles * np.pi / 180.)
            dmu = np.abs(mu[1:nang] - mu[0:nang-1])
            zav = 0.5 * (zscat[i, 1:nang, 0] + zscat[i, 0:nang-1, 0])
            dum = 0.5 * zav * dmu
            kscat_from_z11[i] = dum.sum() * 4 * np.pi
            err = abs(kscat_from_z11[i]/kscat[i]-1.0) * 100.
            if err > errtol:
                error = True
                errmax = max(err, errmax)
        #
        # If the chopforward angle is set >0, then we will remove
        # excessive forward scattering from the opacity. The reasoning
        # is that extreme forward scattering is, in most cases, equivalent
        # to no scattering at all.
        #
        if chopforward > 0:
            iang = np.where(angles < chopforward)
            if angles[0] == 0.0:
                iiang = np.max(iang)+1
            else:
                iiang = np.min(iang)-1
            zscat_nochop[i, :, :] = zscat[i, :, :]  # Backup
            kscat_nochop[i] = kscat[i]      # Backup
            zscat[i, iang, 0] = zscat[i, iiang, 0]
            zscat[i, iang, 1] = zscat[i, iiang, 1]
            zscat[i, iang, 2] = zscat[i, iiang, 2]
            zscat[i, iang, 3] = zscat[i, iiang, 3]
            zscat[i, iang, 4] = zscat[i, iiang, 4]
            zscat[i, iang, 5] = zscat[i, iiang, 5]
            mu = np.cos(angles * np.pi / 180.)
            dmu = np.abs(mu[1:nang] - mu[0:nang-1])
            zav = 0.5 * (zscat[i, 1:nang, 0] + zscat[i, 0:nang-1, 0])
            dum = 0.5 * zav * dmu
            kscat[i] = dum.sum() * 4 * np.pi

            zav = 0.5 * (zscat[i, 1:nang, 0] * mu[1:] + zscat[i, 0:nang-1, 0] * mu[:-1])
            dum = 0.5 * zav * dmu
            gscat[i] = dum.sum() * 4 * np.pi / kscat[i]

    #
    # If error found, then warn (Then shouldn't it be called a warning? If it's a true error
    #  shouldn't we stop the execution and raise an exception?)
    #
    if error:
        msg = " Angular integral of Z11 is not equal to kscat at all wavelength. \n"
        msg += "Maximum error = %13.6e percent" % errmax
        if chopforward > 0:
            msg += "But I am using chopforward to remove strong forward scattering, and then renormalized kapscat."
        print(msg)
#        warnings.warn(msg, RuntimeWarning)
    #
    # Now return what we computed in a dictionary
    #
    package = {"lamcm": lamcm, "kabs": kabs, "kscat": kscat,
               "gscat": gscat, "matdens": matdens, "agraincm": agraincm}
    if theta is not None:
        package["zscat"] = np.copy(zscat)
        package["theta"] = np.copy(angles)
        package["kscat_from_z11"] = np.copy(kscat_from_z11)
    if extrapolate:
        package["wavmic"] = np.copy(wavmic)
        package["ncoef"] = np.copy(ncoef)
        package["kcoef"] = np.copy(kcoef)
    if nagr > 1:
        package["agr"] = np.copy(agr)
        package["wgt"] = np.copy(wgt)
        package["wfact"] = wfact
        package["logawidth"] = logawidth
    if chopforward > 0:
        package["zscat_nochop"] = np.copy(zscat_nochop)
        package["kscat_nochop"] = np.copy(kscat_nochop)

    if return_type == 0:
        return package
    else:
        opac = radmc3dDustOpac()
        opac.nwav = [nlam]
        opac.nfreq = [nlam]
        opac.nang = [nang]
        opac.wav = [lamcm*1e4]
        opac.scatang = [angles]
        opac.freq = [nc.cc/lamcm]
        opac.kabs = [kabs]
        opac.ksca = [kscat]
        opac.phase_g = [gscat]
        opac.z11 = [zscat[:, :, 0]]
        opac.z12 = [zscat[:, :, 1]]
        opac.z22 = [zscat[:, :, 2]]
        opac.z33 = [zscat[:, :, 3]]
        opac.z34 = [zscat[:, :, 4]]
        opac.z44 = [zscat[:, :, 5]]
        opac.therm = [True]
        opac.scatmat = [True]
        opac.ksca_from_z11 = [kscat_from_z11]
        return opac
