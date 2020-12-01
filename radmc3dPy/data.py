"""This module contains a class for handling variable data in radmc-3d

"""
from __future__ import absolute_import
from __future__ import print_function
import traceback
import os
import warnings

try:
    import numpy as np
except ImportError:
    np = None
    print(' Numpy cannot be imported ')
    print(' To use the python module of RADMC-3D you need to install Numpy')
    print(traceback.format_exc())

from . import natconst as nc
from . import crd_trans
from . dustopac import *
from . reggrid import radmc3dGrid
from . octree import radmc3dOctree

class BaseData(object):
    """ base of the data objects
    include only the essential functions shared by all objects

    Attributes
    ----------
    grid      : radmc3dGrid, radmc3dOctree
                Instance of the radmc3dGrid class, contains the spatial and frequency grids

    """
    def __init__(self):
        self.grid = None
    
class radmc3dData(BaseData):
    """RADMC-3D data class.
        Reading and writing dust density/temperature, gas density/temperature/velocity,
        generating a legacy vtk file for visualization.


    Attributes
    ----------

    grid      : radmc3dGrid, radmc3dOctree
                Instance of the radmc3dGrid class, contains the spatial and frequency grids

    rhodust   : ndarray
                Dust density in g/cm^3

    dusttemp  : ndarray
                Dust temperature in K

    rhogas    : ndarray
                Gas density in g/cm^3

    ndens_mol : ndarray
                Number density of the molecule [molecule/cm^3]

    ndens_cp  : ndarray
                Number density of the collisional partner [molecule/cm^3]

    gasvel    : ndarray
                Gas velocity in cm/s

    gastemp   : ndarray
                Gas temperature in K

    vturb     : ndarray
                Mictroturbulence in cm/s

    taux      : ndarray
                Optical depth along the x (cartesian) / r (cylindrical) / r (spherical) dimension

    tauy      : ndarray
                Optical depth along the y (cartesian) / theta (cylindrical) / theta (spherical) dimension

    tauz      : ndarray
                Optical depth along the z (cartesian) / z (cylindrical) / phi (spherical) dimension

    sigmadust : ndarray
                Dust surface density in g/cm^2

    sigmagas  : ndarray
                Gas surface density in molecule/cm^2 (or g/cm^2 depending on the dimension of rhogas)

    alvec     : ndarray
                Dust alignment vector field

    qvis      : ndarray
                heating for heatsource.inp. same dimensions as gas density

    bfield    : ndarray
                magnetic field. same dimensions as gas velocity

    radfield  : ndarray
                the radiation field calculated by "radmc3d mcmono"

    """

    def __init__(self):
        BaseData.__init__(self)

        self.rhodust = np.zeros(0, dtype=np.float64)
        self.dusttemp = np.zeros(0, dtype=np.float64)
        self.rhogas = np.zeros(0, dtype=np.float64)
        self.ndens_mol = np.zeros(0, dtype=np.float64)
        self.ndens_cp = np.zeros(0, dtype=np.float64)
        self.gasvel = np.zeros(0, dtype=np.float64)
        self.gastemp = np.zeros(0, dtype=np.float64)
        self.vturb = np.zeros(0, dtype=np.float64)
        self.taux = np.zeros(0, dtype=np.float64)
        self.tauy = np.zeros(0, dtype=np.float64)
        self.tauz = np.zeros(0, dtype=np.float64)
        self.sigmadust = np.zeros(0, dtype=np.float64)
        self.sigmagas = np.zeros(0, dtype=np.float64)
        self.alvec = np.zeros(0, dtype=np.float64)
        self.qvis = np.zeros(0, dtype=np.float64)
        self.bfield = np.zeros(0, dtype=np.float64)
        self.radfield = np.zeros(0, dtype=np.float64)

    def _scalarfieldWriter(self, fname, arr, binary=False):
        """Writes a scalar field to a file.

        Parameters
        ----------

        arr   : ndarray
                Scalar variable to be written

        fname  : str
                Name of the file containing a scalar variable

        binary : bool
                If True the file will be in binary format, if False the file format is formatted ASCII text

        """

        with open(fname, 'w') as wfile:
            # octree
            if isinstance(self.grid, radmc3dOctree):
                # ! this part is not tested yet !
                if len(arr.shape) == 1:
                    if binary:
                        hdr = np.array([1, 8, arr.shape[0], 1], dtype=np.int64)
                    else:
                        hdr = np.array([1, arr.shape[0], 1], dtype=np.int64)
                elif len(arr.shape) == 2:
                    if binary:
                        hdr = np.array([1, 8, arr.shape[0], arr.shape[1]], dtype=np.int64)
                    else:
                        hdr = np.array([1, arr.shape[0], arr.shape[1]], dtype=np.int64)
                else:
                    raise ValueError('Incorrect shape. Data stored in an Octree-type grid should have 1 or 2 '
                                     + 'dimensions, with the second dimension being the dust species. '
                                     + ' The data to be written has a dimension of ' + ("%d" % len(arr.shape))
                                     + '\n No data has been written')
            else: # regular grid
                # determine header
                if len(arr.shape) == 3:
                    if binary:
                        hdr = np.array([1, 8, self.grid.nx * self.grid.ny * self.grid.nz], dtype=int)
                    else:
                        hdr = np.array([1, self.grid.nx * self.grid.ny * self.grid.nz], dtype=int)
                elif len(arr.shape) == 4:
                    if binary:
                        hdr = np.array([1, 8, self.grid.nx * self.grid.ny * self.grid.nz, arr.shape[3]], dtype=int)
                    else:
                        hdr = np.array([1, self.grid.nx * self.grid.ny * self.grid.nz, arr.shape[3]], dtype=int)
                else:
                    raise ValueError('Incorrect shape. Data stored in a regular grid should have 3 or 4 dimensions'
                                         + ' with the fourth dimension being the dust species. '
                                         + ' The data to be written has a dimension of ' + ("%d" % len(arr.shape))
                                         + '\n No data has been written')

            # write out header and data
            if binary:
                hdr.tofile(wfile)
                arr.flatten(order='f').tofile(wfile)
            else:
                hdr.tofile(wfile, sep="\n", format="%d")
                wfile.write('\n')
                arr.flatten(order='f').tofile(wfile, sep="\n", format="%.9e")

    def _scalarfieldReader(self, fname='', binary=False, octree=False, ndim=3):
        """Reads a scalar field from file.

        Parameters
        ----------

        fname  : str
                Name of the file containing a scalar variable

        binary : bool
                If True the file is in binary format, if False the file format is formatted ASCII text

        octree : bool
                If True data will be read from an octree model and will be stored in a 1D numpy array

        ndim   : int
                Number of dimension of the data field (3 for gas variables, 4 for dust)
        Returns
        -------

        Returns a numpy Ndarray with the scalar field
        """

        data = None
        with open(fname, 'r') as rfile:
            if binary:
                # Read the header
                # hdr[0] = format number
                # hdr[1] = data precision (4=single, 8=double)
                # hdr[2] = nr of cells
                # hdr[3] = nr of dust species
                if ndim == 3:
                    hdr = np.fromfile(rfile, count=3, dtype=np.int64)
                else:
                    hdr = np.fromfile(rfile, count=4, dtype=np.int64)
                if octree:
                    if hdr[2] != self.grid.nLeaf:
                        print(hdr[1], self.grid.nLeaf)
                        msg = ('Number of cells in ' + fname + ' is different from that in amr_grid.inp'
                               + ' nr cells in ' + fname + ' : ' + ("%d" % hdr[2]) + '\n '
                               + ' nr of cells in amr_grid.inp : ' + ("%d" % self.grid.nLeaf))
                        raise ValueError(msg)

                    if hdr[1] == 8:
                        data = np.fromfile(rfile, count=-1, dtype=np.float64)
                    elif hdr[1] == 4:
                        data = np.fromfile(rfile, count=-1, dtype=np.float32)
                    else:
                        msg = 'Unknown datatype/precision in ' + fname + '. RADMC-3D binary files store 4 byte ' \
                              + 'floats or 8 byte doubles. The precision in the file header is ' + ("%d" % hdr[1])
                        raise TypeError(msg)

                    if ndim == 3:
                        if data.shape[0] == hdr[2]:
                            data = np.reshape(data, [self.grid.nLeaf, 1], order='f')
                        else:
                            msg = 'Internal inconsistency in data file, number of cell entries is different from ' \
                                  'indicated in the file header'
                            raise ValueError(msg)

                    else:
                        if data.shape[0] == hdr[2] * hdr[3]:
                            data = np.reshape(data, [self.grid.nLeaf, hdr[3]], order='f')
                        else:
                            msg = 'Internal inconsistency in data file, number of cell entries is different from ' \
                                  'indicated in the file header'
                            raise ValueError(msg)

                else:
                    if hdr[2] != (self.grid.nx * self.grid.ny * self.grid.nz):
                        msg = 'Number of grid cells in ' + fname + ' is different from that in amr_grid.inp ' \
                              + ' nr cells in ' + fname + ' : ' + ("%d" % hdr[2]) + '\n ' \
                              + ' nr of cells in amr_grid.inp : ' \
                              + ("%d" % (self.grid.nx * self.grid.ny * self.grid.nz))
                        raise ValueError(msg)

                    if hdr[1] == 8:
                        data = np.fromfile(rfile, count=-1, dtype=np.float64)
                    elif hdr[1] == 4:
                        data = np.fromfile(rfile, count=-1, dtype=np.float32)
                    else:
                        msg = 'Unknown datatype/precision in ' + fname + '. RADMC-3D binary files store 4 byte ' \
                              + 'floats or 8 byte doubles. The precision in the file header is ' \
                              + ("%d" % hdr[1])
                        raise TypeError(msg)

                    if ndim == 3:
                        if data.shape[0] == hdr[2]:
                            data = np.reshape(data, [1, self.grid.nz, self.grid.ny, self.grid.nx])
                        else:
                            msg = 'Internal inconsistency in data file, number of cell entries is different from ' \
                                  'indicated in the file header'
                            raise ValueError(msg)
                    else:
                        if data.shape[0] == hdr[2] * hdr[3]:
                            data = np.reshape(data, [hdr[3], self.grid.nz, self.grid.ny, self.grid.nx])
                        else:
                            msg = 'Internal inconsistency in data file, number of cell entries is different from ' \
                                  'indicated in the file header'
                            raise ValueError(msg)

                    # data = reshape(data, [hdr[3],self.grid.nz,self.grid.ny,self.grid.nx])
                    # We need to change the axis orders as Numpy always writes binaries in C-order while RADMC-3D
                    # uses Fortran-order
                    data = np.swapaxes(data, 0, 3)
                    data = np.swapaxes(data, 1, 2)

            else: # if not binary
                if ndim == 3:
                    hdr = np.fromfile(rfile, count=3, sep=" ", dtype=np.int64)
                else:
                    hdr = np.fromfile(rfile, count=4, sep=" ", dtype=np.int64)

                if octree:
                    if hdr[1] != self.grid.nLeaf:
                        msg = 'Number of cells in ' + fname + ' is different from that in amr_grid.inp' \
                              + ' nr cells in ' + fname + ' : ' + ("%d" % hdr[1]) + '\n '\
                              + ' nr of cells in amr_grid.inp : ' + ("%d" % self.grid.nLeaf)
                        raise ValueError(msg)

                    data = np.fromfile(rfile, count=-1, sep=" ", dtype=np.float64)

                    if ndim == 3:
                        if data.shape[0] == hdr[2]:
                            data = np.reshape(data, [self.grid.nLeaf, 1], order='f')
                        else:
                            msg = 'Internal inconsistency in data file, number of cell entries is different from ' \
                                  'indicated in the file header'
                            raise ValueError(msg)

                    else:
                        if data.shape[0] == hdr[2] * hdr[3]:
                            data = np.reshape(data, [self.grid.nLeaf, hdr[3]], order='f')
                        else:
                            msg = 'Internal inconsistency in data file, number of cell entries is different from ' \
                                  'indicated in the file header'
                            raise ValueError(msg)

                    # if ndim == 3:
                    #
                    #     data = data.reshape([hdr[1], hdr[2]], order='f')

                else:
                    if hdr[1] != (self.grid.nx * self.grid.ny * self.grid.nz):
                        msg = 'Number of grid cells in ' + fname + ' is different from that in amr_grid.inp '\
                              + ' nr cells in ' + fname + ' : ' + ("%d" % hdr[1]) + '\n '\
                              + ' nr of cells in amr_grid.inp : '\
                              + ("%d" % (self.grid.nx * self.grid.ny * self.grid.nz))
                        raise ValueError(msg)

                    data = np.fromfile(rfile, count=-1, sep=" ", dtype=np.float64)
                    if ndim == 3:
                        if data.shape[0] == hdr[1]:
#                            data = np.reshape(data, [1, self.grid.nz, self.grid.ny, self.grid.nx])
                            data = np.reshape(data, [self.grid.nx, self.grid.ny, self.grid.nz, 1], order='F')
                        else:
                            msg = 'Internal inconsistency in data file, number of cell entries is different from ' \
                                  'indicated in the file header'
                            raise ValueError(msg)
                    else:
                        if data.shape[0] == hdr[1] * hdr[2]:
#                            data = np.reshape(data, [hdr[2], self.grid.nz, self.grid.ny, self.grid.nx])
                            data = np.reshape(data, [self.grid.nx, self.grid.ny, self.grid.nz, hdr[2]], order='F')
                        else:
                            msg = 'Internal inconsistency in data file, number of cell entries is different from ' \
                                  'indicated in the file header'
                            raise ValueError(msg)

                    # data = reshape(data, [hdr[3],self.grid.nz,self.grid.ny,self.grid.nx])
                    # We need to change the axis orders as Numpy always writes binaries in C-order while RADMC-3D
                    # uses Fortran-order
#                    data = np.swapaxes(data, 0, 3)
#                    data = np.swapaxes(data, 1, 2)

        return data

    def _vectorfieldWriter(self, fname, arr, binary=False, ndim=3):
        """
        Parameters
        ----------
        fname : str
            name of the output file
        arr : ndarray 
            vector field
        binary : bool
            binary file or not
        ndim : int
            if only depends spatially : ndim = 3
            if also depends on a fourth parameter : ndim = 4
        """
        with open(fname, 'w') as wfile:
            # octree
            if isinstance(self.grid, radmc3dOctree):
                # ! this is not tested yet !
                # determine header information 
                if len(arr.shape) == 1:
                    if binary:
                        hdr = np.array([1, 8, arr.shape[0], 1], dtype=np.int64)
                    else:
                        hdr = np.array([1, arr.shape[0], 1], dtype=np.int64)
                elif len(arr.shape) == 2:
                    if binary:
                        hdr = np.array([1, 8, arr.shape[0], arr.shape[1]], dtype=np.int64)
                    else:
                        hdr = np.array([1, arr.shape[0], arr.shapre[1]], dtype=np.int64)
                else:
                    raise ValueError('Incorrect shape. Data stored in an Octree-type grid should have 1 or 2 '
                                     + 'dimensions, with the second dimension being the dust species. '
                                     + ' The data to be written has a dimension of ' + ("%d" % len(arr.shape))
                                     + '\n No data has been written')
                # write header and data 
                if binary:
                    hdr.tofile(wfile)
                    arr.flatten(order='f').tofile(wfile)

                else:
                    hdr.tofile(wfile, sep=" ", format="%d\n")
                    arr.flatten(order='f').tofile(wfile, sep=" ", format="%.9e\n")                    

            else: # regular grid
                # determine header information 
                nrcells = self.grid.nx * self.grid.ny * self.grid.nz
                if binary:
                    if ndim == 3:
                        hdr = np.array([1, 8, nrcells], dtype=np.int64)
                    elif ndim == 4:
                        hdr = np.array([1, 8, nrcells, arr.shape[3]], dtype=np.int64)
                    else:
                        raise ValueError('Incorrect shape')
                else:
                    if ndim == 3:
                        hdr = np.array([1, nrcells], dtype=np.int64)
                    elif ndim == 4:
                        hdr = np.array([1, nrcells, arr.shape[3]], dtype=np.int64)
                    else:
                        raise ValueError('Incorrect shape')

                # make the first index to denote the direction
                arr = np.moveaxis(arr, -1, 0)

                # write header and data 
                if binary:
                    hdr.tofile(wfile)
                    arr.flatten(order='f').tofile(wfile)
                else:
                    hdr.tofile(wfile, sep="\n", format='%d')
                    wfile.write('\n')
                    arr = np.reshape(arr, [3, np.prod(arr.shape[1:])], order='f')
                    np.savetxt(wfile, arr.T, fmt='%.9e')

    def _vectorfieldReader(self, fname, binary=False, ndim=3):
        """ read ordingary vector field data 

        Parameters
        ----------
        fname : str
            name of the file
        binary : bool
            binary file or not

        Returns
        -------
        arr : ndarray 
        """
        # begin reading
        with open(fname, 'r') as rfile:
            # if we have an octree grid
            if isinstance(self.grid, radmc3dOctree):
                # ! this part is not tested yet !
                # read header information 
                if binary:
                    hdr = np.fromfile(rfile, count=3, dtype=np.int64)
                    iformat = hdr[0]
                    precis = hdr[1]
                    nLeaf = hdr[2]
                    if precis == 4:
                        dtype = np.float32
                    else:
                        dtype = np.float64
                else:
                    hdr = np.fromfile(rfile, count=2, dtype=np.int64)
                    iformat = hdr[0]
                    nLeaf = hdr[1]
                    dtype = np.float64

                if self.grid.nLeaf != nLeaf:
                    raise ValueError('Number of cells in ' + fname + ' is different from that in amr_grid.inp'
                                             + ' nr cells in ' + fname + ' : ' + ("%d" % nLeaf) + '\n '
                                             + ' nr of cells in amr_grid.inp : '
                                             + ("%d" % self.grid.nLeaf))

                # read data 
                if binary:
                    arr = np.fromfile(rfile, count=-1, sep='', dtype=dtype)
                else:
                    arr = np.fromfile(rfile, count=-1, sep=' ', dtype=dtype)

                arr = np.reshape(arr, [self.nLeaf, 3], order='F')

            else: # regular grid
                # read the header information 
                if binary:
                    if ndim == 3:
                        cnt = 3
                    else:
                        cnt = 4
                    hdr = np.fromfile(rfile, count=cnt, sep='', dtype=np.int64)
                    iformat = hdr[0]
                    precis = hdr[1]
                    nrcells = hdr[2]
                    if ndim == 4:
                        ndust = hdr[3]

                    if precis == 4:
                        dtype = np.float32
                    else:
                        dtype = np.float64
                else:
                    if ndim == 3:
                        cnt = 2
                    else:
                        cnt = 3

                    hdr = np.fromfile(rfile, count=cnt, sep=" ", dtype=np.int64)
                    iformat = hdr[0]
                    nrcells = hdr[1]
                    if ndim == 4:
                        ndust = hdr[2]

                    dtype = np.float64

                # read data 
                if binary:
                    arr = np.fromfile(rfile, count=-1, sep='', dtype=dtype)
                else:
                    arr = np.fromfile(rfile, count=-1, sep=' ', dtype=dtype)

                # first reconstruct the array 
                if ndim == 3:
                    dim = [3,self.grid.nx, self.grid.ny, self.grid.nz]
                else:
                    dim = [3,self.grid.nx, self.grid.ny, self.grid.nz, ndust]

                arr = np.reshape(arr, dim, order='f')

                # want the last index to denote the direction
                arr = np.moveaxis(arr, 0, -1)

        return arr 

    def getTauOneDust(self, idust=0, axis='', kappa=0.):
        """Calculates the optical depth of a single dust species along any given combination of the axes.

        Parameters
        ----------

        idust : int
                Index of the dust species whose optical depth should be calculated

        axis  : str
                Name of the axis/axes along which the optical depth should be calculated
                (e.g. 'x' for the first dimension or 'xyz' for all three dimensions)

        kappa : float
                Mass extinction coefficients of the dust species at the desired wavelength

        Returns
        -------

        Returns a dictionary with the following keys

            taux  : ndarray
                    optical depth along the first dimension
            tauy  : ndarray
                    optical depth along the second dimension

            (tauz is not yet implemented)
        """

        # Check along which axis should the optical depth be calculated
        do_taux = False
        do_tauy = False

        if 'x' in axis:
            do_taux = True
        if 'y' in axis:
            do_tauy = True

        # Calculate the optical depth along the x-axis (r in spherical coordinates)
        if do_taux:
            taux = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz], dtype=np.float64)
            diff_x = self.grid.xi[1:] - self.grid.xi[:-1]
            taux[0, :, :] = self.rhodust[0, :, :, idust] * kappa * diff_x[0]
            for ix in range(1, self.grid.nx):
                taux[ix, :, :] = taux[ix - 1, :, :] + self.rhodust[ix, :, :, idust] * kappa * diff_x[ix]
        else:
            taux = [-1.]

        # Calculate the optical depth along the theta in spherical coordinates
        # Warning the formulation below is valid only in spherical coordinate sytem

        dum_x = np.zeros([self.grid.nx, self.grid.nz], dtype=np.float64)
        for iz in range(self.grid.nz):
            dum_x[:, iz] = self.grid.x

        if do_tauy:
            tauy = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz], dtype=np.float64)
            diff_y = self.grid.yi[1:] - self.grid.yi[:-1]
            tauy[:, 0, :] = self.rhodust[:, 0, :, idust] * kappa * diff_y[0] * dum_x
            for iy in range(1, self.grid.ny):
                tauy[:, iy, :] = tauy[:, iy - 1, :] + self.rhodust[:, iy, :, idust] * kappa * diff_y[iy] * dum_x
        else:
            tauy = [-1.]

        return {'taux': taux, 'tauy': tauy}

    def getTau(self, idust=None, axis='xy', wav=0., kappa=None, old=False):
        """Calculates the optical depth along any given combination of the axes.

        Parameters
        ----------

        idust : list
                List of dust component indices whose optical depth should be calculated
                If multiple indices are set the total optical depth is calculated summing
                over all dust species in idust

        axis  : str
                Name of the axis/axes along which the optical depth should be calculated
                (e.g. 'x' for the first dimension or 'xyz' for all three dimensions)

        wav   : float
                Wavelength at which the optical depth should be calculated

        kappa : list, tuple
                If set it should be a list of mass extinction coefficients at the desired wavelength
                The number of elements in the list should be equal to that in the idust keyword

        old   : bool, optional
                If set to True the file format of the previous, 2D version of radmc will be used
        """

        # Check if the input idust indices can be found in rhoudust
        if len(self.rhodust.shape) == 3:
            ndust = 1
        else:
            ndust = self.rhodust.shape[3]

        if idust is None:
            idust = np.arange(ndust)

        if max(idust) > ndust:
            raise ValueError(' There are less number of dust species than some of the indices in idust')

        scatmat = None
        # If the kappa keyword is set it should be used during the optical depth calculation
        if kappa is not None:
            # Safety check
            if len(kappa) != len(idust):
                raise ValueError(' The number of kappa values should be identical to the number of specified '
                                 + 'dust species ')
        else:
            if not old:
                # Read the master opacity file to get the dustkappa file name extensions
                dum = radmc3dDustOpac()
                mo = dum.readMasterOpac()
                dummy_ext = mo['ext']
                scatmat = mo['scatmat']
                if len(dummy_ext) <= max(idust):
                    raise ValueError('There are less dust species specified in dustopac.inp than '
                                     + 'some of the specified idust indices')
                else:
                    ext = [dummy_ext[i] for i in idust]

        if 'x' in axis:
            self.taux = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz], dtype=np.float64)
        if 'y' in axis:
            self.tauy = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz], dtype=np.float64)

        for i in idust:
            if kappa is None:
                if old:
                    opac = radmc3dDustOpac()
                    opac.readOpac(ext=[("%d" % (i + 1))], old=True)
                    # opac = readOpac(ext=[("%d" % (i + 1))], old=True)
                else:
                    opac = radmc3dDustOpac()
                    opac.readOpac(ext=ext[i], scatmat=scatmat)
                    #opac = readOpac(ext=ext[i], scatmat=scatmat)

                if not opac.ext:
                    return -1
                else:
                    kabs = 10. ** np.interp(np.log10(np.array(wav)), np.log10(opac.wav[0]), np.log10(opac.kabs[0]))
                if opac.ksca[0][0] > 0:
                    ksca = 10. ** np.interp(np.log10(np.array(wav)), np.log10(opac.wav[0]), np.log10(opac.ksca[0]))
                else:
                    ksca = np.array(kabs) * 0.

                print('Opacity at ' + ("%.2f" % wav) + 'um : ', kabs + ksca)
                dum = self.getTauOneDust(i, axis=axis, kappa=kabs + ksca)
            else:
                dum = self.getTauOneDust(i, axis=axis, kappa=kappa[i])

            if 'x' in axis:
                self.taux = self.taux + dum['taux']
            if 'y' in axis:
                self.tauy = self.tauy + dum['tauy']

    def getGasMass(self, mweight=2.3, rhogas=False):
        """Calculates the gas mass in radmc3dData.ndens_mol or radmc3dData.rhogas

        Parameters
        ----------
            mweight   : float
                        Molecular weight [atomic mass unit / molecule, i.e. same unit as mean molecular weight]

            rhogas    : bool, optional
                        If True the gas mass will be calculated from radmc3dData.rhogas, while if set to False
                        the gas mass will be calculated from radmc3dData.ndens_mol. The mweight parameter is only
                        required for the latter.

        Returns
        -------
            A single float being the gas mass in gramm
        """

        vol = self.grid.getCellVolume()
        if not rhogas:
            #
            # I'm not sure if this is the right way of doing it but right now I don't have a better idea
            #
            if isinstance(self.grid, radmc3dOctree):
                return (vol * self.ndens_mol[:, 0] * mweight * nc.mp).sum()
            else:
                return (vol * self.ndens_mol[:, :, :, 0] * mweight * nc.mp).sum()

        else:
            if isinstance(self.grid, radmc3dOctree):
                return (vol * self.rhogas[:, 0]).sum()
            else:
                return (vol * self.rhogas[:, :, :, 0]).sum()


    def getDustMass(self, idust=-1):
        """Calculates the dust mass in radmc3dData.rhodust

        Parameters
        ----------
            idust   : int
                      Dust index whose dust should be calculated. If it is set to -1 (default) the total
                      dust mass is calculated summing up all dust species

        Returns
        -------
            A single float being the dust mass in gramm
        """

        vol = self.grid.getCellVolume()
        if idust > 0:
            #
            # I'm not sure if this is the right way of doing it but right now I don't have a better idea
            #
            if isinstance(self.grid, radmc3dOctree):
                dmass = (vol * self.rhodust[:, idust]).sum()
            else:
                dmass = (vol * self.rhodust[:, :, :, idust]).sum()
        else:
            dmass = 0.
            if isinstance(self.grid, radmc3dOctree):
                for i in range(self.rhodust.shape[1]):
                    dmass += (vol * self.rhodust[:, i]).sum()
            else:
                for i in range(self.rhodust.shape[3]):
                    dmass += (vol * self.rhodust[:, :, :, i]).sum()

        return dmass

    def readDustDens(self, fname='', fdir=None, binary=False, old=False, octree=False):
        """Reads the dust density.

        Parameters
        ----------

        fname   : str, optional
                  Name of the file that contains the dust density. If omitted 'dust_density.inp' is used
                  (or if binary=True the 'dust_density.binp' is used).

        binary  : bool, optional
                  If true the data will be read in binary format, otherwise the file format is ascii

        old     : bool, optional
                  If set to True the file format of the previous, 2D version of radmc will be used

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """

        if self.grid is None:
            if octree:
                self.grid = radmc3dOctree()
            else:
                self.grid = radmc3dGrid()
                self.grid.readGrid(old=old)

        #
        # Read radmc3d output
        #
        if not old:

            if fname == '':
                if binary:
                    fname = 'dust_density.binp'
                else:
                    fname = 'dust_density.inp'

            if fdir is not None:
                fname = os.path.join(fdir, fname)

            print('Reading '+fname)
            self.rhodust = self._scalarfieldReader(fname=fname, binary=binary, octree=octree, ndim=4)
        #
        # Read the output of the previous 2d version of the code
        #
        else:
            fname = 'dustdens.inp'
            print('Reading '+fname)

            data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
            # 4 element header: Nr of dust species, nr, ntheta, ?
            hdr = np.array(data[:4], dtype=np.int64)
            data = np.reshape(data[4:], [hdr[1], hdr[2], 1, hdr[0]])
            self.rhodust = np.zeros([hdr[1], hdr[2]*2, 1, hdr[0]], dtype=np.float64)
            self.rhodust[:, :hdr[2], 0, :] = data[:, :, 0, :]
            self.rhodust[:, hdr[2]:, 0, :] = data[:, ::-1, 0, :]

    def readDustTemp(self, fname='', fdir=None, binary=False, octree=False, old=False):
        """Reads the dust temperature.

        Parameters
        ----------

        fname   : str, optional
                  Name of the file that contains the dust temperature.

        binary  : bool, optional
                  If true the data will be read in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR

        old     : bool, optional
                  If True dust temperature will be written in the old RADMC format
        """

        if self.grid is None:
            if octree:
                self.grid = radmc3dOctree()
            else:
                self.grid = radmc3dGrid()
                self.grid.readGrid(old=old)

        if not old:
            if binary:
                if fname == '':
                    fname = 'dust_temperature.bdat'
                if fdir is not None:
                    fname = fdir + '/' + fname
            else:
                if fname == '': 
                    fname = 'dust_temperature.dat'
                if fdir is not None:
                    fname = fdir + '/' + fname

            print('Reading '+fname)

            self.dusttemp = self._scalarfieldReader(fname=fname, binary=binary, octree=octree, ndim=4)
        else:

            fname = 'dusttemp_final.dat'
            print('Reading '+fname)

            data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
            # 4 element header: Nr of dust species, nr, ntheta, ?
            hdr = np.array(data[:4], dtype=np.int64)
            data = data[4:]
            self.dusttemp = np.zeros([hdr[1], hdr[2]*2, 1, hdr[0]], dtype=np.float64)

            ncell = hdr[1] * hdr[2]
            for idust in range(hdr[0]):
                # We need to get rid of the dust species index that is written inbetween the
                # temperature arrays
                data = data[1:]
                self.dusttemp[:, :hdr[2], 0, idust] = np.reshape(data[:ncell], [hdr[1], hdr[2]])
            self.dusttemp[:, hdr[2]:, 0, :] = self.dusttemp[:, :hdr[2], 0, :][:, ::-1, :]

    def readGasDens(self, fname='', fdir=None, binary=False):
        """ read the gas density 
        """
        
        if fname == '':
            if binary:
                fname = 'gas_density.binp'
            else:
                fname = 'gas_density.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        print('Reading '+fname)
        self.rhogas = self._scalarfieldReader(fname=fname, binary=binary, octree=octree, ndim=3)

    def readGasVel(self, fname='', fdir=None, binary=False, octree=False):
        """Reads the gas velocity.

        Parameters
        -----------

        fname : str, optional
                Name of the file that contains the gas velocity
                If omitted 'gas_velocity.inp' (if binary=True 'gas_velocity.binp')is used.

        binary : bool
                If true the data will be read in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """

        if self.grid is None:
            if octree:
                self.grid = radmc3dOctree()
            else:
                self.grid = radmc3dGrid()
                self.grid.readGrid(old=False)

        if binary:
            if fname == '':
                fname = 'gas_velocity.binp'

            print('Reading '+fname)
            if os.path.isfile(fname):
                # If we have an octree grid
                with open(fname, 'r') as rfile:
                    if isinstance(self.grid, radmc3dOctree):
                        hdr = np.fromfile(rfile, count=3, dtype=np.int64)
                        if hdr[2] != self.grid.nLeaf:
                            raise ValueError('Number of cells in ' + fname + ' is different from that in amr_grid.inp'
                                             + ' nr cells in ' + fname + ' : ' + ("%d" % hdr[2]) + '\n '
                                             + ' nr of cells in amr_grid.inp : '
                                             + ("%d" % self.grid.nLeaf))

                        if hdr[1] == 8:
                            self.gasvel = np.fromfile(rfile, count=-1, dtype=np.float64)
                        elif hdr[1] == 4:
                            self.gasvel = np.fromfile(rfile, count=-1, dtype=np.float32)
                        else:
                            raise TypeError(
                                'Unknown datatype/precision in ' + fname + '. RADMC-3D binary files store 4 byte '
                                + 'floats or 8 byte doubles. The precision in the file header is '
                                + ("%d" % hdr[1]))
                        self.gasvel = np.reshape(self.gasvel, [self.nLeaf, 3], order='f')

                    else:
                        hdr = np.fromfile(rfile, count=3, dtype=np.int64)
                        if hdr[2] != self.grid.nx * self.grid.ny * self.grid.nz:
                            raise ValueError('Number of grid cells in ' + fname + ' is different from that in amr_grid.inp '
                                             + ' nr cells in ' + fname + ' : ' + ("%d" % hdr[2]) + '\n '
                                             + ' nr of cells in amr_grid.inp : '
                                             + ("%d" % (self.grid.nx * self.grid.ny * self.grid.nz)))

                        if hdr[1] == 8:
                            self.gasvel = np.fromfile(rfile, count=-1, dtype=np.float64)
                        elif hdr[1] == 4:
                            self.gasvel = np.fromfile(rfile, count=-1, dtype=float)
                        else:
                            raise TypeError(
                                'Unknown datatype/precision in ' + fname + '. RADMC-3D binary files store 4 byte '
                                + 'floats or 8 byte doubles. The precision in the file header is '
                                + ("%d" % hdr[1]))
                        self.gasvel = np.reshape(self.gasvel, [self.grid.nz, self.grid.ny, self.grid.nx, 3])
                        self.gasvel = np.swapaxes(self.gasvel, 0, 2)
            else:
                raise FileNotFoundError(fname + 'was not found')

        else:

            if fname == '':
                fname = 'gas_velocity.inp'
            if fdir is not None:
                fname = fdir + '/' + fname

            if os.path.isfile(fname):

                print('Reading ' + fname)
                with open(fname, 'r') as rfile:
                    # Two element header 1 - iformat, 2 - Nr of cells
                    hdr = np.fromfile(rfile, count=2, sep=" ", dtype=np.float64)
                    #hdr = np.array(data[:2], dtype=np.int64)
                    if octree:
                        if self.grid.nLeaf != hdr[1]:
                            msg = 'Number of cells in ' + fname + ' is different from that in amr_grid.inp'\
                                  + ' nr cells in ' + fname + ' : ' + ("%d" % hdr[1]) + '\n '\
                                  + ' nr of cells in amr_grid.inp : ' +  ("%d" % self.grid.nLeaf)
                            raise RuntimeError(msg)
                        data = np.fromfile(rfile, count=-1, sep=" ", dtype=np.float64)
                        self.gasvel = np.reshape(data, [hdr[1], 3])
                    else:
                        if (self.grid.nx * self.grid.ny * self.grid.nz) != hdr[1]:
                            msg = 'Number of grid cells in ' + fname + ' is different from that in amr_grid.inp ' \
                                  + ' nr cells in ' + fname + ' : ' + ("%d" % hdr[1]) + '\n '\
                                  + ' nr of cells in amr_grid.inp : '\
                                  + ("%d" % (self.grid.nx * self.grid.ny * self.grid.nz))
                            raise RuntimeError(msg)
                        data = np.fromfile(rfile, count=-1, sep=" ", dtype=np.float64)
# the reshape method doesn't work
#                        self.gasvel = np.reshape(data, [self.grid.nx, self.grid.ny, self.grid.nz, 3])
                        self.gasvel = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 3], dtype=np.float64)
                        ik = 0
                        for iz in range(self.grid.nz):
                            for iy in range(self.grid.ny):
                                for ix in range(self.grid.nx):
                                    for iv in range(3):
                                        self.gasvel[ix,iy,iz,iv] = data[ik]
                                        ik = ik + 1

            else:
                raise FileNotFoundError(fname + 'was not found')

        return True

    def readDustAlign(self, fname='', fdir=None, binary=False):
        """Reads the Dust Alignment.

        Parameters
        -----------

        fname : str, optional
                Name of the file that contains the gas velocity
                If omitted 'grainalign_dir.inp' (if binary=True 'grainalign_dir.binp')is used.

        binary : bool
                If true the data will be read in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """
        # determine default file name
        if fname == '':
            if binary:
                fname = 'grainalign_dir.binp'
            else:
                fname = 'grainalign_dir.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        self.alvec = self._vectorfieldReader(fname, binary=binary, ndim=4)

    def readBfield(self, fname='', fdir=None, binary=False):
        """Reads the magnetic field

        Parameters
        -----------

        fname : str, optional
                Name of the file that contains the gas velocity
                If omitted 'Bfield.inp' (if binary=True 'Bfield.binp')is used.

        binary : bool
                If true the data will be read in binary format, otherwise the file format is ascii

        """
        # check if grid exists 
        if self.grid is None:
            if octree:
                self.grid = radmc3dOctree()
            else:
                self.grid = radmc3dGrid()
                self.grid.readGrid(old=False)

        # determine default file name
        if fname == '':
            if binary:
                fname = 'Bfield.binp'
            else:
                fname = 'Bfield.inp'

        # add on fdir
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        self.bfield = self._vectorfieldReader(fname, binary=binary, ndim=3)

    def readVTurb(self, fname='', fdir=None, binary=True, octree=False):
        """Reads the turbulent velocity field.

        Parameters
        ----------

        fname   : str, optional
                  Name of the file that contains the turbulent velocity field
                  If omitted 'microturbulence.inp' (if binary=True 'microturbulence.binp') is used.

        binary  : bool
                  If true the data will be read in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """
        if self.grid is None:
            if octree:
                self.grid = radmc3dOctree()
            else:
                self.grid = radmc3dGrid()
                self.grid.readGrid(old=False)

        print('Reading microturbulence')

        if binary:
            if fname == '':
                fname = 'microturbulence.binp'
            if fdir is not None:
                fname = fdir + '/' + fname
        else:
            if fname == '':
                fname = 'microturbulence.inp'

            if fdir is not None:
                fname = fdir + '/' + fname

        self.vturb = self._scalarfieldReader(fname=fname, binary=binary, octree=octree, ndim=3)
        if octree:
            self.vturb = np.squeeze(self.vturb)

        return True

    def readGasDens(self, ispec='', fdir=None, binary=True, octree=False):
        """Reads the gas density.

        Parameters
        ----------

        ispec   : str
                  File name extension of the 'numberdens_ispec.inp' (or if binary=True 'numberdens_ispec.binp') file.

        binary  : bool
                  If true the data will be read in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """

        if self.grid is None:
            if octree:
                self.grid = radmc3dOctree()
            else:
                self.grid = radmc3dGrid()
                self.grid.readGrid(old=False)

        if binary:
            fname = 'numberdens_' + ispec + '.binp'

        else:
            fname = 'numberdens_' + ispec + '.inp'

        if fdir is not None:
            fname = fdir + '/' + fname

        print('Reading gas density (' + fname + ')')
        self.ndens_mol = self._scalarfieldReader(fname=fname, binary=binary, octree=octree, ndim=3)
        if octree:
            self.ndens_mol = np.squeeze(self.ndens_mol)

        return True

    def readGasTemp(self, fname='', binary=True, octree=False):
        """Reads the gas temperature.

        Parameters
        ----------

        fname   : str,optional
                  Name of the file that contains the gas temperature. If omitted 'gas_temperature.inp'
                  (or if binary=True 'gas_tempearture.binp') is used.

        binary  : bool
                  If true the data will be read in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """

        if self.grid is None:
            if octree:
                self.grid = radmc3dOctree()
            else:
                self.grid = radmc3dGrid()
                self.grid.readGrid(old=False)

        print('Reading gas temperature')

        if binary:
            if fname == '':
                fname = 'gas_temperature.binp'
        else:
            if fname == '':
                fname = 'gas_temperature.inp'

        self.gastemp = self._scalarfieldReader(fname=fname, binary=binary, octree=octree, ndim=3)
        if octree:
            self.gastemp = np.squeeze(self.gastemp)

        return True

    def readViscousHeating(self, fname='', fdir=None, binary=True, octree=False):
        """Reads the heatsource.inp

        Parameters
        ----------

        fname   : str,optional
                  Name of the file that contains the gas temperature. If omitted 'gas_temperature.inp'
                  (or if binary=True 'gas_tempearture.binp') is used.

        binary  : bool
                  If true the data will be read in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """
        if self.grid is None:
            if octree:
                self.grid = radmc3dOctree()
            else:
                self.grid = radmc3dGrid()
                self.grid.readGrid(old=False)

        print('Reading heatsource.inp')

        if binary:
            if fname == '':
                fname = 'heatsource.binp'
                if fdir is not None:
                    fname= fdir + fname
        else:
            if fname == '':
                fname = 'heatsource.inp'
                if fdir is not None:
                    fname= fdir + fname

        self.qvis = self._scalarfieldReader(fname=fname, binary=binary, octree=octree, ndim=3)
        if octree:
            self.qvis = np.squeeze(self.qvis)

        return True

    def getHeatingLum(self):
        """ Calculates heating luminosity
        Return
        ------
        totlum : float
                 Total heating luminosity in cgs
        """
        vol = self.grid.getCellVolume()
        lum = np.squeeze(vol) * np.squeeze(self.qvis)
        tot_lum = lum.sum()
        return tot_lum
 
    def readRadiationField(self, fname='', fdir=None, binary=False):
        """ read the radiation field
        """
        if fname == '':
            if binary:
                fname = 'mean_intensity.bout'
            else:
                fname = 'mean_intensity.out'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        with open(fname, 'r') as rfile:
            # read header 
            if binary:
                # iformat, precision, nrcells, nfreq
                hdr = np.fromfile(rfile, count=4, dtype=np.int64, sep='')
                iformat = hdr[0] 
                precis = hdr[1]
                nrcells = hdr[2]
                self.grid.radnfreq = hdr[3]
                if precis == 4:
                    dtype = np.float32
                else:
                    dtype = np.float64
            else:
                # iformat, nrcells, nfreq
                hdr = np.fromfile(rfile, count=3, dtype=np.int64, sep=' ')
                iformat = hdr[0]
                nrcells = hdr[1]
                self.grid.radnfreq = hdr[2]
                dtype = np.float64

            self.grid.radnwav = self.grid.radnfreq

            # read frequency 
            if binary:
                freq = np.fromfile(rfile, count=self.grid.radnfreq, dtype=dtype, sep='')
            else:
                freq = np.fromfile(rfile, count=self.grid.radnfreq, dtype=dtype, sep=' ')

            self.grid.radfreq = freq
            self.grid.radwav = nc.cc * 1e4 / freq           
 
            # read data
            if binary:
                data = np.fromfile(rfile, count=-1, dtype=dtype, sep='')
            else:
                data = np.fromfile(rfile, count=-1, dtype=np.float64, sep=' ')

            # reformat
            data = np.reshape(data, [self.grid.nx, self.grid.ny, self.grid.nz, self.grid.radnfreq], order='F')

        self.radfield = data

    def readFluxField(self, fname='', fdir=None, binary=False):
        """ read the flux field. basically the same as mean_intensity.out, but with directions
        """
        if fname == '':
            if binary:
                fname = 'flux_field.bout'
            else:
                fname = 'flux_field.out'
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        if isinstance(self.grid, radmc3dOctree):
            raise ValueError('octree for readfluxfield not implemented yet')

        with open(fname, 'r') as rfile:
            if binary:
                # iformat, precision, nrcells, nfreq
                hdr = np.fromfile(rfile, count=4, dtype=np.int64, sep='')
                iformat = hdr[0]
                precis = hdr[1]
                nrcells = hdr[2]
                self.grid.radnfreq = hdr[3]
                if precis == 4:
                    dtype = np.float32
                else:
                    dtype = np.float64

            else:
                # iformat, nrcells, nfreq
                hdr = np.fromfile(rfile, count=3, dtype=np.int64, sep=sep)
                iformat = hdr[0]
                nrcells = hdr[1]
                self.grid.radnfreq = hdr[2]
                dtype = np.float64

            self.grid.radnwav = self.grid.radnfreq

            # read frequency 
            if binary:
                freq = np.fromfile(rfile, count=self.grid.radnfreq, dtype=dtype, sep='')
            else:
                freq = np.fromfile(rfile, count=self.grid.radnfreq, dtype=dtype, sep=' ')

            self.grid.radfreq = freq
            self.grid.radwav = nc.cc * 1e4 / freq

            # read data
            if binary:
                data = np.fromfile(rfile, count=-1, dtype=dtype, sep='')
            else:
                data = np.fromfile(rfile, count=-1, dtype=dtype, sep=' ')

            # reformat
            data = np.reshape(data, [3,self.grid.nx, self.grid.ny, self.grid.nz, self.grid.radnfreq], order='F')

            # want the last index to denote the direction
            data = np.moveaxis(data, 0, -1)

        self.fluxfield = data 

    def writeDustDens(self, fname='', binary=False, old=False, octree=False,fdir=None):
        """Writes the dust density.

        Parameters
        ----------

        fname   : str, optional
                  Name of the file into which the dust density should be written. If omitted 'dust_density.inp' is used.

        binary  : bool
                  If true the data will be written in binary format, otherwise the file format is ascii

        old     : bool, optional
                  If set to True the file format of the previous, 2D version of radmc will be used

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """

        #
        # Write dust density for radmc3d
        #
        if not old:
            if fname == '':
                if binary:
                    fname = 'dust_density.binp'
                else:
                    fname = 'dust_density.inp'

            if fdir is not None:
                fname = os.path.join(fdir, fname)

            print('Writing ' + fname)
#            if self.grid.crd_sys is 'sph':
#                tot_dmass = self.getDustMass(idust=-1)
#                print(' - Total dust mass: %f (Msun)'%(tot_dmass/nc.ms))

            if octree:
                self._scalarfieldWriter(data=self.grid.convArrTree2Leaf(self.rhodust), fname=fname, binary=binary,
                                        octree=True)
            else:
                self._scalarfieldWriter(fname, self.rhodust, binary=binary)

        #
        # Write dust density for the previous 2D version of the code
        #
        else:
            if self.rhodust.shape[2] > 1:
                raise ValueError('Wrong dimensions for dust density. RADMC (the predecessor code to RADMC-3D is '
                                 + 'strictly 2D, yet the dust density to be written in the RADMC format is 3D.')

            if fname == '':
                fname = 'dustdens.inp'

            with open(fname, 'w') as wfile:

                ntheta = int(self.grid.ny / 2)
                wfile.write("%d %d %d 1\n" % (self.rhodust.shape[3], self.grid.nx, ntheta))
                wfile.write(" \n")
                for idust in range(self.rhodust.shape[3]):
                    for ix in range(self.grid.nx):
                        for iy in range(ntheta):
                            wfile.write("%.7e\n" % self.rhodust[ix, iy, 0, idust])

    def writeDustTemp(self, fname='', binary=False, octree=False, fdir=None):
        """Writes the dust density.

        Parameters
        ----------

        fname   : str, optional
                Name of the file into which the dust density should be written. If omitted 'dust_density.inp' is used.

        binary  : bool
                  If true the data will be written in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """
        if fname == '':
            if binary:
                fname = 'dust_temperature.bdat'
            else:
                fname = 'dust_temperature.dat'
            if fdir is not None:
                if fdir[-1] is '/':
                    fname = fdir + fname
                else:
                    fname = fdir + '/' + fname

        print('Writing ' + fname)
        if octree:
            self._scalarfieldWriter(data=self.grid.convArrTree2Leaf(self.dusttemp), fname=fname, binary=binary,
                                    octree=True)
        else:
            self._scalarfieldWriter(fname, self.dusttemp, binary=binary)

    def writeGasDens(self, fname='', ispec='', binary=True, fdir=None):
        """Writes the gas density.

        Parameters
        ----------

        fname   : str, optional
                  Name of the file into which the data will be written. If omitted "numberdens_xxx.inp" and
                  "numberdens_xxx.binp" will be used for ascii and binary format, respectively
                  (xxx is the name of the molecule).

        ispec   : str
                  File name extension of the 'numberdens_ispec.inp' (if binary=True 'numberdens_ispec.binp')
                  file into which the gas density should be written

        binary  : bool
                  If true the data will be written in binary format, otherwise the file format is ascii

        """
        # specify the species
        if ispec == '':
            raise ValueError('Unknown ispec. Without knowing the name of the gas species the gas density '
                             + 'cannot be written, as the filename "numberdens_ispec.inp" cannot be generated.')

        # determine a default file name
        if fname == '':
            fname = 'numberdens_%s'%ispec
            if binary:
                fname += '.binp'
            else:
                fname += '.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        print('Writing ' + fname)
        if isinstance(self.grid, radmc3dOctree):
            self._scalarfieldWriter(data=self.grid.convArrTree2Leaf(self.ndens_mol), fname=fname, binary=binary)
        else:
            self._scalarfieldWriter(data=self.ndens_mol, fname=fname, binary=binary)

    def writeGasTemp(self, fname='', binary=True, octree=False, fdir=None):
        """Writes the gas temperature.

        Parameters
        ----------

        fname : str, optional
                Name of the file into which the gas temperature should be written. If omitted
                'gas_temperature.inp' (if binary=True 'gas_tempearture.binp') is used.

        binary : bool
                If true the data will be written in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """
        if fname == '':
            if binary:
                fname = 'gas_temperature.binp'
            else:
                fname = 'gas_temperature.inp'
            if fdir is not None:
                if fdir[-1] is '/':
                    fname = fdir + fname
                else:
                    fname = fdir + '/' + fname

        print('Writing ' + fname)

        if octree:
            self._scalarfieldWriter(data=self.grid.convArrTree2Leaf(self.gastemp), fname=fname, binary=binary,
                                    octree=True)
        else:
            self._scalarfieldWriter(data=self.gastemp, fname=fname, binary=binary, octree=False)

    def writeViscousHeating(self, fname='',binary=False, octree=False, fdir=None):
        """Writes the heating 

        Parameters
        ----------

        fname   : str, optional
                  Name of the file into which the data will be written. If omitted "heatsource.inp" will be used

        binary  : bool
                  If true the data will be written in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """
        if fname == '':
            fname = 'heatsource.inp'
            if fdir is not None:
                if fdir[-1] is '/':
                    fname = fdir + fname
                else:
                    fname = fdir + '/' + fname

        vol = self.grid.getCellVolume()
        lum = vol * self.qvis
        tot_lum = lum.sum()
        print('Writing ' + fname)
        print(' - Total luminosity: %f (Lsun)'%(tot_lum/nc.ls))
        if octree:
            self._scalarfieldWriter(data=self.grid.convArrTree2Leaf(self.qvis), fname=fname, binary=binary,
                                    octree=True)
        else:
            self._scalarfieldWriter(data=self.qvis, fname=fname, binary=binary, octree=False)

    def writeGasVel(self, fname='', binary=False, octree=False, fdir=None):
        """Writes the gas velocity.

        Parameters
        ----------

        fname  : str, optional
                Name of the file into which the gas temperature should be written.
                If omitted 'gas_velocity.inp' (if binary=True 'gas_velocity.binp') is used.

        binary : bool
                If true the data will be written in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """

        if fname == '':
            if binary:
                fname = 'gas_velocity.binp'
            else:
                fname = 'gas_velocity.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        self._vectorfieldWriter(fname, self.gasvel, binary=binary, ndim=3)

        """ keep this for reference. can be deleted if checked later on 
        if binary:
            if fname == '':
                fname = 'gas_velocity.binp'

            print('Writing ' + fname)
            with open(fname, 'w') as wfile:

                if octree:
                    #
                    # Check if the gas velocity contains the full tree or only the leaf nodes
                    #
                    if self.gasvel.shape[0] == self.grid.nLeaf:
                        hdr = np.array([1, 8, self.gasvel.shape[0]], dtype=np.int64)
                        hdr.tofile(wfile)
                        self.gasvel.flatten(order='f').tofile(wfile)
                    else:
                        hdr = np.array([1, 8, self.grid.nLeaf], dtype=np.int64)
                        hdr.tofile(wfile)
                        dummy = self.grid.convArrTree2Leaf(self.gasvel)
                        dummy.flatten(order='f').tofile(wfile)
                        # self.gasvel.flatten(order='f').tofile(wfile)

                else:
                    hdr = np.array([1, 8, self.grid.nx * self.grid.ny * self.grid.nz], dtype=int)
                    hdr.tofile(wfile)
                    # Now we need to change the axis orders since the Ndarray.tofile function writes the
                    # array always in C-order while we need Fortran-order to be written
#                    self.gasvel = np.swapaxes(self.gasvel, 0, 2)
#                    self.gasvel.tofile(wfile)

                    # Switch back to the original axis order
#                    self.gasvel = np.swapaxes(self.gasvel, 0, 2)

                    self.gasvel.flatten(order='f').tofile(wfile)

        else:
            if fname == '':
                fname = 'gas_velocity.inp'
                if fdir is not None:
                    fname = os.path.join(fdir, fname)

            print('Writing ' + fname)

            # If we have an octree grid
            if octree:
                #
                # Check if the gas velocity contains the full tree or only the leaf nodes
                #
                if self.gasvel.shape[0] == self.grid.nLeaf:
                    hdr = '1\n'
                    hdr += ("%d\n" % self.gasvel.shape[0])
                    try:
                        np.savetxt(fname, self.gasvel, fmt="%.9e %.9e %.9e", header=hdr, comments='')
                    except Exception as e:
                        print(e)
                else:
                    hdr = '1\n'
                    hdr += ("%d\n" % self.grid.nLeaf)
                    dummy = self.grid.convArrTree2Leaf(self.gasvel)
                    try:
                        np.savetxt(fname, dummy, fmt="%.9e %.9e %.9e", header=hdr, comments='')
                    except Exception as e:
                        print(e)

            else:
                # hdr = "1\n"
                # hdr += ('%d'%(self.grid.nx*self.grid.ny*self.grid.nz))
                # np.savetxt(fname, self.gasvel, fmt="%.9 %.9 %.9", header=hdr, comments='')

                # self.gasvel up to here are all good and normal

                nrcells = self.grid.nx * self.grid.ny * self.grid.nz

                dummy = np.moveaxis(self.gasvel, -1, 0)
                dummy = np.reshape(dummy, [3, nrcells], order='f')

                with open(fname, 'w') as wfile:

                    wfile.write('%d\n' % 1)
                    wfile.write('%d\n' % nrcells )

                    np.savetxt(wfile, dummy.T, fmt='%.9e')

#                    for iz in range(self.grid.nz):
#                        for iy in range(self.grid.ny):
#                            for ix in range(self.grid.nx):
#                                wfile.write("%.9e %.9e %.9e\n" % (self.gasvel[ix, iy, iz, 0], self.gasvel[ix, iy, iz, 1],
#                                                               self.gasvel[ix, iy, iz, 2]))
        """
    def writeDustAlign(self, fname='', binary=False, fdir=None):
        """Writes the dust alignment. Note that the alignment direction is only in cartesian directions 

        Parameters
        ----------

        fname  : str, optional
                Name of the file into which the grain alignment should be written.
                If omitted 'grainalign_dir.inp' (if binary=True 'grainalign_dir.binp') is used.

        binary : bool
                If true the data will be written in binary format, otherwise the file format is ascii

        """
        # determine default file name
        if fname == '':
            if binary:
                fname = 'grainalign_dir.binp'
            else:
                fname = 'grainalign_dir.inp'

        # add on fdir
        if fdir is not None:
            fname = os.path.join(fdir, fname)

        # write out the data 
        self._vectorfieldWriter(fname, self.alvec, binary=binary, ndim=4)

    def writeBfield(self, fname='', binary=False, fdir=None):
        """Writes the Bfield

        Parameters
        ----------

        fname  : str, optional
                Name of the file into which the gas temperature should be written.
                If omitted 'gas_velocity.inp' (if binary=True 'gas_velocity.binp') is used.

        binary : bool
                If true the data will be written in binary format, otherwise the file format is ascii

        """

        if fname == '':
            if binary:
                fname = 'Bfield.binp'
            else:
                fname = 'Bfield.inp'

        if fdir is not None:
            fname = os.path.join(fdir, fname)

        self._vectorfieldWriter(fname, self.bfield, binary=binary, ndim=3)

    def writeVTurb(self, fname='', binary=True, octree=False, fdir=None):
        """Writes the microturbulence file.

        Parameters
        ----------

        fname   : str, optional
                  Name of the file into which the turubulent velocity field should be written.
                  If omitted 'microturbulence.inp' (if binary=True 'microturbuulence.binp') is used.

        binary  : bool
                  If true the data will be written in binary format, otherwise the file format is ascii

        octree  : bool, optional
                  If the data is defined on an octree-like AMR
        """

        if fname == '':
            if binary:
                fname = 'microturbulence.binp'
            else:
                fname = 'microturbulence.inp'
            if fdir is not None:
                if fdir[-1] is '/':
                    fname = fdir + fname
                else:
                    fname = fdir + '/' + fname
        print('Writing ' + fname)

        if octree:
            self._scalarfieldWriter(data=self.grid.convArrTree2Leaf(self.vturb), fname=fname, binary=binary,
                                    octree=True)
        else:
            self._scalarfieldWriter(data=self.vturb, fname=fname, binary=binary, octree=False)

    def writeVTK(self, vtk_fname='', ddens=False, dtemp=False, idust=None, gdens=False, gvel=False, gtemp=False):
        """Writes physical variables to a legacy vtk file.

        Parameters
        ----------

        vtk_fname : str
                    Name of the file to be written, if not specified 'radmc3d_data.vtk' will be used

        ddens     : bool
                    If set to True the dust density will be written to the vtk file

        dtemp     : bool
                    If set to True the dust temperature will be written to the vtk file

        idust     : list
                    List of indices that specifies which dust component should be written
                    if not set then the first dust species (zero index) will be used

        gdens     : bool
                    If set to True the gas density will be written to the vtk file

        gtemp     : bool
                    If set to True the gas temperature will be written to the vtk file

        gvel      : bool
                    If set to True the gas velocity will be written to the vtk file
        """

        if isinstance(idust, int):
            idust = [idust]

        if (ddens is True) | (dtemp is True):
            if idust is None:
                msg = 'Unknown dust species. Dust density or temperature is to be written but  ' \
                      + 'it has not been specified for which dust species.'
                raise ValueError(msg)


        if vtk_fname == '':
            vtk_fname = 'radmc3d_data.vtk'
        else:
            vtk_fname = str(vtk_fname)

        #
        # Get the grid
        #

        x = self.grid.xi
        # For the theta axis I leave out the poles
        #  The current cell type is hexahedron and the cells near the pole are
        #    rather tetrahedra than hexahedra and this is not yet implemented
        y = np.array(self.grid.yi[1:self.grid.nyi - 1])
        z = self.grid.zi
        nxi = x.shape[0]
        nyi = y.shape[0]
        nzi = z.shape[0]

        #
        # Gas velocity field (Should be corner centered)
        # TODO
        #  The lines below should be double checked and re-implemented as
        #  the re-mapping of the cell centered velocity field to the cell corners
        #  is physically not correct and very messy...
        #
        if gvel:
            vgas = np.zeros([nxi, nyi, nzi, 3], dtype=np.float64)
            vgas[0:nxi - 1, 0:nyi, 0:nzi - 1, :] = self.gasvel[:, 1:nyi + 1, :, :]
            vgas[nxi - 1, :, :, :] = vgas[nxi - 2, :, :, :]
            vgas[:, nyi - 1, :, :] = vgas[:, nyi - 2, :, :]
            vgas[:, :, nzi - 1, :] = vgas[:, :, nzi - 2, :]

        #
        # Header
        #
        with open(vtk_fname, 'w') as wfile:
            wfile.write('%s\n' % '# vtk DataFile Version 3.0')
            wfile.write('%s\n' % 'RADMC-3D Data')
            wfile.write('%s\n' % 'ASCII')
            wfile.write('%s\n' % 'DATASET UNSTRUCTURED_GRID')

            #
            # Write out the coordinates of the cell corners
            #
            wfile.write('%s\n' % ('POINTS ' + str(nxi * nyi * nzi).strip() + ' double'))
            print('Writing POINTS: ')
            for ix in range(nxi):
                print(ix, nxi)
                for iy in range(nyi):
                    for iz in range(nzi):
                        crd = crd_trans.ctransSph2Cart([x[ix], z[iz], y[iy]])
                        wfile.write('%.9e %9e %9e\n' % (crd[0], crd[1], crd[2]))

                        # ---------------------------------------------------------------------------------------------
                        # Write out the indices of the cell interface mesh that define a
                        # hexahedron (VTK cell type #12)
                        #
                        # The indexing of a hexahedron is as follows
                        #
                        #                  7________6
                        #                 /|      / |
                        #                / |     /  |
                        #               4_------5   |             z ^   ^ y
                        #               |  3____|___2               |  /
                        #               | /     |  /                | /
                        #               |/      | /                 |/
                        #               0-------1                   0-----> x
                        #
                        # ---------------------------------------------------------------------------------------------

            wfile.write('%s %d %d\n' % ('CELLS ', ((nxi - 1) * (nyi - 1) * (nzi - 1)),
                                        ((nxi - 1) * (nyi - 1) * (nzi - 1)) * 9))

            for ix in range(nxi - 1):
                print('Writing CELL COORDINATES: ', ix, self.grid.nxi - 2)
                for iy in range(nyi - 1):
                    for iz in range(nzi - 1):
                        id1 = nzi * nyi * ix + nzi * iy + iz
                        id2 = nzi * nyi * ix + nzi * (iy + 1) + iz
                        id4 = nzi * nyi * ix + nzi * iy + ((iz + 1) % (nzi - 1))
                        id3 = nzi * nyi * ix + nzi * (iy + 1) + ((iz + 1) % (nzi - 1))
                        id5 = nzi * nyi * (ix + 1) + nzi * iy + iz
                        id6 = nzi * nyi * (ix + 1) + nzi * (iy + 1) + iz
                        id7 = nzi * nyi * (ix + 1) + nzi * (iy + 1) + ((iz + 1) % (nzi - 1))
                        id8 = nzi * nyi * (ix + 1) + nzi * iy + ((iz + 1) % (nzi - 1))

                        line = np.array([8, id1, id2, id3, id4, id5, id6, id7, id8])
                        line.tofile(wfile, sep=' ', format='%d')
                        wfile.write('\n')
                        #
                        # Now write out the type of each cell (#12)
                        #
            wfile.write('%s %d\n' % ('CELL_TYPES', ((nxi - 1) * (nyi - 1) * (nzi - 1))))

            for ix in range(nxi - 1):
                for iy in range(nyi - 1):
                    for iz in range(nzi - 1):
                        wfile.write('%d\n' % 12)
                        #
                        # Now write out the corner centered velocities
                        #

            if gvel:
                wfile.write('%s %d\n' % ('POINT_DATA', (nxi * nyi * nzi)))
                wfile.write('%s\n' % 'VECTORS gas_velocity double')
                for ix in range(nxi):
                    print('Writing velocity : ', ix, nxi - 1)
                    for iy in range(nyi):
                        for iz in range(nzi):
                            vsph = np.array([vgas[ix, iy, iz, 0], vgas[ix, iy, iz, 2], vgas[ix, iy, iz, 1]])
                            vxyz = crd_trans.vtransSph2Cart([x[ix], z[iz], y[iy]], vsph)

                            wfile.write('%.9e %.9e %.9e\n' % (vxyz[0], vxyz[1], vxyz[2]))

            #
            # Write out the cell centered scalars
            #
            wfile.write('%s %d\n' % ('CELL_DATA', ((nxi - 1) * (nyi - 1) * (nzi - 1))))

            if ddens:
                for ids in idust:
                    wfile.write('%s\n' % ('SCALARS dust_density_' + str(int(ids)) + ' double'))
                    wfile.write('%s\n' % 'LOOKUP_TABLE default')

                    for ix in range(nxi - 1):
                        print('Writing dust density : ', ix, nxi - 2)
                        for iy in range(nyi - 1):
                            for iz in range(nzi - 1):
                                wfile.write('%.9e\n' % self.rhodust[ix, iy, iz, ids])

            if dtemp:
                for ids in idust:
                    wfile.write('%s\n' % ('SCALARS dust_temperature_' + str(int(ids)) + ' double'))
                    wfile.write('%s\n' % 'LOOKUP_TABLE default')

                    for ix in range(nxi - 1):
                        print('Writing dust temperature : ', ix, nxi - 2)
                        for iy in range(nyi - 1):
                            for iz in range(nzi - 1):
                                wfile.write('%.9e\n' % self.dusttemp[ix, iy, iz, ids])

            if gdens:
                wfile.write('%s\n' % 'SCALARS gas_numberdensity double')
                wfile.write('%s\n' % 'LOOKUP_TABLE default')

                for ix in range(nxi - 1):
                    print('Writing gas density : ', ix, nxi - 2)
                    for iy in range(nyi - 1):
                        for iz in range(nzi - 1):
                            wfile.write('%.9e\n' % self.ndens_mol[ix, iy, iz])

            if gtemp:
                wfile.write('%s\n' % 'SCALARS gas_temperature double')
                wfile.write('%s\n' % 'LOOKUP_TABLE default')

                for ix in range(nxi - 1):
                    print('Writing dust temperature : ', ix, nxi - 2)
                    for iy in range(nyi - 1):
                        for iz in range(nzi - 1):
                            wfile.write('%.9e\n' % self.gastemp[ix, iy, iz])

    def getSigmaDust(self, idust=-1):
        """Calculates the dust surface density.

        Parameters
        ----------

        idust : int, optional
                Index of the dust species for which the surface density should be calculated
                if omitted the calculated surface density will be the sum over all dust species
        """

        # Calculate the volume of each grid cell
        vol = self.grid.getCellVolume()
        # Dustmass in each grid cell
        if len(self.rhodust.shape) > 3:
            if idust >= 0:
                mass = vol * self.rhodust[:, :, :, idust]
            else:
                mass = vol * self.rhodust.sum(3)
        else:
            mass = vol * self.rhodust

        # Calculate the surface of each grid facet in the midplane
        surf = np.zeros([self.grid.nx, self.grid.nz], dtype=np.float64)
        diff_r2 = (self.grid.xi[1:] ** 2 - self.grid.xi[:-1] ** 2) * 0.5
        diff_phi = self.grid.zi[1:] - self.grid.zi[:-1]
        for ix in range(self.grid.nx):
            surf[ix, :] = diff_r2[ix] * diff_phi

        # Now get the surface density
        dum = mass.sum(1)
        self.sigmadust = dum / surf

    def getSigmaGas(self):
        """Calculates the gas surface density.
        This method uses radmc3dData.rhogas to calculate the surface density, thus the
        unit of surface density depends on the unit of radmc3dData.rhogas (g/cm^2 or molecule/cm^2)
        """

        # Calculate the volume of each grid cell
        vol = self.grid.getCellVolume()
        # Total number of molecules / gas mass in each grid cell
        mass = vol * self.rhogas
        # Calculate the surface are of each grid facet in the midplane
        surf = np.zeros([self.grid.nx, self.grid.nz], dtype=np.float64)
        diff_r2 = (self.grid.xi[1:] ** 2 - self.grid.xi[:-1] ** 2) * 0.5
        diff_phi = self.grid.zi[1:] - self.grid.zi[:-1]
        for ix in range(self.grid.nx):
            surf[ix, :] = diff_r2[ix] * diff_phi

        # Now get the surface density
        dum = np.squeeze(mass.sum(1))
        self.sigmagas = dum / np.squeeze(surf)

def integrateOverFrequency(dat_nu, nuCell=None, nuWall=None, phys_nu=None):
    """
    integrate a gridded data over frequency, used to calculate energy density
    Parameters
    ----------
    dat_nu : ndarray
        can be of any dimension, but the frequency dependence is always the last dimension

    nuCell : 1d ndarray, optional 
        frequency at the cell locations. the walls will be extrapolated
    nuWall : 1d ndarray, optional
        frequency at the cell walls in Hz. This is preferred, 
    phys_nu : 1d ndarray, optional
        a physical quantity as a function of frequency to multiple the data 
    """
    if (nuCell is None) & (nuWall is None):
        raise ValueError('at least the frequency cell or wall should be given')
    if nuWall is None:
        nuWall = extrapolate_geo(nuCell)

    dim = dat_nu.shape
    ndim = len(dim)

    nf = len(nuCell)
    df = np.abs(np.diff(nuWall))

    if ndim > 1:
        # reshape ito a 2d array
        ndatcells = np.prod(dim[:-1])
        flat_dat_nu = np.reshape(dat_nu, (ndatcells, nf))
    else:
        flat_dat_nu = dat_nu

    # integrate
    if phys_nu is None:
        flat_dat_sum = np.sum(flat_dat_nu * df, axis=-1)
    else:
        flat_dat_sum = np.sum(flat_dat_nu * phys_nu * df, axis=-1)

    if ndim > 1:
        dat_sum = np.reshape(flat_dat_sum, dim[:-1])
    else:
        dat_sum = flat_dat_sum
    return dat_sum

def integrateOverWavelength(dat_w, wavCell=None, wavWall=None, phys_w=None):
    """
    calculate a value summed over wavelength, eg radE, 
    and also calculate the average wavelength
    Parameters
    ----------
    wav : 1d ndarray
        the wavelength axis where the radiation field is defined
    dat_w : ndarray
        can be of any dimension, but the wavelength dependence is always the last dimension
    """
    if (wavCell is None) & (wavWall is None):
        raise ValueError('at least the wavelength cell or wall should be given')

    if wavCell is not None:
        # extrapolate to obtain the wavelength walls
        wavWall = grid.extrapolate_geo(wavCell)

    dim = dat_w.shape
    ndim = len(dim)

    nw = len(wavWall) - 1
    dw = np.diff(wavWall)

    # flatten the array 
    if ndim > 1:
        ncells = np.prod(dim[:-1])
        flat_dat_w = np.reshape(dat_w, (ncells, nw))
    else:
        flat_dat_w = dat_w

    # integrate
    if phys_w is None:
        flat_dat_sum = np.sum(flat_dat_w * dw, axis=-1)
    else:
        flat_dat_sum = np.sum(flat_dat_w * phys_w * dw, axis=-1)

    if ndim > 1:
        dat_sum = np.reshape(flat_dat_sum, dim[:-1])
    else:
        dat_sum = flat_dat_sum

    return dat_sum

def integrate_x(dat_x, xCell=None, xWall=None, phys_x=None):
    """
    calculate a value summed over x 
    Parameters
    ----------
    xCell : 1d ndarray
    xWall : 1d ndarray

    dat_x : ndarray
        can be of any dimension, but the x dependence is always the last dimension
    """
    if (xCell is None) & (xWall is None):
        raise ValueError('at least the cell or wall should be given')

    if xCell is not None:
        # extrapolate to obtain the walls
        xWall = grid.extrapolate_geo(xCell)

    dim = dat_x.shape
    ndim = len(dim)

    nx = len(xWall) - 1
    dx = np.diff(xWall)

    # flatten the array 
    if ndim > 1:
        ncells = np.prod(dim[:-1])
        flat_dat_x = np.reshape(dat_x, (ncells, nx))
        if phys_x is not None:
            flat_phys_x = np.reshape(phys_x, (ncells, nx))
    else:
        flat_dat_x = dat_x

    # integrate
    if phys_x is None:
        flat_dat_sum = np.sum(flat_dat_x * dx, axis=-1)
    else:
        flat_dat_sum = np.sum(flat_dat_x * flat_phys_x * dx, axis=-1)

    if ndim > 1:
        dat_sum = np.reshape(flat_dat_sum, dim[:-1])
    else:
        dat_sum = flat_dat_sum

    return dat_sum
