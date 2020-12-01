"""This module contains a class for handling regular wavelength and spatial grids
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import traceback

try:
    import numpy as np
except ImportError:
    np = None
    print(' Numpy cannot be imported ')
    print(' To use the python module of RADMC-3D you need to install Numpy')
    print(traceback.format_exc())

from . import natconst as nc

import os
import pdb

class radmc3dGrid(object):
    """ Class for spatial and frequency grid used by RADMC-3D.

    Attributes
    ----------

    act_dim    : ndarray
                A three element vector the i-th element is 1 if the i-th dimension is active,
                otherwize the i-th element is zero

    crd_sys    : {'sph', 'cyl', 'car'}
                coordinate system of the spatial grid

    nx         : int
                Number of grid points in the x (cartesian) / r (cylindrical) / r (spherical) dimension

    ny         : int
                Number of grid points in the y (cartesian) / theta (cylindrical) / theta (spherical) dimension

    nz         : int
                Number of grid points in the z (cartesian) / z (cylindrical) / phi (spherical) dimension

    nxi        : int
                Number of cell interfaces in the x (cartesian) / r (cylindrical) / r (spherical) dimension

    nyi        : int
                Number of cell interfaces in the y (cartesian) / theta (cylindrical) / theta (spherical) dimension

    nzi        : int
                Number of cell interfaces in the z (cartesian) / z (cylindrical) / phi (spherical) dimension

    nwav       : int
                Number of wavelengths in the wavelength grid

    nfreq      : int
                Number of frequencies in the grid (equal to nwav)

    x          : ndarray
                Cell centered x (cartesian) / r (cylindrical) / r (spherical)  grid points

    y          : ndarray
                Cell centered y (cartesian) / theta (cylindrical) / theta (spherical)  grid points

    z          : ndarray
                Cell centered z (cartesian) / z (cylindrical) / phi (spherical)  grid points

    xi         : ndarray
                Cell interfaces in the x (cartesian) / r (cylindrical) / r (spherical)  dimension

    yi         : ndarray
                Cell interfaces in the y (cartesian) / theta (cylindrical) / theta (spherical)  dimension

    zi         : ndarray
                Cell interfaces in the z (cartesian) / z (cylindrical) / phi (spherical)  dimension

    xtype      : list !! Not done yet !!
                The type of axis spacing
                 - 'u' = equal spacing
                 - 'l' = logarithmic. smaller spacing from beginning of interval. Should not cross any zeros within the interval, e.g. [-1, 1] cannot be in logarithmic spacing
                 - 'd' = logarithmic. smaller spacing towards the end of interval. Should not cross any zeros within the interval. 

    ytype      : list
                The type of axis spacing. see 'xtype' argument

    ztype      : list
                The type of axis spcing

    wav        : ndarray
                Wavelengh  grid

    freq       : ndarray
                Frequency  grid


    """

    # --------------------------------------------------------------------------------------------------

    def __init__(self):

        self.crd_sys = 'sph'
        self.act_dim = [1, 1, 1]
        self.grid_style = 0
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.nxi = 0
        self.nyi = 0
        self.nzi = 0
        self.x = np.zeros(0, dtype=np.float64)
        self.y = np.zeros(0, dtype=np.float64)
        self.z = np.zeros(0, dtype=np.float64)
        self.xi = np.zeros(0, dtype=np.float64)
        self.yi = np.zeros(0, dtype=np.float64)
        self.zi = np.zeros(0, dtype=np.float64)
        self.xtype = []
        self.ytype = []
        self.ztype = []

        self.nwav = 0
        self.nfreq = 0
        self.wav = np.zeros(0, dtype=np.float64)
        self.freq = np.zeros(0, dtype=np.float64)

        # frequency grid for radiation field
        self.radnfreq = 0
        self.radnwav = 0
        self.radfreq = np.zeros(0, dtype=np.float64)
        self.radwav = np.zeros(0, dtype=np.float64)

    # --------------------------------------------------------------------------------------------------
    def makeWavelengthGrid(self, wbound=None, nw=None, ppar=None):
        """Creates the wavelength/frequency grid.

        Parameters
        ----------

        wbound : list
                 Contains the wavelength boundaries of the wavelength grid (should contain at least two elements)

        nw     : list
                 Contains len(wbound)-1 elements containing the number of wavelengths between the bounds
                 set by wbound

        ppar   : dictionary, optional
                 Contains all input parameters with the parameter names as keys
        """
        if ppar:
            if wbound is None:
                wbound = ppar['wbound']
            if nw is None:
                nw = ppar['nw']
        else:
            if wbound is None:
                raise ValueError(
                    'Unknown wbound. Without the grid boundaries the wavelength grid cannot be generated')
            if nw is None:
                raise ValueError('Unknown nw. Without the number of grid points the wavelength grid cannot be '
                                 + 'generated')

        # use the walls, so we'll have nw+1 points
        dum, self.wav= getAxis(nw, wbound, 'geo')

        self.nwav = len(self.wav)
        self.freq = nc.cc / self.wav * 1e4
        self.nfreq = self.nwav

    # --------------------------------------------------------------------------------------------------
    def writeWavelengthGrid(self, fname='', old=False, fdir=None):
        """Wriites the wavelength grid to a file (e.g. wavelength_micron.inp).

        Parameters
        ----------

        fname  : str, optional
                 File name into which the wavelength grid should be written. If omitted 'wavelength_micron.inp'
                 will be used

        old    : bool, optional
                 If set to True the file format of the previous, 2D version of radmc will be used
        """

        if not old:
            if fname == '':
                fname = 'wavelength_micron.inp'
            if fdir is not None:
                fname = os.path.join(fdir, fname)

            print('Writing ' + fname)
            with open(fname, 'w') as wfile:
                wfile.write('%d\n' % self.nwav)
                for ilam in range(self.nwav):
                    wfile.write('%.9e\n' % self.wav[ilam])
        else:
            if fname == '':
                fname = 'frequency.inp'

            with open(fname, 'w') as wfile:

                print('Writing ' + fname)
                wfile.write("%d\n" % self.nfreq)
                wfile.write(" \n")
                #
                # Reverse the order of the frequency grid as it is ordered in frequency in radmc
                #
                freq = self.freq[::-1]
                for i in range(self.nfreq):
                    wfile.write("%.7e\n" % freq[i])

                    # --------------------------------------------------------------------------------------------------

    def makeSpatialGrid(self, ppar=None, 
            crd_sys=None, act_dim=None, 
            xbound=None, ybound=None, zbound=None, 
            nx=None, ny=None, nz=None ):
        """Calculates the spatial grid.

        Parameters
        ----------

        crd_sys : {'ppl', 'sph','car'}
                    Coordinate system of the spatial grid

        act_dim : 3 element list of int
                    whether or not to activate dimension

        xbound  : list
                    (with at least two elements) of boundaries for the grid along the first dimension

        ybound  : list
                    (with at least two elements) of boundaries for the grid along the second dimension

        zbound  : list
                    (with at least two elements) of boundaries for the grid along the third dimension

        nx      : list of int
                    Number of grid cells along the first dimension. List with len(xbound)-1 elements with nx[i] being the number of grid points between xbound[i] and xbound[i+1]

        ny      : list of int
                    Same as nx but for the second dimension

        nz      : list of int
                    Same as nx but for the third dimension

        ppar    : Dictionary containing all input parameters of the model (from the problem_params.inp file)
                   if ppar is set all keyword arguments that are not set will be taken from this dictionary
        """

        # read in the parameters from ppar, if they're not set explicitly here
        if ppar:
            if crd_sys is None:
                if 'crd_sys' in ppar:
                    crd_sys = ppar['crd_sys']

            if act_dim is None:
                if 'act_dim' in ppar:
                    act_dim = ppar['act_dim']

            if xbound is None:
                if 'xbound' in ppar:
                    xbound = ppar['xbound']

            if nx is None:
                if 'nx' in ppar:
                    nx = ppar['nx']

            if ybound is None:
                if 'ybound' in ppar:
                    ybound = ppar['ybound']

            if ny is None:
                if 'ny' in ppar:
                    ny = ppar['ny']
            if zbound is None:
                if 'zbound' in ppar:
                    zbound = ppar['zbound']
            if nz is None:
                if 'nz' in ppar:
                    nz = ppar['nz']

        # save information to class attributes
        self.crd_sys = crd_sys
        self.act_dim = act_dim

        # ==== plane parallel coordinates ====
        if crd_sys == 'ppl':
            self.z, self.zi = getAxis(nz, zbound, 'lin')

            self.x = [0]
            self.xi = [-1e90, 1e90]
            self.y = [0]
            self.yi = [-1e90, 1e90]

            # turn off x and y after creating all the relevant axis
            self.act_dim[0] = 0
            self.act_dim[1] = 0

        # ==== cartesian coordinates ====
        if crd_sys == 'car':
            #
            # First check whether the grid boundaries are specified
            #
            if xbound is None:
                raise ValueError('Unknown xbound. Boundaries for the cartesian x-axis are not specified. '
                                 + 'Without the boundaries the grid cannot be generated')

            if ybound is None:
                raise ValueError('Unknown ybound. Boundaries for the cartesian y-axis are not specified. '
                                 + 'Without the boundaries the grid cannot be generated')
            if zbound is None:
                raise ValueError('Unknown zbound. Boundaries for the cartesian z-axis are not specified. '
                                 + 'Without the boundaries the grid cannot be generated')

            if nx is None:
                raise ValueError('Unknown nx. Number of grid points for the cartesian x-axis is not specified. '
                                 + 'The grid cannot be generated')

            if ny is None:
                raise ValueError('Unknown ny. Number of grid points for the cartesian y-axis is not specified. '
                                 + 'The grid cannot be generated')

            if nz is None:
                raise ValueError('Unknown nz. Number of grid points for the cartesian z-axis is not specified. '
                                 + 'The grid cannot be generated')

            #
            # Create the x-axis
            #
            if self.act_dim[1] == 1:
                self.x, self.xi = getAxis(nx, xbound, 'lin')
            else:
                self.x = [0.]
                self.xi = [0., 0.]

            #
            # Create the y axis
            #
            if self.act_dim[1] == 1:
                self.y, self.yi = getAxis(ny, ybound, 'lin')
            else:
                self.y = [0.]
                self.yi = [0., 0.]


            #
            # Create the z-azis
            #
            if self.act_dim[1] == 1:
                self.z, self.zi = getAxis(nz, zbound, 'lin')
            else:
                self.z = [0.]
                self.zi = [0., 0.]

        # ==== spherical coordinates ====
        if crd_sys == 'sph':
            #
            # r->x, theta->y, phi-z
            #

            #
            # Create the x axis
            #

            if self.act_dim[0] == 1:
                self.x, self.xi = getAxis(nx, xbound, 'geo')

                # Refinement of the inner edge of the grid
                # This has to be done properly
                if ppar is not None:
                    if 'xres_nlev' in ppar:
                        if ppar['xres_nlev'] > 0:
                            ri_ext = np.array([self.xi[0], self.xi[ppar['xres_nspan']]])
                            for i in range(ppar['xres_nlev']):
                                dum_ri = ri_ext[0] + (ri_ext[1] - ri_ext[0]) * np.arange(ppar['xres_nstep'] + 1,

     dtype=np.float64) / float(
                                ppar['xres_nstep'])
                                # print ri_ext[0:2]/au
                                # print dum_ri/au
                                ri_ext_old = np.array(ri_ext)
                                ri_ext = np.array(dum_ri)
                                ri_ext = np.append(ri_ext, ri_ext_old[2:])

                            r_ext = (ri_ext[1:] + ri_ext[:-1]) * 0.5

                            self.xi = np.append(ri_ext, self.xi[ppar['xres_nspan'] + 1:])
                            self.x = np.append(r_ext, self.x[ppar['xres_nspan']:])

            else:
                self.x = [0.]
                self.xi = [0., 0.]

            #
            # Create the y axis
            #
            if self.act_dim[1] == 1:
                self.y, self.yi = getAxis(ny, ybound, 'lin')
            else:
                self.y = [0.]
                self.yi = [0., np.pi]

            #
            # Create the z axis
            # 
            if self.act_dim[2] == 1:
                self.z, self.zi = getAxis(nz, zbound, 'lin')
            else:
                self.z = [0.]
                self.zi = [0., 2*np.pi]

        self.nx = len(self.x)
        self.nxi = len(self.xi)
        self.ny = len(self.y)
        self.nyi = len(self.yi)
        self.nz = len(self.z)
        self.nzi = len(self.zi)

    def writeSpatialGrid(self, fname='', old=False, fdir=None):
        """Writes the wavelength grid to a file (e.g. amr_grid.inp).

        Parameters
        ----------

        fname : str, optional
                File name into which the spatial grid should be written. If omitted 'amr_grid.inp' will be used.

        old   : bool, optional
                If set to True the file format of the previous, 2D version of radmc will be used
        """

        #
        # Write the spatial grid for radmc3d
        #
        if not old:
            if fname == '':
                fname = 'amr_grid.inp'
                if fdir is not None:
                    fname = os.path.join(fdir, fname)

            print('Writing ' + fname)
            with open(fname, 'w') as wfile:
                # Format number
                wfile.write('%d\n' % 1)
                # AMR self.style (0=regular self. NO AMR)
                wfile.write('%d\n' % 0)
                # Coordinate system (0-99 cartesian, 100-199 spherical, 200-299 cylindrical)
                if self.crd_sys == 'car':
                    wfile.write('%d\n' % 0)

                # Coordinate system (10 - plane-parallel)
                if self.crd_sys == 'ppl':
                    wfile.write('%d\n' % 10)

                # Coordinate system (0-99 cartesian, 100-199 spherical, 200-299 cylindrical)
                if self.crd_sys == 'sph':
                    wfile.write('%d\n' % 100)
                # Coordinate system (0-99 cartesian, 100-199 spherical, 200-299 cylindrical)
                if self.crd_sys == 'cyl':
                    wfile.write('%d\n' % 200)
                # Gridinfo
                wfile.write('%d\n' % 0)

                # Active dimensions
                wfile.write('%d %d %d \n' % (self.act_dim[0], self.act_dim[1], self.act_dim[2]))
                # Grid size (x,y,z or r,phi,theta, or r,phi,z)
                wfile.write('%d %d %d \n' % (self.nx, self.ny, self.nz))
                for i in range(self.nxi):
                    wfile.write('%.9e\n' % self.xi[i])
                for i in range(self.nyi):
                    wfile.write('%.9e\n' % self.yi[i])
                for i in range(self.nzi):
                    wfile.write('%.9e\n' % self.zi[i])
            wfile.close()
        #
        # Write the spatial grid for radmc
        #
        else:

            fname = 'radius.inp'
            with open(fname, 'w') as wfile:

                print('Writing ' + fname)
                x = np.sqrt(self.xi[1:] * self.xi[:-1])
                wfile.write("%d\n" % self.nx)
                wfile.write(" \n")
                for i in range(self.nx):
                    wfile.write("%.7e\n" % x[i])

            fname = 'theta.inp'
            with open(fname, 'w') as wfile:
                print('Writing ' + fname)
                wfile.write("%d 1\n" % (self.ny / 2))
                wfile.write(" \n")
                for i in range(int(self.ny / 2)):
                    wfile.write("%.7e\n" % self.y[i])

    def readWavelengthGrid(self, fname=None, old=False, fdir=None):
        """Reads the wavelength grid

        Parameters
        ----------

        fname : str, optional
                File name from which the spatial grid should be read. If omitted 'wavelength_micron.inp' will be used.

        old   : bool, optional
                If set to True the file format of the previous, 2D version of radmc will be used
        """
        #
        # Read the radmc3d format
        #
        if not old:
            if fname is None:
                fname = 'wavelength_micron.inp'
                if fdir is not None:
                    if fdir[-1] is '/':
                        fname = fdir + fname
                    else:
                        fname = fdir + '/' + fname
            #
            # Read the frequency grid
            #
            print('Reading ' + fname)
            data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
            self.nfreq = np.int(data[0])
            self.nwav = self.nfreq
            self.wav = data[1:]
            self.freq = nc.cc / self.wav * 1e4
        #
        # Read the old radmc format
        #
        else:
            if fname is None:
                fname = 'frequency.inp'

            print('Reading '+fname)
            data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
            self.nfreq = np.int(data[0])
            self.nwav = self.nwav
            self.freq = data[1:]
            self.wav = nc.cc / self.freq * 1e4

        return

    def readSpatialGrid(self, fname='', old=False, fdir=None):
        """Reads the spatial grid

        Parameters
        ----------

        fname : str, optional
                File name from which the spatial grid should be read. If omitted 'amr_grid.inp' will be used.

        old   : bool, optional
                If set to True the file format of the previous, 2D version of radmc will be used
        """
        #
        # Read the radmc3d format
        #
        if not old:
            if fname == '':
                fname = 'amr_grid.inp'
                if fdir is not None:
                    if fdir[-1] is '/':
                        fname = fdir + fname
                    else:
                        fname = fdir + '/' + fname
                #
                # Read the spatial grid
                #

            print('Reading '+fname)
            data = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
            hdr = np.array(data[:10], dtype=np.int)
            data = data[10:]

            # Check the file format
            if hdr[0] != 1:
                msg = 'Unkonwn format number in amr_grid.inp'
                raise RuntimeError(msg)

            # Check the coordinate system
            if hdr[2] < 100:
                self.crd_sys = 'car'
            elif (hdr[2] >= 100) & (hdr[2] < 200):
                self.crd_sys = 'sph'
            elif (hdr[2] >= 200) & (hdr[2] < 300):
                self.crd_sys = 'cyl'
            else:
                raise ValueError('Unsupported coordinate system identification in the ' + fname + ' file.')

            # Get active dimensions
            self.act_dim = hdr[4:7]

            # Get the number of cells in each dimensions
            self.nx = hdr[7]
            self.ny = hdr[8]
            self.nz = hdr[9]
            self.nxi, self.nyi, self.nzi = self.nx + 1, self.ny + 1, self.nz + 1

            # Get the cell interfaces
            self.xi = data[:self.nxi]
            data = data[self.nxi:]
            self.yi = data[:self.nyi]
            data = data[self.nyi:]
            self.zi = data[:self.nzi]

            if self.crd_sys == 'car':
                self.x = (self.xi[0:self.nx] + self.xi[1:self.nx + 1]) * 0.5
                self.y = (self.yi[0:self.ny] + self.yi[1:self.ny + 1]) * 0.5
                self.z = (self.zi[0:self.nz] + self.zi[1:self.nz + 1]) * 0.5
            else:
                self.x = np.sqrt(self.xi[0:self.nx] * self.xi[1:self.nx + 1])
                self.y = (self.yi[0:self.ny] + self.yi[1:self.ny + 1]) * 0.5
                self.z = (self.zi[0:self.nz] + self.zi[1:self.nz + 1]) * 0.5

        #
        # Read the old radmc format
        #
        else:
            self.crd_sys = 'sph'
            self.act_dim = [1, 1, 0]

            #
            # Read the radial grid
            #
            data = np.fromfile('radius.inp', count=-1, sep=" ", dtype=np.float64)
            self.nx = np.int(data[0])
            self.nxi = self.nx + 1
            self.x = data[1:]
            self.xi = np.zeros(self.nxi, dtype=float)
            self.xi[1:-1] = 0.5 * (self.x[1:] + self.x[:-1])
            self.xi[0] = self.x[0] - (self.xi[1] - self.x[0])
            self.xi[-1] = self.x[-1] + (self.x[-1] - self.xi[-2])

            #
            # Read the poloidal angular grid
            #

            data = np.fromfile('theta.inp', count=-1, sep=" ", dtype=np.float64)
            self.ny = np.int(data[0]) * 2
            self.nyi = self.ny + 1
            self.y = np.zeros(self.ny, dtype=float)
            self.y[:self.ny//2] = data[2:]
            self.y[self.ny//2:] = np.pi - data[2:][::-1]
            self.yi = np.zeros(self.nyi, dtype=float)
            self.yi[1:-1] = 0.5 * (self.y[1:] + self.y[:-1])
            self.yi[self.ny] = np.pi * 0.5

            #
            # Create the azimuthal grid
            #
            self.nz = 1
            self.zi = np.array([0., 2. * np.pi], dtype=float)

        return

    def readGrid(self, old=False, fdir=None):
        """Reads the spatial (amr_grid.inp) and frequency grid (wavelength_micron.inp).

        Parameters
        ----------

        old   : bool, optional
                If set to True the file format of the previous, 2D version of radmc will be used
        """

        self.readSpatialGrid(old=old, fdir=fdir)
        self.readWavelengthGrid(old=old, fdir=fdir)

        return

    def getCellVolume(self):
        """Calculates the volume of grid cells.

        """
        if self.crd_sys == 'sph':

            if self.act_dim[0] == 0:
                raise ValueError('The first (r) dimension of a shserical grid is switched off')
            elif self.act_dim[1] == 0:
                if self.act_dim[2] == 0:
                    vol = np.zeros([self.nx, self.ny, self.nz], dtype=np.float64)
                    diff_r3 = self.xi[1:] ** 3 - self.xi[:-1] ** 3
                    diff_cost = 2.0
                    diff_phi = 2. * np.pi
                    for ix in range(self.nx):
                        vol[ix, 0, 0] = 1. / 3. * diff_r3[ix] * diff_cost * diff_phi

                else:
                    vol = np.zeros([self.nx, self.ny, self.nz], dtype=np.float64)
                    diff_r3 = self.xi[1:] ** 3 - self.xi[:-1] ** 3
                    diff_cost = 2.0
                    diff_phi = self.zi[1:] - self.zi[:-1]
                    for ix in range(self.nx):
                        for iz in range(self.nz):
                            vol[ix, 0, iz] = 1. / 3. * diff_r3[ix] * diff_cost * diff_phi[iz]

            elif self.act_dim[2] == 0:
                vol = np.zeros([self.nx, self.ny, self.nz], dtype=np.float64)
                diff_r3 = self.xi[1:] ** 3 - self.xi[:-1] ** 3
                diff_cost = np.cos(self.yi[:-1]) - np.cos(self.yi[1:])
                diff_phi = 2. * np.pi
                for ix in range(self.nx):
                    for iy in range(self.ny):
                        vol[ix, iy, :] = 1. / 3. * diff_r3[ix] * diff_cost[iy] * diff_phi

            else:
                vol = np.zeros([self.nx, self.ny, self.nz], dtype=np.float64)
                diff_r3 = self.xi[1:] ** 3 - self.xi[:-1] ** 3
                diff_cost = np.cos(self.yi[:-1]) - np.cos(self.yi[1:])
                diff_phi = self.zi[1:] - self.zi[:-1]
                for ix in range(self.nx):
                    for iy in range(self.ny):
                        vol[ix, iy, :] = 1. / 3. * diff_r3[ix] * diff_cost[iy] * diff_phi
        else:
            raise ValueError('Coordinate system ' + self.crd_sys + ' is not yet supported.')

        return vol

def getAxis(ncell, xbound, xtype):
    """
    calculate an axis

    Parameters
    ----------
    ncell : list of int, or int
        for the number of cells per interval
    xbound : list
        at least two elements to define wall
    xtype : str
            'lin' = linear 
            'geo' = geometric
    """
    if not isinstance(ncell, list):
        ncell = [ncell]

    # check if the number of cells are consistent with xbound
    if len(ncell) != (len(xbound)-1):
        raise ValueError('number of cells should be one less than the boundaries')

    nwall = [i + 1 for i in ncell]
    totcell = np.sum(ncell)

    x = np.zeros([totcell], dtype=np.float64)	# cell
    xi = np.zeros([totcell + 1], dtype=np.float64)	# wall

    # mark the first one
    xi[0] = xbound[0]
    istart = 1
    for ii in range(len(nwall)):
        if xtype == 'lin':
            iwall = xbound[ii] + (xbound[ii+1] - xbound[ii]) * (
                np.arange(nwall[ii], dtype=np.float64) / float(ncell[ii]))
        elif xtype == 'geo':
            iwall = xbound[ii] * (xbound[ii+1] / xbound[ii])**(
                np.arange(nwall[ii], dtype=np.float64) / float(ncell[ii])) 
        else:
            raise ValueError('xtype can only be lin or geo')
        xi[istart:istart+ncell[ii]] = iwall[1:]
        istart = istart + ncell[ii]

    # now calculate cells
    if xtype == 'lin':
        x = 0.5 * (xi[1:] + xi[:-1])
    else:
        x = np.sqrt(xi[1:] * xi[:-1])

    return x, xi

def extrapolate_lin(cell):
    """
    calculate wall coordinate based on the cell coordinates
    this extrapolates linearly
    """
    ncell = len(cell)
    nwall = ncell + 1
    wall = np.zeros([nwall])
    wall[1:-1] = 0.5 * (cell[:-1] + cell[1:])

    # extrapolate for the first wall
    wall[0] = wall[1] - (cell[1] - cell[0])
    # extrapolate for the last wall
    wall[-1] = wall[-2] + (cell[-1] - cell[-2])
    return wall

def extrapolate_geo(cell):
    """
    calculate the wall coordinate based on the cell coordinates
    this extrapolates linearly
    """
    ncell = len(cell)
    nwall = ncell + 1
    wall = np.zeros([nwall])
    wall[1:-1] = np.sqrt(cell[:-1] * cell[1:])

    # extrapolate for the end points
    wall[0] = wall[1] * (cell[1] / cell[0])**(-1)
    wall[-1] = wall[-2] * (cell[-1] / cell[-2])
    return wall
