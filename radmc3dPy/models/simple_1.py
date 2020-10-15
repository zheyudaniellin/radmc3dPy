"""A 3D spherical Gaussian blob in a 3D cartesian grid

Original IDL model by Kees Dullemond, Python translation by Attila Juhasz
"""
from __future__ import absolute_import
from __future__ import print_function
import traceback
try:
    import numpy as np
except ImportError:
    np = None
    print(' Numpy cannot be imported ')
    print(' To use the python module of RADMC-3D you need to install Numpy')
    print(traceback.format_exc())


def getModelDesc():
    """Provides a brief description of the model
    """

    return "Example model: A 3D spherical Gaussian blob in a 3D cartesian grid"
           

def getDefaultParams():
    """Provides default parameter values 

    Returns a list whose elements are also lists with three elements:
    1) parameter name, 2) parameter value, 3) parameter description
    All three elements should be strings. The string of the parameter
    value will be directly written out to the parameter file if requested,
    and the value of the string expression will be evaluated and be put
    to radmc3dData.ppar. The third element contains the description of the
    parameter which will be written in the comment field of the line when
    a parameter file is written. 
    """

    defpar = [['mstar', '1.0*ms', 'Mass of the star(s)'],
              ['pstar', '[0., 0., 0.]', 'Position of the star(s) (cartesian coordinates)'],
              ['rstar', '1.0*rs', 'Radius of the star(s)'],
              ['tstar', '1.0*ts', 'Effective temperature of the star(s)'],
              ['crd_sys', "'car'", 'Coordinate system used (car/sph)'],
              ['nx', '100', 'Number of grid points in the first dimension'],
              ['ny', '100', 'Number of grid points in the second dimension'],
              ['nz', '100', 'Number of grid points in the third dimension'],
              ['xbound', '[-100.0*au, 100.0*au]', 'Boundaries for the x-grid'],
              ['ybound', '[-100.0*au, 100.0*au]', 'Boundaries for the y-grid'],
              ['zbound', '[-100.0*au, 100.0*au]', 'Boundaries for the z-grid'],
              ['nw', '[20,100,30]', 'Number of points in the wavelength grid'],
              ['wbound', '[0.1, 7., 25., 1e4]', 'Boundaries for the wavelength grid'],
              ['dustkappa_ext', "['silicate']", 'Dust opacity file name extension'],
              ['nphot', '1000000', 'Number of photons in the thermal Monte Carlo simulation'],
              ['scattering_mode_max', '0', '0 - no scattering, 1 - isotropic scattering, 2 - anizotropic scattering'],
#              ['iranfreqmode', '1', '', 'Code parameters'],
              ['radius', '25.0*au', ''],
              ['rho0', '1e-15', ''], 
              ['T0', '100.', 'Temperature']
             ]

    return defpar


def getDustDensity(grid=None, ppar=None):
    """Calculates the dust density distribution 
    
    Parameters
    ----------
    grid : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid
    
    ppar : dictionary
            Dictionary containing all parameters of the model 
    
    Returns
    -------
    Returns the dust volume density in g/cm^3
    """
    xx, yy = np.meshgrid(grid.x, grid.y)
    xx = xx.swapaxes(0, 1)
    yy = yy.swapaxes(0, 1)
    rho = np.zeros([grid.nx, grid.ny, grid.nz, 1], dtype=np.float64)

    for iz in range(grid.nz):
        r = np.sqrt(xx**2 + yy**2 + grid.z[iz]**2)
        rho[:, :, iz, 0] = ppar['rho0'] * np.exp(-0.5*r**2/ppar['radius']**2)

    return rho

def getDustTemperature(grid=None, ppar=None):
    """ Calculates the dust temperature

    Parameters
    ----------
    grid : radmc3dGrid
    ppar : dictionary 
           containing all the parameters of the model
    Returns
    -------
    Returns dust temperature
    """
    mesh = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    if ppar['crd_sys'] == 'sph':
        rr = mesh[0]
        tt = mesh[1]
        pp = mesh[2]
    elif ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        cyrr = np.sqrt(xx**2. + yy*2.)
    else:
        raise ValueError('crd_sys not specified in ppar')

    tdust = np.zeros([grid.nx, grid.ny, grid.nz, 1], dtype=np.float64)+ppar['T0']
    return tdust
