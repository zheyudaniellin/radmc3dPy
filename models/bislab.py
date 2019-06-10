"""bislab
two layers of slab to help with analytical solutions
not done yet

This template is an empty model, i.e. all model functions return zeros in the appropriate arrays and dimensions. 
The purpose of this model is to illustrate the names and syntax of the model functions. Hence, this file can
be a starting point for implementing new models in the library. 

A radmc3dPy model file can contain any / all of the functions below

    * getDefaultParams()
    * getModelDesc()
    * getDustDensity()
    * getDustTemperature()
    * getGasAbundance()
    * getGasDensity()
    * getGasTemperature()
    * getVelocity()
    * getVTurb()

The description of the individual functions can be found in the docstrings below the function name.
If a model does not provide a variable or the variable should be calculated by RADMC-3D 
(e.g. dust temperature) the corresponding function (e.g. get_dust_temperature) should be removed from
or commented out in the model file. 

NOTE: When using this template it is strongly advised to rename the template model (to e.g. mydisk.py)
as the get_model_names() function in the setup module removes the name 'template' from the list of available
models. 

"""
from __future__ import absolute_import
from __future__ import print_function
import traceback

from .. import dustopac
import pdb

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

    return "Example model template"
           

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

    defpar = [
        ['crd_sys', "'car'", 'Coordinate system'], 
        ['nx', '[100]', 'Number of grid points in the first dimension'],
        ['xbound', '[-100*au, 100*au]', 'Number of radial grid points'],
        ['ny', '[100]', 'Number of grid points in the second dimension'],
        ['ybound', '[-100*au, 100*au]',
           'Number of radial grid points'],
        ['nz', '[100]', 'Number of grid points in the third dimension'],
        ['zbound', '[-100*au, 100*au]', 'Number of radial grid points'],
        # star related
        ['tstar', '[3900.0]', 'Temperature of star'],
        ['mstar', '[1.0*ms]', 'Mass of the star(s)'],
        ['rstar', '[2.5*rs]', 'Radius of star'],
        # dust density related
        ['width', '80*au', 'length in x and y'],
        ['zlen', '[40*au, 40*au]', 'thickness for each slab'],
        ['rho0', '[1e-15, 1e-15]', 'rho for each slab'], 
        ['T0', '[50, 50]', 'temperature']
              ]

    return defpar


def getGasTemperature(grid=None, ppar=None):
    """Calculates/sets the gas temperature
    
    Parameters
    ----------
    grid : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid
    
    ppar : dictionary
            Dictionary containing all parameters of the model 
    
    Returns
    -------
    Returns the gas temperature in K
    """

    tgas = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) + 1.0
    return tgas


def getDustTemperature(grid=None, ppar=None):
    """Calculates/sets the dust temperature
    
    Parameters
    ----------
    grid : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid
    
    ppar : dictionary
            Dictionary containing all parameters of the model 
    
    Returns
    -------
    Returns the dust temperature in K
    """
    mesh = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    if ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
    else:
        raise ValueError('crd_sys not specified in ppar')

    op = dustopac.radmc3dDustOpac()
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])

    tdust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64)
    for ig in range(ngs):
        tdust[:,:,:,ig] = ppar['T0'][ig]

    return tdust


def getGasAbundance(grid=None, ppar=None, ispec=''):
    """Calculates/sets the molecular abundance of species ispec 
    The number density of a molecule is rhogas * abun 
   
    Parameters
    ----------
    grid  : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid

    ppar  : dictionary
            Dictionary containing all parameters of the model 

    ispec : str
            The name of the gas species whose abundance should be calculated

    Returns
    -------
    Returns the abundance as an ndarray
    """
   
    gasabun = -1
    if ppar['gasspec_mol_name'].__contains__(ispec):
        ind = ppar['gasspec_mol_name'].index(ispec)
        gasabun[:, :, :] = ppar['gasspec_mol_abun'][ind]
 
    elif ppar['gasspec_colpart_name'].__contains__(ispec):
        gasabun = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) 
        ind = ppar['gasspec_colpart_name'].index(ispec)
        gasabun[:, :, :] = ppar['gasspec_colpart_abun'][ind]
    else:
        raise ValueError(' The abundance of "'+ispec+'" is not specified in the parameter file')
   
    return gasabun


def getGasDensity(grid=None, ppar=None):
    """Calculates the total gas density distribution 
    
    Parameters
    ----------
    grid : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid
    
    ppar : dictionary
            Dictionary containing all parameters of the model 
    
    Returns
    -------
    Returns the gas volume density in g/cm^3
    """
    
    rhogas = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) + 1e-20

    return rhogas


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
    mesh = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    if ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        cyrr = np.sqrt(xx**2. + yy**2.)
    else:
        raise ValueError('crd_sys not specified in ppar')

    op = dustopac.radmc3dDustOpac()
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])

    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64) 
    zwidth = sum(ppar['zlen'][:])
    zstart = - 0.5 * zwidth
    for ig in range(ngs):
        zend = zstart + ppar['zlen'][ig]
        reg = ( 
               (abs(xx) <= ppar['width']) 
               & (abs(yy) <= ppar['width']) 
               & (zz > zstart)
               & (zz <= zend)
              )
        zstart = zend
        rhoii = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
        rhoii[reg] = ppar['rho0'][ig]
        rhodust[:,:,:,ig] = rhoii

    return rhodust


def getVTurb(grid=None, ppar=None):
    """Calculates/sets the turbulent velocity field
    
    Parameters
    ----------
    grid : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid
    
    ppar : dictionary
            Dictionary containing all parameters of the model 
    
    Returns
    -------
    Returns the turbulent velocity in cm/s
    """

    vturb = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) + ppar['gasspec_vturb']
    return vturb


def getVelocity(grid=None, ppar=None):
    """Calculates/sets the gas velocity field
    
    Parameters
    ----------
    grid : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid
    
    ppar : dictionary
            Dictionary containing all parameters of the model 
    
    Returns
    -------
    Returns the turbulent velocity in cm/s
    """

    vel = np.zeros([grid.nx, grid.ny, grid.nz, 3], dtype=np.float64)
    return vel
