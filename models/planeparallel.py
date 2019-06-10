"""planeparallel.py

to set up planeparallel model. 
Constant density and constant temperature
can include different grain sizes at various locations

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
        # coordinate system
        ['crd_sys', "'ppl'", 'Coordinate system'],
        ['nx', '[1]', 'Number of grid points in the first dimension'],
        ['xbound', '[-1e90, 1e90]', 'Number of x grid points'],
        ['ny', '[1]', 'Number of grid points in the second dimension'],
        ['ybound', '[-1e90, 1e90]', 'Number of y grid points'],
        ['nz', '[100]', 'Number of grid points in the third dimension'],
        ['zbound', '[0., 1*au]', 'value for z grid walls'],
        # density
        ['rho0', '[1e-15]', 'density for various grain sizes'], 
        # temperature
        ['T0', '[100]', 'temperature'],
        # locations for various dust grains
        ['dztop', '[1*au]', 'location of top for various grains'], 
        ['dzbot', '[0]', 'location of bottom for various grains']
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
    if ppar['crd_sys'] != 'ppl':
        raise ValueError('crd_sys must be "ppl" for this model')

    tcut = 1. 
    op = dustopac.radmc3dDustOpac()
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])

    tdust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64)
    for ig in range(ngs):
        for iz in range(grid.nz):
            if (grid.z[iz] <= ppar['dztop'][ig]) and (grid.z[iz] >= ppar['dzbot'][ig]): 
                tdust[:,:,iz,ig] = ppar['T0'][ig]
            else:
                tdust[:,:,iz,ig] = tcut

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
    if ppar['crd_sys'] != 'ppl':
        raise ValueError('crd_sys must be "ppl" for this model')

    cutddens = 1e-30
    op = dustopac.radmc3dDustOpac()
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])

    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64)
    for ig in range(ngs):
        for iz in range(grid.nz):
            if (grid.z[iz] <= ppar['dztop'][ig]) and (grid.z[iz] >= ppar['dzbot'][ig]):
                rhodust[:,:,iz,ig] = ppar['rho0'][ig]
            else:
                rhodust[:,:,iz,ig] = cutddens

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
