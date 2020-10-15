"""Blob of density with power-law density, power-law temperature, velocity

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
try:
    import numpy as np
except ImportError:
    np = None
    print(' Numpy cannot be imported ')
    print(' To use the python module of RADMC-3D you need to install Numpy')
    print(traceback.format_exc())

from .. import dustopac
import pdb

def getModelDesc():
    """To produce blob with power-law density and power-law temperature
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
              ['crd_sys', "'sph'", 'Coordinate system'],
              ['nx', '[35,50]', 'Number of grid points in the first dimension'],
              ['xbound', '[0.01*au,20.*au, 100.0*au]', 'Number of radial grid points'],
              ['ny', '[10,30,30,10]', 'Number of grid points in the first dimension'],
              ['ybound', '[0., pi/3., pi/2., 2.*pi/3., pi]', 'Number of radial grid points'],
              ['nz', '[61]', 'Number of grid points in the first dimension'],
              ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
              ['gasspec_vturb', '1e5', 'Microturbulent line width'], 
              ['gasspec_mol_name', "['12co','13co','c18o','c17o']", 'name of molecule'],
              ['gasspec_mol_abun', '[4e-5, 5.78e-7, 7.18e-8, 1e-8]', 'abundance of molecule'],
              ['gasspec_mol_dbase_type', "['leiden','leiden','leiden','leiden']", 'data base type'],
              ['gasspec_mol_dissoc_taulim', '[1.0, 1.0, 1.0, 1.0]', 'Continuum optical depth limit below which all molecules dissociate'],
              ['gasspec_mol_freezeout_temp', '[0.0, 0.0, 0.0, 0.0]', 'Freeze-out temperature of the molecules in Kelvin'],
              ['gasspec_mol_freezeout_dfact', '[1e-8, 1e-8, 1e-8, 1e-8]',
               'Factor by which the molecular abundance should be decreased in the frezze-out zone'],
              ['R0', '50.*au', 'characteristic radius'],
              ['T0', '100.', 'tempterature at R0'],
              ['qtemp', '-0.5', 'exponent for temperature'],
              ['cuttemp', '1.', 'lowest temperature'],
              ['mblob', '0.05*ms', 'total mass of the blob'],
              ['cutdens', '1e-30', 'lowest gas density'],
              ['sigp', '-2.0', 'power-law of density'],
              ['Rinner', '0.01*au', 'inner radius of blob'],
              ['Router', '200.*au', 'outer radius of blob'],
              ['V0', '5*1e5', 'rotational velocity at R0'],
              ['qvel', '-0.5', 'exponent for velocity']
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
    if ppar['crd_sys'] == 'sph':
        for ix in range(grid.nx):
            tgas[ix,:,:] = ppar['T0'] * (grid.x[ix] / ppar['R0'])**(ppar['qtemp'])

    if ppar['crd_sys'] == 'car':
        for ix in range(grid.nx):
            for iy in range(grid.ny):
                for iz in range(grid.nz):
                    xii = grid.x[ix]
                    yii = grid.y[iy]
                    zii = grid.z[iz]
                    rii = np.sqrt(xii**2 + yii**2 + zii**2)
                    tii = ppar['T0'] * (rii / ppar['R0'])**(ppar['qtemp'])
                    tgas[ix,iy,:] = tii
    
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
    op = dustopac.radmc3dDustOpac()
    res = op.readDustInfo()
    ngs = len(res['gsize'])

    tgas = getGasTemperature(grid=grid, ppar=ppar)

    tdust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64)
    for ig in range(ngs):
        tdust[:,:,:,ig] = tgas

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
        gasabun = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
        ind = ppar['gasspec_mol_name'].index(ispec)
        gasabun[:, :, :] = ppar['gasspec_mol_abun'][ind]
 
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
    def eq_rho0(mblob, rinner, r0, router, sigp):
        if sigp == -3:
            rho0 = mblob * r0**sigp / 4. / np.pi / (np.log(router / rinner))
        else:
            rho0 = mblob * r0**sigp / 4. / np.pi / (router**sigp - rinner**sigp)
        return rho0

    rhogas = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) + 1e-30

    rho0 = eq_rho0(ppar['mblob'], ppar['Rinner'], ppar['R0'], ppar['Router'], ppar['sigp'])

    if ppar['crd_sys'] == 'sph':
        for ix in range(grid.nx):
            rhogas[ix,:,:] = rho0 * (grid.x[ix] / ppar['R0'])**ppar['sigp']

    if ppar['crd_sys'] == 'car':
        for ix in range(grid.nx):
            for iy in range(grid.ny):
                for iz in range(grid.nz):
                    rii = np.sqrt(grid.x[ix]**2 + grid.y[iy]**2 + grid.z[iz]**2)
                    rhogas[ix,iy,:] = rho0 * (rii / ppar['R0'])**ppar['sigp']

    reg = rhogas < ppar['cutdens']
    rhogas[reg] = ppar['cutdens']

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

    op = dustopac.radmc3dDustOpac()
    res = op.readDustInfo()
    ngs = len(res['gsize'])

#    rhogas = getGasDensity(grid=grid, ppar=ppar)
    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64) + 1e-30

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
    if ppar['crd_sys'] == 'sph':
        for ix in range(grid.nx):
            vel[ix,:,:,2] = ppar['V0'] * (grid.x[ix] / ppar['R0'])**ppar['qvel']

    if ppar['crd_sys'] == 'car':
        for ix in range(grid.nx):
            for iy in range(grid.ny):
                for iz in range(grid.nz):
                    rii = np.sqrt(grid.x[ix]**2 + grid.y[iy]**2 + grid.z[iz]**2)
                    vel[ix,:,:,2] = ppar['V0'] * (rii / ppar['R0'])**ppar['qvel']
    return vel
