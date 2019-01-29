""" Calculate a thin disk which would be:
      - vertically iso thermal, with some radial profile
      - gaussian disk, given the temperature profile

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

from .. import natconst
from .. import dustopac
import pdb

def getModelDesc():
    """Provides a brief description of the model
    """

    return "Thin disk"
           

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
        ['nx', '[50,70]', 'Number of grid points in the first dimension'],
        ['xbound', '[0.1*au, 30.*au, 200.*au]', 'Number of radial grid points'],
        ['ny', '[10,30, 30, 10]',
           'Number of grid points in the second dimension'],
        ['ybound', '[0.1, pi/3., pi/2., 2.*pi/3., 3.04]',
           'Number of radial grid points'],
        ['nz', '[181]', 'Number of grid points in the third dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
        # star related
        ['tstar', '[5000.0]', 'Temperature of star'],
        ['mstar', '[2.0*ms]', 'Mass of the star(s)'],
        ['rstar', '[2.0*rs]', 'Radius of star'],
        # density related
        ['mdisk', '0.2*ms', 'Total mass of disk'],
        ['sigp', '0.1', 'power index of Lynden-Bell surface density'],
        ['Rin', '0.01*au', 'Inner radii of disk '],
        ['Rsig', '50*au', 'Characteristic radius for surface density'],
        ['g2d', '0.01', 'gas to dust ratio'],
        ['cutgdens', '1e-30', 'cut for density, also ambient density'],
        # temperature related
        ['Rt', '50.*au', 'characteristic radius'], 
        ['T0', '50', 'Temperature at Rt'],
        ['qtemp', '-0.5', 'exponent for temperature'], 
        ['cuttemp', '10', 'temperature cut'],
        # alignment
        ['altype', "'toroidal'", 'alignment type']
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
    mesh = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    if ppar['crd_sys'] == 'sph':
        rr = mesh[0]
        tt = mesh[1]
        pp = mesh[2]
        xx = rr * np.sin(tt) * np.sin(pp)
        yy = rr * np.sin(tt) * np.cos(pp)
        zz = rr * np.cos(tt)
        cyrr = np.sqrt(xx**2. + yy**2)
    elif ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        cyrr = np.sqrt(xx**2. + yy*2.)
    else:
        raise ValueError('crd_sys not specified in ppar')

    tgas = ppar['T0'] * (cyrr / ppar['Rt'])**(ppar['qtemp'])
    reg = tgas <= ppar['cuttemp']
    tgas[reg] = ppar['cuttemp']

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
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])

    tdust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64)
    tgas = getGasTemperature(grid=grid, ppar=ppar)
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
    mesh = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    if ppar['crd_sys'] == 'sph':
        rr = mesh[0]
        tt = mesh[1]
        pp = mesh[2]
        xx = rr * np.sin(tt) * np.sin(pp)
        yy = rr * np.sin(tt) * np.cos(pp)
        zz = rr * np.cos(tt)
        cyrr = np.sqrt(xx**2. + yy**2)
    elif ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        cyrr = np.sqrt(xx**2. + yy*2.)
    else:
        raise ValueError('crd_sys not specified in ppar')

    # get the temperature 
    tgas = getGasTemperature(grid=grid, ppar=ppar)
    cs = np.sqrt(natconst.kk * tgas / natconst.muh2 / natconst.mp)
    # get the wk
    wk = np.sqrt(natconst.gg * ppar['mstar'][0] / cyrr**3.)

    hh = cs / wk

    sig0=(2.-ppar['sigp'])*ppar['mdisk']/(2.*np.pi)/ppar['Rsig']**2
    sig_cyrr = (sig0 * (cyrr / ppar['Rsig'])**(-ppar['sigp'])
                     * np.exp(-(cyrr/ppar['Rsig'])**(2.-ppar['sigp'])))
    rhogas = sig_cyrr / np.sqrt(2.*np.pi) / hh * np.exp(-0.5*(zz/hh)**2.)

    vol = grid.getCellVolume()
    rhovol = rhogas * vol
    tot_rho = rhovol.sum()
    print('Total Gas mass=%f (Msun)'%(tot_rho/natconst.ms))

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
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])

    rhogas = getGasDensity(grid=grid, ppar=ppar)
    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64) 
    for ig in range(ngs):
        rhodust[:, :, :, ig] = rhogas * ppar['g2d']
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

def getDustAlignment(grid=None, ppar=None):
    """calculates the dust alignment

    Parameters
    ----------
        grid : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid
    
    ppar : dictionary
            Dictionary containing all parameters of the model.
           altype = '0': all zeros
                  = 'x': aligned to x direction. or 'y', 'z'
                  = 'toroidal' : toroidal alignment
                  = 'poloidal' : poloidal alignment
                  = 'radial'   : radial alignment, in spherical coordinates
                  = 'cylradial': radial in cylindrical coordinates
    
    Returns
    -------
    Returns the dust grain alignment
    """
    # check inputs in ppar
    if 'crd_sys' not in ppar:
        raise ValueError('crd_sys is not in ppar')
    else:
        crd_sys = ppar['crd_sys']
    if 'altype' not in ppar:
        altype = '0'
    else:
        altype = ppar['altype']

    # meshgrid for space
    mesh = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')

    alvec = np.zeros([grid.nx, grid.ny, grid.nz, 3], dtype=np.float64)

    if crd_sys is 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
    elif crd_sys is 'sph':
        rr = mesh[0]
        tt = mesh[1]
        pp = mesh[2]
        xx = rr * np.sin(tt) * np.cos(pp)
        yy = rr * np.sin(tt) * np.sin(pp)
        zz = rr * np.cos(tt)
    else:
        raise ValueError('incorrect input for crd_sys')
    # different modes for alignment
    # x
    if altype is 'x':
        alvec[:,:,:,0] = 1.0
    # y
    if altype is 'y':
        alvec[:,:,:,1] = 1.0
    # z 
    if altype is 'z':
        alvec[:,:,:,2] = 1.0
    # radial
    if altype is 'radial':
        alvec[:,:,:,0] = xx / rr
        alvec[:,:,:,1] = yy / rr
        alvec[:,:,:,2] = zz / rr

    # cylradial
    if altype is 'cylradial':
        cyl_rr = np.sqrt(xx**2 + yy**2)
        alvec[:,:,:,0] = xx / cyl_rr
        alvec[:,:,:,1] = yy / cyl_rr
    # poloidal
    if altype is 'poloidal':
        raise ValueError('poloidal no implemented yet')
    # toroidal
    if altype is 'toroidal':
        cyl_rr = np.sqrt(xx**2 + yy**2)
        alvec[:,:,:,0] = yy / cyl_rr
        alvec[:,:,:,1] = -xx / cyl_rr

    if altype is not '0':
    # Normalize
        length = np.sqrt(alvec[:,:,:,0]*alvec[:,:,:,0] +
                     alvec[:,:,:,1]*alvec[:,:,:,1] +
                     alvec[:,:,:,2]*alvec[:,:,:,2])
        alvec[:,:,:,0] = np.squeeze(alvec[:,:,:,0]) / ( length + 1e-60 )
        alvec[:,:,:,1] = np.squeeze(alvec[:,:,:,1]) / ( length + 1e-60 )
        alvec[:,:,:,2] = np.squeeze(alvec[:,:,:,2]) / ( length + 1e-60 )

    return alvec

