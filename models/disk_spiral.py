"""Disk model for spiral protoplanetary disk
particularly for Elias 2-27

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
from .. import natconst
import pdb
import fneq
from . import DiskEqs

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
        ['crd_sys', "'sph'", 'Coordinate system'],
        ['nx', '[30, 60, 50]', 'Number of grid points in the first dimension'],
        ['xbound', '[0.1*au, 30.*au, 120.*au, 500.*au]', 'Number of radial grid points'],
        ['ny', '[10,30, 30, 10]',
           'Number of grid points in the second dimension'],
        ['ybound', '[2*pi/9., pi/3., pi/2., 2*pi/3., 7*pi/9]',
           'Number of radial grid points'],
        ['nz', '[721]', 'Number of grid points in the third dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'], 
        # star related
        ['tstar', '[4000.0]', 'Temperature of star'],
        ['mstar', '[0.5*ms]', 'Mass of the star(s)'],
        ['rstar', '[2.5*rs]', 'Radius of star'],
        # density
        ['mdisk', '0.1*ms', 'Total mass of disk'], 
        ['g2d', '0.01', ' Dust to gas ratio'],
        ['Rsig', '200*au', ' characteristic radius'], 
        ['sigp', '0.7', ' exponent value for sigma'],
        ['cutgdens', '1e-30', 'cut for gas density'], 
        # spiral
        ['Rspir', '[84*au, 84*au]', 'R0 of spiral'],
        ['Rend', '[120*au, 120*au]', 'end of spiral'], 
        ['bspir', '[0.138, 0.138]', 'R=R0 exp(b theta)'],
        ['wspir', '[0.3, 0.3]', 'Width of spiral (in rad)'], 
        ['pspir', '[0, pi]', 'initial phase for the spiral'], 
        # gaps
        ['Rgap', '[71*au]', 'Radius of gap'], 
        ['wgap', '[4*au]', 'FWHM of gap'], 
        # height
        ['Ht', '13*au', 'Height'],
        ['Rt', '100*au', 'Radius for height'], 
        ['qheight', '1.125', 'powerlaw for height'], 
        # temperature
        ['T0', '13.4', 'Temperature'],
        ['R0', '200*au', 'Characteristic radius'],
        ['qtemp', '0.45', 'temperature powerlaw'], 
        ['cuttemp', '10', 'cut for temperature'],
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

    mesh = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    if ppar['crd_sys'] == 'sph':
        rr = mesh[0]
        tt = mesh[1]
        pp = mesh[2]
        xx = rr * np.sin(tt) * np.sin(pp)
        yy = rr * np.sin(tt) * np.cos(pp)
        cyrr = np.sqrt(xx**2. + yy**2)
    elif ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        cyrr = np.sqrt(xx**2. + yy**2.)
    else:
        raise ValueError('crd_sys not specified in ppar')

    tgas = ppar['T0'] * (cyrr / ppar['R0'])**(-ppar['qtemp'])
    reg = tgas < ppar['cuttemp']
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

def getSpiral(cyrr, raxis,ttaaxis,paxis, R0, b0, w0, phase0, Rend):
    """ R0, b0, w0, Rend, phase0 all in list or numpy array
        phase0 in radians
    """
    nx,ny,nz = cyrr.shape
    spir = np.zeros([nx,ny,nz], dtype=np.float64)
    narms = len(R0)
    if phase0 is None:
        dphase = 2.*np.pi / narms
        phase0 = np.arange(narms)*dphase

    # let inner component be azimuthally smooth
    norm = 1. / narms

    for ia in range(narms):
        spirii = np.zeros([nx,ny,nz], dtype=np.float64)
        for ix in range(nx):
            rii = raxis[ix]
            if rii < R0[ia]:
                spirii[ix,:,:] = norm
            elif (rii >= R0[ia] and (rii <= Rend[ia])):
                for iz in range(nz):
                    pii = paxis[iz]
                    phispiral = np.log(rii/R0[ia]) / b0[ia] - phase0[ia]
                    phispiral = np.mod(phispiral, 2*np.pi)

                    wsig = w0[ia] / (2.*np.sqrt(2*np.log(2.))) #fwhm to wsig

                    pset = np.array([pii-2*np.pi, pii, pii+2.*np.pi], dtype=np.float64)
                    delp = min(abs(pset -phispiral))

                    spirii[ix,:,iz] = max(np.exp(-0.5*(delp/wsig)**2), 0.5*norm)
            else:
                spirii[ix,:,:] = norm

        spir = spir + spirii

    return spir
                
def getGap(cyrr, Rgap, wgap):
    nx,ny,nz = cyrr.shape
    gap = np.ones([nx,ny,nz], dtype=np.float64)
    ngaps = len(Rgap)
    for ii in range(ngaps):
        sigma = wgap[ii] / (2.*np.sqrt(2*np.log(2.)))
        gapii = 1. - np.exp(-0.5*((cyrr - Rgap[ii])/sigma)**2)
        gap = gap * gapii

    return gap

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
        xx = rr * np.sin(tt) * np.cos(pp)
        yy = rr * np.sin(tt) * np.sin(pp)
        zz = rr * np.cos(tt)
        cyrr = np.sqrt(xx**2. + yy**2.)
    elif ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2 + yy*2 + zz**2)
        cyrr = np.sqrt(xx**2 + yy**2)
    else:
        raise ValueError('crd_sys not specified')
    
    hh = ppar['Ht'] * (cyrr / ppar['Rt'])**(ppar['qheight'])
    sig_cyrr = fneq.eq_sig(cyrr, ppar['mdisk'], xx.min(), ppar['Rsig'],
        xx.max(), ppar['sigp'], 1)

#    armpart = getSpiral(cyrr, pp, ppar['Rspir'], ppar['bspir'], ppar['wspir'])
    armpart = getSpiral(cyrr, grid.x, grid.y, grid.z, ppar['Rspir'], ppar['bspir'], ppar['wspir'], ppar['pspir'], ppar['Rend']) 

    gappart = getGap(cyrr, ppar['Rgap'], ppar['wgap'])

    rhogas = sig_cyrr * armpart * gappart / np.sqrt(2.*np.pi) / hh * np.exp(-0.5*(zz/hh)**2)
    vol = grid.getCellVolume()
    sumd = rhogas * vol
    rat = ppar['mdisk'] / sumd.sum()
    rhogas = rhogas * rat

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
    if ppar['crd_sys'] == 'sph':
        rr = mesh[0]
        tt = mesh[1]
        pp = mesh[2]
        xx = rr * np.sin(tt) * np.cos(pp)
        yy = rr * np.sin(tt) * np.sin(pp)
        zz = rr * np.cos(tt)
        cyrr = np.sqrt(xx**2. + yy**2.)
    elif ppar['crd_sys'] == 'car':
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
    dweights = dinfo['dweights']

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
    # check inputs from ppar
    if 'crd_sys' not in ppar:
        raise ValueError('crd_sys is not in ppar')
    else:
        crd_sys = ppar['crd_sys']
    if 'altype' not in ppar:
        altype = '0'
    else:
        altype = ppar['altype']

    alvec = DiskEqs.eqDustAlignment(crd_sys, grid.x, grid.y, grid.z, altype, ppar)
    return alvec

