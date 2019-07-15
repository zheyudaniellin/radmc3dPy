"""
disk_ring.py

consider a disk with rings and flat power-law regions

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
        ['crd_sys', "'sph'", 'Coordinate system'],
        ['nx', '[60, 40, 30]', 'Number of grid points in the first dimension'],
        ['xbound', '[0.1*au, 30.*au, 110.*au, 250.*au]', 'Number of radial grid points'],
        ['ny', '[10,30, 30, 10]',
           'Number of grid points in the second dimension'],
        ['ybound', '[0.1, pi/6., pi/2., 5.*pi/6., 3.04]',
           'Number of radial grid points'],
        ['nz', '[361]', 'Number of grid points in the third dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
        # star related
        ['tstar', '[3900.0]', 'Temperature of star'],
        ['mstar', '[1.0*ms]', 'Mass of the star(s)'],
        ['rstar', '[2.5*rs]', 'Radius of star'],
        # gas density 
        ['Rin', '[0.1*au, 80*au]', 'inner bounding edge'],
        ['Rin_w', '[0, 1*au]', 'gaussian taper before inner edge'], 
        ['Rout', '[30*au, 120*au]', 'outer bounding edge'],
        ['Rout_w', '[1*au, 1*au]', 'gaussian taper after outer edge'], 
        ['sigp', '[-1.0, -1.5]', 'power-law surface density'],
        ['sig0', '[1e2, 1e1]', 'surface density at Rin in g/cm^2'], 
        ['ring_r', '[50*au]', 'location of gaussian ring'], 
        ['ring_win', '[5*au]', 'width of gaussian ring in inner radius'],
        ['ring_wout', '[5*au]', 'width of gaussian ring in outer radius'], 
        ['ring_a', '[1e2]', 'surface density at center of ring in g/cm^2]'], 
        ['cutgdens', '1e-30', 'cut for density'], 
        ['Rt', '100*au', 'radius for scale height'], 
        ['Ht', '10*au', 'scale height'],         
        ['qheight', '1.25', 'height power-law'], 
        # gas species
        ['gasspec_mol_name', "['12co']", 'name of molecule'],
        ['gasspec_mol_abun', '[5e-5]', 'mass abundance '],
        ['gasspec_mol_dbase_type', "['leiden']", ''],
        ['gasspec_mol_freezeout_dfact', '[1e-3]',
         'Factor by which the molecular abundance should be decreased in the freeze-out zone'],
        ['mol_freeze_Ht', '[24*au]', 'Height at Rt, with index=qheight, for freeze out to happen'],
        ['mol_freeze_del_hfrac', '0.2', 'Gaussian taper for freeze-out. del H = h * hfrac'],
        ['mol_snowR', '[20*au]', 'Radius when freeze out begins to happen'],
        # dust density
        # flat power-law parts
        ['dRin', '[0.1*au, 80*au]', 'inner bounding edge'],
        ['dRin_w', '[0, 1*au]', 'gaussian taper before inner edge'], 
        ['dRout', '[30*au, 120*au]', 'outer bounding edge'],
        ['dRout_w', '[1*au, 1*au]', 'gaussian taper after outer edge'], 
        ['dsigp', '[-1.0, -1.5]', 'power-law surface density'],
        ['dsig0', '[1e2, 1e1]', 'surface density at Rin'],
        # ring parts
        ['dring_r', '[50*au]', 'location of gaussian ring'],
        ['dring_win', '[5*au]', 'width of gaussian ring in inner radius'],
        ['dring_wout', '[5*au]', 'width of gaussian ring in outer radius'], 
        ['dring_a', '[1e2]', 'surface density at center of ring in g/cm^2]'],
        ['cutddens', '1e-30', 'cut for dust density'],
        ['dRt', '[100*au]', 'radius for scale height for each grain size'], 
        ['dHt', '[10*au]', 'scale height for each grain size'], 
        ['dqheight', '[1.25]', 'scale height power-law for dust'], 
        # temperature
        ['T0mid', '50', 'mid plane temperature at Rt'],
        ['T0atm', '50', 'atmosphere temperature at Rt'],
        ['zqratio', '3', 'factor of Ht of where temperature transition occurs'],
        ['qmid', '-0.5', 'midplane temperature exponent'],
        ['qatm', '-0.5', 'atmosphere temperature exponent'],
        ['hdel', '2', 'temperature transition exponent '],
        ['cuttemp', '10', 'temperature cut']
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

    ztrans = ppar['zqratio'] * ppar['Ht'] * (cyrr / ppar['Rt'])**(ppar['qheight'])

    tatm = ppar['T0atm'] * (rr / ppar['Rt'])**ppar['qatm']
    tmid = ppar['T0mid'] * (cyrr / ppar['Rt'])**ppar['qmid']

    if ppar['zqratio'] > 0:
        tgas = tatm
        reg = abs(zz) < ztrans

        tgas[reg] = tatm[reg] + (tmid[reg] - tatm[reg]) * ((np.cos(np.pi*0.5 * abs(zz[reg])/ztrans[reg]))**(2*ppar['hdel']))
    elif ppar['zqratio'] == 0:
        tgas = tatm
    elif ppar['zqratio'] == -1:
        tgas = tmid
    else:
        raise ValueError('zqratio value not accepted')

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

def fn_getring(cyrr, ring_r, ring_win, ring_wout, ring_a):
    ring = cyrr * 0
    reg = cyrr <= ring_r
    ring[reg] = ring_a * np.exp(-0.5 * ((cyrr[reg] - ring_r) / ring_win)**2)
    reg = cyrr > ring_r
    ring[reg] = ring_a * np.exp(-0.5 * ((cyrr[reg] - ring_r) / ring_wout)**2)

    return ring

def fn_getflat(cyrr, Rin_w, Rin, Rout, Rout_w, sigp, sig0):
    if Rin >= Rout:
        raise ValueError('Rin must be less than Rout: %.1e, %.1e'%(Rin, Rout))
    flat = cyrr * 0
    reg = (cyrr >= Rin) & (cyrr <= Rout)
    flat[reg] = sig0 * (cyrr[reg] / Rin)**(sigp)

    if Rin_w != 0:
        reg = cyrr < Rin
        flat[reg] = sig0 * np.exp(-((cyrr[reg] - Rin) / Rin_w)**2)

    if Rout_w != 0:
        reg = cyrr > Rout
        flat[reg] = sig0 * (Rout / Rin)**(sigp) * np.exp(-((cyrr[reg] - Rout) / Rout_w)**2)

    return flat

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
        cyrr = np.sqrt(xx**2. + yy**2.)
    else:
        raise ValueError('crd_sys not specified in ppar')

    # calculate scale height
    hh = ppar['Ht'] * (cyrr / ppar['Rt'])**ppar['qheight']

    # calculate surface density
    nflat = len(ppar['Rin'])
    flat = cyrr * 0.
    for ii in range(nflat):
        flatii = fn_getflat(cyrr, ppar['Rin_w'][ii], ppar['Rin'][ii], 
            ppar['Rout'][ii], ppar['Rout_w'][ii], 
            ppar['sigp'][ii], ppar['sig0'][ii])
        flat = flat + flatii

    nring = len(ppar['ring_r'])
    ring = cyrr * 0
    for ii in range(nring):
        ringii = fn_getring(cyrr, ppar['ring_r'][ii], 
            ppar['ring_win'][ii],ppar['ring_wout'][ii], 
            ppar['ring_a'][ii])
        ring = ring + ringii

    sig = flat + ring

    rhogas = sig / np.sqrt(2.*np.pi) / hh * np.exp(-0.5 * (zz / hh)**2)
    reg = rhogas < ppar['cutgdens']
    rhogas[reg] = ppar['cutgdens']

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
        xx = rr * np.sin(tt) * np.sin(pp)
        yy = rr * np.sin(tt) * np.cos(pp)
        zz = rr * np.cos(tt)
        cyrr = np.sqrt(xx**2. + yy**2)
    elif ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        cyrr = np.sqrt(xx**2. + yy**2.)
    else:
        raise ValueError('crd_sys not specified in ppar')

    # calculate surface density
    nflat = len(ppar['dRin'])
    flat = cyrr * 0.
    for ii in range(nflat):
        flatii = fn_getflat(cyrr, ppar['dRin_w'][ii], ppar['dRin'][ii], 
            ppar['dRout'][ii], ppar['dRout_w'][ii], 
            ppar['dsigp'][ii], ppar['dsig0'][ii])
        flat = flat + flatii

    nring = len(ppar['dring_r'])
    ring = cyrr * 0
    for ii in range(nring):
        ringii = fn_getring(cyrr, ppar['dring_r'][ii], 
            ppar['dring_win'][ii], ppar['dring_wout'][ii], 
            ppar['dring_a'][ii])
        ring = ring + ringii

    sig = flat + ring

    # calculate the dust density
    op = dustopac.radmc3dDustOpac()
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])
    dweights = dinfo['dweights']

    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64) 
    for ig in range(ngs):
        hhii = ppar['dHt'][ig] * (cyrr / ppar['dRt'][ig])**ppar['dqheight'][ig]
        rho_ig = sig / np.sqrt(2.*np.pi) / hhii * np.exp(-0.5*(zz/hhii)**2)
        rhodust[:,:,:,ig] = rho_ig * dweights

    reg = rhodust < ppar['cutddens']
    rhodust[reg]= ppar['cutddens']

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
