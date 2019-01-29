"""Disk with analytical multiple layers of dust
the dust grain sizes go from small to large

This template is an empty model, i.e. all model functions return zeros in the appropriate arrays and dimensions. 
The purpose of this model is to illustrate the names and syntax of the model functions. Hence, this file can
be a starting point for implementing new models in the library. 

Can look at Cleeves' paper on IM Lup with dust and gas modeling

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
        # gas density related
        ['mdisk', '0.17*ms', 'Total mass of disk'],
        ['g2d', '0.01', 'gas to dust ratio'],
        ['sigp', '1.0', 'power index of Lynden-Bell surface density'],
        ['Rin', '0.2*au', 'Inner radii of disk '],
        ['Rsig', '100*au', 'Characteristic radius for surface density'],
        ['cutgdens', '1e-30', 'cut for density, also ambient density'],
        ['Rt', '100.*au', 'radius for scale height'],
        ['Ht', '12.*au', 'Scale height'],
        ['qheight', '1.15', 'power index for scale height'],

        # dust density related
        # should be in number of dust species. increase from small to large
        ['dsigp', '[1.0, 0.3]', 'power index of Lynden-Bell surface density'],
        ['dRsig', '[100*au, 50*au]', 'Characteristic radius for surface density'],
        ['dRt', '[100.*au, 100*au]', 'radius for scale height'],
        ['dHt', '[12.*au, 3.*au]', 'Scale height'],
        ['dqheight', '[1.15, 1.15]', 'power index for scale height'],

        # temperature related
        ['cuttemp', '10', 'temperature cut'],
        ['alph', '0.01', 'viscosity parameter. set to -1 to use dM (not implemented yet).'],
        ['dM', '0.', 'constant accretion rate across disk. Uses this if alph is -1'],
        # alignment
        ['altype', "'toroidal'", 'alignment type']
             ]

    return defpar


def getGasTemperature(grid=None, ppar=None):
    """Calculates/sets the gas temperature
    Estimated by vertically isothermal disk
    
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

    tgas = np.sqrt(0.5 * ppar['rstar'][0] / cyrr) * ppar['tstar'][0]
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

def fn_getsig(cyrr, sigp, mdisk, Rsig):
    sig0 = (2.-sigp)*mdisk / 2. / np.pi / Rsig**2.
    sig_cyrr = sig0 * (cyrr / Rsig)**(-sigp) * np.exp(-(cyrr/Rsig)**(2.-sigp))
    return sig_cyrr

def fn_getrhoxyz(cyrr,zz, Ht, Rt, qheight, sigp, mdisk, Rsig):
    hh = Ht * (cyrr / Rt)**(qheight)
    sig_cyrr = fn_getsig(cyrr, sigp, mdisk, Rsig)
    rho = sig_cyrr / np.sqrt(2.*np.pi) / hh * np.exp(-0.5*(zz/hh)**2.)
    return rho

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

    rhogas = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
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

    rhogas = fn_getrhoxyz(cyrr, zz, ppar['Ht'], ppar['Rt'], ppar['qheight'], 
        ppar['sigp'], ppar['mdisk'], ppar['Rsig'])
    reg = rhogas < ppar['cutgdens']
    rhogas[reg]= ppar['cutgdens']

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

    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64)

    # names of parameters that will be used
    # want to check if the number of parameters are same as number of dust species
    parnames = ['dsigp', 'dRsig', 'dRt', 'dHt', 'dqheight']
    nparnames = len(parnames)
    for ii in range(nparnames):
        if parnames[ii] in ppar:
            nlen = len(ppar[parnames[ii]])
            if nlen is not ngs:
                raise ValueError('%s must be same as number of dust species %d'%(parnames[ii], ngs))
        else:
            raise ValueError('ppar does not have %s'%(parnames[ii]))

    # now iterate through the arguments and produce each layer of dust
    for ig in range(ngs):
        hii = ppar['dHt'][ig]
        rtii = ppar['dRt'][ig]
        qhii = ppar['dqheight'][ig]
        sigii = ppar['dsigp'][ig]
        mdiskii = ppar['mdisk'] * ppar['g2d'] * dweights[ig]
        rsigii = ppar['dRsig'][ig]

        rho_ig = fn_getrhoxyz(cyrr, zz, hii, rtii, qhii, sigii, mdiskii, rsigii)
        reg = rho_ig < ppar['g2d'] * ppar['cutgdens']
        rho_ig[reg] = ppar['g2d'] * ppar['cutgdens']
        rhodust[:,:,:,ig] = rho_ig

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
    elif altype is 'y':
        alvec[:,:,:,1] = 1.0
    # z 
    elif altype is 'z':
        alvec[:,:,:,2] = 1.0
    # radial
    elif altype is 'radial':
        alvec[:,:,:,0] = xx / rr
        alvec[:,:,:,1] = yy / rr
        alvec[:,:,:,2] = zz / rr

    # cylradial
    elif altype is 'cylradial':
        cyl_rr = np.sqrt(xx**2 + yy**2)
        alvec[:,:,:,0] = xx / cyl_rr
        alvec[:,:,:,1] = yy / cyl_rr
    # poloidal
    elif altype is 'poloidal':
        raise ValueError('poloidal no implemented yet')
    # toroidal
    elif altype is 'toroidal':
        cyl_rr = np.sqrt(xx**2 + yy**2)
        alvec[:,:,:,0] = yy / cyl_rr
        alvec[:,:,:,1] = -xx / cyl_rr
    elif altype is '0':
        alvec = alvec
    else:
        raise ValueError('no acceptable altype argument')

    if altype is not '0':
    # Normalize
        length = np.sqrt(alvec[:,:,:,0]*alvec[:,:,:,0] +
                     alvec[:,:,:,1]*alvec[:,:,:,1] +
                     alvec[:,:,:,2]*alvec[:,:,:,2])
        alvec[:,:,:,0] = np.squeeze(alvec[:,:,:,0]) / ( length + 1e-60 )
        alvec[:,:,:,1] = np.squeeze(alvec[:,:,:,1]) / ( length + 1e-60 )
        alvec[:,:,:,2] = np.squeeze(alvec[:,:,:,2]) / ( length + 1e-60 )

    return alvec

def getViscousHeating(grid=None, ppar=None):
    """ Calculate the viscous accretion heating
     needs:
         mstar = star mass
         alph = viscosity parameter
         scale height
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

    hh = ppar['Ht'] * (cyrr / ppar['Rt'])**ppar['qheight']
    wk = np.sqrt(natconst.gg * ppar['mstar'][0] / cyrr**3.)

    rhogas = getGasDensity(grid=grid, ppar=ppar)
    reg = rhogas <= ppar['cutgdens']

    if ppar['alph'] >= 0.:
        # use alpha
        qvis = rhogas * ppar['alph'] * hh**2 * wk**3 * 9./4.
    else:
        # use accretion rate [Msun / year]
        acc = ppar['dM'] * natconst.ms / natconst.year
        qvis = 3. * acc * wk**2 / 4. / np.pi / np.sqrt(2.*np.pi) / hh * np.exp(-0.5*(zz/hh)**2)
    qvis[reg] = 0.   

    chkqvis = np.isfinite(qvis)
    if False in chkqvis:
        raise ValueError('qvis is not finite')

#    vol = grid.getCellVolume()
#    tol_lum = np.sum(vol*qvis)
#    lum = vol*qvis
#    tot_lum = lum.sum()
#    print('setting up heatsource.inp')
#    print('total heating luminosity: %f Lsun'%(tot_lum/natconst.ls))

    return qvis

