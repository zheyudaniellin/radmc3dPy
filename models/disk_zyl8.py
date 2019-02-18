"""zyl8
disk, envelope system
always assume there are at most two different grains
the first one is for the envelope, the second one is for disk
if only one, then the same for both

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
    * getExternalSource()

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
from .. import analyze
from .. import dustopac
import pdb
import fneq
from . import DiskEqs

def getModelDesc():
    """Provides a brief description of the model
    """

    return "A ppdisk model for HH212 disk. Dust density distribution like Lee 2017, and temperature like Rosenfield 2012"
           

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
	['nx', '[30, 60, 10]', 'Number of grid points in the first dimension'],
        ['xbound', '[0.1*au,30.*au, 120.0*au, 200*au]', 'Number of radial grid points'],
        ['ny', '[10,30,30,10]', 'Number of grid points in the first dimension'],
        ['ybound', '[0.1, pi/3., pi/2., 2.*pi/3., 3.04]', 'Number of radial grid points'],
        ['nz', '[61]', 'Number of grid points in the first dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
        # gas 
      	['gasspec_mol_name', "['12co','13co','c18o','c17o']", 'name of molecule'],
	['gasspec_mol_abun', '[4e-5, 5.78e-7, 7.18e-8, 1e-8]', 'abundance of molecule'],
	['gasspec_mol_dbase_type', "['leiden','leiden','leiden','leiden']", 'data base type'],
        ['gasspec_mol_dissoc_taulim', '[1.0, 1.0, 1.0, 1.0]', 'Continuum optical depth limit below which all molecules dissociate'],
        ['gasspec_mol_freezeout_temp', '[19.0, 19.0, 19.0, 19.0]', 'Freeze-out temperature of the molecules in Kelvin'],
        ['gasspec_mol_freezeout_dfact', '[1e-8, 1e-8, 1e-8, 1e-8]',
         'Factor by which the molecular abundance should be decreased in the frezze-out zone'],
        ['gasspec_vturb', '0.2e5', 'Microturbulent line width'],
        # density 
	['g2d', '0.01', ' Dust to Gas ratio'],
        ['mstar', '[0.25*ms]', 'Mass of the star(s)'],
        ['Rinner', '0.01*au', ' Inner radius of the disk'],
        ['Router', '70.0*au', ' Outer radius of the disk'],	      	
	['Rsig', '30.*au', ' characteristic radius for exponential tapering sigam'], 
	['sigp', '0.2', 'exponent value for sigma'],
	['mdisk', '0.05*ms', 'mass of disk'],
	['sigma_type', '0', '0-polynomial, 1-exponential tapering'],
        ['cutgdens', '1e-30', 'cut for gas density'],
        # envelope
        ['dMenv', '5e-6', 'envelope accretion rate onto disk [Msun/year]'], 
        ['rhoRc', '5e-16', 'Density at Rc. This is used if dMenv is < 0'], 
        ['Rc', '50.*au', 'radius where envelope meets disk'],
        ['topen', '30', 'opening angle [deg]'],
        ['deltopen', '5', 'level of tapering angle [deg]'], 
        # temperature
        ['Rt','20.*au', ' characteristic radius for temperature, height'],
	['T0mid', '105.', 'midplane temperature at Rt'], #with dM=5e-6, T=30 at 20au
	['qmid', '-0.75', 'midplane temperature exponent'],
	['T0atm', '145.', 'atmosphere temperature value at Rt'], 
	['qatm', '-0.5', 'atmosphere temperature exponent'],
        ['cuttemp', '10', 'temperature cut'],
        # height
	['H0', '7.5*au', 'height at Rt'],
	['qheight', '1.125', 'height exponent'],
	['zqratio', '3.0', 'multiple of scale height for temperature transition'],
	['hdel', '2.0', 'power of transition for temperature'],
        # alignment
        ['altype', "'toroidal'", 'alignment type'], 
        # velocity
        ['vsys', '0.0', 'systemic velocity']
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
    tgas = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    xaxis = grid.x; nx = grid.nx
    yaxis = grid.y; ny = grid.ny
    zaxis = grid.z; nz = grid.nz

    # spherical coordinates
    if ppar['crd_sys'] == 'sph':
        for ix in range(nx):
            for iy in range(ny):
                xii = xaxis[ix] * np.sin(yaxis[iy])
    	        zii = xaxis[ix] * abs(np.cos(yaxis[iy]))
                rii = (xii**2 + zii**2)**0.5
                hii = ppar['H0'] * (xii / ppar['Rt'])**(ppar['qheight'])
                tmid = ppar['T0mid'] * (xii/ppar['Rt'])**(ppar['qmid'])
                tatm = ppar['T0atm'] * (rii/ppar['Rt'])**(ppar['qatm'])
                zq = hii * ppar['zqratio']
                if zii >= zq:
                    tgasii = tatm
                else:
                    tgasii = tatm + (tmid - tatm)*((np.cos(np.pi*0.5 * zii/zq))**(2.0*ppar['hdel']))
                if tgasii < ppar['cuttemp']:
                    tgas[ix,iy,:] = ppar['cuttemp']
                else:
                    tgas[ix,iy,:] = tgasii


    # cartesian coordinates
    if ppar['crd_sys'] == 'car':
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    xii = (xaxis[ix]**2 + yaxis[iy]**2)**0.5
                    zii = abs(zaxis[iz])
                    rii = (xii**2 + zii**2)**0.5
                    hii = ppar['H0'] * (xii / ppar['Rt'])**(ppar['qheight'])
                    tmid = ppar['T0mid'] * (xii/ppar['Rt'])**(ppar['qmid'])
                    tatm = ppar['T0atm'] * (rii/ppar['Rt'])**(ppar['qatm'])
                    zq = hii * ppar['zqratio']
                    if zii >= zq:
                        tgasii = tatm
                    else:
                        tgasii = tatm + (tmid - tatm)*((np.cos(np.pi*0.5 * zii/zq))**(2.0*ppar['hdel']))
                    if tgasii < ppar['cuttemp']:
                        tgas[ix,iy,iz] = ppar['cuttemp']
                    else:
                        tgas[ix,iy,iz] = tgasii

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

    tgas = getGasTemperature(grid=grid, ppar=ppar)
    tdust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64)
    for ii in range(ngs):
        tdust[:,:,:,ii] = tgas
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
    
    # Read the dust density and temperature
    try:
        data = analyze.readData(ddens=True, dtemp=True, binary=True)
    except:
        try:
            data = analyze.readData(ddens=True, dtemp=True, binary=False)
        except:
            msg = 'Gas abundance cannot be calculated as the required dust density and/or temperature '\
                  + 'could not be read in binary or in formatted ascii format.'
            raise RuntimeError(msg)

    nspec = len(ppar['gasspec_mol_name'])
    if ppar['gasspec_mol_name'].__contains__(ispec):

        sid = ppar['gasspec_mol_name'].index(ispec)
        # Check where the radial and vertical optical depth is below unity
        gasabun = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)

        for spec in range(nspec):
            gasabun[:, :, :] = ppar['gasspec_mol_abun'][sid]

        for iz in range(data.grid.nz):
            for iy in range(data.grid.ny):
                ii = (data.dusttemp[:, iy, iz, 0] < ppar['gasspec_mol_freezeout_temp'][sid])
                gasabun[ii, iy, iz] = ppar['gasspec_mol_abun'][sid] * ppar['gasspec_mol_freezeout_dfact'][sid]

    else:
        gasabun = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) + 1e-10
        txt = 'Molecule name "'+ispec+'" is not found in gasspec_mol_name \n A default 1e-10 abundance will be used'
        warnings.warn(txt, RuntimeWarning)

    return gasabun

def fn_scaleheight(cyrr, Rt, Ht, qheight, Router, Hd, mode='1'):
    """
    scale height as a function of cylindrical radius

    cyrr : meshgrid. Cylindrical radius

    mode : string
           '0': flare
           '1': flare + exponential taper
           '2': flare x exponential taper
           '3': flare x root taper

    Returns
    -------
    hh : meshgrid. scale height
    """
    if isinstance(mode, int):
        mode = str(mode)
    if mode is '0':
        hh = Ht * (cyrr / Rt)**qheight
    elif mode is '1':
        hh = Ht * (cyrr / Rt)**qheight
        reg = (cyrr > Rt) & (cyrr <= Router)
        hh[reg] = Ht * ( np.exp(-((cyrr[reg] - Rt) / (Router - Rt))**Hd) )
        reg = cyrr > Router
        hh[reg] = 0.
    elif mode is '2':
        flare = (cyrr / Rt)**qheight
        tap = np.exp(-((cyrr - Rt) / (Router - Rt))**Hd)
        hh = Ht * flare * tap
    elif mode is '3':
        flare = (cyrr / Rt)**qheight
        tap = (np.abs((cyrr - Router) / (Rt - Router)))**(1./Hd)
        hh = Ht * flare * tap
        reg = cyrr > Router
        hh[reg] = 0.
    else:
        raise ValueError('mode for scaleheight not found')

    return hh
   
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

    rhogas = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) + 1e-30
    hh = fn_scaleheight(cyrr, ppar['Rt'], ppar['H0'], ppar['qheight'], 
            ppar['Router'], 0, mode='0')

    sig_cyrr = fneq.eq_sig(cyrr, ppar['mdisk'], ppar['Rinner'], ppar['Rsig'], 
        ppar['Router'], ppar['sigp'], ppar['sigma_type'])

    reg = hh > 0
    rhogas[reg] = sig_cyrr[reg] / np.sqrt(2.*np.pi) / hh[reg] * np.exp(-0.5*(zz[reg]/hh[reg])**2.)

    # flag out regions with 0 scale height
    reg = hh <= 0.
    rhogas[reg] = ppar['cutgdens']

    # include envelope
    r2d = rr[:,:,0]
    t2d = tt[:,:,0]
    if ppar['dMenv'] >= 0:
        GMR3 = natconst.gg * ppar['mstar'][0] * ppar['Rc']**3
        dMenv = ppar['dMenv'] * natconst.ms / natconst.year
        rhoRc = dMenv / (4.*np.pi) / np.sqrt(GMR3)
    else:
        rhoRc = ppar['rhoRc']
    envdens2d = DiskEqs.eqEnvelopeDens(r2d, t2d, ppar['Rc'], rhoRc)
    for ip in range(grid.nz):
        rhoin = rhogas[:,:,ip]
        rhogas[:,:,ip] = rhogas[:,:,ip] + envdens2d

    # flag out regions lower than cutgdens
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

    # read dust grain information
    op = dustopac.radmc3dDustOpac()
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])

    # create dust distribution
    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64) 

    # disk component
    hh = fn_scaleheight(cyrr, ppar['Rt'], ppar['H0'], ppar['qheight'],
            ppar['Router'], 0, mode='0')

    sig_cyrr = fneq.eq_sig(cyrr, ppar['mdisk'], ppar['Rinner'], ppar['Rsig'],
        ppar['Router'], ppar['sigp'], ppar['sigma_type'])

    rho_disk = sig_cyrr / np.sqrt(2.*np.pi) / hh * np.exp(-0.5*(zz/hh)**2.)
    rho_disk = rho_disk * ppar['g2d']

    # envelope component
    r2d = rr[:,:,0]
    t2d = tt[:,:,0]
    if ppar['dMenv'] >= 0:
        GMR3 = natconst.gg * ppar['mstar'][0] * ppar['Rc']**3
        dMenv = ppar['dMenv'] * natconst.ms / natconst.year
        rhoRc = dMenv / (4.*np.pi) / np.sqrt(GMR3)
    else:
        rhoRc = ppar['rhoRc']
    rho_env2d = DiskEqs.eqEnvelopeDens(r2d, t2d, ppar['Rc'], rhoRc)
    rho_env2d = rho_env2d * ppar['g2d']

    rho_env3d = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    for ip in range(grid.nz):
        rho_env3d[:,:,ip] = rho_env2d

    # decide on how to add it together
    if ngs == 1:
        rhodust[:,:,:,0] = rho_env3d + rho_disk
    elif ngs == 2:
        ig = 0 # small grains
        rhodust[:,:,:,ig] = rho_env3d

        ig = 1 # big grains
        rhodust[:,:,:,ig] = rho_disk
    else:
        raise ValueError('number of grains can only be less than 2 for this model')

    # flag out regions lower than cutgdens
    reg = rhodust < (ppar['cutgdens'] * ppar['g2d'])
    rhodust[reg] = ppar['cutgdens'] * ppar['g2d']
    
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
    xaxis = grid.x; nx = grid.nx
    yaxis = grid.y; ny = grid.ny
    zaxis = grid.z; nz = grid.nz

    vel = np.zeros([nx, ny, nz, 3], dtype=np.float64)

    if ppar['crd_sys'] == 'sph':
        for ix in range(nx):
            for iy in range(ny):
                xii = xaxis[ix] * np.sin(yaxis[iy])
                zii = xaxis[ix] * abs(np.cos(yaxis[iy]))
                rii = xaxis[ix]
                vkep = np.sqrt(natconst.gg*ppar['mstar'][0] / xii) * (pow(1.+(zii/xii)**2., -0.75))
                vel[ix, iy,:,2] = vkep + ppar['vsys']

    if ppar['crd_sys'] == 'car':
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    xii = xaxis[ix]
                    yii = yaxis[iy]
                    zii = zaxis[iz]
                    rii = np.sqrt(xii**2 + yii**2)
                    vkep = np.sqrt(natconst.gg*ppar['mstar'][0] / xii) * (pow(1.+(zii/xii)**2., -0.75))
                    vel[ix,iy,:,2] = vkep + ppar['vsys']


    return vel

def getDustAlignment(grid=None, ppar=None):
    """ calculates the dust alignment
    Parameters
    ----------
    grid : radmc3dGrid
            An instance of radmc3dGrid class
    ppar : dictionary
            Dictionary containing all parameters of the model

    Returns
    ------
    Returns the alignment grid
    """
    # check inputs from ppar
    if 'crd_sys' not in ppar:
        raise ValueError('crd_sys is not in ppar')
    else:
        crd_sys = ppar['crd_sys']
    if 'altype' not in ppar:
        altype = '0'
    else:
        altype = ppar['altype']

    alvec = DiskEqs.eqDustAlignment(crd_sys, grid.x, grid.y, grid.z, altype)
    return alvec
