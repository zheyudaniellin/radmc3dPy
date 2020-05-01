"""kwon 2011 to model HL Tau. column density like power-law with tapering, temperature like Rosenfeld 2012.
   can accept 

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
from .. import analyze
import pdb
import fneq

def getModelDesc():
    """Provides a brief description of the model
    """

    return "Kwon 2011 disk"
           

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
	['nx', '[30,60]', 'Number of grid points in the first dimension'],
        ['xbound', '[0.1*au,20.*au, 250.0*au]', 'Number of radial grid points'],
        ['ny', '[10,30,30,10]', 'Number of grid points in the first dimension'],
        ['ybound', '[0., pi/3., pi/2., 2.*pi/3., pi]', 'Number of radial grid points'],
        ['nz', '[30]', 'Number of grid points in the first dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
      	['gasspec_mol_name', "['12co']", 'name of molecule'],
	['gasspec_mol_abun', '[4e-5]', 'abundance of molecule'],
	['gasspec_mol_dbase_type', "['leiden']", 'data base type'],
        ['gasspec_mol_dissoc_taulim', '[1.0]', 'Continuum optical depth limit below which all molecules dissociate'],
        ['gasspec_mol_freezeout_temp', '[19.0]', 'Freeze-out temperature of the molecules in Kelvin'],
        ['gasspec_mol_freezeout_dfact', '[1e-3]',
         'Factor by which the molecular abundance should be decreased in the frezze-out zone'],
        ['gasspec_vturb', '0.2e5', 'Microturbulent line width'],
        ['ngs', '1', 'number of grain sizes'],
        ['mixgsize', '0', '1 to mix grain size, 0 to not'],
	['g2d', '0.01', ' Dust to Gas ratio'],
        ['mstar', '0.55*ms', 'Mass of the star(s)'],
        ['Rinner', '2.4*au', ' Inner radius of the disk'],
        ['Router', '200.0*au', ' Outer radius of the disk'],	      	
	['Rc','79.*au', ' characteristic radius for density and height'], 
	['sigp', '1.064', 'exponent value for sigma'],
	['mdisk', '0', 'mass of disk. set to 0 to use rhogas0'],
        ['rhogas0', '1.124e-14','value of gas density at Rc'],
	['sigma_type', '1', '0-polynomial, 1-exponential tapering'],
        ['R0', '10*au', 'radius for midplane temperature. cylindrical'],
	['T0mid', '70.', 'midplane temperature at R0'],
        ['Rs0', '3*au', 'radius for atmosphere temperature. spherical'],
	['T0atm', '400.', 'atmosphere temperature value at Rs0'],
	['qtemp', '0.43', 'temperature exponent'],
	['H0', '16.8*au', 'height at Rc'],
	['wqratio', '3.0', 'multiple of scale height for temperature transition'],
	['cuttemp', '10', 'temperature cut'],
	['cutgdens', '1e-30', 'cut for gas density'], 
        ['altype', '0', 'type of alignment for grains']
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
                rii = xaxis[ix]
                hii = ppar['H0'] * (xii / ppar['Rc'])**(1.5 - ppar['qtemp']*0.5)
                wii = np.e**(- 0.5*(zii / ppar['wqratio'] / hii)**2)
                tmid = ppar['T0mid'] * (xii/ppar['R0'])**(-ppar['qtemp'])
                tatm = ppar['T0atm'] * (rii/ppar['Rs0'])**(-ppar['qtemp'])
                tgasii = wii * tmid + ( 1. - wii ) * tatm
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
                    hii = ppar['H0'] * (xii / ppar['Rc'])**(1.5 - ppar['qtemp']*0.5)
                    wii = np.e**(- 0.5 * (zii / ppar['wqratio'] / hii)**2)
                    tmid = ppar['T0mid'] * (xii/ppar['R0'])**(-ppar['qtemp'])
                    tatm = ppar['T0atm'] * (rii/ppar['Rs0'])**(-ppar['qtemp'])
                    tgasii = wii * tmid + (1.-wii)*tatm
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
    tgas = getGasTemperature(grid=grid, ppar=ppar)
    if 'ngs' in ppar:
        ngs = ppar['ngs']
        if 'mixgsize' in ppar:
            if ppar['mixgsize'] == 1:
                ngs = 1
    else:
        ngs = 1
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
    xaxis = grid.x
    yaxis = grid.y
    zaxis = grid.z
    nx = grid.nx
    ny = grid.ny
    nz = grid.nz
   
    gam = -1.5 + ppar['sigp'] + ppar['qtemp']*0.5

    # for spherical coordinates
    if ppar['crd_sys'] == 'sph':
        for xx in range(nx):
            for yy in range(ny):
                xii = xaxis[xx] * np.sin(yaxis[yy])
                zii = xaxis[xx] * abs(np.cos(yaxis[yy]))
                if ppar['mdisk'] != 0:
                    hii = fneq.eq_taperheight(xii, ppar['Rc'], ppar['H0'], 
                        1.5 - ppar['qtemp']*0.5, ppar['Router'], -1)
                    sigii = fneq.eq_sig(xii,ppar['mdisk'],ppar['Rinner'],ppar['Rc'],
#                        ppar['Router'],ppar['sigp'],ppar['sigma_type'])
                         ppar['Router'],gam, ppar['sigma_type']) #use Kwon's def of gamma
                    gdensii = sigii / (2.0*np.pi)**0.5 / hii * np.e**(-0.5*zii**2 / hii**2)
                else:
                    hii = ppar['H0'] * ((xii/ppar['Rc'])**(1.5-ppar['qtemp']*0.5))
                    dum = ppar['rhogas0'] * (xii/ppar['Rc'])**(-ppar['sigp'])
                    dum = dum * np.exp(-(xii/ppar['Rc'])**(3.5-ppar['sigp']-0.5*ppar['qtemp']))
                    gdensii = dum * np.exp(-(zii / hii)**2)

                if xii < ppar['Rinner']:
                    gdensii = ppar['cutgdens']

                if gdensii < ppar['cutgdens']:
                    rhogas[xx,yy,:] = ppar['cutgdens']
                else:
                    rhogas[xx,yy,:] = gdensii

    # for cartesian coordinates
    if ppar['crd_sys'] == 'car':
        for xx in range(nx):
            for yy in range(ny):
                for zz in range(nz):
                    xii = (xaxis[xx]**2 + yaxis[yy]**2)**0.5
                    zii = abs(zaxis[yy])

                    if ppar['mdisk'] != 0:
                        hii = fneq.eq_taperheight(xii, ppar['Rc'], ppar['H0'],
                            1.5 - ppar['qtemp']*0.5, ppar['Router'], -1)
                        sigii = fneq.eq_sig(xii,ppar['mdisk'],ppar['Rinner'],ppar['Rc'],
#                            ppar['Router'],ppar['sigp'],ppar['sigma_type'])
                         ppar['Router'],gam, ppar['sigma_type']) #use Kwon's def of gamma
                        gdensii = sigii / (2.0*np.pi)**0.5 / hii * np.e**(-0.5*zii**2 / hii**2)
                    else:
                        hii = ppar['H0'] * (xii/ppar['Rc'])**(1.5-ppar['qtemp']*0.5)
                        dum = ppar['rhogas0'] * (xii/ppar['Rc'])**(-ppar['sigp'])
                        dum = dum * np.e**(-(xii/ppar['Rc'])**(3.5-ppar['sigp']-0.5*ppar['qtemp']))
                        gdensii = dum * np.e**(-(zii / hii)**2)
                    if xii < ppar['Rinner']:
                        gdensii = ppar['cutgdens']
                    if gdensii < ppar['cutgdens']:
                        rhogas[xx,yy,zz] = ppar['cutgdens']
                    else:
                        rhogas[xx,yy,zz] = gdensii
                
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

    rhogas = getGasDensity(grid=grid, ppar=ppar)
    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, 1], dtype=np.float64) 
    rhodust[:, :, :, 0] = rhogas * ppar['g2d']
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

    # same keplerian velocity in cylinders >> in hyrdo static equilibrium

    if ppar['crd_sys'] == 'sph':
        for ix in range(nx):
            for iy in range(ny): 
                xii = xaxis[ix] * np.sin(yaxis[iy])
                vkep = np.sqrt(natconst.gg*ppar['mstar'] / xii)
                vel[ix, iy,:,2] = vkep

    if ppar['crd_sys'] == 'car':
        for ix in range(nx):
            for iy in range(ny):
                xii = xaxis[ix]
                yii = yaxis[iy]
                rii = np.sqrt(xii**2 + yii**2)
                vkep = np.sqrt(natconst.gg*ppar['mstar'] / rii)
                vel[ix,iy,:,2] = vkep

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
        print('altype not found in ppar. Using zero alignment')
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


