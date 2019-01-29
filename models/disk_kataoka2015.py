"""kataoka2015 models a gaussian ringed disk in his 2015 paper

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

from .. import analyze
import pdb
import fneq

def getModelDesc():
    """Provides a brief description of the model
    """

    return "A ppdisk model like Kataoka 2015"

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
	['nx', '[30,50]', 'Number of grid points in the first dimension'],
        ['xbound', '[1.0*au,20.*au, 100.0*au]', 'Number of radial grid points'],
        ['ny', '[10,30,30,10]', 'Number of grid points in the first dimension'],
        ['ybound', '[0., pi/3., pi/2., 2.*pi/3., pi]', 'Number of radial grid points'],
        ['nz', '[30]', 'Number of grid points in the first dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
      	['gasspec_mol_name', "['co']", 'name of molecule'],
	['gasspec_mol_abun', '[1e-4]', 'abundance of molecule'],
	['gasspec_mol_dbase_type', "['leiden']", 'data base type'],
        ['gasspec_mol_dissoc_taulim', '[1.0]', 'Continuum optical depth limit below which all molecules dissociate'],
        ['gasspec_mol_freezeout_temp', '[19.0]', 'Freeze-out temperature of the molecules in Kelvin'],
        ['gasspec_mol_freezeout_dfact', '[1e-3]',
         'Factor by which the molecular abundance should be decreased in the frezze-out zone'],
        ['gasspec_vturb', '0.2e5', 'Microturbulent line width'],
	['g2d', '0.01', ' Dust to Gas ratio'],
        ['sig0', '[0.6]', 'column density at Rd'],
        ['Rd', '[173*au]', 'radius'], 
        ['mstar', '0.3*ms', 'Mass of the star(s)'],
        ['Rinner', '0.01*au', ' Inner radius of the disk'],
        ['Router', '100.0*au', ' Outer radius of the disk'],	      	
	['Rt','30.*au', ' characteristic radius for temperature, height'], 
	['Rsig', '30.*au', ' characteristic radius for exponential tapering sigam'], 
	['sigp', '1.0', 'exponent value for sigma'],
	['mdisk', '0.05*ms', 'mass of disk'],
	['sigma_type', '1', '0-polynomial, 1-exponential tapering'],
	['T0mid', '100.', 'midplane temperature at Rt'],
	['qmid', '-0.75', 'midplane temperature exponent'],
	['T0atm', '125.', 'atmosphere temperature value at Rt'],
	['qatm', '-0.5', 'atmosphere temperature exponent'],
	['H0', '7.0*au', 'height at Rt'],
	['qheight', '1.125', 'height exponent'],
        ['Hd', '-1', 'exponent for decrease of height. -1 to turn off'],
	['zqratio', '3.0', 'multiple of scale height for temperature transition'],
	['hdel', '5.0', 'power of exponential decrease for gas density'],
	['cuttemp', '10', 'temperature cut'],
	['cutgdens', '1e-30', 'cut for gas density']
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
    tgas = getGasTemperature(grid=grid, ppar=ppar)
    tdust = np.zeros([grid.nx, grid.ny, grid.nz, 1], dtype=np.float64)
    tdust[:,:,:,0] = tgas
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
    
    # for spherical coordinates
    if ppar['crd_sys'] == 'sph':
        for xx in range(nx):
    	    for yy in range(ny):
                xii = xaxis[xx] * np.sin(yaxis[yy])
                zii = xaxis[xx] * abs(np.cos(yaxis[yy]))
                # hii = ppar['H0'] * (xii / ppar['Rt'])**(ppar['qheight'])
                hii = fneq.eq_taperheight(xii, ppar['Rt'], ppar['H0'], 
                    ppar['qheight'], ppar['Router'], ppar['Hd'])
                sigii = fneq.eq_sig(xii,ppar['mdisk'],ppar['Rinner'],ppar['Rt'],
                    ppar['Router'],ppar['sigp'],ppar['sigma_type'])
                gdensii = sigii / (2.0*np.pi)**0.5 / hii * np.e**(-0.5*zii**2 / hii**2)
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
                    #hii = ppar['H0'] * (xii / ppar['Rt'])**(ppar['qheight'])
                    hii = fneq.eq_taperheight(xii, ppar['Rt'], ppar['H0'], 
                        ppar['qheight'], ppar['Router'], ppar['Hd'])
                    sigii = fneq.eq_sig(xii,ppar['mdisk'], ppar['Rinner'], ppar['Rt'],
                        ppar['Router'], ppar['sigp'], ppar['sigma_type'])
                    gdensii = sigii / (2.0*np.pi)**0.5 / hii * np.e**(-0.5*zii**2 / hii**2)
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
                vkep = np.sqrt(gg*ppar['mstar'][0] / xii)
                vel[ix, iy,:,2] = vkep

    if ppar['crd_sys'] == 'car':
        for ix in range(nx):
            for iy in range(ny):
                xii = xaxis[ix]
                yii = yaxis[iy]
                rii = np.sqrt(xii**2 + yii**2)
                vkep = np.sqrt(gg*ppar['mstar'][0] / rii)
                vel[ix,iy,:,2] = vkep

    return vel
