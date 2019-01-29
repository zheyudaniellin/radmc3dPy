"""model a disk with hydro eq and dust settling. temperature like Rosenfield 2012
 in calculations, all will be in cylindrical coordinates, then output to 3d spherical coordinates.

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
from .. import dustopac
from .. import crd_trans
import pdb
import fneq

def getModelDesc():
    """Provides a brief description of the model
    """

    return "A ppdisk model including dust settling by temperature like Rosenfield 2012"
           

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
        ['xbound', '[0.05*au,20.*au, 100.0*au]', 'Number of radial grid points'],
        ['ny', '[10,30,30,10]', 'Number of grid points in the first dimension'],
        ['ybound', '[0., pi/3., pi/2., 2.*pi/3., pi]', 'Number of radial grid points'],
        ['nz', '[61]', 'Number of grid points in the first dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
      	['gasspec_mol_name', "['12co','13co','c18o','c17o']", 'name of molecule'],
	['gasspec_mol_abun', '[4e-5, 5.78e-7, 7.18e-8, 1e-8]', 'abundance of molecule'],
	['gasspec_mol_dbase_type', "['leiden','leiden','leiden','leiden']", 'data base type'],
        ['gasspec_mol_dissoc_taulim', '[1.0, 1.0, 1.0, 1.0]', 'Continuum optical depth limit below which all molecules dissociate'],
        ['gasspec_mol_freezeout_temp', '[19.0, 19.0, 19.0, 19.0]', 'Freeze-out temperature of the molecules in Kelvin'],
        ['gasspec_mol_freezeout_dfact', '[1e-8, 1e-8, 1e-8, 1e-8]',
         'Factor by which the molecular abundance should be decreased in the frezze-out zone'],
        ['gasspec_vturb', '0.2e5', 'Microturbulent line width'],
	['g2d', '0.01', ' Dust to Gas ratio'],
        ['mstar', '0.3*ms', 'Mass of the star(s)'],
        ['Rinner', '0.01*au', ' Inner radius of the disk'],
        ['Router', '60.0*au', ' Outer radius of the disk'],	      	
	['Rt','21.*au', ' characteristic radius for temperature, height'], 
	['Rsig', '20.*au', ' characteristic radius for exponential tapering sigam'], 
	['sigp', '0.3', 'exponent value for sigma'],
	['mdisk', '0.05*ms', 'mass of disk'],
	['sigma_type', '1', '0-polynomial, 1-exponential tapering'],
        ['alpha', '0.01', 'viscosity parameter'],
        ['gam', '5./3.', 'powerlaw of the energy spectrum'],
        ['dohydroeq', '1', '0-to not do hydrostatic equilibrium (scale height will use H0 and qheight)'],
        ['dosettle', '1', '0-to not do dust settling (scale height will use H0 and qheight)'],
	['T0mid', '100.', 'midplane temperature at Rt'],
	['qmid', '-0.75', 'midplane temperature exponent'],
	['T0atm', '145.', 'atmosphere temperature value at Rt'],
	['qatm', '-0.5', 'atmosphere temperature exponent'],
        ['H0', '7*au', 'height at Rt'],
        ['qheight', '1.125', 'powerlaw index for height'],
        ['Hd', '-1', 'outer tapering of height'],
	['zqratio', '5.0', 'multiple of scale height for temperature transition'],
	['hdel', '2.0', 'power of transition for temperature'],
	['cuttemp', '10', 'temperature cut'],
	['cutgdens', '1e-30', 'cut for gas density'],
        ['vsys', '0.0', 'systemic velocity']
              ]

    return defpar

def getGasTemperature(grid=None, ppar=None, walls=False):
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
    if walls is False:
        xaxis = grid.x; nx = grid.nx
        yaxis = grid.y; ny = grid.ny
        zaxis = grid.z; nz = grid.nz
    else:
        xaxis = grid.xi; nx = grid.nxi
        yaxis = grid.yi; ny = grid.nyi
        zaxis = grid.zi; nz = grid.nzi
    tgas = np.zeros([nx, ny, nz], dtype=np.float64)

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


def getDustTemperature(grid=None, ppar=None, walls=False):
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

    tgas = getGasTemperature(grid=grid, ppar=ppar, walls=walls)
    tshape = list(tgas.shape)
    tshape.append(ngs)
    tdust = np.zeros(tshape, dtype=np.float64)
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
   
def getGasDensity(grid=None, ppar=None, walls=False):
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
    if walls is False:
        xaxis = grid.x; nx = grid.nx
        yaxis = grid.y; ny = grid.ny
        zaxis = grid.z; nz = grid.nz
    else:
        xaxis = grid.xi; nx = grid.nxi
        yaxis = grid.yi; ny = grid.nyi
        zaxis = grid.zi; nz = grid.nzi
    rhogas = np.zeros([nx, ny, nz], dtype=np.float64) + 1e-30

    if ppar['dohydroeq'] == 0:
        if ppar['crd_sys'] == 'sph':        
            for xx in range(nx):
                for yy in range(ny):
                    xii = xaxis[xx] * np.sin(yaxis[yy])
                    zii = xaxis[xx] * abs(np.cos(yaxis[yy]))
                    # hii = ppar['H0'] * (xii / ppar['Rt'])**(ppar['qheight'])
                    hii = fneq.eq_taperheight(xii, ppar['Rt'], ppar['H0'],
                        ppar['qheight'], ppar['Router'], ppar['Hd'])
                    sigii = fneq.eq_sig(xii,ppar['mdisk'],ppar['Rinner'],ppar['Rsig'],
                        ppar['Router'],ppar['sigp'],ppar['sigma_type'])
                    gdensii = sigii / (2.0*np.pi)**0.5 / hii * np.exp(-0.5*(zii/hii)**2)
                    if gdensii < ppar['cutgdens']:
                        rhogas[xx,yy,:] = ppar['cutgdens']
                    else:
                        rhogas[xx,yy,:] = gdensii

        if ppar['crd_sys'] == 'car':
            for xx in range(nx):
                for yy in range(ny):
                    for zz in range(nz):
                        xii = (xaxis[xx]**2 + yaxis[yy]**2)**0.5
                        zii = abs(zaxis[yy])
                        #hii = ppar['H0'] * (xii / ppar['Rt'])**(ppar['qheight'])
                        hii = fneq.eq_taperheight(xii, ppar['Rt'], ppar['H0'], 
                            ppar['qheight'], ppar['Router'], ppar['Hd'])
                        sigii = fneq.eq_sig(xii,ppar['mdisk'], ppar['Rinner'], ppar['Rsig'],
                            ppar['Router'], ppar['sigp'], ppar['sigma_type'])
                        gdensii = sigii / (2.0*np.pi)**0.5 / hii * np.exp(-0.5*(zii/hii)**2)
                        if gdensii < ppar['cutgdens']:
                            rhogas[xx,yy,zz] = ppar['cutgdens']
                        else:
                            rhogas[xx,yy,zz] = gdensii

    else: #if want to do hydo equilibrium
        tgas = getGasTemperature(grid=grid, ppar=ppar)

        if ppar['crd_sys'] == 'sph':
            # need to interpolate the temperature to cartesian first
            res = crd_trans.contSph2Cart(contSph=tgas[:,:,0], raxis=grid.x, ttaaxis=grid.y,
                log=True, pad=ppar['cuttemp'], midplane=False)
            tgas_xz = res['contCart']
            tgas_x = res['xaxis']
            tgas_z = res['zaxis']
            tgas_zi = np.zeros([len(tgas_z)+1], dtype=np.float64)
            tgas_zi[1:-1] = 0.5*(tgas_z[1:] + tgas_z[0:-1])
            tgas_zi[0] = tgas_zi[1] - abs(tgas_zi[2]-tgas_zi[1])
            tgas_zi[-1] = tgas_zi[-2] + abs(tgas_zi[-3]-tgas_zi[-2])

        if ppar['crd_sys'] == 'car':
            inx = np.argmin(yaxis)
            reg = grid.x >= 0.0
            tgas_xz = tgas[reg,inx,:]
            tgas_x = grid.x[reg]
            tgas_z = grid.z
            tgas_zi = grid.zi

        sig_x = fneq.eq_sig(tgas_x,ppar['mdisk'], ppar['Rinner'], ppar['Rsig'],
                            ppar['Router'], ppar['sigp'], ppar['sigma_type'])

        rhogasxz = hydro_xz(tgas_x, tgas_zi, tgas_z, sig_x, tgas_xz, ppar['mstar'], ppar['cutgdens'])

        if ppar['crd_sys'] == 'sph':
            rhogas_slice = crd_trans.contCart2Sph(contCart=rhogasxz, xaxis=tgas_x, zaxis=tgas_z, raxis=xaxis, ttaaxis=yaxis,
                log=True, pad=ppar['cutgdens'])
            for zz in range(nz):
                rhogas[:,:,zz] = rhogas_slice['contSph']

        if ppar['crd_sys'] == 'car':
            finterp = interpolate.interp2d(tgas_x, tgas_z, np.log(rhogasxz), kind='cubic', fill_value=np.log(ppar['cutgdens']))
            for xx in range(nx):
                for yy in range(ny):
                    rii = np.sqrt(xaxis[xx]**2 + yaxis[yy]**2)
                    rhogas[xx,yy,:] = np.exp(finterp(np.array([rii]), zaxis))

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

    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64) 
    rhogas = getGasDensity(grid=grid, ppar=ppar)

    if ppar['dosettle'] == 0:
        for gg in range(ngs):
            rhodust[:, :, :, gg] = rhogas * dinfo['dweights'][gg] * ppar['g2d']
    else: # if we want to do dust settling
        tgas = getGasTemperature(grid=grid, ppar=ppar)
        if ppar['crd_sys'] == 'sph':
            # need to interpolate the temperature to cartesian first
            res = crd_trans.contSph2Cart(contSph=tgas[:,:,0], raxis=grid.x, ttaaxis=grid.y,
                log=True, pad=ppar['cuttemp'], midplane=False)
            tgas_xz = res['contCart']
            tgas_x = res['xaxis']
            tgas_z = res['zaxis']
            tgas_zi = np.zeros([len(tgas_z)+1], dtype=np.float64)
            tgas_zi[1:-1] = 0.5*(tgas_z[1:] + tgas_z[0:-1])
            tgas_zi[0] = tgas_zi[1] - abs(tgas_zi[2]-tgas_zi[1])
            tgas_zi[-1] = tgas_zi[-2] + abs(tgas_zi[-3]-tgas_zi[-2])
            tgas_dz = tgas_zi[1:] - tgas_zi[0:-1]
            res = crd_trans.contSph2Cart(contSph=rhogas[:,:,0], raxis=grid.x, ttaaxis=grid.y,
                log=True, pad=ppar['cutgdens'], midplane=False)
            rhogas_xz = res['contCart']

        if ppar['crd_sys'] == 'car':
            inx = np.argmin(yaxis)
            reg = grid.x >= 0.0
            tgas_xz = tgas[reg,inx,:]
            tgas_x = grid.x[reg]
            tgas_z = grid.z
            tgas_zi = grid.zi
            tgas_dz = tgas_zi[1:] - tgas_zi[0:-1]
            rhogas_xz = rhogas[reg, inx, :]

        sig_x = fneq.eq_sig(tgas_x,ppar['mdisk'], ppar['Rinner'], ppar['Rsig'],
                            ppar['Router'], ppar['sigp'], ppar['sigma_type'])

        alphaxz = ppar['alpha'] + np.zeros(list(tgas_xz.shape), dtype=np.float64)

        print('--calculating dust density--')
        for gg in range(ngs):
            print('.')
            matdens = dinfo['matdens'][gg]
            gsize = dinfo['gsize'][gg]
            gam = ppar['gam']
            dsig_x = sig_x * ppar['g2d'] * dinfo['dweights'][gg]

            dustgg = dubrulle_settle(tgas_x, tgas_z, tgas_dz, tgas_xz, rhogas_xz, ppar['mstar'], dsig_x, 
                alphaxz, gam, matdens, gsize, cutdens=ppar['cutgdens']*ppar['g2d']*dinfo['dweights'][gg])

            if ppar['crd_sys'] == 'sph':
                dust_slice = crd_trans.contCart2Sph(contCart=dustgg, xaxis=tgas_x, zaxis=tgas_z, raxis=grid.x,
                    ttaaxis=grid.y, log=True, pad=ppar['cutgdens']*ppar['g2d']*dinfo['dweights'][gg])
                for zz in range(grid.nz):
                    rhodust[:,:,zz,gg] = dust_slice['contSph']
            if ppar['crd_sys'] == 'car':
                finterp = interpolate.interp2d(tgas_x, tgas_z, np.log(dustgg), kind='cubic',
                    fill_value=np.log(ppar['cutgdens']))
                for xx in range(grid.nx):
                    for yy in range(grid.ny):
                        rii = np.sqrt(xaxis[xx]**2 + yaxis[yy]**2)
                        rhodust[xx,yy,:,gg] = np.exp(finterp(np.array([rii]), zaxis))
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
                vkep = np.sqrt(natconst.gg*ppar['mstar'] / xii) * (pow(1.+(zii/xii)**2., -0.75))
                vel[ix, iy,:,2] = vkep + ppar['vsys']

    if ppar['crd_sys'] == 'car':
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    xii = xaxis[ix]
                    yii = yaxis[iy]
                    zii = zaxis[iz]
                    rii = np.sqrt(xii**2 + yii**2)
                    vkep = np.sqrt(natconst.gg*ppar['mstar'] / xii) * (pow(1.+(zii/xii)**2., -0.75))
                    vel[ix,iy,:,2] = vkep + ppar['vsys']


    return vel

def hydro_xz(xstg, zaxis, zstg, sig_x, tempxz, Ms, cutgdens):
    """Calculates gas density in hydro static equilibrium given a temperature distribution
    xstg = x on staggered 
    zaxis = zaxis of walls in cm, zstg = zaxis on staggered grid
    sig_x = gas column density as a function of cm
    tempxz = temperature on staggered grid
    Ms = star mass in grams
    """

    # the actual function for calculations
    def hydro_half(xstg, zaxis, zstg, sig_x, tempxz, Ms):
        """ solves the equation assuming z is all positive and increasing
        zaxis is the cell walls
        xaxis can be cell wall or cell center
        output rhohalf will be on cell center for z, and anythin on x
        """
        nxs = len(xstg)
        nz = len(zaxis)
        nzs = len(zstg)
        rhohalf = np.zeros([nxs, nzs], dtype=np.float64)

        dzaxis = zaxis[1:] - zaxis[0:-1]

        for xx in range(nxs):
            xconst = natconst.muh2 * natconst.mp * natconst.gg * Ms / natconst.kk / (abs(xstg[xx]))**3.
            gd = np.zeros([nzs], dtype=np.float64)
            gd[0] = 1.
            for zz in range(nzs-1):
                dz = dzaxis[zz]
                gterm = xconst * 0.5 * (zstg[zz+1]/tempxz[xx,zz+1] + zstg[zz]/tempxz[xx,zz])
                tterm = np.log(tempxz[xx,zz+1]/tempxz[xx,zz])
                gd[zz+1] = gd[zz] * np.exp(-dz*gterm-tterm)

            sigdum = 0.0
            for zz in range(nzs-1):
                sigdum = sigdum + gd[zz] * dz
            if np.isfinite(sigdum) == False:
                pdb.set_trace()
            sigrat = 0.5 * sig_x[xx] / sigdum
            rhohalf[xx, :] = gd * sigrat 

        return rhohalf
    # --------------------------------------------------

    # deal with z higher than midplane
    zs_upreg = zstg >= 0.0
    zstgin = zstg[zs_upreg]
    z_upreg = zaxis >= 0.0
    zaxisin = zaxis[z_upreg]
    rhoup = hydro_half(xstg, zaxisin, zstgin, sig_x, tempxz[:,zs_upreg], Ms)

    # deal with z lower than midplane
    zs_loreg = zstg <= 0.0
    neg_zstgin = zstg[zs_loreg]
    zs_sortlo = np.argsort(-neg_zstgin)
    zstgin =  -neg_zstgin[zs_sortlo]

    z_loreg = zaxis <= 0.0
    neg_zaxisin = zaxis[z_loreg]
    z_sortlo = np.argsort(-neg_zaxisin)
    zaxisin = -neg_zaxisin[z_sortlo]

    lotempxz = tempxz[:,zs_loreg]
    tempxzin = lotempxz[:,zs_sortlo]
    rholo = hydro_half(xstg, zaxisin, zstgin, sig_x, tempxzin, Ms)

    # resort the results for output
    nxs = len(xstg)
    nzs = len(zstg)

    gdensxz = np.zeros([nxs, nzs], dtype=np.float64)

    # upper part
    gdensxz[:,zs_upreg] = rhoup

    # lower part
    resortlo = np.argsort(neg_zstgin)
    gdensxz[:,zs_loreg] = rholo[:,zs_sortlo]

    reg = gdensxz < cutgdens
    gdensxz[reg] = cutgdens

    return gdensxz

def dubrulle_settle(xstg,zstg,dzstg,tempxz,gdensxz,Ms,dsig_x,alphaxz,gam,swgt,asize, cutdens=None):

    nxs = len(xstg)
    nzs = len(zstg)

    ddensxz = np.zeros([nxs, nzs], dtype=np.float64)

    wk = np.sqrt(natconst.gg * Ms / (xstg **3))
    cs = np.sqrt(natconst.kk * tempxz / natconst.muh2 / natconst.mp)

    lowreg = np.argwhere(zstg < 0)
    nlow = len(lowreg)
    lowregbool = zstg < 0

    upreg = np.argwhere(zstg > 0)
    nup = len(upreg)
    upregbool = zstg > 0

    for xx in range(nxs):
        tau = swgt * asize * 1e-4 / gdensxz[xx,:] / cs[xx,:]
        fac = (wk[xx] * tau)**2 / (1. + wk[xx] * tau)
        k0 = wk[xx] / cs[xx,:] / np.sqrt(alphaxz[xx,:])
        k02 = k0**2
        kt0 = np.sqrt(np.arctan(fac * k0 * abs(zstg)) / k0 / abs(zstg)) / wk[xx] / tau

        kt = wk[xx] * kt0 / np.sqrt(1. + gam) / k02

        # lower part. z<0
        dsliclow = np.zeros([nlow], dtype=np.float64)
        dsliclow[-1] = 1.0
        for ii in range(nlow-1):
            if dsliclow[nlow-1-ii] == 0.0:
                break
            zz = lowreg[nlow-1-ii]
            zii = abs(zstg[zz-1])
            dz1 = dzstg[zz]
            dz2 = dzstg[zz-1]
            kt1 = kt[zz]
            kt2 = kt[zz-1]
            if np.isfinite(tau[zz-1] * dz2 / kt2) == 0:
                break
            if np.isfinite(tau[zz] * dz1 / kt1) == 0:
                break
            gterm = np.log(gdensxz[xx,zz-1] / gdensxz[xx,zz])
            dterm = wk[xx]**2 * zii * 0.5 * (tau[zz-1] * dz2 / kt2 + tau[zz] * dz1 / kt1)
            dsliclow[nlow-1-ii-1] = dsliclow[nlow-1-ii] * np.exp(gterm-dterm)
            if dterm < 0:
                pdb.set_trace()
            if np.isfinite(dsliclow[nlow-1-ii-1]) is False:
                pdb.set_trace()

        sigdumlow = 0.0
        for ii in range(nlow):
            zz = lowreg[nlow-1-ii]
            sigdumlow = sigdumlow + dsliclow[nlow-1-ii] * dzstg[zz]

        if np.isfinite(sigdumlow) is False:
            pdb.set_trace()
        dsigrat = 0.5 * dsig_x[xx] / sigdumlow
        ddensxz[xx,lowregbool] = dsliclow * dsigrat

        # upper part
        dslic = np.zeros([nup], dtype=np.float64)
        dslic[0] = 1.0
        for ii in range(nup-1):
            if dslic[ii] == 0:
                break
            zz = upreg[ii]
            zii = zstg[zz+1]
            dz1 = dzstg[zz]
            dz2 = dzstg[zz+1]
            kt1 = kt[zz]
            kt2 = kt[zz+1]
            if np.isfinite(tau[zz+1]*dz2/kt2) == 0:
                break
            if np.isfinite(tau[zz]*dz1/kt1) == 0:
                break
            gterm = np.log(gdensxz[xx,zz+1] / gdensxz[xx,zz])
            dterm = wk[xx]**2 * zii * 0.5 * (tau[zz+1]*dz2/kt2 + tau[zz]*dz1/kt1)
            dslic[ii+1] = dslic[ii] * np.exp(gterm-dterm)
            if np.isfinite(dslic[ii+1]) == 0:
                print('there is an inifinte value')
                pdb.set_trace()
        sigdum = 0.0
        for ii in range(nup):
            zz = upreg[ii]
            sigdum = sigdum + dslic[ii]*dzstg[zz]
        if np.isfinite(sigdum) == 0:
            print('there is an infinite value')
            pdb.set_trace()
        dsigrat = 0.5 * dsig_x[xx] / sigdum
        ddensxz[xx,upregbool] = dslic * dsigrat

    if cutdens is None:
        cutdens = 0.0
    reg = ddensxz < cutdens
    ddensxz[reg] = cutdens

    return ddensxz
