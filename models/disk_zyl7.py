"""Models for zyl7 disk:
    two component temperature, hydrostatic equilibrium, 
    2 parameter exponent surface density
    Dust-coupled fully coupled to gas (no settling considered, 1 dust species)
    For speed, only assume 2D spherical, mirror symmetric.
    Ignoring scattering, alignment, etc; so that I can use emcee

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
from scipy import interpolate
import pdb
import fneq

def getModelDesc():
    """Provides a brief description of the model
    """

    return "two component temperature, hydrostatic equilibrium, empirical surface profile."
           

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
        ['xbound', '[0.1*au,20.*au, 100.*au]', 'Number of radial grid points'],
        ['ny', '[20,50]', 'Number of grid points in the second dimension'],
        ['ybound', '[0.5, pi/3., pi/2.]', 'Number of radial grid points'],
        ['nz', '[0]', 'Number of grid points in the first dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
        # star related
        ['tstar', '[4000.0]', 'Temperature of star'], 
        ['mstar', '[0.2*ms]', 'Mass of the star(s)'],
        # density related
        ['Mdisk', '0.05*ms', 'Total mass of disk'],
        ['Rin', '0.01*au', 'Inner radii of disk '],
        ['Rsig', '30.*au', 'upper radius boundary of each segment'],
        ['sigp', '-0.7', 'exponent of power-law'], 
        ['sigq', '1.0', 'exponent for exponential part'], 
        ['g2d', '0.01', 'gas to dust ratio'],
        ['cutgdens', '1e-30', 'cut for density, also ambient density'],  
        # temperature related
        ['Rt', '20.*au', 'Characteristic radius for temperature'], 
        ['T0atm', '115', 'Atmospheric temperature at Rt'], 
        ['qatm', '-0.5', 'exponent for atmosphere'], 
        ['T0mid', '70', 'Midplane temperature at Rt'],
        ['qmid', '-0.75', 'exponent for midplane'],
        ['Ht', '7.5*au', 
           'Transition height at Rt. Note this is not pressure scale height'],
        ['qheight', '1.125', 'exponent for height'],
        ['zdel', '2.', 'exponent for cosine for temperature transition'],
        ['cuttemp', '10', 'temperature cut, also ambient temperature'],
        ['alph', '0.01', 'viscosity parameter. careful with this']
              ]

    return defpar

def fn_getTempCar(xx, zz, Rt, Ht, qheight, 
                  T0atm, qatm, T0mid, qmid, zdel, cuttemp):
    # get the temperature in cartesian coordinates
    # xx,yy,zz = meshgrid in cartesian 
    nx = xx.shape[0]
    nz = xx.shape[1]
    temp = np.zeros([nx, nz], dtype=np.float64)
    rr = np.sqrt(xx**2. + zz**2.)

    hh = Ht * (abs(xx / Rt))**(qheight)
    Tatm = T0atm * (abs(rr / Rt))**qatm
    Tmid = T0mid * (abs(xx / Rt))**qmid

    reg = zz > hh
    temp[reg] = Tatm[reg]
    reg = zz <= hh
    temp[reg] = Tatm[reg]+(Tmid[reg] - Tatm[reg])*(np.cos(np.pi*0.5*zz[reg]/hh[reg]))**(2.*zdel)
    reg = temp < cuttemp
    temp[reg] = cuttemp
    return temp

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
    mesh = np.meshgrid(grid.x, grid.y, indexing='ij')

    if ppar['crd_sys'] == 'sph':
        rr = mesh[0]
        tt = mesh[1]
        xx = rr * np.sin(tt)
        zz = rr * np.cos(tt)
    elif ppar['crd_sys'] == 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
    else:
        raise ValueError('crd_sys not specified in ppar')

    # since getDustDensity will also be needing temperature, 
    # might as well just define a function for it. 
    tgaspp = fn_getTempCar(xx,zz,ppar['Rt'], ppar['Ht'], ppar['qheight'], 
            ppar['T0atm'], ppar['qatm'], ppar['T0mid'], ppar['qmid'], 
            ppar['zdel'], ppar['cuttemp'])
    for ip in range(grid.nz):
        tgas[:,:,ip] = tgaspp

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

def fn_hydro_rt(xstg, ttastg,ttaaxis,sig_x,tgas,Ms):
    """Calculates gas density under hydrostatic equilibrium
    sig_x = surface density along xstg
    tgas = mesh of temperature of gas 
    """
    nxs = len(xstg)
    nttas = len(ttastg)
    gdens = np.zeros([nxs,nttas], dtype=np.float64)

    # assume only upper region
    xconst = natconst.muh2 * natconst.mp * natconst.gg * Ms / natconst.kk / xstg**3.
    gd = np.zeros([nxs, nttas], dtype=np.float64)
    gd[:,-1] = 1.0
    for ix in range(nxs):
        zstg = xstg[ix] * np.cos(ttastg) #tta is increasing
        zaxis = xstg[ix] * np.cos(ttaaxis)
        dzstg = zaxis[:-1] - zaxis[1:]
        for iz in range(nttas-1):
            itt = nttas-1-iz
            gterm = xconst[ix] * 0.5 * (zstg[itt-1]/tgas[ix,itt-1] + zstg[itt]/tgas[ix,itt])
            tterm = np.log(tgas[ix,itt-1]/tgas[ix,itt])
            gd[ix,itt-1] = gd[ix,itt] * np.exp(-dzstg[itt] * gterm - tterm)
        sigdum = 0.0
        for iz in range(nttas-1):
            sigdum = sigdum + gd[ix,iz] * dzstg[iz]

        if np.isfinite(sigdum) == False:
            pdb.set_trace()
        sigrat = 0.5 * sig_x[ix] / sigdum
        gdens[ix,:] = gd[ix,:] * sigrat

    return gdens

def fn_hydro_xz(xstg, ttastg,ttaaxis,sig_x,ppar,Ms):
    """Calculate hydrostatic equilibrium by first converting to Cartesian
    Not in effect yet
    """
    Rt = ppar['Rt']
    Ht = ppar['Ht']
    qheight = ppar['qheight']
    T0atm = ppar['T0atm']
    qatm = ppar['qatm']
    T0mid = ppar['T0mid']
    qmid = ppar['qmid']
    zdel = ppar['zdel']
    cuttemp = ppar['cuttemp']
    ntta = len(ttaaxis)
    nttas = ntta - 1

    nxs = len(xstg)
    dttastg = abs(ttaaxis[1:] - ttaaxis[:-1])
    dz = dttastg.min() * xstg[nxs/2]
    minz = min(xstg[0] * np.cos(ttaaxis))
    maxz = max(xstg[-1]*0.75 * np.cos(ttaaxis))
    nz = 2.*len(ttaaxis)
    z1 = minz * (maxz/minz)**(np.arange(nz, dtype=np.float64)/float(nz-1))
    zaxis = np.concatenate((z1, [0.]))
    zaxis.sort()
    nz = len(zaxis)
    nzs = nz-1
    zstg = 0.5*(zaxis[1:] + zaxis[:-1])

    mesh = np.meshgrid(xstg, zstg, indexing='ij')
    xx = mesh[0]
    zz = mesh[1]
    tgas = fn_getTempCar(xx,zz,Rt,Ht,qheight,
            T0atm, qatm, T0mid, qmid,
            zdel,cuttemp)

    gdens_xz = fneq.eq_hydro_xz(xstg, zstg, zaxis, sig_x, tgas, Ms)
    reg = gdens_xz < ppar['cutgdens']
    gdens_xz[reg] = ppar['cutgdens']
    lngdens_xz = np.log(gdens_xz)

    f_gdens_xz = interpolate.interp2d(xstg, zstg, lngdens_xz.T, kind='linear', 
                     fill_value=np.log(ppar['cutgdens']))

    mesh = np.meshgrid(xstg, ttastg, indexing='ij')
    rr = mesh[0]
    tt = mesh[1]
    xx = rr * np.sin(tt)
    zz = rr * np.cos(tt)
    lngdens_rt = np.zeros(rr.shape, dtype=np.float64)
    for ix in range(nxs):
        for itta in range(nttas):
            lngdens_rt[ix,itta] = f_gdens_xz(xx[ix, itta], zz[ix,itta])
        
    gdens = np.exp(lngdens_rt)

    return gdens

def getSigma(Rsig, sigp, sigq, xx):
    sig = (xx/Rsig)**(sigp) * np.exp(-(xx/Rsig)**sigq)
    return sig

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
    mesh = np.meshgrid(grid.x, grid.y, indexing='ij')

    # generate the density field
    if ppar['crd_sys'] == 'sph':
        rr = mesh[0]
        tt = mesh[1]
        xx = rr * np.sin(tt)
        yy = rr * np.sin(tt)
        zz = rr * np.cos(tt)

    elif ppar['crd_sys'] == 'car':
        raise ValueError('cartesian not allowed')
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]

        mesh = np.meshgrid(grid.x, grid.y, grid.zi, indexing='ij')
        zzi = mesh[2]
    else:
        raise ValueError('crd_sys not specified in ppar')

    xstg = grid.x #since whether in Spherical or Cartesian, raxis is the same
    xaxis = grid.xi
    dxstg = xaxis[1:] - xaxis[:-1]
    ttastg = grid.y
    ttaaxis = grid.yi

    # solve for sigma
    Rin = ppar['Rin']
    Rsig = ppar['Rsig']
    sigp = ppar['sigp']
    sigq = ppar['sigq']

    Mdisk = ppar['Mdisk']
   
    sigc = getSigma(Rsig, sigp, sigq, xstg)

    # the total mass
    tot_norm= np.sum(sigc * xstg * dxstg) #do this until I find some analytical solution
    
    norm_fac = Mdisk / tot_norm

    # check if mirror symmetry
    if ppar['ybound'][-1] <= (np.pi/2. * 1.0001):
        norm_fac = norm_fac / 2.0

    sigc = sigc * norm_fac #now sigc will be in units of surface density

    sig_x = sigc
   
    # get the temperature 
    Rt = ppar['Rt']
    Ht = ppar['Ht']
    qheight = ppar['qheight']
    T0atm = ppar['T0atm']
    qatm = ppar['qatm']
    T0mid = ppar['T0mid']
    qmid = ppar['qmid']
    zdel = ppar['zdel']
    cuttemp = ppar['cuttemp']
    tgas = fn_getTempCar(xx,zz,Rt,Ht,qheight,
            T0atm, qatm, T0mid, qmid, 
            zdel,cuttemp)

    # solve hydro static equilibrim
    Ms = ppar['mstar'][0]
#    gdens_rt = fn_hydro_rt(xstg, ttastg, ttaaxis,sig_x,tgas,Ms) # assuming thin disk
    
    gdens_rt = fn_hydro_xz(xstg, ttastg, ttaaxis, sig_x, ppar,Ms)
    cutgdens = ppar['cutgdens']
    reg = gdens_rt < cutgdens
    gdens_rt[reg] = cutgdens
    gdens = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    for iz in range(grid.nz):
        gdens[:,:,iz] = gdens_rt

    # interpolate onto sph coordinates
#    if ppar['crd_sys'] is 'sph':
#        
#    if ppar['crd_sys'] is 'car':
#        rhogas = gdens
    rhogas = gdens
    
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
    ngs = 1
    rhodust = np.zeros([grid.nx, grid.ny, grid.nz, ngs], dtype=np.float64) 
    rhodust[:,:,:,0] = ppar['g2d'] * rhogas
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
    qvis = rhogas * ppar['alph'] * hh**2 * wk**3 * 9./4.
#    reg = rhogas <= ppar['cutgdens']
#    qvis[reg] = 0.0

#    vol = grid.getCellVolume()
#    tol_lum = np.sum(vol*qvis)
#    lum = vol*qvis
#    tot_lum = lum.sum()
#    print('setting up heatsource.inp')
#    print('total heating luminosity: %f Lsun'%(tot_lum/natconst.ls))

    return qvis

