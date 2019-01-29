"""Models for zyl6 disk:
    two component temperature, hydrostatic equilibrium, 
    empirical surface profile.
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
from scipy.interpolate import RegularGridInterpolator
import pdb

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
        ['Rseg', '[2.0*au, 80.*au]', 'upper radius boundary of each segment'],
        ['pseg', '[-0.3, -0.5]', 
         'power-law within each segment.Btw, should not be -2'],
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
        ['cuttemp', '10', 'temperature cut, also ambient temperature']
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

def fn_hydro_xz(xstg, ttastg,ttaaxis, ppar, Rc, sigc, pseg,Ms):
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

    nx = len(xstg)

    tgas = fn_getTempCar(xx,zz,Rt,Ht,qheight,
            T0atm, qatm, T0mid, qmid,
            zdel,cuttemp)
    return gdens

def getSigma(Rc, sigc, pseg, xx):
    sig = np.zeros(xx.shape, dtype=np.float64)
    nc = len(Rc)
    for ic in range(nc-1):
        reg = (Rc[ic] <= xx) & (xx <= Rc[ic+1])
        sig[reg] = sigc[ic] * (xx[reg] / Rc[ic])**(pseg[ic])
    reg = sig <= 0.
    sig[reg] = 1e-30
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
    ttastg = grid.y
    ttaaxis = grid.yi

    # solve for sigma
    Rin = ppar['Rin']
    Rseg = ppar['Rseg'] #this is a list
    Rc = np.array(Rseg)
    Rc = np.insert(Rc, 0, Rin)

    #characteristic radius, including Rin. so like grid walls
    nseg = len(Rseg)
    pseg = ppar['pseg']
    if nseg is not len(pseg):
        raise ValueError('Number of segments for Radius is not same as exponents')
    Mdisk = ppar['Mdisk']
   
    #to store all the sigmas defined on Rc
    sigc = np.zeros(nseg+1, dtype=np.float64)
    sigc[0] = 1.0 #start with 1st sigma=1, and then normalize to mdisk
    # solve for all the other sigmas. since it should be continuous
    for ic in range(nseg):
        sigc[ic+1] = sigc[ic] * (Rc[ic+1]/Rc[ic])**(pseg[ic])
    # the total mass in each segment
    def getMass(sig0, R0, R1, R2, p):
        #sig0=sigma at R0. R1,R2 lower and upper limit. p=exponent
        Mass = 2.*np.pi * sig0 / (p+2.) * ((R2**2.)*(R2/R0)**p - (R1**2.)*(R1/R0)**p)
        return Mass
    Mseg = np.zeros(nseg, dtype=np.float64)
    for iseg in range(nseg):
        Mseg[iseg] = getMass(sigc[iseg], Rc[iseg], 
                         Rc[iseg], Rc[iseg+1], pseg[iseg])
    tot_norm = sum(Mseg)
    norm_fac = Mdisk / tot_norm
    sigc = sigc * norm_fac #now sigc will be in units of surface density

    sig_x = getSigma(Rc ,sigc, pseg, xstg)
   
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
    gdens_rt = fn_hydro_rt(xstg, ttastg, ttaaxis,sig_x,tgas,Ms) # assuming thin disk
#    gdens_rt = fn_hydro_xz(xstg, ttastg, ttaaxis, sig_x, ppar,Rc,sigc,pseg, Ms)
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
