# for calculating line-of-sight properties
import numpy as np
import scipy.interpolate as sip
from scipy.interpolate import RegularGridInterpolator
import pdb
from radmc3dPy import natconst, crd_trans

def interpcont(yvec, zvec, cont, rstg, ttastg, pad):
    """
    Parameters
    ----------
    yvec, zvec = the points to find values in disk coordinates
    cont = 2d array
    rstg, ttastg = radius and theta cell grid
    pad = the value for regions outside of interpolation
    Returns
    ------
    vals = values along yvec and zvec
    """
    nvec = len(yvec)
    vshape = yvec.shape
    if len(vshape) is not 1:
        raise ValueError('shape of vector is incorrect')

    nrs = len(rstg)
    nttas = len(ttastg)
    cshape = cont.shape
    if nrs != cshape[0]:
        raise ValueError('shape of 2d array is different from radius')
    if nttas != cshape[1]:
        raise ValueError('shape of 2d array is different from theta')

    # find the corresponding radius and theta vectors
    # y = r sin(theta)
    # z = r cos(theta)
    rvec = np.sqrt(yvec**2. + zvec**2.)
    tvec = np.arctan2(yvec, zvec)
    reg = tvec < 0.
    tvec[reg] = tvec[reg] + 2.*np.pi

    # create the inteprolation object
    fcont = sip.interp2d(rstg, ttastg, cont.T, kind='linear',
        fill_value=pad, bounds_error=False)
    vals = np.zeros(nvec, dtype=np.float64)
    for iv in range(nvec):
        vals[iv] = fcont(rvec[iv], tvec[iv])

#    vals = crd_trans.findValin2D(rstg, ttastg, cont, rvec, tvec, padvalue=pad)

    return vals, rvec, tvec

def getZlos(raxis, ttaaxis, zlosmax=None, nzlos=None):
    # determine a good grid for line of sight z
    nr = len(raxis)
    if nzlos is None:
        nzlos = nr / 2
#    minr = min([0.01*natconst.au, raxis[0]])
    minr = raxis[0]
    if zlosmax is None:
        maxr = max(raxis) * 0.75
    else:
        maxr = zlosmax
    if maxr < minr:
        raise ValueError('minr cannot be greater than maxr')

    rlen = np.geomspace(minr, maxr, nzlos)

    # rlen = raxis
    zlosi = np.concatenate((rlen, np.array([0.]), -rlen), axis=0)
    zlosi.sort()
    zlos = 0.5 * (zlosi[1:] + zlosi[:-1])

    return zlosi, zlos


def extract(inc, ym, cont2d, raxis, rstg, ttaaxis, ttastg, pad, zlosmax=None, nzlos=None):
    """
    extract profiles specifically along the minor axis 

    Parameters:
    inc = float
        inclination in rad
    ym = float
        the y coordinate in image plane
    cont2d = 2d array
        the 2d array to be interpolated
    raxis = 1d array
        the radius wall grid
    rstg = 1d array
        the radius cell grid
    ttastg = 1d array
        the theta cell grid
    zlos = 1d array
        the z coordinates in image plane to extract values.
        default is to use rstg
    Returns
    -------
    los = dictionary
        'zlosi' : the z wall coordinate along line of sight
        'zlos' : the z cell coordinates along line of sight
        'valcell': values in cell 
        'valwall': values in walls
        'rcell' : cell radius in disk coordinates, 'rwall': in wall 
        'tcell' : cell theta in disk coordinates, 'twall': in wall
        'ycell' : cell y in disk coordinates, 'ywall' : in wall
        'zcell' : cell z in disk coordinates, 'zwall' : in wall
    """
    # determine the symmetries
    # mirror symmetry across midplane
    mirrorsym = 0
    if ttastg.max() < np.pi/2.:
        # mirror symmetric
        mirrorsym = 1
    # assume axisymmetric
    axisym = 1

    # calculate zlos
    zlosi, zlos = getZlos(raxis, ttaaxis, zlosmax=zlosmax, nzlos=nzlos)

    # calculate sky to disk coordinates
    yvec = np.cos(inc) * ym + np.sin(inc) * zlos
    zvec = np.sin(inc) * ym - np.cos(inc) * zlos


    if axisym:
        yvec = np.abs(yvec)

    if axisym and mirrorsym:
        #y and z are positive
        zvec = np.abs(zvec)
    valcell, rvec, tvec = interpcont(yvec, zvec, cont2d, rstg, ttastg, pad)

    # use zwalls
    ywall = np.cos(inc) * ym + np.sin(inc) * zlosi
    zwall = np.sin(inc) * ym - np.cos(inc) * zlosi
    if axisym:
        ywall = np.abs(ywall)
    if axisym and mirrorsym:
        zwall = np.abs(zwall)
    valwall, rwall, twall = interpcont(ywall, zwall, cont2d, rstg, ttastg, pad)

    losout = {'valcell':valcell, 'zlos':zlos, 'rcell':rvec, 'tcell':tvec,
                  'ycell':yvec, 'zcell':zvec, 
              'valwall':valwall, 'zlosi':zlosi, 'rwall':rwall, 'twall':twall, 
                  'ywall':ywall, 'zwall':zwall}
    return losout

def extract3d(xaxis, yaxis, zaxis, dat3d, crd_sys, xvec,yvec, zvec, pad=0.):
    """
    extract profiles based on given Cartesian coordinates.
    The 3d data can be spherical or cartesian
    
    Parameters 
    ----------
    xaxis : first axis
    yaxis : second axis
    zaxis : third axis
    dat3d : the three dimensional data
    crd_sys : string
        'sph' for spherical coordinates
        'car' for cartesian coordinates
    xvec, yvec, zvec : the coordinates in Cartesian coordinates
    
    """
    func = RegularGridInterpolator((xaxis, yaxis, zaxis), dat3d, 
        method='linear', bounds_error=False, fill_value=pad)

    # convert x,y,z coordinates to spherical coordinates
    if crd_sys == 'car':
        profx = xvec
        profy = yvec
        profz = zvec
    elif crd_sys == 'sph':
        # radius
        profx = np.sqrt(xvec**2 + yvec**2 + zvec**2)

        # theta
        tvec = np.arctan2(zvec, np.sqrt(xvec**2 + yvec**2))
        reg = tvec < 0.
        tvec[reg] = tvec[reg] + 2.*np.pi
        profy = tvec

        # azimuth
        pvec = np.arctan2(yvec, xvec)
        reg = pvec < 0
        pvec[reg] = pvec[reg] + 2*np.pi
        profz = pvec

    nvec = len(xvec)
    prof = np.zeros([nvec], dtype=np.float64)
    for ii in range(nvec):
        prof[ii] = func([profx[ii], profy[ii], profz[ii]])

    return prof

def getTauz(zlosi, zlos, rho, kap):
    """ calculate optical depth as a function of line of sight
    Parameter
    --------
    zlosi : line of sight wall coordinate. increases further from the observer
    zlos  : line of sight cell coordinate
    rho   : line of sight cell density 
    kap   : line of sight opacity
    """
    nz = len(zlosi)
    nzs = nz - 1
    dzlos = zlosi[1:] - zlosi[:-1]

    dtaustg = kap * rho * dzlos
    tau = np.zeros(nz, dtype=np.float64)
    for iz in range(nzs):
        tau[iz+1] = sum(dtaustg[:iz+1])
    taustg = 0.5 * (tau[1:] + tau[:-1])
    return tau, dtaustg, taustg

def getTauLos(losdens, kap):
    """
    losdens = losout package the calculated density
    kap = float. opacity
    """
    rhocell = losdens['valcell']
    nzs =len(rhocell)
    nz = nzs + 1
    zlos = losdens['zlos'] # cell
    zlosi = losdens['zlosi'] # wall

    tau, dtaustg, taustg = getTauz(zlosi, zlos, rhocell, kap)


#    dzlos = zlosi[1:] - zlosi[:-1] # zlos is in the same direction as tau
#    dtaustg = kap * rhocell * dzlos

#    tau = np.zeros(nz, dtype=np.float64)
#    for iz in range(nzs):
#        tau[iz+1] = sum(dtaustg[:iz+1])

#    taustg = 0.5*(tau[1:] + tau[:-1])

    return tau, dtaustg, taustg

def getdTdTau(tau, dtaustg, lostemp, tauval=None, tauvalthres=None, dolnT=False):
    """
    Parameters
    ----------
    tau		: ndarray
        the optical depth axis calculated by getTauLos

    dtaustg	: ndarray
        dtau calculated by getTauLos

    lostemp 	: losout package
        losout package for temperature

    tauval	: float
        the optical depth value where we want the dTdTau. if not given, the full profile will be given 

    tauvalthres	: float
        the threshold maximum optical depth. if actual tau is less than this threshold, dTdTau will be zero. 
    
    """
    T0thres = 3.
    dtauthres = 1e-3
    temp = lostemp['valwall']
    # take out any values that has temperature less than 3
    flagreg = lostemp['valcell'] < T0thres

    if dolnT:
        dT = np.log(temp[1:] / temp[:-1])
    else:
        dT = temp[1:] - temp[:-1]
    taustg = 0.5*(tau[1:] + tau[:-1])
    T0 = lostemp['valcell']
    dTdTau = dT / dtaustg

    if tauval is None:
#        reg = abs(dtaustg) < dtauthres
#        dTdTau[reg] = 0.
#        dTdTau[flagreg] = 0.
        T0 = T0
    else:
        if tauvalthres is None: 
            tauvalthres = tauval
        if max(tau) > (tauvalthres):
            fdTau = sip.interp1d(taustg, dtaustg)
            fdTdTau = sip.interp1d(taustg, dTdTau)
            fT0 = sip.interp1d(taustg, T0)
            if (fdTau(tauval) < dtauthres) or (fT0(tauval) < T0thres):
                dTdTau = 0.
            else:
#                dTdTau = fdT(tauval) / fdTau(tauval)
                dTdTau = fdTdTau(tauval)
            T0 = fT0(tauval)
        else:
            dTdTau = 0.0 
            T0 = np.mean(T0)
    return dTdTau, T0

""" example use
import numpy as np
import matplotlib.pyplot as plt
from radmc3dPy import *

dat = analyze.readData(binary=False, ddens=True, dtemp=True)

rad = np.pi / 180.
raxis = dat.grid.xi
rstg = dat.grid.x
ttaaxis = dat.grid.yi
ttastg = dat.grid.y

inc = 0.*rad
ym = 1.*natconst.au

tcont = np.squeeze(dat.dusttemp[:,:,0])
lostemp = zylutils.los.extract(inc, ym, tcont, raxis, rstg, ttaaxis, ttastg, 0.)
# smooth the temperature
kern = np.array([1., 2., 3., 2., 1.], dtype=np.float64) / 9.
lostemp['valcell'] = np.convolve(lostemp['valcell'], kern, mode='same')
lostemp['valwall'] = np.convolve(lostemp['valwall'], kern, mode='same')

rhocont = np.log10(np.squeeze(dat.rhodust[:,:,0,0]))
reg = rhocont < -30.
rhocont[reg] = -30.
losdens = zylutils.los.extract(inc, ym, rhocont, raxis, rstg, ttaaxis, ttastg, -30.)
losdens['valcell'] = 10.**(losdens['valcell'])
losdens['valwall'] = 10.**(losdens['valwall'])
rhocont = 10.**(rhocont)

kap = 1.0
tau, dtaustg, taustg = zylutils.los.getTauLos(losdens, kap)

dTdTau, T0 = zylutils.los.getdTdTau(tau, dtaustg, lostemp)

plt.pcolormesh(rstg/natconst.au, ttastg, tcont.T)
plt.plot(lostemp['rcell']/natconst.au, lostemp['tcell'])
plt.xscale('log')
plt.show()

plt.pcolormesh(rstg/natconst.au, ttastg, np.log10(rhocont.T))
plt.plot(losdens['rcell']/natconst.au, losdens['tcell'])
plt.xscale('log')
plt.show()

plt.subplot(1,2,1)
plt.plot(losdens['zlos']/natconst.au, np.log10(losdens['valcell']))
plt.subplot(1,2,2)
plt.plot(losdens['zlos']/natconst.au, lostemp['valcell'])
plt.show()

plt.plot(losdens['zlosi']/natconst.au, tau)
plt.show()

if len(dTdTau) > 1:
    plt.plot(losdens['zlos']/natconst.au, dTdTau)
    plt.show()

"""
