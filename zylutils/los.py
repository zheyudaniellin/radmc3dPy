# for calculating line-of-sight properties
import numpy as np
import scipy.interpolate as sip
import pdb
from radmc3dPy import natconst

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

    # create the inteprolation object
    fcont = sip.interp2d(rstg, ttastg, cont.T, kind='linear',
        fill_value=pad, bounds_error=False)

    # find the corresponding radius and theta vectors
    # y = r sin(theta)
    # z = r cos(theta)
    rvec = np.sqrt(yvec**2. + zvec**2.)
    tvec = np.arctan2(yvec, zvec)
    reg = tvec < 0.
    tvec[reg] = tvec[reg] + 2.*np.pi

    vals = np.zeros(nvec, dtype=np.float64)
    for iv in range(nvec):
        vals[iv] = fcont(rvec[iv], tvec[iv])

    return vals, rvec, tvec

def getZlos(raxis, ttaaxis, zlosmax=None):
    # determine a good grid for line of sight z
    nr = len(raxis)
    minr = min([0.01*natconst.au, raxis[0]])
    if zlosmax is None:
        maxr = max(raxis) * 0.75
    else:
        maxr = zlosmax
    if maxr < minr:
        raise ValueError('minr cannot be greater than maxr')

    rlen = maxr * (minr / maxr)**(np.arange(nr)/float(nr-1))
    # rlen = raxis
    zlosi = np.concatenate((rlen, np.array([0.]), -rlen), axis=0)
    zlosi.sort()
    zlos = 0.5 * (zlosi[1:] + zlosi[:-1])

    return zlosi, zlos


def extract(inc, ym, cont2d, raxis, rstg, ttaaxis, ttastg, pad, zlosmax=None):
    """
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
    mirrorsym = 0
    if ttastg.max() < np.pi/2.:
        # mirror symmetric
        mirrorsym = 1
    # assume axisymmetric
    axisym = 1

    # calculate zlos
    zlosi, zlos = getZlos(raxis, ttaaxis, zlosmax=zlosmax)

    # calculate sky to disk coordinates
    yvec0 = np.cos(inc) * ym - np.sin(inc) * zlos
    yvec = yvec0
    zvec0 = np.sin(inc) * ym + np.cos(inc) * zlos
    zvec = zvec0

    if axisym:
        yvec = np.abs(yvec)

    if axisym and mirrorsym:
        #y and z are positive
        zvec = np.abs(zvec)
    valcell, rvec, tvec = interpcont(yvec, zvec, cont2d, rstg, ttastg, pad)

    # use zwalls
    ywall = np.cos(inc) * ym - np.sin(inc) * zlosi
    zwall = np.sin(inc) * ym + np.cos(inc) * zlosi
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

def getTauLos(losdens, kap):
    """
    losdens = losout package the calculated density
    kap = float. opacity
    """
    rhocell = losdens['valcell']
    rhowall = losdens['valwall']
    nz = len(rhowall)
    nzs = nz - 1
    zlos = losdens['zlos'] # cell
    zlosi = losdens['zlosi'] # wall
    dzlos = zlosi[:-1] - zlosi[1:]
    dtaustg =  - kap * rhocell * dzlos
    tau = np.zeros(nz, dtype=np.float64)
    for iz in range(nzs):
        jj = nzs-1-iz
        tau[jj] = sum(dtaustg[jj:])
    taustg = 0.5*(tau[1:] + tau[:-1])

    return tau, dtaustg, taustg

def getdTdTau(tau, dtaustg, lostemp, tauval=None, tauvalthres=None):
    """
    lostemp = losout package with temperature
    dtaustg = dtau calculated by getTauLos
    """
    T0thres = 3.
    dtauthres = 1e-3
    temp = lostemp['valwall']
    # take out any values that has temperature less than 3
    flagreg = lostemp['valcell'] < T0thres

    dT = temp[1:] - temp[:-1]
    taustg = 0.5*(tau[:-1] + tau[1:])
    T0 = lostemp['valcell']
    if tauval is None:
        dTdTau = dT / dtaustg
        reg = abs(dtaustg) < dtauthres
        dTdTau[reg] = 0.
        dTdTau[flagreg] = 0.
        T0 = T0
    else:
        if tauvalthres is None: 
            tauvalthres = tauval
        if max(tau) > (tauvalthres):
            fdTau = sip.interp1d(taustg, dtaustg)
            fdT = sip.interp1d(taustg, dT)
            fT0 = sip.interp1d(taustg, T0)
            if (fdTau(tauval) < dtauthres) or (fT0(tauval) < T0thres):
                dTdTau = 0.
            else:
                dTdTau = fdT(tauval) / fdTau(tauval)
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
