"""
 DiskEqs.py
 common disk equations that can be used for multiple models

 * eqDustAlignment()
"""
from .. import natconst
import numpy as np
from scipy.interpolate import interp1d
import pdb

def eqLyndenSurf(r, mdisk, R0, p):
    # r in AU, mdisk in Msun
    if p == 2:
        raise ValueError('cannot accept p=2')
    else:
        sig0 = (2. - p) * mdisk * natconst.ms / 2. / np.pi / (R0*natconst.au)**2.
    sig = sig0 * (r / R0)**(-p) * np.exp(-(r/R0)**(2.-p))
    if np.size(r) > 1:
        reg = r < 0.01
        sig[reg] = 0.
    else:
        if r < 0.01:
            sig = 0.
    return sig

def eqDustAlignment(crd_sys,xaxis,yaxis,zaxis, 
        altype):
    """
    Parameters
    ----------
    crd_sys : string
              Cartesian: 'car'
              Spherical: 'sph'
    xaxis, yaxis, zaxis : 1d array for each axis
    altype	= '0': all zeros
                = 'x': aligned to x direction. or 'y', 'z'
                = 'toroidal' : toroidal alignment
                = 'poloidal' : poloidal alignment
                = 'radial'   : radial alignment, in spherical coordinates
                = 'cylradial': radial in cylindrical coordinates
    Returns
    -------
    alvec : 4 dimensional array. in (x,y,z,direction)
    """
    mesh = np.meshgrid(xaxis, yaxis, zaxis, indexing='ij')
    nx = len(xaxis)
    ny = len(yaxis)
    nz = len(zaxis)
    alvec = np.zeros([nx, ny, nz, 3], dtype=np.float64)

    if crd_sys is 'car':
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        rr = np.sqrt(xx**2. + yy**2. + zz**2.)
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

def eqEnvelopeDens(rr, tt, Rrot, rho0, topen=0, deltopen=5):
    """
    rr, tt = radius and theta meshgrid. [cm], [rad] in 2D
    par = [Rrot, rho0]
    rho0 = rho0 at Rrot. can be found by dm/4/pi/wkc where wkc=sqrt(GM*Rrot**3)
    topen = opening angle [deg]
    deltopen = change in opening angle [deg]
    """
    # ---------------solution to stream line---------
    def findsol(r, t, Rrot):
        mu = np.cos(t)
        coeff = [1., 0., (r/Rrot - 1.), -r/Rrot*mu]
        res = np.roots(coeff)
        flg = [False, False, False]
        for ii in range(3):
           resii = res[ii]
           if np.imag(resii) != 0.: flg[ii] = True
           if np.real(resii) > 1.: flg[ii] = True
           if np.real(resii) < -1.: flg[ii] = True
        if (False in flg) is False:
            raise ValueError('solution not found')

        if (mu > 0):
            reg = np.real(res) > 0
            goodres = np.real(res[reg])
        elif (mu == 0):
            goodres = 0.
        else:
            reg = np.real(res) < 0
            goodres = np.real(res[reg])
        if len(goodres) != 1:
            raise ValueError('solution not found')

        return goodres
    # ------------------------------------------------

    rshape = rr.shape
    dens = rr * 0.

    mu02d = rr*0.

    for ir in range(rshape[0]):
        for it in range(rshape[1]):
            # find solution to theta0
            mu0 = findsol(rr[ir,it], tt[ir,it], Rrot)
            mu02d[ir,it] = mu0

            # density
            densii = ( rho0
                            * (rr[ir,it] / Rrot)**(-1.5)
                            * (1. + np.cos(tt[ir,it]) / mu0)**(-0.5)
                            * (np.cos(tt[ir,it])/mu0 
                               + 0.5 * Rrot / rr[ir,it] * (mu0)**2
                              )**(-1.) ) 
            # opening angle
            if tt[ir,it] < (topen*natconst.rad):
                fac = np.exp(-(
                               (topen*natconst.rad - tt[ir,it])
                               / (deltopen*natconst.rad) 
                              )**2)
            elif tt[ir,it] > (np.pi-(topen*natconst.rad)):
                fac = np.exp(-(
                               ((np.pi-topen*natconst.rad) - tt[ir,it]) 
                               / (deltopen*natconst.rad)
                              )**2)
            else:
                fac = 1.
            dens[ir, it] = fac * densii

    return dens
