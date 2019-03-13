"""
 DiskEqs.py
 common disk equations that can be used for multiple models

 * eqDustAlignment()
"""
from radmc3dPy import natconst
from radmc3dPy import crd_trans
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
        altype, param):
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
                = 'wght_sph' : weight radial, poloidal, toroidal 
                    - weights
                = 'wght_cyr' : weight cyradial, toroidal, vertical 
                    - weights
                = 'wght_car' : weighted x, y, z
                    - weights
                = 'bi_Bsph' : bilayer with setup in spherical coordinates
                    - bi_Bsph
    param : dictionary
            settings are searched in this depending on altype

    Arguments for param
    ------------------
    weights : array
              mesh for the weightings of different orientations

    bi_Bsph : dictionary
              'R0' : list in (r, theta, phi, magnitude) for R0
              'zq0' : list for zq0
              'qheight' : list for qheight
              'Bmid0' :  list for Bmid0
              'Batm0' : list for Batm0
              'qmid' : list for qmid
              'qatm' : list for qatm
              'flip' : list for flipping directions depending on z

    Returns
    -------
    alvec : 4 dimensional array. in (x,y,z,direction)
    """
    # collect some functions common to various altype
    # normalize at each coordinate
    def alvecNorm(alvec):
        # Normalize
        length = np.sqrt(alvec[:,:,:,0]*alvec[:,:,:,0] +
                     alvec[:,:,:,1]*alvec[:,:,:,1] +
                     alvec[:,:,:,2]*alvec[:,:,:,2])
        alvec[:,:,:,0] = np.squeeze(alvec[:,:,:,0]) / ( length + 1e-60 )
        alvec[:,:,:,1] = np.squeeze(alvec[:,:,:,1]) / ( length + 1e-60 )
        alvec[:,:,:,2] = np.squeeze(alvec[:,:,:,2]) / ( length + 1e-60 )
        return alvec

    # 2 layer model
    def getB2layer(cyrr, rr, zz, R0, zq0, qheight, Bmid0, Batm0, qmid, qatm, flip=False):
        """ the 2 layer model for B field
        """
        Bmid = Bmid0 * (cyrr / R0)**(-qmid)
        Batm = Batm0 * (rr / R0)**(-qatm)
        zq = zq0 * (cyrr / R0)**qheight
        crdshape = cyrr.shape
        Bfield = np.zeros(crdshape, dtype=np.float64)
        Bfield = Batm + (Bmid - Batm) * (np.cos(np.pi/2. * zz/zq))**2
        reg = abs(zz) >= zq
        Bfield[reg] = Batm[reg]
        # flip
        if flip is not False:
            if flip is 'upper':
                reg = zz > 0.
            elif flip is 'lower':
                reg = zz < 0.
            else:
                raise ValueError('flip should be "pos" or "neg"')
            Bfield[reg] = - Bfield[reg]
 
        return Bfield

    # ---------------------------------------------------------------

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
        cyrr = np.sqrt(xx**2 + yy**2)
        tt = np.arctan2(cyrr, zz)
        pp = np.arctan2(yy, xx)
    elif crd_sys is 'sph':
        rr = mesh[0]
        tt = mesh[1]
        pp = mesh[2]
        xx = rr * np.sin(tt) * np.cos(pp)
        yy = rr * np.sin(tt) * np.sin(pp)
        zz = rr * np.cos(tt)
        cyrr = rr * np.sin(tt)
    else:
        raise ValueError('incorrect input for crd_sys')

    # different modes for alignment
    # x
    if altype is 'x':
        alvec[:,:,:,0] = 1.0
        alvec = alvecNorm(alvec)

    # y
    elif altype is 'y':
        alvec[:,:,:,1] = 1.0
        alvec = alvecNorm(alvec)

    # z 
    elif altype is 'z':
        alvec[:,:,:,2] = 1.0
        alvec = alvecNorm(alvec)

    # radial
    elif altype is 'radial':
        alvec[:,:,:,0] = xx / rr
        alvec[:,:,:,1] = yy / rr
        alvec[:,:,:,2] = zz / rr
        alvec = alvecNorm(alvec)

    # cylradial
    elif altype is 'cylradial':
        alvec[:,:,:,0] = xx / cyrr
        alvec[:,:,:,1] = yy / cyrr
        alvec = alvecNorm(alvec)

    # poloidal
    elif altype is 'poloidal':
        raise ValueError('poloidal no implemented yet')
    # toroidal
    elif altype is 'toroidal':
        alvec[:,:,:,0] = yy / cyrr
        alvec[:,:,:,1] = -xx / cyrr
        alvec = alvecNorm(alvec)

    # bilayer in spherical coordinates
    elif altype in 'bi_Bsph':
        if param.has_key('bi_Bsph') is False:
            raise ValueError('no key, bi_Bsph, found for altype=bi_Bsph')
        par = param['bi_Bsph']
        Bfield, btags = [], ['r', 't', 'p']
        for ii in range(len(btags)):
            bii = getB2layer(cyrr,rr,zz,
                par['R0'][ii], par['zq0'][ii], par['qheight'][ii], 
                par['Bmid0'][ii], par['Batm0'][ii], par['qmid'][ii], par['qatm'][ii],
                flip=par['flip'][ii])
            Bfield.append(bii)
        bout = crd_trans.vtransSph2Cart(crd=[rr,tt,pp], v=Bfield)

        for ii in range(3):
            alvec[:,:,:,ii] = bout[ii]
        alvec = alvecNorm(alvec)

    # just 0
    elif altype is '0':
        alvec = alvec
    else:
        raise ValueError('no acceptable altype argument')

    return alvec

def fncavprof(cyr, cavpar):
    # z profile for cavity
    zcav = cavpar[1] * (cyr / cavpar[0])**(cavpar[2]) + cavpar[3]
    # always positive
    return zcav

def eqEnvelopeDens(rr, tt, Rrot, rho0, cavpar=None):
    """
    rr, tt = radius and theta meshgrid. [cm], [rad] in 2D
    Rrot = radius where infall velocity equals rotational velocity
        the radius where infall velocity equals zero is 0.5*Rrot
    rho0 = rho0 at Rrot. can be found by dm/4/pi/wkc where wkc=sqrt(GM*Rrot**3)
    cavpar = [Rc, Hc, qcav, Hoff, delH] 
        the location of cavity by H = Hc * (R/Rc)**qcav + Hoff
        where R is in cylindrical coordinates
        delH is the length scale in height for exponential tapering
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
    # ---------------------------------------------

    rshape = rr.shape
    dens = rr * 0.

    mu02d = rr*0.

    # mask. mask for multiplying to the density array
    cavmask = rr * 0 + 1.

    for ir in range(rshape[0]):
        for it in range(rshape[1]):
            cyr = rr[ir,it] * np.sin(tt[ir,it])
            zii = rr[ir,it] * np.cos(tt[ir,it])

            # find solution to theta0
            mu0 = findsol(rr[ir,it], tt[ir,it], Rrot)
            mu02d[ir,it] = mu0

            # density
            densii = ( rho0
                            * (rr[ir,it] / Rrot)**(-1.5)
                            * (1. + np.cos(tt[ir,it]) / mu0)**(-0.5)
                            * (np.cos(tt[ir,it])/mu0 / 2.
                               + Rrot / rr[ir,it] * (mu0)**2
                              )**(-1.) ) 
            # cavity
            if cavpar is not None:
                zcav = fncavprof(cyr, cavpar)
                delH = cavpar[4]
                if zii > zcav:
                    fac = np.exp( - ((zii - zcav) / delH)**2)
                elif zii < -zcav:
                    fac = np.exp( - ((zii + zcav) / delH)**2)
                else:
                    fac = 1.
                cavmask[ir,it] = fac
            else:
                fac = 1.
                cavmask[ir,it] = fac

            dens[ir, it] = fac * densii

    return dens, cavmask
