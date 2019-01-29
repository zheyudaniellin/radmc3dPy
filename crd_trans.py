"""
This module contains functions for coordinate transformations (e.g. rotation).
For help on the syntax or functionality of each function see the help of the individual functions

some additions:
- added contSph2Cart and contCart2Sph to convert contours of spherical to cartesian and vice versa

"""
from __future__ import absolute_import
from __future__ import print_function
import traceback

try:
    import numpy as np
    from scipy import interpolate 
except ImportError:
    np = None
    print(traceback.format_exc())

import pdb

def ctransSph2Cart(crd=None, reverse=False):
    """Transform coordinates between spherical to cartesian systems

    Parameters
    ----------
    crd      : ndarray
               Three element array containing the input
               coordinates [x,y,z] or [r,theta,phi] by default
               the coordinates assumed to be in the cartesian system

    reverse  : bool
               If True calculates the inverse transformation
               (cartesian -> spherical). In this case crd should be [r,theta,phi]

    Returns
    -------
    Returns a three element array containig the output coordinates [r,theta,phi] or [x,y,z]
    """

    if crd is None:
        raise ValueError('Unknown crd. Cannot do coordinate transformation without knowing the coordinates.')

    if reverse is False:
        r = crd[0]
        theta = crd[1] + 1e-50
        phi = crd[2]

        x = np.sin(theta) * np.cos(phi) * r
        y = np.sin(theta) * np.sin(phi) * r
        z = np.cos(theta) * r

        crdout = [x, y, z]

    else:

        x = crd[0]
        y = crd[1]
        z = crd[2]

        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arccos(x / np.sqrt(x**2 + y**2) + 1e-90)
        theta = np.arccos(z / r)

        if y < 0.0:
            phi = 2.0 * np.pi - phi

        crdout = [r, theta, phi]

    return crdout


def vtransSph2Cart(crd=None, v=None, reverse=False):
    """Transform velocities between spherical to cartesian systems

    Parameters
    ----------
    crd     : ndarray
              Three element array containing the input
              coordinates [x,y,z] or [r,theta,phi] by default
              the coordinates assumed to be in the cartesian system

    v       : ndarray
              Three element array containing the input
              velocities in the same coordinate system as crd


    reverse : bool
              If True it calculates the inverse trasnformation (cartesian -> spherical)

    Returns
    -------

    Returns a three element array containg the output velocities [vr,vphi,vtheta] or [vx,vy,vz]


    """

    # NOTE!!!!! The velocities in the spherical system are not angular velocities!!!!
    # v[1] = dphi/dt * r
    # v[2] = dtheta/dt * r

    if crd is None:
        raise ValueError('Unknown crd. Cannot do coordinate transformation without knowing the coordinates.')

    if v is None:
        raise ValueError('Unknown v. Cannot transform vectors without knowing the vectors themselves.')

    if reverse is False:
        # r = crd[0]
        theta = crd[1]
        phi = crd[2]

        vr = v[0]
        vtheta = v[1]
        vphi = v[2]

        vx = vr * np.sin(theta) * np.cos(phi) - vphi * np.sin(phi) + vtheta * np.cos(theta) * np.cos(phi)
        vy = vr * np.sin(theta) * np.sin(phi) + vphi * np.cos(phi) + vtheta * np.cos(theta) * np.sin(phi)
        vz = vr * np.cos(theta) - vtheta * np.sin(theta)

        vout = [vx, vy, vz]

    else:

        # crd_sph = ctrans_sph2cart(crd, reverse=True)
        # r       = crd_sph[0]
        # theta   = crd_sph[1]
        # phi     = crd_sph[2]

        # a       = [[np.sin(theta)*np.cos(phi), -np.sin(phi), np.cos(theta)*np.cos(phi)],\
        # [np.sin(theta)*np.sin(phi), np.cos(phi), np.cos(theta)*np.sin(phi)],\
        # [np.cos(theta), 0., -np.sin(theta)]]

        # a       = [[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],\
        # [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)],\
        # [np.cos(theta), -np.sin(theta),0.]]
        # a       = np.array(a, dtype=np.float64)
        # vout = np.linalg.solve(a,v)

        #
        # New stuff
        #
        vout = np.zeros(3, dtype=np.float64)
        r = np.sqrt((crd**2).sum())
        rc = np.sqrt(crd[0]**2 + crd[1]**2)

        # Vr
        vout[0] = (crd * v).sum() / r
        # Vtheta
        vout[1] = (crd[2] * (crd[0] * v[0] + crd[1] * v[1]) - v[2] * rc**2) / (r * rc)
        # Vphi
        vout[2] = (crd[0] * v[1] - crd[1] * v[0]) / rc

    return vout


def csrot(crd=None, ang=None, xang=0.0, yang=0.0, zang=0.0, deg=False):
    """ Performs coordinate system rotation.

    Parameters
    ----------

    crd : numpy ndarray
          Three element vector containing the coordinates of a given point in a cartesian system

    ang : list, ndarray
          Three element list/ndarray describing the rotation angles around the x, y and z axes, respectively
          
    xang: float
          Rotation around the x-axis
    
    yang: float
          Rotation around the y-axis

    zang: float
          Rotation around the z-axis
    
    deg : float, optional 
          If True angles should be given in degree instead of radians (as by default)


    Returns
    -------
    list
        Returns a three element list with the rotated coordinates


    Notes
    -----

    Rotation matrices

    Around the x-axis:

    .. math:: 
         \\left(\\begin{matrix} 
                 1 & 0 & 0 \\\\
                 0 & cos(\\alpha) & -sin(\\alpha)\\\\
                 0 & sin(\\alpha) & cos(\\alpha)
                 \\end{matrix}\\right)

    Around the y-axis:

    .. math:: 
         \\left(\\begin{matrix} 
                 cos(\\beta) & 0 & -sin(\\beta) \\\\
                 0 & 1 & 0\\\\
                 sin(\\beta)& 0 & cos(\\beta)
                 \\end{matrix}\\right)

    Around the z-axis

    .. math:: 
         \\left(\\begin{matrix} 
                 cos(\\gamma) & -sin\\gamma) & 0 \\\\
                 sin(\\gamma) &  cos(\\gamma) & 0 \\\\
                 0  & 0 & 1
                 \\end{matrix}\\right)


    """

    if crd is None:
        raise ValueError('Unknown crd. Cannot do coordinate transformation without knowing the coordinates.')

    if ang is None:

        if (xang == 0.) & (yang == 0.) & (zang == 0.):
            return crd

    if ang is not None:
        xang = ang[0]
        yang = ang[1]
        zang = ang[2]

    #
    # Convert degree into radian if the angles are given in degree
    #
    if deg:
        xang = xang / 180.0 * np.pi
        yang = yang / 180.0 * np.pi
        zang = zang / 180.0 * np.pi

    crd_new = np.zeros(len(crd), dtype=np.float64)
    #
    # Rotation around the x axis
    #
    if xang != 0.0:
        dumx = crd[0]
        dumy = np.cos(xang) * crd[1] - np.sin(xang) * crd[2]
        dumz = np.sin(xang) * crd[1] + np.cos(xang) * crd[2]

        crd_new = [dumx, dumy, dumz]

    #
    # Rotation around the y axis
    #
    if yang != 0.0:
        dumx = np.cos(yang) * crd[0] + np.sin(yang) * crd[2]
        dumy = crd[1]
        dumz = -np.sin(yang) * crd[0] + np.cos(yang) * crd[2]

        crd_new = [dumx, dumy, dumz]

    #
    # Rotation around the z axis
    #
    if zang != 0.0:
        dumx = np.cos(zang) * crd[0] - np.sin(zang) * crd[1] + 0.0
        dumy = np.sin(zang) * crd[0] + np.cos(zang) * crd[1] + 0.0
        dumz = crd[2]

        crd_new = [dumx, dumy, dumz]

    return crd_new


def vrot(crd=None, v=None, ang=None):
    """Rotates a vector in spherical coordinate system.
    First transforms the vector to cartesian coordinate system, then does the rotation then 
    makes the inverse transformation

    Parameters 
    ----------
    crd  : ndarray
           Three element array containing the coordinates of a
           given point in the cartesian system

    v    : ndarray
           Three element array, angles of rotation around the x,y,z axes

    ang  : ndarray 
           Three element arrray containing the angles to rotate around the x, y, z, axes, respectively

    """
    if crd is None:
        raise ValueError('Unknown crd. Cannot do coordinate transformation without knowing the coordinates.')
    if v is None:
        raise ValueError('Unknown v. Vector rotation cannot be done without knowing the vectors themselves.')
    if ang is None:
        raise ValueError('Unknown ang. Vector rotation cannot be done without knowing the rotation angles.')

    # Convert the position vector to cartesian coordinate system
    crd_xyz = ctransSph2Cart(crd=crd)
    # Convert the velocity vector to cartesian coordinate system
    v_xyz = vtransSph2Cart(crd=crd, v=v)
    # Rotate the vector
    v_xyz_rot = csrot(crd=v_xyz, ang=ang)
    # Transform the rotated vector back to the spherical coordinate system
    v_rot = vtransSph2Cart(crd=crd_xyz, v=v_xyz_rot, reverse=True)

    return v_rot

def contSph2Cart(contSph=None, raxis=None, ttaaxis=None, xaxis=None, zaxis=None, 
        log=False, pad=0., midplane=True):
    """takes in 2d contours in spherical coordinates and outputs to cartesian coordinates

    Parameters
    ----------
    cont  : ndarray
           two dimensional array 
    raxis : ndarray
           the one dimensional axis for radius
    ttaaxis : ndarray
	     one dimensional axis for theta from 0~pi. 0 points towards Cartesian zaxis. 
    xaxis : ndarray, optional
           axis for x coordinates. if not given, the default will be the same as raxis
    zaxis : ndarray, optional
           axis for z coordinates. if not given, the default will cover the range of heights
           expanded by ttaaxis, and number of elements will be that of ttaaxis
    log : bool, optional
         turn on to interpolate on logarithmic of contour
    logaxis : bool, optional
         turn on to use log of raxis, xaxis, zaxis. not implemented yet
    pad : float
         the value to take for interpolation if xaxis and zaxis are outside raxis,ttaaxis
    midplane : bool, optional
               If true, default zaxis will have 0.0 in it, else it will not be included
   
    Return
    ---------
    {contCart:contCart, xaxis:xaxis, zaxis:zaxis, dzstg:dzstg} dictionary
   
    """
    if contSph is None:
        raise ValueError('Unknown contSph. Cannot do interpolation')
    if raxis is None:
        raise ValueError('Unknown raxis. Cannot do contSph2Cart')
    if ttaaxis is None:
        raise ValueError('Unknown ttaaxis. Cannot do contSph2Cart')

    # set up log keywords
    if log:
        d = np.log10(contSph)
    else:
        d = contSph

    if log:
        dpad = np.log10(pad)
    else:
        dpad = pad

    nr = raxis.size
    ntta = ttaaxis.size

    if xaxis is None:
        xaxis = raxis
    nx = xaxis.size

    # since using spherical coordinates usually sample heights in inner radius more, 
    # the default for zaxis will be in log step size
    if zaxis is None:
        maxz = raxis.max() * np.cos(ttaaxis.min())
        minz = raxis.max() * np.cos(ttaaxis.max())
        if maxz*minz > 0: #does not cross zero
            nz = ntta
            zaxis = minz * (abs(maxz) / abs(minz))**(np.arange(nz,dtype=np.float64)/(nz-1))
            zwalls = minz * (abs(maxz) / abs(minz))**(np.arange(nz+1, dtype=np.float64)/(nz))
            dzstg = zwalls[1:] - zwalls[:-1]
        else: 
            nz1 = ntta // 2
            minzpos = min(abs(raxis[0] * np.cos(ttaaxis)-0))
            z1 = abs(minzpos) * (abs(maxz) / abs(minzpos))**(np.arange(nz1,dtype=np.float64)/(nz1-1))
            z1walls = abs(minzpos) * (abs(maxz) / abs(minzpos))**(np.arange(nz1+1,dtype=np.float64)/(nz1))
            if midplane is True:
                zaxis = np.concatenate((z1, -z1, [0.0]))
                zwalls = np.concatenate((z1walls, -z1walls))
            else:
                zaxis = np.concatenate((z1, -z1))
                zwalls= np.concatenate((z1walls[1:], -z1walls[1:], [0.0]))
            zaxis.sort()
            zwalls.sort()
            dzstg = zwalls[1:] - zwalls[:-1]
    else:
        zwalls = np.zeros([len(zaxis)+1], dtype=np.float64)
        zwalls[1:-1] = 0.5 * (zaxis[:-1] + zaxis[1:])
        zwalls[0] = zaxis[0] - abs(zaxis[1] - zaxis[2])
        zwalls[-1] = zaxis[-1] + abs(zaxis[-1] - zaxis[-2])
        dzstg = zwalls[1:] - zwalls[:-1]
    nz = zaxis.size

    # create 1d vectors
    xvec = np.zeros([nr*ntta], dtype=np.float64)
    zvec = np.zeros([nr*ntta], dtype=np.float64)
    points = np.zeros([nr*ntta, 2], dtype=np.float64)
    dvec = np.zeros([nr*ntta], dtype=np.float64)

    ii = 0
    for ir in range(nr):
        for it in range(ntta):
            xvec[ii] = raxis[ir] * np.sin(ttaaxis[it])
            zvec[ii] = raxis[ir] * np.cos(ttaaxis[it])
            points[ii,0] = raxis[ir] * np.sin(ttaaxis[it])
            points[ii,1] = raxis[ir] * np.cos(ttaaxis[it])
            dvec[ii] = d[ir, it]
            ii = ii+1

    # create the interpolation class?
#    f = interpolate.interp2d(xvec, zvec, dvec, kind='linear', fill_value=pad)
    
    # the coordinates to interpolate onto. in 1d
    xnew = np.zeros([nx*nz], dtype=np.float64)
    znew = np.zeros([nx*nz], dtype=np.float64)
    newpoints = np.zeros([nx*nz, 2], dtype=np.float64)
    ii = 0
    for ix in range(nx):
        for iz in range(nz):
            xnew[ii] = xaxis[ix]
            znew[ii] = zaxis[iz]
            newpoints[ii,0] = xaxis[ix]
            newpoints[ii,1] = zaxis[iz]
            ii = ii+1

    # do interpolation
#    doutvec = f(xnew, znew)
    doutvec = interpolate.griddata(points, dvec, newpoints, method='linear',fill_value=dpad)

    # redo onto gridded data
    contout = np.zeros([nr, nz], dtype=np.float64)
    ii = 0
    for ix in range(nx):
        for iz in range(nz):
            contout[ix,iz] = doutvec[ii]
            ii = ii + 1

    # unlog the output 
    if log:
        contout = 10.**(contout)
    
    contCart = {'contCart':contout, 'xaxis':xaxis, 'zaxis':zaxis, 'dzstg':dzstg}
    return contCart

def contCart2Sph(contCart=None, xaxis=None, zaxis=None, raxis=None, ttaaxis=None,
        log=False, pad=0.):
    """takes in 2d contours in cartesian coordinates and outputs to spherical coordinates

    Parameters
    ----------
    cont  : ndarray
            two dimensional array
    xaxis : ndarray
            1 dimensional axis for xaxis. this must be greater than 0
    zaxis : ndarray
            1 dimensional axis for zaxis.
    raxis : ndarray, optional 
            if not given, the default wil be the same as xaxis
    ttaaxis : ndarray, optional
            if not given, the default will cover the range of zaxis with same elements as zaxis
    log   : bool, optional
            to interpolate on logarithmic of contour
    logaxis : bool, optional
         turn on to use log of raxis, xaxis, zaxis. not implemented yet
    pad : float
         the value to take for interpolation if xaxis and zaxis are outside raxis,ttaaxis

    Return
    ---------
    {contCart:contSph, raxis:raxis, ttaaxis:ttaaxis} dictionary
    """
    if contCart is None:
        raise ValueError('Unknown contCart. Cannot do interpolation')
    if xaxis is None:
        raise ValueError('Unknown xaxis. Cannot do contCart2Sph')
    if zaxis is None:
        raise ValueError('Unknown zaxis. Cannot do contCart2Sph')

    # set up log keywords
    if log:
        din = np.log10(contCart)
    else:
        din = contCart

    if log:
        dpad = np.log10(pad)
    else:
        dpad = pad

    nx = xaxis.size
    nz = zaxis.size

    if raxis is None:
        raxis = xaxis
    nr = raxis.size

    if ttaaxis is None:
        mintta = 0.
        maxtta = np.pi
        ttaaxis = np.arange(nz, dtype=np.float64)/(nz-1) * (maxtta-mintta)
    ntta = ttaaxis.size

    # create 1d vectors
    xvec = np.zeros([nx*nz], dtype=np.float64)
    yvec = np.zeros([nx*nz], dtype=np.float64)
    points = np.zeros([nx*nz, 2], dtype=np.float64)
    dvec = np.zeros([nx*nz], dtype=np.float64)

    ii = 0
    for ix in range(nx):
        for iz in range(nz):
            xvec[ii] = xaxis[ix]
            yvec[ii] = zaxis[iz]
            points[ii,0] = xaxis[ix]
            points[ii,1] = zaxis[iz]
            dvec[ii] = din[ix,iz]
            ii = ii + 1

    # the coordinates to interpolate onto
    xnew = np.zeros([nr*ntta], dtype=np.float64)
    ynew = np.zeros([nr*ntta], dtype=np.float64)
    newpoints = np.zeros([nr*ntta, 2], dtype=np.float64)
    ii = 0
    for ir in range(nr):
        for it in range(ntta):
            xnew[ii] = raxis[ir] * np.sin(ttaaxis[it])
            ynew[ii] = raxis[ir] * np.cos(ttaaxis[it])
            newpoints[ii,0] = raxis[ir] * np.sin(ttaaxis[it])
            newpoints[ii,1] = raxis[ir] * np.cos(ttaaxis[it])
            ii = ii + 1

    # do interpolation
#    doutvec = f(xnew, ynew)
    doutvec = interpolate.griddata(points, dvec, newpoints, method='cubic',fill_value=dpad)

    # redo onto gridded data
    contout = np.zeros([nr, ntta], dtype=np.float64)
    ii = 0
    for ir in range(nr):
        for it in range(ntta):
            contout[ir,it] = doutvec[ii]
            ii = ii + 1

    # unlog the output
    if log:
        contout = 10.**contout

    contSph = {'contSph':contout, 'raxis':raxis, 'ttaaxis':ttaaxis}
    return contSph


