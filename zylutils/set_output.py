# output_im.py
# upgrades:
# - polarization intensity
# - overplot tau=1 surface to density and temperature
# - show polarization level based on temperature at tau=1 surface
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from astropy.convolution import convolve
from radmc3dPy import *
import los
import pdb
import fntools
import copy

def getIm2Tb(im, wav):
    ld2 = (wav*1e-4)**2.
    hmu = natconst.hh * natconst.cc / wav / 1e-4
    hmu3_c2 = natconst.hh * (natconst.cc / wav / 1e-4)**3. / natconst.cc**2
    tb = hmu / natconst.kk / np.log(2.*hmu3_c2/(im+1e-90) +1.)
    return tb

def getOpticalDepth(rho, xaxis, dzstg, kext):
    nx = len(xaxis)
    opdepth = np.zeros([nx], dtype=np.float64)
    for xx in range(nx):
        opdepth[xx] = np.sum(rho[xx,:]*dzstg*kext)

    return opdepth

def getTatTau(dtemp, xx, yy, zz, k0):
    # dtemp = radmc3dPy.data
    # xx, yy, zz = the locations of tau surface
    # k0 = the alignment factor. ex kpara or korth
    return tau_t

def getImPol(I, Q, U, alignang=None):
    #alignang = alignment angle relative to +x of image
    qq = Q / I.clip(1e-60)
    uu = U / I.clip(1e-60)
    pol = np.sqrt(qq**2 + uu**2).clip(1e-60) #fraction of linear polarization 
    kpol = -np.sign(qq) #the sign
    pol = kpol * pol
    
    # determine angles
    qqr = qq / pol
    uur = uu / pol
    ang = np.arccos(qqr) / 2.0
    ii = (uur < 0)
    if True in ii:
        ang[ii] = np.pi - ang[ii]

#    if alignang is not None:
#        angdis = abs(ang-alginang)
#        reg = (angdis<=np.pi/4.) or (angdis<)
#        pol[reg] = -pol[reg]

    return pol

def getOptPolEm(opdepth, kpara90, korth90):
    # polarization by emission
    #opdepth= opticaldepth. kpara,korth at one wavelength and 90 deg
    p0 = (korth90 - kpara90) / (kpara90 + korth90)
    pol = np.zeros(opdepth.shape, dtype=np.float64) + p0
    pol = -np.exp(-opdepth)*np.sinh(p0*opdepth) / (1.-np.exp(-opdepth)*np.cosh(p0*opdepth) + 1e-90)
#    pol = abs(pol) #don't care about the signs just yet
    return pol

def getOutput_im(ifreq, dis, im, optim, tauim, polunitlen, fkabs, fksca,pngname, polmax=None, imTblim=None, xlim=None, ylim=None):
    """ calculates image
    xlim : tuple
    ylim : tuple
    """
    # determine if there is stokes data
    dostokes = False
    if im.stokes:
        if im.image[:,:,1,ifreq].max() == 0 and im.image[:,:,2,ifreq].max() ==0:
            dostokes = False
        else:
            dostokes = True
    # image, polarized intensity, pol frac, tau, optical depth, pol vectors
    nsub = 1 #image
    if dostokes: # polarized intensity
        nsub = nsub + 1
    if dostokes: # polarization fraction
        nsub = nsub + 1
    if optim is not None:
        nsub = nsub + 1
    if tauim is not None:
        nsub = nsub + 1
    if dostokes: #vectors
        nsub = nsub + 1

    nrow = np.floor(np.sqrt(nsub))
    ncol = np.ceil(nsub / nrow)
    nrow, ncol = int(nrow), int(ncol)

    fig = plt.figure(num=ifreq, figsize=(4*ncol, 3*nrow))

    axes = []
    isubplot = 0
    # image
    if imTblim is not None:
        imTbvmin, imTbvmax = imTblim[0], imTblim[1]
    else:
        imTbvmin, imTbvmax = 0, None
    isubplot = isubplot + 1
    ax = fig.add_subplot(nrow, ncol, isubplot)
    axes.append(ax)
    image.plotImage(image=im, cmap=plt.cm.jet, interpolation='bilinear',
        arcsec=False, au=True, dpc=dis, oplotbeam='w', 
        stokes='I', bunit='Tb',ifreq=ifreq, saturate='90percent', 
        clevs=[1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], clcol='w', 
        vmin=imTbvmin, vmax=imTbvmax, ax=ax)
    if im.stokes is True:
        image.plotPolDir(image=im, arcsec=False, au=True, dpc=dis, color='w',
            nx=16, ny=16, polunitlen=polunitlen, ifreq=ifreq, ax=ax)
    #
    # polarized intensity
    if dostokes:
        isubplot = isubplot + 1
        ax = fig.add_subplot(nrow,ncol,isubplot)
        axes.append(ax)
        image.plotImage(image=im, cmap=plt.cm.jet, interpolation='bilinear',
            arcsec=False, au=True, dpc=dis, oplotbeam='w',
            stokes='PI', bunit='Tb',ifreq=ifreq, saturate='90percent',
            clevs=[0, 1.0, 10., 100., 200., 500.], clcol='w', 
            ax=ax)

    # polarized degree
    if dostokes:
        isubplot = isubplot + 1
        ax = fig.add_subplot(nrow,ncol,isubplot)
        axes.append(ax)
        image.plotImage(image=im, cmap=plt.cm.jet, interpolation='bilinear', 
            arcsec=False, au=True, dpc=dis, ifreq=ifreq,
            saturate='100percent', stokes='P',bunit='percent' , vmin=0, vmax=polmax,
            clevs=[1., 5.0, 10., 20.], clcol='k', ax=ax)

    # optical depth
    if optim is not None:
        isubplot = isubplot + 1
        ax = fig.add_subplot(nrow,ncol,isubplot)
        axes.append(ax)
        kext = fkabs(im.wav[ifreq]) + fksca(im.wav[ifreq])
        image.plotImage(image=optim, cmap=plt.cm.jet, interpolation='bilinear',
            arcsec=False, au=True, dpc=dis, ifreq=ifreq,
            saturate='100percent', stokes='I', bunit='optdepth',
            clevs=[0.1,1.,5., 10.,100.], clcol='w', ax=ax
            )
 
    # tau image
    if tauim is not None:
        isubplot = isubplot + 1
        ax = fig.add_subplot(nrow,ncol,isubplot)
        axes.append(ax)
        reg = tauim.image[:,tauim.ny/2,0,ifreq] > tauim.image.min()
        if True in reg:
            xmax = tauim.x[reg].max()/natconst.au
            vmax = tauim.image[:,:,0,ifreq].max()/natconst.au
        else:
            xmax = tauim.x.max()/natconst.au
            vmax = tauim.image[:,:,0,:].max()/natconst.au
        # tausurface
        image.plotImage(image=tauim, cmap=plt.cm.jet, interpolation='bilinear',
            arcsec=False, au=True, dpc=dis,
            stokes='I', bunit='length', ifreq=ifreq,
            vmin=-vmax,
            vmax=vmax, ax=ax)
        nx = int(2.*tauim.x.max()/natconst.au / (xmax / 8))
        if (nx % 2) == 1:
            nx = nx + 1
        if nx > 50: nx = 50
        ny = nx
        # polarization vectors
        image.plotPolDir(image=im, arcsec=False, au=True, dpc=dis, color='k',
            nx=nx, ny=ny, polunitlen=polunitlen, ifreq=ifreq, ax=ax)
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)

    # only polarized vectors that varies with polarized fraction, no image
    if dostokes:
        isubplot = isubplot + 1
        ax = fig.add_subplot(nrow,ncol,isubplot)
        axes.append(ax)
        image.plotPolDir(image=im, arcsec=False, au=True, dpc=dis, color='k',
            nx=16, ny=16, polunitlen=-1, ifreq=ifreq, ax=ax)
        ax.set_title('Polarization E Vectors')

    # xlim and ylim
    if xlim is not None:
        for ii in range(len(axes)):
            axes[ii].set_xlim(xlim)
    if ylim is not None:
        for ii in range(len(axes)):
            axes[ii].set_ylim(ylim)

    # set aspect radtio
    for ii in range(len(axes)):
        axes[ii].set_aspect('equal')

    fig.tight_layout()
    fig.savefig(pngname)
    plt.close()

def getOutput_xy(ifreq, tau3d, dat, polunitlen, acclum, parobj, pngname):
    totdmass = dat.getDustMass()
    plt.figure(num=ifreq, figsize=(14,8))

    if tau3d is not None:
        # get the tau3d x,y,z coordinates. 
        imshape = tau3d['x'].shape
        imnx = imshape[0]
        imny = imshape[1]

        # consider only along minor axis
        mintau3d = tau3d['x'].min()/natconst.au

        tau_y = tau3d['y'][imnx/2, :, ifreq]/natconst.au
#        tau_y = np.ma.masked_where(tau_y==mintau3d, tau_y)
        tau_y = np.ma.masked_values(tau_y, mintau3d, rtol=1e-5, shrink=False)

        tau_z = tau3d['z'][imnx/2, :, ifreq]/natconst.au
#        tau_z = np.ma.masked_where(tau_z==mintau3d, tau_z)
        tau_z = np.ma.masked_values(tau_z, mintau3d, rtol=1e-5, shrink=False)

        tau_r = np.sqrt(tau_y**2 + tau_z**2)
        tau_tta = np.arctan2(tau_y, tau_z)

        # x,y,limits
        if False in tau_y.mask:
            xmax = tau_y.max() #in natconst.au
            if abs(tau_y.min()) > xmax:
                xmax = abs(tau_y.min())
            xmin = -xmax
            if xmax < 0.1:
                xmax = dat.grid.x.max()
                xmin = -xmax
        else:
            xmax = dat.grid.x.max()
            xmin = -xmax

        if False in tau_z.mask:
            ymin = -3. * abs(tau_z).max()
            if abs(ymin) < 0.1:
                ymin = None
            ymax = 3. * abs(tau_z).max()
            if ymax < 0.1:
                ymax = None
        else:
            ymax = xmax
            ymin = -xmax
    else:
        tau_y = None
        tau_z = None
        tau_r = None
        tau_tta = None
        xmin, xmax = -dat.grid.x.max()/natconst.au, dat.grid.x.max()/natconst.au
        ymin, ymax = xmin, xmax	# cartesian coordinates

    npltrow = 2
    npltcol = 3

    # density structure in x,z
    plt.subplot(npltrow, npltcol,1)
    sliceplt = analyze.plotSlice2D(data=dat,
        var='ddens', ispec=-1,
        plane='xy', crd3=0.0,
        log=True, vmin=None, vmax=None,
        linunit='au', angunit='deg',
        gridcolor='r', gridalpha=1, showgrid=False,
        contours=False, coverplot=True,
        clmin=1e-20, clmax=dat.rhodust.max(), cllog=True, ncl=11, clcol='w',
        cllabel_fontsize=10, cllabel=True, cllabel_fmt='%.1e',
        lattitude=False, Sph2Cart=True, mirror=True)
    if (xmax is not None) and (ymax is not None):
        plt.text(xmax*0.9, ymin*0.9,
             'Total Dust=%.1e Msun'%(totdmass/natconst.ms),
             va='bottom', ha='right', color='w')
    if (tau_y is not None) and (tau_z is not None):
        plt.plot(tau_y, tau_z, 'w')
    plt.legend()
#    plt.ylim(-dat.grid.x.max()/natconst.au, dat.grid.x.max()/natconst.au)
    if xmax is not None:
        plt.xlim(xmin , xmax)
    if (ymin is not None) and (ymax is not None):
        plt.ylim(ymin, ymax)

    # density structure in radius, theta
    plt.subplot(npltrow, npltcol,4)
    analyze.plotSlice2D(data=dat, 
        var='ddens', ispec=-1,
        plane='xy', crd3=0.0,
        log=True, vmin=None, vmax=None,
        linunit='au', angunit='deg',
        gridcolor='r', gridalpha=1, showgrid=False,
        contours=False, coverplot=True,
        clmin=1e-20, clmax=1e-10, cllog=True, ncl=11, clcol='w',
        cllabel_fontsize=10, cllabel=True, cllabel_fmt='%.1e',
        lattitude=False, Sph2Cart=False, mirror=True)
    if (tau_r is not None) and (tau_tta is not None):
        plt.plot(tau_r, tau_tta*180./np.pi, 'w')
    plt.legend()
#    plt.ylim(-dat.grid.x.max()/natconst.au, dat.grid.x.max()/natconst.au)
    plt.xscale('symlog')

    # temperature structure
    plt.subplot(npltrow, npltcol,2)
    temp_slice = analyze.plotSlice2D(data=dat, var='dtemp', plane='xy', crd3=0.,
        ispec=0, log=True, linunit='au', angunit='deg',
        gridcolor='r', gridalpha=1, showgrid=False,
        vmin=10.,
        contours=False, coverplot=True,
        clmin=1., clmax=5e2, cllog=True, ncl=20, clcol='k',
        cllabel_fontsize=10, cllabel=True, cllabel_fmt='%d',
        lattitude=False, Sph2Cart=True, mirror=True)
    if (tau_y is not None) and (tau_z is not None):
        plt.plot(tau_y, tau_z, 'w')
    plt.legend()
#    plt.ylim(-dat.grid.x.max()/natconst.au, dat.grid.x.max()/natconst.au)
    if xmax is not None:
        plt.xlim(xmin , xmax)
    if (ymin is not None) and (ymax is not None):
        plt.ylim(ymin, ymax)
    if (xmax is not None) and (ymin is not None) and (acclum > 0):
        plottxt = 'AccLum=%.1e Lsun'%(acclum/natconst.ls)
        plt.text(xmax*0.9,ymin*0.9, plottxt, 
            va='bottom',ha='right', color='w')

    # temperature structure in radius, theta
    plt.subplot(npltrow, npltcol,5)
    dum = analyze.plotSlice2D(data=dat, var='dtemp', plane='xy', crd3=0.,
        ispec=0, log=True, linunit='au', angunit='deg',
        gridcolor='r', gridalpha=1, showgrid=False,
        vmin=10.,
        contours=False, coverplot=True,
        clmin=1., clmax=5e2, cllog=True, ncl=20, clcol='k',
        cllabel_fontsize=10, cllabel=True, cllabel_fmt='%d',
        lattitude=False, Sph2Cart=False, mirror=True)
    if (tau_r is not None) and (tau_tta is not None):
        plt.plot(tau_r, tau_tta*180./np.pi, 'w')
    plt.legend()
    plt.xscale('symlog')

    # temperature profile
    plt.subplot(npltrow, npltcol, 3)
    # take index nearest theta=np.pi/2
    inx = np.argmin(abs(dat.grid.y - np.pi/2.))
    Tmidprof = dat.dusttemp[:,inx,0,0]
    # take index near theta=30 degrees
    inx = np.argmin(abs(dat.grid.y - 30. * natconst.rad))
    Tatmprof = dat.dusttemp[:,inx,0,0]

    plt.plot(dat.grid.x/natconst.au, Tmidprof, label='Midplane')
    plt.plot(dat.grid.x/natconst.au, Tatmprof, label='atm')
    Tirr = np.sqrt(parobj.ppar['rstar'][0] / 2. / dat.grid.x) * parobj.ppar['tstar'][0]
    plt.plot(dat.grid.x/natconst.au, Tirr, label='Tirr')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Temperature')

    # surface density
    plt.subplot(npltrow, npltcol, 6)
    ax_sig = plt.gca()
    dat.getSigmaDust(idust=-1)
    phi_inx = 0
    totsigma = dat.sigmadust[:,phi_inx]
    ax_sig.plot(dat.grid.x/natconst.au, totsigma, 'k-', label=r'Total $\Sigma_{d}$')
    ax_sig.set_ylim(totsigma.min()/2., totsigma.max()*2.)
    ngs = dat.rhodust.shape[-1]
    if ngs != 1:
        for ig in range(ngs):
            dat.getSigmaDust(idust=ig)
            ax_sig.plot(dat.grid.x/natconst.au, dat.sigmadust[:,phi_inx], linestyle='--', 
                label='idust=%d'%ig)
        # leave total sigma to be the final 
        dat.getSigmaDust(idust=-1)
    ax_sig.legend(loc='center left')
    ax_sig.set_xscale('log')
    ax_sig.set_yscale('log')
    ax_sig.set_title('Dust Surface Density')
    ax_sig.text(0.01,0.01, 'Total Dust=%.1e Msun'%(totdmass/natconst.ms),
        va='bottom', ha='left', color='k', transform=ax_sig.transAxes)

#    plt.tight_layout()
    plt.savefig(pngname)
    plt.close()

def getOutput_minor(ifreq, im, optim, kpara90, korth90, kext, ylostemp, ylosdens, pngname):
    # polarization, optical depth, brightness temperature, temperature 
    p0 = (korth90 - kpara90) / (korth90 + kpara90)

    if abs(p0-1e-5) <= 1e-5:
        noalign = True
    else:
        noalign = False

    ydTdTau = np.zeros(im.ny, dtype=np.float64)
    ypol = np.zeros(im.ny, dtype=np.float64)
    for iy in range(im.ny):
        lostemp = ylostemp[iy]
        losdens = ylosdens[iy]
        tau, dtaustg, taustg = los.getTauLos(losdens, kext)
        dTdTauii, T0ii = los.getdTdTau(tau, dtaustg, lostemp, tauval=1., tauvalthres=3.)
        ydTdTau[iy] = dTdTauii
        ypol[iy] = dTdTauii / T0ii * p0
    ydTdTau = - ydTdTau # change sign

    plt.figure(num=ifreq, figsize=(13,8))

    nsubx = 2
    nsuby = 3
    isub = 0

    # image
    isub = isub + 1
    plt.subplot(nsubx,nsuby,isub)
    tb = getIm2Tb(im.image[im.nx/2,:,0,ifreq], im.wav[ifreq])
    plt.plot(im.y/natconst.au, tb)
    plt.title('Image [K]')
    plt.xlabel('Y [AU]')

    if optim is not None:
        opdepth = optim.image[optim.nx/2,:,0,ifreq]
        optpolem = getOptPolEm(opdepth, kpara90, korth90)
    else:
        opdepth = None
        optpolem = None

    # polarization fraction
    isub = isub + 1
    plt.subplot(nsubx, nsuby, isub)
    pol = getImPol(im.image[im.nx/2,:,0,ifreq], #I
                   im.image[im.nx/2,:,1,ifreq], #Q
                   im.image[im.nx/2,:,2,ifreq]) #U
#                   alignang=np.pi/2.)
    plt.plot(im.y/natconst.au, pol*100., label='Image')
    plt.plot(im.y/natconst.au, ypol * 100., label=r'$p0/T*dT/d\tau$')

    if optpolem is not None:
        plt.plot(optim.y/natconst.au, optpolem * 100., label='IsoT Pol')

    plt.legend()
    plt.title('Polarization Fraction [percent]')
    plt.xlabel('Y [AU]')
    if noalign is False:
        plt.ylim(-p0*100., p0*100.)

    # linear polarization intensity 
    isub = isub + 1
    plt.subplot(nsubx, nsuby, isub)
    polI = np.sqrt(im.image[im.nx/2,:,1,ifreq]**2 
                 + im.image[im.nx/2,:,2,ifreq]**2)
    polIunit = '[Snu]'
    # convolved data: convert to Jy/beam
    if len(im.fwhm) != 0:
        beam = im.fwhm[ifreq]
        beam_area = (beam[0] /3600. * natconst.rad) * (beam[1]/3600.*natconst.rad) * np.pi / 4. / np.log(2.0)
        polI = polI * beam_area * 1e23
        polIunit = '[Jy/beam]'

    plt.plot(im.y/natconst.au, polI)
    plt.title('Linear Polarized Intensity '+polIunit)

    # optical depth
    isub = isub + 1
    plt.subplot(nsubx, nsuby, isub)
    if optim is not None:
        plt.plot(optim.y/natconst.au, opdepth)
        plt.title('Optical Depth')
        plt.xlabel('Y [AU]')

    # dT/dtau
    isub = isub + 1
    plt.subplot(nsubx, nsuby, isub)
    if noalign is False:
        plt.plot(im.y/natconst.au, pol/p0 * tb, label='p/p0*Tb')
    plt.plot(im.y/natconst.au, ydTdTau, label='dT/dTau')
    plt.legend()
    plt.title('dT/dtau')
    plt.xlabel('Y [AU]')
    if p0 != 0:
        plt.ylim(-max(abs(pol/p0*tb)), max(abs(pol/p0*tb)))

    plt.tight_layout()
    plt.savefig(pngname)
    plt.close()

def getOutput_stokes(ifreq, dis, im, pngname, polmax=None):
    # plot stokes values: I, Q, U, V, Polarized Linear Intensity, fraction
    plt.figure(num=ifreq, figsize=(14,8))
    # image
    plt.subplot(2,3,1)
    image.plotImage(image=im, cmap=plt.cm.jet, interpolation='bilinear',
        arcsec=False, au=True, dpc=dis, oplotbeam='w',
        stokes='I', bunit='inu',ifreq=ifreq,saturate='90percent')
    plt.subplot(2,3,2)
    image.plotImage(image=im, cmap=plt.cm.bwr, interpolation='bilinear',
        arcsec=False, au=True, dpc=dis, textcolor='k', 
        stokes='Q', bunit='norm',ifreq=ifreq, vmin=-1, vmax=1)
    plt.subplot(2,3,3)
    image.plotImage(image=im, cmap=plt.cm.bwr, interpolation='bilinear',
        arcsec=False, au=True, dpc=dis, textcolor='k',
        stokes='U', bunit='norm',ifreq=ifreq, vmin=-1,vmax=1)
    plt.subplot(2,3,4)
    image.plotImage(image=im, cmap=plt.cm.bwr, interpolation='bilinear',
        arcsec=False, au=True, dpc=dis, textcolor='k',
        stokes='V', bunit='norm',ifreq=ifreq, vmin=-1, vmax=1)
    plt.subplot(2,3,5)
    image.plotImage(image=im, cmap=plt.cm.jet, interpolation='bilinear',
        arcsec=False, au=True, dpc=dis,
        stokes='PI', bunit='inu',ifreq=ifreq,saturate='90percent')
    image.plotPolDir(image=im, arcsec=False, au=True, dpc=dis, color='w',
        nx=16, ny=22, polunitlen=-2, ifreq=ifreq)
    plt.subplot(2,3,6)
    image.plotImage(image=im, cmap=plt.cm.jet, interpolation='bilinear',
        arcsec=False, au=True, dpc=dis, ifreq=ifreq,
        saturate='100percent', stokes='P',bunit='percent' , vmin=0,vmax=polmax,
        clevs=[5.0, 10., 20.], clcol='k')
    image.plotPolDir(image=im, arcsec=False, au=True, dpc=dis, color='w',
        nx=16, ny=22, polunitlen=-2, ifreq=ifreq)
    plt.tight_layout()
    plt.savefig(pngname)
    plt.close()

def getOutput_los(wav, fkabs, fksca, losdens, lostemp, pngname):
    kap = fkabs(wav) + fksca(wav)
    tau, dtaustg, taustg = los.getTauLos(losdens, kap)
    dTdTau, T0 = los.getdTdTau(tau, dtaustg, lostemp)
    plt.plot(taustg, dTdTau)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'dT / $\tau$')
    plt.tight_layout()
    plt.savefig(pngname)
    plt.close()

def getOutput_op(op, mopac, dinfo, pngname, pltopwav=None):
    """
    opacity outputs
    
    """
    ngs = len(dinfo['gsize'])
    nxplots = np.ceil(np.sqrt(ngs))
    nyplots = np.floor(ngs / nxplots)
    fig, ax = plt.subplots(int(nxplots), int(nyplots), sharex='col', sharey='row')
    for ig in range(ngs):
        if isinstance(ax, np.ndarray):
            axii = ax[ig]
        else:
            axii = ax
        axii.plot(op.wav[ig], op.kabs[ig], 'r-', label='kabs')
        axii.plot(op.wav[ig], op.ksca[ig], 'b-', label='ksca')
        axii.set_xscale('log')
        axii.set_yscale('log')
        if ig == ngs-1:
            axii.set_xlabel(r'wavelength [$\mu$m]')
            plt.legend()
        ylim = axii.get_ylim()
#        axii.text(op.wav[ig].min(),ylim[0]*1.1, 'a=%.2e'%dinfo['gsize'][ig])
        if dinfo['gsize'][ig] < 1.:
            gsizetxt = 'a=%.1e'%dinfo['gsize'][ig]
        else:
            gsizetxt = 'a=%.1f'%dinfo['gsize'][ig]
        axii.set_title(gsizetxt + r'$\mu$m')
        axii.set_ylabel('Opacity')

        # overplot the beckwith opacity
        axii.plot(op.wav[ig], 10.*(op.freq[ig] / 1e12)**1., linestyle='--', label=r'$\beta$=1')

        # over plot certain wavlengths
        if pltopwav is not None:
            for wav in pltopwav:
                axii.axvline(x=wav, color='k', linestyle='--')

    fig.tight_layout()
    fig.savefig(pngname)
    plt.close()

def getOutput_beta(ifreq, im,op, pngname):
    """
    calculate beta index
    ifreq = index for wavelength/frequency. should always be greater than 0. 
        the frequency decrease in increasing index
    """
    if im.nfreq == 1:
        raise ValueError('image should be multiwavelength for beta index calculations')
    dum_image = copy.deepcopy(im)
    dalognu = np.log(dum_image.freq[ifreq-1]) - np.log(dum_image.freq[ifreq])
    if dum_image.stokes:
        dalogI = np.log(dum_image.image[:,:,0,ifreq-1]) - np.log(dum_image.image[:,:,0,ifreq])
    else:
        dalogI = np.log(dum_image.image[:,:,ifreq-1]) - np.log(dum_image.image[:,:,ifreq])

    beta = dalogI / dalognu - 2.

    # opacity index
    dinfo = op.readDustInfo()
    ngs = len(dinfo['gsize'])

    fig = plt.figure(0, figsize=(8, 5))
    ax = fig.add_subplot(111)
    pc = ax.pcolormesh(dum_image.x/natconst.au, dum_image.y/natconst.au, 
        beta.T)
    cbar = plt.colorbar(pc)
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_title('Beta Index: %d - %d GHz'%(dum_image.freq[ifreq-1]/1e9, dum_image.freq[ifreq]/1e9))
    ax.set_aspect('equal')

#    ax = fig.add_subplot(122)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for ig in range(ngs):
        fext = interpolate.interp1d(op.wav[ig], op.kabs[ig]+op.ksca[ig])
        dlogkap = np.log(fext(dum_image.wav[ifreq-1])) - np.log(fext(dum_image.wav[ifreq]))
        opbeta = dlogkap / dalognu
        ax.text(xlim[1], ylim[0]*ig/float(ngs), ('a=%.1f, beta=%.1f'%(dinfo['gsize'][ig],opbeta)), 
            va='bottom', ha='right', color='w')
    fig.savefig(pngname)
    plt.close()

def getOutput_sed(spec, pngname, star=None, pltsed=None):
    fig = plt.figure(figsize=(8,5))

    pltdict = image.plotSpectrum(image=spec, 
        pltx='wav', plty='fnu', pltyunit='jy', 
        marker='.', linestyle='-', color='k')
    ax = pltdict['specplot']
    if star is not None:
        ax.plot(star.grid.wav, star.fnustar/natconst.jy / spec.dpc**2)
    if pltsed is not None:
        ax.plot(pltsed[0,:], pltsed[1,:], 'r+')
    ylim = ax.get_ylim()
    ylimmin = min([1e-8 * ylim[1], 1e-6])
    ax.set_ylim(ylimmin, ylim[1])
    fig.savefig(pngname)
    plt.close()


def getOutput_compreal():
    """ compares the real images to calculated images
    """
# ------------------------------------------------------------
def commence(rundir, polunitlen=-2, dis=400, polmax=None, 
        dooutput_op=1, pltopwav=None, 
        dooutput_im=1, imTblim=None, imxlim=None, imylim=None, 
        dooutput_xy=0, xyinx=None, 
        dooutput_minor=0, 
        dooutput_stokes=0,
        dooutput_conv=0, fwhm=None, pa=[[0]], 
        dooutput_beta=0, 
        dooutput_fits=0, bwidthmhz=2000., coord='03h10m05s -10d05m30s',
        dooutput_los=0, dokern=False, 
        dooutput_sed=0, pltsed=None, 
        dooutput_compreal=0
        ):
    """
    dooutput_conv : bool
        to output the convolved images
    imTblim : list
           list of two elements for vmin, vmax of output_im image
    imxlim : tuple
        x coordinate limit for image in AU
    imylim : tuple
        y coordinate limit for image in AU
    xyinx : tuple
            2 element tuple to specify which image to plot xy. 
            In (a,b) where a is the index for inclination and b is for wavelength
    fwhm : list
         number of different resolutions by number of wavelengths in image.out
    pa   : list
         number of different resolutions by number of wavelengths in image.out

    pltsed : 2d array
             [0,:] for wavelength
             [1,:] for observed flux
    """

    if os.path.isdir(rundir) is False:
        raise ValueError('rundir does not exist: %s'%rundir)

    # stokes inputs for fits files
    stokespref = ['I', 'Q', 'U', 'V']

    # inclinations for image
    imageinc = fntools.zylreadvec(rundir + '/inp.imageinc')
    ninc = len(imageinc)

    # read parameter file
    parobj = params.radmc3dPar()
    parobj.readPar(fdir=rundir)

    # opacity
    op = dustopac.radmc3dDustOpac()
    res = op.readMasterOpac(fdir=rundir)
    op.readOpac(fdir=rundir, ext=res['ext'], scatmat=res['scatmat'], alignfact=res['align'][0])
    fkabs = interpolate.interp1d(op.wav[0], op.kabs[0], kind='linear')
    fksca = interpolate.interp1d(op.wav[0], op.ksca[0], kind='linear')
    dinfo = op.readDustInfo()
    if res['align'][0]:
        kpara90 = op.kpara[0][0,-1]
        korth90 = op.korth[0][0,-1]
    else:
        kpara90 = 1.
        korth90 = 1.

    # read dust density and temperature
    dat = analyze.readData(fdir=rundir, binary=False, ddens=True, dtemp=True)
    # see if heatsource.inp exists to read accretion luminosity
    if os.path.isfile(rundir + '/heatsource.inp'):
        qvisdat = analyze.readData(fdir=rundir, binary=False, qvis=True)
        acclum = qvisdat.getHeatingLum()
    else:
        acclum = 0.

    temp2d = np.squeeze(dat.dusttemp[:,:,0,0])
    if dokern:
        kern = np.array([[1,1,1,1,1], 
                         [1,2,2,2,1],
                         [1,2,3,2,1], 
                         [1,2,2,2,1],
                         [1,1,1,1,1]], dtype=np.float64)
        kern = kern / sum(kern)
        temp2d = convolve(temp2d, kern, boundary='fill', fill_value=0.)
    rho2d = np.squeeze(dat.rhodust[:,:,0,0])
    reg = rho2d < 1e-32
    rho2d[reg] = 1e-32
    rholog = np.log10(rho2d)
    raxis = dat.grid.xi
    rstg = dat.grid.x
    ttaaxis = dat.grid.yi
    ttastg = dat.grid.y

    # output opacity
    if dooutput_op:
        pngname = rundir+'/out_opacity.png'
        getOutput_op(op, res, dinfo, pngname, pltopwav=pltopwav)

    # spectrum
    if dooutput_sed and os.path.isfile('inp.spectruminc'):
        star = star = radsources.radmc3dRadSources()
        star.readStarsinp()
        sedinc = fntools.zylreadvec('inp.spectruminc')
        nsedinc = len(sedinc)
        for ii in range(nsedinc):
           fname = 'myspectrum.i%d.out'%(sedinc[ii])
           spec = image.radmc3dImage()
           spec.readSpectrum(fname=fname, dpc=dis)
           pngname = rundir + '/out_sed.i%d.png'%(sedinc[ii])
           getOutput_sed(spec, pngname, star=star, pltsed=pltsed)

    # start iterating image related 
    for ii in range(ninc):
        fname = rundir + '/myimage.i%d.out'%(imageinc[ii])
        im = image.readImage(fname=fname)

        fname = rundir + '/mytausurf1.i%d.out'%(imageinc[ii])
        if os.path.isfile(fname):
            tauim = image.readImage(fname=fname)
        else:
            tauim = None

        fname = rundir + '/mytau3d.i%d.out'%(imageinc[ii])
        if os.path.isfile(fname):
            tau3d = image.readTauSurf3D(fname=fname)
        else:
            tau3d = None

        fname = rundir + '/myoptdepth.i%d.out'%(imageinc[ii])
        if os.path.isfile(fname):
            optim = image.readImage(fname=fname)
        else:
            optim = None
 
        camwav = image.readCameraWavelength(fname=rundir+'/camera_wavelength_micron.inp.image')
        ncamwav = len(camwav)

        # line of sight properties along minor axis
        if dooutput_minor or dooutput_los:
            ylostemp = range(im.ny)
            ylosdens = range(im.ny)
            for iy in range(im.ny):
                ym = im.y[iy]
                lostemp = los.extract(imageinc[ii]*natconst.rad, ym, temp2d, 
                    raxis, rstg,ttaaxis,ttastg,0.)
                losdens = los.extract(imageinc[ii]*natconst.rad, ym, rholog, 
                    raxis, rstg,ttaaxis,ttastg,-32.)
                losdens['valcell'] = 10.**(losdens['valcell'])
                losdens['valwall'] = 10.**(losdens['valwall'])
                ylostemp[iy] = lostemp
                ylosdens[iy] = losdens

        ifreq = 0
        for ifreq in range(ncamwav):
            kext = fkabs(camwav[ifreq]) + fksca(camwav[ifreq])

            if dooutput_im:
                pngname = rundir+'/out_im.i%d.f%d.png'%(imageinc[ii],camwav[ifreq])
                getOutput_im(ifreq, dis, im, optim, tauim, polunitlen, fkabs, fksca, 
                    pngname, polmax=polmax, imTblim=imTblim, xlim=imxlim, ylim=imylim)

            if dooutput_xy:
                if xyinx is not None:
                    if (xyinx[0] == ii) and (xyinx[1] == ifreq):
                        pngname = rundir+'/out_xy.i%d.f%d.png'%(imageinc[ii],camwav[ifreq])
                        getOutput_xy(ifreq, tau3d, dat, polunitlen, acclum, parobj, pngname)

                else:
                    pngname = rundir+'/out_xy.i%d.f%d.png'%(imageinc[ii],camwav[ifreq])
                    getOutput_xy(ifreq, tau3d, dat, polunitlen, acclum, parobj, pngname)

            if dooutput_minor:
                pngname=rundir+'/out_minor.i%d.f%d.png'%(imageinc[ii],camwav[ifreq])
                getOutput_minor(ifreq, im, optim, kpara90, korth90, kext, 
                            ylostemp, ylosdens, pngname)
            if dooutput_stokes:
                pngname=rundir+'/out_stokes.i%d.f%d.png'%(imageinc[ii],camwav[ifreq])
                getOutput_stokes(ifreq, dis, im, pngname, polmax=polmax)

            if dooutput_los:
                pngname = rundir + '/out_los.i%d.f%d.png'%(imageinc[ii],camwav[ifreq])
                getOutput_los(camwav[ifreq], fkabs, fksca, losdens, lostemp, pngname)

            if (dooutput_beta) and (ifreq > 0):
                pngname = rundir + '/out_beta.i%d.f%dx%d.png'%(imageinc[ii], camwav[ifreq], camwav[ifreq-1])
                getOutput_beta(ifreq, im, op, pngname)

            if dooutput_fits:
            # output to fits file
                if im.stokes: #if stokes image
                    for isk in range(4):
                        fitsname = rundir + '/myimage.i%d.f%d.%s.fits'%(imageinc[ii], camwav[ifreq],stokespref[isk])
                        im.writeFits(fname=fitsname, dpc=dis, casa=True, 
                            bandwidthmhz=bwidthmhz, coord=coord, 
                            stokes=stokespref[isk], ifreq=ifreq)
                else:
                    fitsname = rundir + '/myimage.i%d.f%d.fits'%(imageinc[ii], camwav[ifreq])
                    im.writeFits(fname=fname, dpc=dis, casa=True,
                        bandwidthmhz=bwidthmhz, coord=coord, 
                        stokes='I', ifreq=ifreq)


        if (dooutput_conv) & (fwhm is not None):
            npa = len(pa)
            for ipa in range(npa):
                conv = im.imConv(dpc=dis, psfType='gauss', fwhm=fwhm[ipa], pa=pa[ipa])
                for ifreq in range(ncamwav):
                    if dooutput_im:
                        pngname = rundir+'/out_im.i%d.f%d.b%d.png'%(imageinc[ii],camwav[ifreq], ipa)
                        getOutput_im(ifreq, dis, conv, optim, tauim, polunitlen, 
                            fkabs, fksca, pngname, polmax=polmax, imTblim=imTblim, 
                            xlim=imxlim, ylim=imylim)

#                    if dooutput_xy:
#                        pngname = rundir+'/out_xy.i%d.f%d.b%d.png'%(imageinc[ii],camwav[ifreq], ipa)
#                        getOutput_xy(ifreq, tauim, tau3d, dat, polunitlen, pngname)

                    if dooutput_minor:
                        pngname=rundir+'/out_minor.i%d.f%d.b%d.png'%(imageinc[ii],camwav[ifreq], ipa)
                        getOutput_minor(ifreq, conv, optim, kpara90, korth90, kext, 
                            ylostemp, ylosdens, pngname)
                    if dooutput_stokes:
                        pngname=rundir+'/out_stokes.i%d.f%d.b%d.png'%(imageinc[ii],camwav[ifreq], ipa)
                        getOutput_stokes(ifreq, dis, conv, pngname, polmax=polmax)

                    if dooutput_fits:
                        if conv.stokes:
                            for isk in range(4):
                                fitsname = 'myimage.i%d.f%d.%s.b%d.fits'%(imageinc[ii], camwav[ifreq],stokespref[isk], ipa)
                                conv.writeFits(fname=fitsname, dpc=dis, casa=True, 
                                    bandwidthmhz=bwidthmhz, coord=coord, 
                                    stokes=stokespref[isk], ifreq=ifreq)
                        else:
                            fitsname = 'myimage.i%d.f%d.b%d.fits'%(imageinc[ii], camwav[ifreq], ipa)
                            conv.writeFits(fname=fname, dpc=dis, casa=True, 
                                bandwidthmhz=bwidthmhz, coord=coord, 
                                stokes='I', ifreq=ifreq)
                # free up this memory
                del conv

        # free up the memory before going to next iteration
        del im
        del tauim
        del tau3d
        del optim

