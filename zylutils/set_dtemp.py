# set up the temperature
# reform the inputs
# calculate 2d, axisymmetric, mirror symmetric
# copy the 2d results to whatever the original setting was
#
#    """
#    Class to handle calculating temperature.
#    To make radmc3d temperature calculations faster, 
#    assume 2d axisymmetric, mirror symmetric. 
#    Densities are in 3d spherical coordinates,
#    Change amr_grid, change density to 2d, calculate 2d temperature,
#    then make a 3d temperature by copying. 
#    To make things clear, 
#    this will create a new directory to store all the relevant 2d calculations
#    Do not change any of the original files. Do all new calculations in new directory
#    """
# assume isotropic scattering

import os
from radmc3dPy import *
import pdb
import numpy as np

def create_params(temp2ddir, mopac0, scatmode):
    # set up the problem_params.inp
    # par = the par file that can be changed
    # par0 = the most original file. do not change
    # inputs:
    # temp2ddir = string. directory of 2d structure
    # mopac0 = dictionary. of master opac
    # scatmode = string. scattering_mode_max
    opac_align = mopac0['align'][0]

    par0 = analyze.readParams()
    par = analyze.readParams()
    if opac_align is True:
        par.setPar(parlist=['alignment_mode', '0', '', 'Code parameters'])
    # set up scattering mode max
    par.setPar(parlist=['scattering_mode_max', scatmode, '', 'Code parameters'])

    par.writeParfile(fdir=temp2ddir)
    return par, par0

def create_opac(temp2ddir, mopac0, op, scatmode):
    os.system('cp dustinfo.zyl '+temp2ddir)
    ndust = len(mopac0['ext'])
    opac_align = mopac0['align']

    for ig in range(ndust):
        if mopac0['scatmat'][ig]:
            fname = 'dustkapscatmat_%s.inp'%(mopac0['ext'][ig])
        else:
            fname = 'dustkappa_%s.inp'%(mopac0['ext'][ig])
        os.system('cp %s %s'%(fname, temp2ddir))
    if opac_align:
        op.writeMasterOpac(ext=mopac0['ext'], therm=mopac0['therm'],
            scattering_mode_max=int(scatmode), alignment_mode=0, fdir=temp2ddir)
    return 0

def create_grid(temp2ddir, par, par0):
    # create grid specifically for 2d temperature calculations
    grid2d = reggrid.radmc3dGrid()

    # radial coordinate
    xbound = par0.pvalstr['xbound']
    nxi = par0.pvalstr['nx']
    par.setPar(parlist=['xbound', xbound, '', ''])
    par.setPar(parlist=['nx', nxi, '', ''])

    # theta coordinate. 
    # use mirror symmetry. thus only take theta to pi/2
    ybound = []
    nyi = []
    iy = 0
    for yval in par.ppar['ybound']:
        if yval <= (np.pi/2.-1e-4):
            ybound.append(yval)
            nyi.append(par.ppar['ny'][iy])
            iy = iy + 1
    ybound.append(np.pi / 2.)
    if len(nyi) is not (len(ybound)-1):
        raise ValueError('Error in calculating ybound')
    par.setPar(parlist=['ybound', str(ybound), '', ''])
    par.setPar(parlist=['ny', str(nyi), '', ''])

    # phi coordinate. turn off
    zbound = [0.]
    nzi = [0]
    par.setPar(parlist=['zbound', str(zbound), '', ''])
    par.setPar(parlist=['nz', str(nzi), '', ''])

    fname = temp2ddir + '/amr_grid.inp'
    grid2d.makeSpatialGrid(ppar=par.ppar)
    grid2d.writeSpatialGrid(fname=fname)

    # directly copy the wavelength grid
    os.system('cp wavelength_micron.inp '+temp2ddir)

    return grid2d

def create_data(temp2ddir, grid2d, grid3d, ndust):
    # dat3d is assumed not to have mirror symmetry 

    dat3d = data.radmc3dData(grid=grid3d)
    dat2d = data.radmc3dData(grid=grid2d)

    # density
    dat3d.readDustDens(binary=False, fname='dust_density.inp')

    # manipulate the 2d versions
    ddens2d = np.zeros([grid2d.nx, grid2d.ny, grid2d.nz, ndust], dtype=np.float64)

    for ig in range(ndust):
        ddens2d[:,:,0,ig] = np.squeeze(dat3d.rhodust[:,:grid3d.ny/2,0,ig])

    dat2d.rhodust = ddens2d

    fname = temp2ddir + '/dust_density.inp'
    dat2d.writeDustDens(fname=fname, binary=False)

    # viscous heating
    fname = 'heatsource.inp'
    exist_qvis = os.path.isfile(fname)
    if exist_qvis:
        dat3d.readViscousHeating(binary=False, fname='heatsource.inp')
        # manipulate 2d versions
        qvis2d = np.zeros([grid2d.nx, grid2d.ny, grid2d.nz], dtype=np.float64)
        qvis2d[:,:,0] = np.squeeze(dat3d.qvis[:,:grid3d.ny/2, 0])

        # output the 2d data object
        dat2d.qvis = qvis2d
        fname = temp2ddir + '/heatsource.inp'
        dat2d.writeViscousHeating(fname=fname, binary=False)

    return dat2d, dat3d

def create_temp2d(temp2ddir, dat2d):
    os.chdir(temp2ddir)
    os.system('radmc3d mctherm')
    os.chdir('..')
    if os.path.isfile(temp2ddir + '/dust_temperature.dat') is 0:
        raise ValueError('Did not succeed in radmc3d mctherm calculations')
    fname = temp2ddir + '/dust_temperature.dat'
    dat2d.readDustTemp(fname=fname, binary=False)
    return dat2d

def reform_dat3d(dat2d, dat3d):
    # specifically, output temperature in 3d
    nx, ny, nz, ndust = dat3d.rhodust.shape
    temp2d = dat2d.dusttemp
    temp3d = np.zeros([nx, ny, nz, ndust], dtype=np.float64)
    for ig in range(ndust):
        dtempslice = np.zeros([nx, ny], dtype=np.float64)
        dtempslice[:,:ny/2] = np.squeeze(temp2d[:,:,0,ig])
        dtempslice[:,ny/2:] = np.squeeze(temp2d[:,::-1,0,ig])
        for iz in range(nz):
            temp3d[:,:,iz, ig] = dtempslice
    dat3d.dusttemp = temp3d
    fname = 'dust_temperature.dat'
    dat3d.writeDustTemp(fname=fname, binary=False)

    return dat3d

def pipeline(dohydroeq=0, itermax=0, scatmode='1'):
    # call all the procedure

    # create a directory to store data
    temp2ddir = 'temp2d'
    os.system('rm -rf '+temp2ddir)
    os.system('mkdir '+temp2ddir)

    # read some original settings
    op0 = dustopac.radmc3dDustOpac()
    mopac0 = op0.readMasterOpac() # the most original
    opac_align = mopac0['align'][0]
    opac_align = '1' if opac_align else '0' # the basis of whether alignment mode is on
    ndust = len(mopac0['ext'])
    grid3d = reggrid.radmc3dGrid()
    grid3d.readSpatialGrid()

    # set up the problem_params.inp
    par, par0 = create_params(temp2ddir, mopac0, scatmode)

    # recreate opacity
    dum = create_opac(temp2ddir, mopac0, op0, scatmode)

    # recreate radmc3d.inp
    setup.writeRadmc3dInp(modpar=par, fdir=temp2ddir)
    os.system('cp stars.inp '+temp2ddir)

    # recreate amr_grid, wavlength grid
    grid2d = create_grid(temp2ddir, par, par0)

    # recreate data: dust density, heating, velocity 
    dat2d, dat3d = create_data(temp2ddir, grid2d, grid3d, ndust)

    # calculate temperature
    dat2d = create_temp2d(temp2ddir, dat2d)

    # hydrostatic equilibrium
#    if dohydroeq:
#        for ii in range(itermax):
            # not implemented yet

    # output temperature to original settings
    dat3d = reform_dat3d(dat2d, dat3d)

    return dat3d    
