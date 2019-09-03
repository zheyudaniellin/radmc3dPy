# panel.py
from radmc3dPy import zylutils
import os
import time

clock_begin = time.time()

runname = 'run1/'

if os.path.isdir(runname) is False: 
    os.system('mkdir '+runname)

# files to be copied
datacp = ['dustopac.inp', 'dustinfo.zyl', 'dustkap*.inp']
# data to be moved
datamv = ['dust_temperature.dat', 'dust_density.inp',
          'myimage*.out', 'mytausurf*.out', 'myspectrum*.out', 
          'mytau3d*.out', 'myoptdepth*.out', 
          'myimag*.fits',
          'amr_grid.inp', 'camera_wavelength_micron.inp',
          'grainalign_dir.inp', 'inp.imageinc', 'inp.spectruminc', 
          'camera_wavelength_micron.inp.image', 'radmc3d.inp',
          'stars.inp', 'wavelength_micron.inp', 'problem_params.inp', 
          'heatsource.inp'
         ]
dirmv = ['temp2d']

for idata in datacp:
    os.system('cp %s %s'%(idata, runname))
for idata in datamv:
    os.system('mv %s %s'%(idata, runname))
for idata in dirmv:
    if os.path.isdir(idata):
        fname = runname+'/'+idata
        if os.path.isdir(fname):
            os.system('rm -rf '+fname)
        os.system('mv %s %s'%(idata, runname))

# start to do outputs
zylutils.set_output.commence(runname,polunitlen=-2, dis=400.,polmax=5.,
    dooutput_op=1, 
    dooutput_im=1, imTblim=[0,100],
    dooutput_xy=1, xyinx=(0,0), 
    dooutput_minor=0,
    dooutput_stokes=0,
    dooutput_conv=1, 
        fwhm=[ [[0.08,0.08], [0.02,0.02], [0.02,0.02], [0.04,0.04],[0.055,0.055]] ]
#         fwhm = [ [[0.02,0.02]] ]
         )

clock_stop = time.time()
print 'Total time elapsed for panel_out.py'
print clock_stop-clock_begin
