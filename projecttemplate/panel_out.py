# panel.py
from radmc3dPy import zylutils
import os
import time

clock_begin = time.time()

runname = 'run1/'
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


os.system('mv dust_temperature.dat dust_density.inp my*.out '+runname)
os.system('mv amr_grid.inp camera_wavelength_micron.inp '+runname)
os.system('cp dustinfo.zyl dustopac.inp '+runname)
os.system('cp dustkappa_*.inp dustkapscatmat_*.inp ' + runname)
#os.system('cp dustkapalignfact_*.inp grainalign_dir.inp '+runname)
os.system('mv problem_params.inp radmc3d.inp stars.inp wavelength_micron.inp '+runname)
os.system('mv inp.imageinc '+runname)

# extra
if os.path.isdir('temp2d'):
    os.system('mv temp2d '+runname)

clock_stop = time.time()
print 'Total time elapsed for panel_out.py'
print clock_stop-clock_begin
