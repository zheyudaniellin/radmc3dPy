# panel.py
from radmc3dPy import zylutils
import os
import time

os.system('rm -rf RIVANNA_DONE.txt')

with open('RIVANNA_RUNNING.txt', 'w') as wfile:
    wfile.write('Currently running with rivanna')

clock_begin = time.time()

execfile('set_opac.py')

execfile('panel4radmc.py')

zylutils.set_image.makemcImage(inc=[86.],
    camwav=[433, 851, 1327., 2853., 9098.], 
    sizeau=200., npix=300,  wavfname='camera_wavelength_micron.inp.image',
    dotausurf=False, dooptdepth=True)

clock_stop = time.time()
print 'Total time elapsed for panel_setup.py'
print clock_stop-clock_begin

with open('RIVANNA_DONE.txt', 'w') as wfile:
    wfile.write('Done in %d'%(clock_stop-clock_begin))

