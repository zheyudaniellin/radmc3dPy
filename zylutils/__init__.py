"""
this module contains some common tools meant for radmc3dPy
"""

try:
    from . import set_image
except:
    print('failed to import set_image.py')

try:
    from . import set_output
except:
    print('failed to import set_output.py')

try:
    from . import set_dustspec
except:
    print('failed to import set_dustspec.py')

try:
    from . import los
except:
    print('failed to import los.py')


try:
    from . import set_dtemp
except:
    print('failed to import set_dtemp.py')

print('imported zylutils')
