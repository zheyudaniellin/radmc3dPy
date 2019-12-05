# set_dustspec.py

class dustspec(object):
    """
    Class to get the dust species
    """
    # file to store dust specifics

    def __init__(self):
        self.lnkfiles = []	# the full name of lnkfile 
        self.optconst = []	# the filename prefix *.lnk
        self.graintype = []	# type of grain
        self.swgt = []		# specific grain density
        self.mabun = []		# mass abundance

    def fnSpec(self, lnkname, optconst, graintype, swgt, mabun):
        self.lnkfiles.append(lnkname)
        self.optconst.append(optconst)
        self.graintype.append(graintype)
        self.swgt.append(swgt)
        self.mabun.append(mabun)

    def getSpec(self, specs=None):
        """
        wrapper to get the attributes

        Parameters
        ----------
        specs        : list
                       list containing the names of the species. order matters!
        """
        
        nspecs = len(specs)
        for ispec in specs:
            # Waterice from Semenov
            if ispec == 'Waterice':
                self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Semenov/RefractiveIndex/watericek.lnk', 
                    'watericek', 
                    'waterice', 
                    0.92, 
                    5.55)


            # Waterice from Warren2008
            elif ispec == 'WarrenWater':
                self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Warren/RefractiveIndex/waterice.lnk', 
                    'waterice',
                    'waterice', 
                    0.92,
                    5.55)

            # organics from Semenov
            elif ispec == 'Organics':
                self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Semenov/RefractiveIndex/organicsk.lnk', 
                    'organicsk', 
                    'organics', 
                    1.5, 
                    3.53)

            # carbon from Jena (Jaeger) 
            elif ispec == 'Carbon':
                self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Jena/Carbon/cel400.lnk',
                    'cel400.lnk', 
                    'carbon', 
                    1.5, 
                    3.53)

            # organics from Henning 1996
            elif ispec == 'Henning_Organics':
                self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Henning/RefractiveIndex/organicsk.lnk',
                    'organicsk',
                    'organics',
                    1.5,
                    3.53)

            # troilite from Henning 1996
            elif ispec == 'Henning_Troilite':
                self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Henning/RefractiveIndex/troilitek.lnk',
                    'troilitek',
                    'troilite',
                    4.83,
                    0.65) #details from Birnstiel 2018

            ## silicates from Semenov
            elif ispec == 'Sil_Sem':
                self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Semenov/RefractiveIndex/pyrmg70k.lnk', 
                    'pyrmg70', 
                    'silicate', 
                    3.49, 
                    2.64)

            # silicates from Draine (from Dullemond Radmc3d)
            elif ispec == 'Sil_Draine':
                self.fnSpec('/home/zdl3gk/coding/fortran/radmc3d_code/radmc3d-0.41/opac/dust_continuum/astrosil_draine03/astrosilicate_draine03.lnk', 
                    'astrosilicate_draine03',
                    'silicate', 
                    3.3, 
                    2.64
                    )
            else:
                raise ValueError('species not found in available data base: %s'%ispec)

