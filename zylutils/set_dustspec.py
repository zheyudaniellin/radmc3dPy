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
                       list containing the names of the species
        """
        

        # Waterice from Semenov
        if 'Waterice' in specs:
            self.lnkfiles.append('/home/zdl3gk/mainProjects/my_opacity/Semenov/RefractiveIndex/watericek.lnk')
            self.optconst.append('watericek')
            self.graintype.append('waterice')
            self.swgt.append(0.92)
            self.mabun.append(5.55)

        # Waterice from Warren2008
        if 'WarrenWater' in specs:
            self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Warren/RefractiveIndex/waterice.lnk', 
                'waterice',
                'waterice', 
                0.92,
                5.55)

        # organics from Semenov
        if 'Organics' in specs:
            self.lnkfiles.append('/home/zdl3gk/mainProjects/my_opacity/Semenov/RefractiveIndex/organicsk.lnk')
            self.optconst.append('organicsk')
            self.graintype.append('organics')
            self.swgt.append(1.5)
            self.mabun.append(3.53)

        # carbon from Jena (Jaeger) 
        if 'Carbon' in specs:
            self.lnkfiles.append('/home/zdl3gk/mainProjects/my_opacity/Jena/Carbon/cel400.lnk')
            self.optconst.append('cel400.lnk')
            self.graintype.append('carbon')
            self.swgt.append(1.5)
            self.mabun.append(3.53)

        # organics from Henning 1996
        if 'Henning_Organics' in specs:
            self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Henning/RefractiveIndex/organicsk.lnk',
                'organicsk',
                'organics',
                1.5,
                3.53)

        # troilite from Henning 1996
        if 'Henning_Troilite' in specs:
            self.fnSpec('/home/zdl3gk/mainProjects/my_opacity/Henning/RefractiveIndex/troilitek.lnk',
                'troilitek',
                'troilite',
                4.83,
                0.65) #details from Birnstiel 2018

        ## silicates from Semenov
        if 'Sil_Sem' in specs:
            self.lnkfiles.append('/home/zdl3gk/mainProjects/my_opacity/Semenov/RefractiveIndex/pyrmg70k.lnk')
            self.optconst.append('pyrmg70')
            self.graintype.append('silicate')
            self.swgt.append(3.49)
            self.mabun.append(2.64)

        # silicates from Draine (from Dullemond Radmc3d)
        if 'Sil_Draine' in specs:
            self.fnSpec('/home/zdl3gk/coding/fortran/radmc3d_code/radmc3d-0.41/opac/dust_continuum/astrosil_draine03/astrosilicate_draine03.lnk', 
                'astrosilicate_draine03',
                'silicate', 
                3.3, 
                2.64
                )

