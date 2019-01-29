# set_dustspec.py

class dustspec(object):
    """
    Class to get the dust species
    """
    # file to store dust specifics

    def __init__(self):
        self.lnkfiles = []
        self.optconst = []
        self.graintype = []
        self.swgt = []
        self.mabun = []

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
            self.lnkfiles.append('/scratch/zdl3gk/data/my_opacity/Semenov/RefractiveIndex/watericek.lnk')
            self.optconst.append('watericek')
            self.graintype.append('waterice')
            self.swgt.append(0.92)
            self.mabun.append(5.55)

        # organics from Semenov
        if 'Organics' in specs:
            self.lnkfiles.append('/scratch/zdl3gk/data/my_opacity/Semenov/RefractiveIndex/organicsk.lnk')
            self.optconst.append('organicsk')
            self.graintype.append('organics')
            self.swgt.append(1.5)
            self.mabun.append(3.53)

        # carbon from Jena (Jaeger) 
        if 'Carbon' in specs:
            self.lnkfiles.append('/scratch/zdl3gk/data/my_opacity/Jena/Carbon/cel400.lnk')
            self.optconst.append('cel400.lnk')
            self.graintype.append('carbon')
            self.swgt.append(1.5)
            self.mabun.append(3.53)

        ## silicates from Semenov
        if 'Sil_Sem' in specs:
            self.lnkfiles.append('/scratch/zdl3gk/data/my_opacity/Semenov/RefractiveIndex/pyrmg70k.lnk')
            self.optconst.append('pyrmg70')
            self.graintype.append('silicate')
            self.swgt.append(3.49)
            self.mabun.append(2.64)

        # silicates from Draine (from Dullemond Radmc3d)
        if 'Sil_Draine' in specs:
            self.lnkfiles.append('/home/zdl3gk/coding/fortran/radmc3d_code/radmc3d-0.41/opac/dust_continuum/astrosil_draine03/astrosilicate_draine03.lnk')
            self.optconst.append('astrosilicate_draine03')
            self.graintype.append('silicate')
            self.swgt.append(3.3)
            self.mabun.append(2.64)
