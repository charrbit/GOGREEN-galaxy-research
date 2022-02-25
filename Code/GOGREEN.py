from astropy.table import Table
from astropy.io import fits, ascii
import pandas as pd 

class GOGREEN:
    def init():
        pass

    def generateDF(path:'Path to data file', isDAT:'If data is stored in .dat, set this flag') -> 'Pandas Dataframe':
        data = 0
        if (isDAT):
            data = pd.read_csv(path, sep='\s+', engine='python', header=1)
        else: # Otherwise, assumes .fits
            data = Table( fits.getdata( data ) ).to_pandas() 
            data[0] = data[0].str.rstrip().values
        return data

    def __init__(self, path:'Path to directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/ directories') -> 'GOGREEN DATA OBJECT':
        self.clusters_cat = path + 'DR1/CATS/Clusters.fits'
        self.photo_cat = path + 'DR1/CATS/Photo.fits'
        self.redshift_cat = path + 'DR1/CATS/Redshift_catalogue.fits'
        self.galfit_cat = path + 'STRUCTURAL_PARA_v1.1_CATONLY/GALFIT_ORG_CATS/'
        self.matched_cat = path + 'STRUCTURAL_PARA_v1.1_CATONLY/STRUCTCAT_MATCHED/'

        self.init