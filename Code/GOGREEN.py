from inspect import GEN_CREATED
from astropy.table import Table
from astropy.io import fits, ascii
import pandas as pd

catalogDict = {
    'clusters': 0,
    'photo': 1,
    'redshift': 2,
    'galfit': 3,
    'matched': 4,
}

class GOGREEN:
    def __init__(self, dataPath:'Path to directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/ directories') -> 'GOGREEN DATA OBJECT':
        self.path = dataPath
        self.cats = []
        self.clusters_cat = self.path + 'DR1/CATS/Clusters.fits'
        self.photo_cat = self.path + 'DR1/CATS/Photo.fits'
        self.redshift_cat = self.path + 'DR1/CATS/Redshift_catalogue.fits'
        self.galfit_cat = None
        self.matched_cat = None

        self.init()

    # HELPER METHODS
    def init(self):
        self.clusters_cat = self.generateDF(self.clusters_cat),
        self.photo_cat = self.generateDF(self.photo_cat),
        self.redshift_cat = self.generateDF(self.redshift_cat)
        # Structural paramater catalogs to load are not known at startup
        # Need to call loadStruct() to initialize

        self.cats = [self.clusters_cat, self.photo_cat, self.redshift_cat, self.galfit_cat, self.matched_cat]

    def generateDF(self, path: 'Path to data file', ftype: 'Data file extension type' ='fits') -> 'Pandas Dataframe':
        data = 0
        if ftype == 'fits':
            data = Table( fits.getdata( path ) ).to_pandas() 
        elif ftype == 'dat':
            data = pd.read_csv(path, sep='\s+', engine='python', header=1)

        return data

    # USER METHODS
    def loadStruct(self, targetCluster:'"Shortname" of structural cluster to load', cat:'galfit/matched'='matched') -> 'Pandas Dataframe':
        galfitPath = self.path + 'STRUCTURAL_PARA_v1.1_CATONLY/GALFIT_ORG_CATS/'
        matchedPath = self.path + 'STRUCTURAL_PARA_v1.1_CATONLY/STRUCTCAT_MATCHED/'

        if cat == 'matched': 
            self.cats[catalogDict[cat]] = self.generateDF(galfitPath + 'structcat_photmatch_' + targetCluster.lower() + '.dat', ftype='dat')
        else:
            self.cats[catalogDict[cat]] = self.generateDF(matchedPath + 'gal_' + targetCluster.lower() + '_orgcat.fits')

        return self.cats[catalogDict[cat]]

    def get(self, cat: 'Catalog to get ("clusters/photo/redshift/galfit/matched/all")' = 'clusters') -> 'Pandas Dataframe':
        if (cat == 'all'):
            return self.cats
        return self.cats[catalogDict[cat]]

    def search():
        pass
    
    def match():
        pass