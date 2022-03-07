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
    # dataPath : Path to directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/ subdirectories
    def __init__(self, dataPath:str):
        self.path = dataPath
        self.cats = []
        self.clusters_cat = self.path + 'DR1/CATS/Clusters.fits'
        self.photo_cat = self.path + 'DR1/CATS/Photo.fits'
        self.redshift_cat = self.path + 'DR1/CATS/Redshift_catalogue.fits'
        self.galfit_cat = 'NOT INITIALIZED, call loadStruct()' # This error message will print if the user tries to access
        self.matched_cat = 'NOT INITIALIZED, call loadStruct()'# structural parameter cats before specifying target cluster
        self.init()

    def init(self):
        self.clusters_cat = self.generateDF(self.clusters_cat)
        self.photo_cat = self.generateDF(self.photo_cat)
        self.redshift_cat = self.generateDF(self.redshift_cat)
        # Structural paramater catalogs to load are not known at startup
        # Need to call loadStruct() to initialize
        self.cats = [self.clusters_cat, self.photo_cat, self.redshift_cat, self.galfit_cat, self.matched_cat]

    # path: Path to data file
    # ftype: Data file extension type
    def generateDF(self, path:str, ftype:str ='fits'):
        data = pd.DataFrame()
        if ftype == 'fits':
            data = Table( fits.getdata( path ) ).to_pandas()
        elif ftype == 'dat':
            data = pd.read_csv(path, sep='\s+', engine='python', header=1)
        return data

    # targetCluster: "Shortname" of structural cluster to load
    # cat: 'Structural catalog to search (galfit/matched). Default="matched"
    #
    def loadStruct(self, targetCluster:str, cat:str='matched'):
        galfitPath = self.path + 'STRUCTURAL_PARA_v1.1_CATONLY/GALFIT_ORG_CATS/'
        matchedPath = self.path + 'STRUCTURAL_PARA_v1.1_CATONLY/STRUCTCAT_MATCHED/'
        if cat == 'matched': 
            self.cats[catalogDict[cat]] = self.generateDF(matchedPath + 'structcat_photmatch_' + targetCluster.lower() + '.dat', ftype='dat')
        else:
            self.cats[catalogDict[cat]] = self.generateDF(galfitPath + 'gal_' + targetCluster.lower() + '_orgcat.fits')
        return self.cats[catalogDict[cat]]


    # cat: Catalog to get ("clusters/photo/redshift/galfit/matched/all"). Default="clusters"
    def get(self, cat:str='clusters'):
        if (cat == 'all'):
            return self.cats
        return self.cats[catalogDict[cat]]

    # cat: Catalog to search ("clusters/photo/redshift/galfit/matched/all"). Default="clusters"
    # targetClusters: List of clusters of interest
    # targetProperties: List of properties of interest
    # criteria: List of criteria the properties should have
    def search(self, targetCluster:list, targetProperties:list, criteria:list, cat:str='clusters'):
        searchResults = self.get(cat)
        queryString = ''
        if catalogDict[cat] < 3:
            queryString = 'Cluster==@targetCluster'
            if catalogDict[cat] == 0:
                queryString = queryString[0].lower() + queryString[1:]
            searchResults = searchResults.query(queryString)
        # searchResults should only contain the targetCluster data at this point
        for constraint in criteria:
            searchResults = searchResults.query(constraint)
        return searchResults[targetProperties]

    # query1: Dataframe containing photo/redshift cat data
    # query2: Dataframe containing matched cat data 
    def merge(self, query1:pd.DataFrame, query2:pd.DataFrame):
        query2copy = query2.copy(deep=True)
        query2copy.rename(columns = {'PHOTCATID':'cPHOTID'}, inplace = True)
        query2copy.loc[:,'cPHOTID'] += int(1.03e8)
        matched = query1.merge(query2copy, on='cPHOTID')
        return matched
