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
    # __init__() - Constructor to define class members
        # dataPath : Path to directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/ subdirectories
    def __init__(self, dataPath:str):
        self.path = dataPath
        self.cats = []
        self.clusters_cat = self.path + 'DR1/CATS/Clusters.fits'
        self.photo_cat = self.path + 'DR1/CATS/Photo.fits'
        self.redshift_cat = self.path + 'DR1/CATS/Redshift_catalogue.fits'
        # **Structural paramater catalog retrieval is pending redesign**
        self.galfit_cat = 'NOT INITIALIZED, call loadStruct()' # This error message will print if the user tries to access
        self.matched_cat = 'NOT INITIALIZED, call loadStruct()'# structural parameter cats before specifying target cluster
        self.init()

    # init() - Internal function for initializing the catalogs
    def init(self):
        self.clusters_cat = self.generateDF(self.clusters_cat)
        self.photo_cat = self.generateDF(self.photo_cat)
        self.redshift_cat = self.generateDF(self.redshift_cat)
        # **Structural paramater catalog retrieval is pending redesign**
        self.cats = [self.clusters_cat, self.photo_cat, self.redshift_cat, self.galfit_cat, self.matched_cat]

    # generateDF() - Generates a pandas dataframe from a .fits or .dat file
        # path: Path to data file
        # ftype: Data file extension type
    def generateDF(self, path:str, ftype:str ='fits'):
        data = pd.DataFrame()
        if ftype == 'fits':
            data = Table( fits.getdata( path ) ).to_pandas()
        elif ftype == 'dat':
            data = pd.read_csv(path, sep='\s+', engine='python', header=1)
        return data

    # **Structural paramater catalog retrieval is pending redesign**
    # loadStruct() - Loads the structural parameter catalog for a specific cluster
        # targetCluster: "Shortname" of structural cluster to load
        # cat: 'Structural catalog to search (galfit/matched). Default="matched"
    def loadStruct(self, targetCluster:str, cat:str='matched'):
        galfitPath = self.path + 'STRUCTURAL_PARA_v1.1_CATONLY/GALFIT_ORG_CATS/'
        matchedPath = self.path + 'STRUCTURAL_PARA_v1.1_CATONLY/STRUCTCAT_MATCHED/'
        if cat == 'matched': 
            # Builds a string matching the catalog naming convention within the matched structural parameter directory
            # Then generate a pandas dataframe from the cluster catalog specified by targetCluster
            self.cats[catalogDict[cat]] = self.generateDF(matchedPath + 'structcat_photmatch_' + targetCluster.lower() + '.dat', ftype='dat')
        else:
            # Same as above but for the galfit structural parameter directory
            self.cats[catalogDict[cat]] = self.generateDF(galfitPath + 'gal_' + targetCluster.lower() + '_orgcat.fits')
        return self.cats[catalogDict[cat]]

    # get() - Returns a specific catalog
        # cat: Catalog to retrieve
            # Values = "clusters/photo/redshift/galfit/matched/all"
            # Default = "clusters"
    def get(self, cat:str='clusters'):
        if (cat == 'all'):
            return self.cats
        return self.cats[catalogDict[cat]]

    # search() - Searches a specific catalog for members meeting some specified criteria
        # targetClusters: List of clusters of interest
        # targetProperties: List of properties of interest
        # criteria: List of criteria the properties should have
        # cat: Catalog to retrieve
            # Values = "clusters/photo/redshift/galfit/matched/all"
            # Default = "clusters"
    def search(self, targetCluster:list, targetProperties:list, criteria:list, cat:str='clusters'):
        # Get the specific catalog to search within
        searchResults = self.get(cat)
        # Initialize an empty query string for specifying the cluster of interest
        # This will vary depending on the catalog being searched
        queryString = ''
        # If the catalog to search is clusters, photo, or redshift
        if catalogDict[cat] < 3:
            # Using @targetCluster within a string places the value stored in targetCluster into the string 
            queryString = 'Cluster==@targetCluster'
            # If the catalog to search is clusters
            if catalogDict[cat] == 0:
                # The first column in this catalog is "cluster"
                # Modify the query string to cluster==@targetCluster (convert first letter to lowercase c)
                queryString = queryString[0].lower() + queryString[1:]
                print(queryString)
            # Reduce the search results to contain only targetCluster data    
            searchResults = searchResults.query(queryString)
        # Reduce the search results to members of targetCluster that meet the constraints specified in criteria
        for constraint in criteria:
            searchResults = searchResults.query(constraint)
        # Return the targetProperties for the constrained members of targetCluster
        return searchResults[targetProperties]
        
    # merge() - Merges a matched structural parameter catalog with either photo or redshift catalogs
    # ** Currently only matches by cPHOTID **
        # query1: Dataframe containing photo/redshift cat data
        # query2: Dataframe containing matched cat data 
    def merge(self, query1:pd.DataFrame, query2:pd.DataFrame):
        # Create a copy of the matched catalog dataframe to preserve the original
        query2copy = query2.copy(deep=True)
        # Modify the first column to match that of the photo/redshift catalogs
        query2copy.rename(columns = {'PHOTCATID':'cPHOTID'}, inplace = True)
        # Convert from PHOTCATID to cPHOTID
        query2copy.loc[:,'cPHOTID'] += int(1.03e8)
        # Take the intersection of the two pandas dataframes (only keep members contained in both)
        matched = query1.merge(query2copy, on='cPHOTID')
        return matched
