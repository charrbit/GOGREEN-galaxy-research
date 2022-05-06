from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rng
import os

class GOGREEN:
    def __init__(self, dataPath:str):
        """
        __init__ Constructor to define and initialize class members

        :param dataPath: absolute path to the directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/
                         subdirectories
        """ 
        self.catalog = pd.DataFrame()
        self.standardCriteria = []
        # Private Members
        self._path = dataPath
        self._structClusterNames = ['SpARCS0219', 'SpARCS0035','SpARCS1634', 'SpARCS1616', 'SPT0546', 'SpARCS1638',
                                    'SPT0205', 'SPT2106', 'SpARCS1051', 'SpARCS0335', 'SpARCS1034']
        self._clustersCatalog = pd.DataFrame()
        self._photoCatalog = pd.DataFrame()
        self._redshiftCatalog = pd.DataFrame()
        self._galfitCatalog = pd.DataFrame()
        self._matchedCatalog = pd.DataFrame()

        self.init()
    # END __INIT__

    def init(self):
        """
        init Helper method for initializing catalogs
        """ 
        # Build path string to the desired catalog
        clusterCatPath = self._path + 'DR1/CATS/Clusters.fits'
        # Generate a DataFrame of the catalog data
        self._clustersCatalog = self.generateDF(clusterCatPath)
        # Remove whitespaces included with some cluster names
        self._clustersCatalog['cluster'] = self._clustersCatalog['cluster'].str.strip()

        photoCatPath = self._path + 'DR1/CATS/Photo.fits'
        self._photoCatalog = self.generateDF(photoCatPath)

        redshiftCatPath = self._path + 'DR1/CATS/Redshift_catalogue.fits'
        self._redshiftCatalog = self.generateDF(redshiftCatPath)

        # Build a DataFrame for each galfit and matched structural parameter cluster (11 total)
        # Then combine them into a single galfit catalog and a single matched catalog
        galfitCatPath = self._path + 'STRUCTURAL_PARA_v1.1_CATONLY/GALFIT_ORG_CATS/'
        matchedCatPath = self._path + 'STRUCTURAL_PARA_v1.1_CATONLY/STRUCTCAT_MATCHED/'
        for clusterName in self._structClusterNames:
            # Build filename strings
            galfitClusterFilename = 'gal_' + clusterName.lower() + '_orgcat.fits'
            matchedClusterFilename = 'structcat_photmatch_' + clusterName.lower() + '.dat'
            # Filename for SpARCSXXXX clusters in photmatched catalogs are spjXXXX
            if (clusterName[:6] == 'SpARCS'):
                galfitClusterFilename = 'gal_spj' + clusterName[-4:] + '_orgcat.fits'
                matchedClusterFilename = 'structcat_photmatch_spj' + clusterName[-4:] + '.dat'

            galfitClusterDF = self.generateDF(galfitCatPath + galfitClusterFilename)
            # Combine it with the main galfit DataFrame
            self._galfitCatalog = self._galfitCatalog.append(galfitClusterDF)

            matchedClusterDF = self.generateDF(matchedCatPath + matchedClusterFilename)
            # Convert PHOTCATID to cPHOTID
            # Find a cPHOTID of the cluster in the photometric catalog 
            tempCPHOTID = self._photoCatalog[self._photoCatalog['Cluster'] == clusterName].iloc[0]['cPHOTID']
            # Extract the source ID and cluster ID from the temporary cPHOTID
            idPrefix = int(str(tempCPHOTID)[:3])*int(1e6)
            # Convert the structural catalog PHOTCATID into the photometric catalog cPHOTID
            matchedClusterDF.rename(columns = {'PHOTCATID':'cPHOTID'}, inplace = True)
            matchedClusterDF.loc[:,'cPHOTID'] += idPrefix
            self._matchedCatalog = self._matchedCatalog.append(matchedClusterDF)

        # Merge photomatched structural catalog with photometric catalog
        self.catalog = self.merge(self._photoCatalog, self._matchedCatalog, 'cPHOTID')
    # END INIT

    def generateDF(self, filePath:str) -> pd.DataFrame:
        """
        generateDF Generates a Pandas DataFrame from a .fits or .dat file

        :param filePath: Relative path from the directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/
                         subdirectories to the desired data file to load into a DataFrame
        :return:         Pandas DataFrame containing the data stored in param:filePath
        """
        # Different data formats require different DataFrame initializations
        # Extract the data file format from the file path
        fileType = os.path.splitext(filePath)[1]
        if fileType == '.fits':
            return Table( fits.getdata( filePath ) ).to_pandas()
        elif fileType == '.dat':
            return pd.read_csv(filePath, sep='\s+', engine='python', header=1)
        else:
            print("The ", fileType, " data format is not current implemented!")
            return pd.DataFrame()
    # END GENERATEDF

    def merge(self, frame1:pd.DataFrame, frame2:pd.DataFrame, columnName:str=None) -> pd.DataFrame:
        """
        merge Combines two Pandas DataFrames along the axis specified by param:columnName
              Only values of param:columnName contained in both frames will be kept

        :param frame1:     Pandas DataFrame to combine with param:frame2
        :param frame2:     Pandas DataFrame to combine with param:frame1
        :param columnName: Name of column shared between param:frame1 and param:frame2 to join on
                            Default: None
        :return:           Pandas DataFrame containing param:frame1 and param:frame2 merged on the param:columnName axis
        """
        return pd.merge(frame1, frame2, how='left', on=columnName)
    # END MERGE

    def getClusterZ(self, clusterName:str) -> float:
        """
        getClusterZ Gets the best estimate of the cluster redshift for the cluster specified by param:clusterName

        :param clusterName: Name of the cluster whose redshift should be returned
        :return:            Cluster redshift estimate as a float
        """
        targetCluster = self._clustersCatalog[self._clustersCatalog['cluster'] == clusterName]
        return targetCluster['Redshift'].values[0]
    # END GETCLUSTERZ

    def getMembers(self, clusterName:str) -> pd.DataFrame:
        """
        getMembers Gets the member galaxies of a cluster based on the galaxys redshift with respect to the
                   best estimate of the cluster redshift

        :param clusterName: Name of the cluster whose members should be returned
        :return:            Pandas DataFrame containing the galaxies whose redshift match the membership requirements
        """
        clusterZ = self.getClusterZ(clusterName)
        allClusterGalaxies = self.getClusterGalaxies(clusterName)
        # Find spectroscopic and photometric members seperately
        specZthreshold = np.abs(allClusterGalaxies['zspec'].values-clusterZ) < 0.02*(1+allClusterGalaxies['zspec'].values)
        specZgalaxies = allClusterGalaxies[specZthreshold]
        photZthreshold = np.abs(allClusterGalaxies['zphot'].values-clusterZ) < 0.08*(1+allClusterGalaxies['zphot'].values)
        photZgalaxies = allClusterGalaxies[photZthreshold]
        # Remove photZgalaxies with a specZ
        photZgalaxies = photZgalaxies[~photZgalaxies['cPHOTID'].isin(specZgalaxies['cPHOTID'])]
        # Combine into a single DataFrame
        memberGalaxies = specZgalaxies.append(photZgalaxies)
        return memberGalaxies
    # END GETMEMBERS

    def reduceDF(self, frame:pd.DataFrame, additionalCriteria:list, useStandards:bool) -> pd.DataFrame:
        """
        reduceDF Reduces the DataFrame param:frame to contain only galaxies that meet the criteria provided in
                 param:additionalCriteria and the standard criteria (if param:useStandards is True)

        :param additionalCriteria: List of criteria to apply to param:frame
        :param useStandards:       Flag to specify whether the standard criteria should be applied to param:frame
        :return:                   Pandas DataFrame containing the galaxies whose values meet the criteria within param:additionalCriteria
                                   and the standard criteria (if param:useStandards is True)
        """
        if (additionalCriteria != None):
            for criteria in additionalCriteria:
                frame = frame.query(criteria)
        if useStandards:
            for criteria in self.standardCriteria:
                frame = frame.query(criteria)
        return frame
    # END REDUCEDF
        
    def getClusterGalaxies(self, clusterName:str) -> pd.DataFrame:
        """
        getClusterGalaxies Get all galaxies associated with the cluster provided by param:clusterName

        :param clusterName: Name of the cluster whose galaxies should be returned
        :return:            Pandas DataFrame containing only galaxies associated with cluster param:clusterName 
        """
        return self.catalog[self.catalog['Cluster'] == clusterName]
    # END GETCLUSTERGALAXIES

    def plot(self, xQuantityName:str, yQuantityName:str, plotType:int, clusterName:str=None, additionalCriteria:list=None, onlyMembers:bool=True, colorType:str=None,
             colors:list=None, useStandards:bool=True, xRange:list=None, yRange:list=None, xLabel:str='', yLabel:str='', useLog:list=[False,False]):
        """
        plot Generates a plot(s) of param:xQuantityName vs param:yQuantityName according to param:plotType
             
        :param xQuantityName:      Name of the column whose values are to be used as the x
        :param yQuantityName:      Name of the column whose values are to be used as the y
        :param plotType:           How to plots should be generated
                                    Value: 1 - plot only the cluster provided in param:clusterName
                                    Value: 2 - plot all the clusters on seperate plots (subplot)
                                    Value: 3 - plot all the clusters on a single plot
        :param clusterName:        Name of the cluster to plot (if param:plotType is 1)
                                    Default: None
        :param additionalCriteria: List of desired criteria the plotted galaxies should meet
                                    Default: None
        :param onlyMembers:        Flag to indicate whether only cluster members should be plotted
                                    Default: True
        :param colorType:          Specifies how to color code the plotted galaxies
                                    Default: None
                                    Value:   'membership' - member vs non-member
                                    Value:   'passive' - passive vs star forming
        :param colors:             Specifies what colors should be used when plotting
                                    Default: None - random colors are generated
                                    Value:   [(r,g,b), (r,g,b)]
        :param useStandards:       Flag to indicate whether the standard search criteria should be applied
                                    Default: True
        :param xRange:             List containing the desired lower and upper bounds for the x-axis
                                    Default: None
        :param yRange:             List containing the desired lower and upper bounds for the y-axis
                                    Default: None
        :param xLabel:             Label to put on the x-axis
                                    Default: Empty string
        :param yLabel:             Label to put on the y-axis
                                    Default: Empty string
        :param useLog:             Flag to indicate whether the x- or y-axis should be in log scale
                                    Default: [False,False] - neither axis in log scale
                                    Value:   [False,True] - y axis in log scale
                                    Value:   [True,False] - x axis in log scale
                                    Value:   [True,True] - both axis in log scale
        :return:                  The generated plot(s) will be displayed
        """
        # Generate random colors
        color1 = [1, rng.random(), rng.random()]
        color2 = [0, rng.random(), rng.random()]
        if (colors != None):
            color1 = colors[0]
            color2 = colors[1]
        # Plot only the cluster specified
        if plotType == 1:
            if clusterName == None:
                print("No cluster name provided!")
                return
            # Get all galaxies associated with this cluster
            data = self.getClusterGalaxies(clusterName)
            if onlyMembers:
                # Reduce data to only contain galaxies with
                # (zphot-zclust) < 0.08(1+zphot) and (zspec-zclust) < 0.02(1+zspec)
                data = self.getMembers(clusterName)
            # Apply other specified reducing constraints
            data = self.reduceDF(data, additionalCriteria, useStandards)
            # Plot depending on how the values should be colored
            if colorType == None:
                xData = data[xQuantityName].values
                yData = data[yQuantityName].values
                # Check if either axis needs to be put in log scale
                if useLog[0] == True:
                    xData = np.log10(xData)
                if useLog[1] == True:
                    yData = np.log10(yData)
                plt.scatter(xData, yData, color=color1)
            elif colorType == 'membership':
                # .copy() is needed here to prevent a warning about making modifications if log scale is to be used
                specZ = data[~data['zspec'].isna()].copy()
                photZ = data[~data['cPHOTID'].isin(specZ['cPHOTID'])].copy()
                if useLog[0] == True:
                    specZ.loc[:, xQuantityName] = np.log10(specZ.loc[:, xQuantityName])
                    photZ.loc[:, xQuantityName] = np.log10(photZ.loc[:, xQuantityName])
                if useLog[1] == True:
                    specZ.loc[:, yQuantityName] = np.log10(specZ.loc[:, yQuantityName])
                    photZ.loc[:, yQuantityName] = np.log10(photZ.loc[:, yQuantityName])
                plt.scatter(specZ[xQuantityName].values, specZ[yQuantityName].values, color=color1, label='Spectroscopic z')
                plt.scatter(photZ[xQuantityName].values, photZ[yQuantityName].values, color=color2, label='Photometric z')
            elif colorType == 'passive':
                passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)'
                passive = data.query(passiveQuery).copy()
                starForming = data[~data['cPHOTID'].isin(passive['cPHOTID'])].copy()
                if useLog[0] == True:
                    passive.loc[:, xQuantityName] = np.log10(passive.loc[:, xQuantityName])
                    starForming.loc[:, xQuantityName] = np.log10(starForming.loc[:, xQuantityName])
                if useLog[1] == True:
                    passive.loc[:, yQuantityName] = np.log10(passive.loc[:, yQuantityName])
                    starForming.loc[:, yQuantityName] = np.log10(starForming.loc[:, yQuantityName])
                plt.scatter(passive[xQuantityName].values, passive[yQuantityName].values, color=color1, label='Quiescent')
                plt.scatter(starForming[xQuantityName].values, starForming[yQuantityName].values, color=color2, label='Star Forming')
            else:
                print(colorType, ' is not a valid coloring scheme!')

        # Plot all clusters individually in a subplot
        elif plotType == 2:
            # Generate the subplot
            _, axes = plt.subplots(4,3,figsize=(15,12))
            currentIndex = 0
            # Loop over each subplot
            for i in range(4):
                for j in range(3):
                    # Exclude the 12th subplot (there are only 11 clusters in self.catalog)
                    if (currentIndex == len(self._structClusterNames)):
                        break
                    currentClusterName = self._structClusterNames[currentIndex]
                    data = self.getClusterGalaxies(currentClusterName)
                    if onlyMembers:
                        data = self.getMembers(currentClusterName)
                    data = self.reduceDF(data, additionalCriteria, useStandards)

                    if colorType == None:
                        xData = data[xQuantityName].values
                        yData = data[yQuantityName].values
                        # Check if either axis needs to be put in log scale
                        if useLog[0] == True:
                            xData = np.log10(xData)
                        if useLog[1] == True:
                            yData = np.log10(yData)
                        axes[i][j].scatter(xData, yData, c=color1)
                    elif colorType == 'membership':
                        specZ = data[~data['zspec'].isna()].copy()
                        photZ = data[~data['cPHOTID'].isin(specZ['cPHOTID'])].copy()
                        if useLog[0] == True:
                            specZ.loc[:, xQuantityName] = np.log10(specZ.loc[:, xQuantityName])
                            photZ.loc[:, xQuantityName] = np.log10(photZ.loc[:, xQuantityName])
                        if useLog[1] == True:
                            specZ.loc[:, yQuantityName] = np.log10(specZ.loc[:, yQuantityName])
                            photZ.loc[:, yQuantityName] = np.log10(photZ.loc[:, yQuantityName])
                        axes[i][j].scatter(specZ[xQuantityName].values, specZ[yQuantityName].values, color=color1, label='Spectroscopic z')
                        axes[i][j].scatter(photZ[xQuantityName].values, photZ[yQuantityName].values, color=color2, label='Photometric z')
                    elif colorType == 'passive':
                        passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)'
                        passive = data.query(passiveQuery).copy()
                        starForming = data[~data['cPHOTID'].isin(passive['cPHOTID'])].copy()
                        if useLog[0] == True:
                            passive.loc[:, xQuantityName] = np.log10(passive.loc[:, xQuantityName])
                            starForming.loc[:, xQuantityName] = np.log10(starForming.loc[:, xQuantityName])
                        if useLog[1] == True:
                            passive.loc[:, yQuantityName] = np.log10(passive.loc[:, yQuantityName])
                            starForming.loc[:, yQuantityName] = np.log10(starForming.loc[:, yQuantityName])
                        axes[i][j].scatter(passive[xQuantityName].values, passive[yQuantityName].values, color=color1, label='Quiescent')
                        axes[i][j].scatter(starForming[xQuantityName].values, starForming[yQuantityName].values, color=color2, label='Star Forming')
                    else:
                        print(colorType, ' is not a valid coloring scheme!')

                    # Plot configurations
                    axes[i][j].set(xlabel=xLabel, ylabel=yLabel)
                    if (xRange != None):
                        axes[i][j].set(xlim=xRange)
                    if (yRange != None):
                        axes[i][j].set(ylim=yRange)
                    axes[i][j].set(title=currentClusterName)
                    axes[i][j].legend()
                    currentIndex += 1
            # Remove the 12th unused subplot from the figure
            plt.delaxes(axes[3][2])
            # Configure the subplot spacing so axes aren't overlapping
            plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

        # Plot all clusters on the same plot            
        elif plotType == 3:
            # Loop over every cluster
            for clusterName in self._structClusterNames:
                data = self.getClusterGalaxies(clusterName)
                if onlyMembers:
                    data = self.getMembers(clusterName)
                data = self.reduceDF(data, additionalCriteria, useStandards)

                if colorType == None:
                    xData = data[xQuantityName].values
                    yData = data[yQuantityName].values
                    # Check if either axis needs to be put in log scale
                    if useLog[0] == True:
                        xData = np.log10(xData)
                    if useLog[1] == True:
                        yData = np.log10(yData)
                    plt.scatter(xData, yData, c=color1)
                elif colorType == 'membership':
                    specZ = data[~data['zspec'].isna()].copy()
                    photZ = data[data['zspec'].isna()].copy()
                    if useLog[0] == True:
                        specZ.loc[:, xQuantityName] = np.log10(specZ.loc[:, xQuantityName])
                        photZ.loc[:, xQuantityName] = np.log10(photZ.loc[:, xQuantityName])
                    if useLog[1] == True:
                        specZ.loc[:, yQuantityName] = np.log10(specZ.loc[:, yQuantityName])
                        photZ.loc[:, yQuantityName] = np.log10(photZ.loc[:, yQuantityName])
                    # Only add legend labels for the last plot
                    if (clusterName != self._structClusterNames[-1]):
                        plt.scatter(specZ[xQuantityName].values, specZ[yQuantityName].values, color=color1)
                        plt.scatter(photZ[xQuantityName].values, photZ[yQuantityName].values, color=color2)
                    else:
                        plt.scatter(specZ[xQuantityName].values, specZ[yQuantityName].values, color=color1, label='Spectroscopic z')
                        plt.scatter(photZ[xQuantityName].values, photZ[yQuantityName].values, color=color2, label='Photometric z')
                elif colorType == 'passive':
                    passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)'
                    passive = data.query(passiveQuery).copy()
                    starForming = data[~data['cPHOTID'].isin(passive['cPHOTID'])].copy()
                    if useLog[0] == True:
                        passive.loc[:, xQuantityName] = np.log10(passive.loc[:, xQuantityName])
                        starForming.loc[:, xQuantityName] = np.log10(starForming.loc[:, xQuantityName])
                    if useLog[1] == True:
                        passive.loc[:, yQuantityName] = np.log10(passive.loc[:, yQuantityName])
                        starForming.loc[:, yQuantityName] = np.log10(starForming.loc[:, yQuantityName])
                    if (clusterName != self._structClusterNames[-1]):
                        plt.scatter(passive[xQuantityName].values, passive[yQuantityName].values, color=color1)
                        plt.scatter(starForming[xQuantityName].values, starForming[yQuantityName].values, color=color2)
                    else:
                        plt.scatter(passive[xQuantityName].values, passive[yQuantityName].values, color=color1, label='Quiescent')
                        plt.scatter(starForming[xQuantityName].values, starForming[yQuantityName].values, color=color2, label='Star Forming')
        else:
            print(plotType, " is not a valid plotting scheme!")

        # plotType == 2 handles plot configurations for each individual subplot
        if plotType != 2:
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            if (xRange != None):
                plt.xlim(xRange[0], xRange[1])
            if (yRange != None):
                plt.ylim(yRange[0], yRange[1])
            plt.legend()
            plt.show()
    # END PLOT