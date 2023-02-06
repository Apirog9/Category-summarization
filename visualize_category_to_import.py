# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:30:05 2022
Visualize as heatmap with Z-scores and missing values:
    One entity in "entity name" column 
    Values in 'quantcolumns'
    order of values in will be as in 'quantcolumns'
    
@author: APirog
"""

import pandas as pd
import copy
import seaborn
import scipy.stats as sts
import numpy as np
from itertools import combinations
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform, pdist as distance
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import os


#characters to be excluded from category name to make proper filename for plot
invalid_characters = ';:,[]{}~-<>/\=+^%&*()'

def take_items(entrylist):
    ''' Takes simple text file in form type of category \t item name per line to create list of lists
    returns item list of lists used for further plotting
    entrylist - file name to get dictionary from
    Returns list of two-element lists, [item category type,item] '''
    data = open(entrylist,'r').read().splitlines()
    itemlist = []
    for line in data:
        line = line.split("\t")
        itemlist.append(line)
    return itemlist

def validate_rows(dataframe,quantcolumns,conditions,max_ANOVA,mindiff):
    '''performs simple ANOVA based validation of quatitative rows. For now simple and 
    uncorrected, for visualization purposes only
    dataframe - input dataframefor calculation
    quantcolumns - all columns used for quantitative analysis
    conditions - condition:columns dictionary
    max_ANOVA - max ANOVA p value for valid row
    mindiff minimum maximal mean difference in experiment for valid row
    returns dataframe with Valid Row column for marking reasonable quants in plot'''
    
    def validate_anova(serieslike,groups,max_ANOVA,mindiff):
        arr = []
        for group in groups.keys():
            groupdata = list(serieslike[groups[group]])
            arr.append(groupdata)
        stat = sts.f_oneway(*arr)
        stat =stat.pvalue
        if stat <= max_ANOVA:
            validstat = True
        else:
            validstat = False
            
        means = [serieslike[groups[group]].mean() for group in groups.keys()]
        comparisons = combinations(means,2)
        differences = [abs(x[0]-x[1]) for x in comparisons]
        maxdiff = max(differences)
        if maxdiff < mindiff:
            validdiff = False
        else:
            validdiff = True
        valid = validstat and validdiff
        if valid:
            returnval = [0,1,0]
        else:
            returnval =[1,0,0]
        return returnval
    
    dataframe['Valid Row'] = dataframe.apply(validate_anova,args = [conditions,max_ANOVA,mindiff],axis =1)
    return dataframe

def impute(dataframe,quantcolumns,quantified_entity,n):
    '''KNN imputation wth n neighbors
    imputed data is used only to enable row clustering for better per protein visualization
    dataframe - input dataframe
    quantcolumns - all columns used for quantitative analysis
    quantified_entity - actually qiantified entity in data. Most commonly, protein group
    n - neighbor number for imputation
    Returns imputed data frame only for clustering purposes
    
    '''
    proteintest_df = copy.copy(dataframe)
    proteintest_df = proteintest_df.reset_index()
    cols = list(proteintest_df.columns)
    proteintest_quant = proteintest_df[quantcolumns + [quantified_entity]] #take quantitative set
    #proteintest_df = proteintest_df.drop_duplicates(subset = 'Protein.Group')
    rest_cols = [x for x in cols if not x in quantcolumns]
    rest_cols_frame = proteintest_df[rest_cols]#take rest of data
    rest_cols_frame = rest_cols_frame.set_index(quantified_entity)#make identical index
    proteintest_quant = proteintest_quant.set_index(quantified_entity)          #make identical index
    imputer = KNNImputer(n_neighbors=n,weights="distance")
    imputed_proteintest_quant = imputer.fit_transform(proteintest_quant)
    imputed_proteintest_quant = pd.DataFrame(imputed_proteintest_quant,columns = proteintest_quant.columns,index = proteintest_quant.index)
    imputed_data = imputed_proteintest_quant.join(rest_cols_frame)
    imputed_data = imputed_data.reset_index()
    return imputed_data

def check_existence_single(string,substrings):
    '''accessory function to check if item is in item list given as string separated by ;
    string - string to split and search
    substring - item to search for
    Returns bool, if item is or is not present'''
    if pd.isna(string):
        val = False
    else:
        stringlist = string.split(';')
        if any([x in stringlist for x in substrings]):
            val = True
        else:
            val = False
    return val

'''make sure that both chunks have the same index!'''
def takechunk(datasource,datasource_imputed,quantcolumns,item,column,quantified_entity,valid_column,description_column):
     ''' Generates a proper data for plotting
     datasource - raw data source for taking data for plot
     datasource_imputed - imputed counterpart to take data for row clustering function - result of impute function
     quantcolumns - all columns used for quantitative analysis
     item - ; separated list of items to plot. ; separation is needed when the concatenated categories are used as input
     column - column to look for item in
     quantified_entity - actually qiantified entity in data. Most commonly, protein group
     valid_column - column with source of data for marking valid (mean somehow useful quantitation) rows
     description_column - column with description for row labels
     Returns data chunk for plotting, data chunk counterpart withoun NaN for row clustering, item name for plot title
     
     '''
     splititem = item.split(';')
     datasource['lookup'] = datasource[column].apply(check_existence_single, args = [splititem])
     datachunk = datasource[datasource['lookup'] == True]
     cols = quantcolumns+[quantified_entity,valid_column,description_column]
     datachunk = datachunk[cols]
     datasource_imputed['lookup'] = datasource_imputed[column].apply(check_existence_single, args = [splititem])
     datachunk_imputed = datasource_imputed[datasource_imputed['lookup'] == True]
     columns = quantcolumns + [quantified_entity]
     datachunk_imputed = datachunk_imputed[columns]
     datachunk = datachunk.set_index(quantified_entity)
     datachunk_imputed = datachunk_imputed.set_index(quantified_entity)
     
     return [datachunk,datachunk_imputed,item]
     
def clustermap_single(data,data_imputed,quantcolumns,description,quantified_entity,valid_data,description_column):
    '''Makes a clustermap for individual category with per-entity(usually per-protein) rows and additional column
    containing valid marker
    data -  chunk of raw data
    data_imputed - chunk of imputed data
    description -  plot title string, usually categroy name
    quantified_entity - actually qiantified entity in data. Most commonly, protein group
    valid_data - column containing markers for valid and not valid rows, as rgb 3-element lists!
    description_column - column with description for row labels
    Returns seaborn.ClusterGrid instance
    
    '''
    def normalize_mean(serieslike):
        try:
            serieslike = serieslike - serieslike.mean()
        except:
            print(serieslike)
            serieslike = serieslike
        return serieslike
    
    
    numrows = data.shape[0]
    transform = lambda x: x[0:70] if len(x)>70 else x
    description = transform(description)
    print(data.columns)
    print(data.index)
    if  description_column == quantified_entity:
        data[description_column] = data.index
    cols = list(data.columns)
    cols.remove(valid_data)
    cols.remove(description_column)
    #data = data.set_index(quantified_entity)
    for_clustering = data[cols]
    #0 mean
    for_clustering[quantcolumns] = for_clustering[quantcolumns].apply(normalize_mean,axis=1)
    matrix = data_imputed[quantcolumns]
    matrix = np.array(matrix)
    dist_matrix = distance(matrix)
    link = linkage(dist_matrix)
    seaborn.plotting_context(rc={'xtick.labelsize': 10,'ytick.labelsize': 10,"font.size" :10})
    cluster = seaborn.clustermap(for_clustering,cmap="coolwarm",z_score=None,metric="correlation",figsize = (15,(numrows/4)+2),col_cluster=False,\
    dendrogram_ratio=(0,0.1),colors_ratio = (0.05),cbar_pos = (0,1,0.05,0.2),tree_kws={"linewidths":0},\
    row_colors = data[valid_data],row_linkage = link,yticklabels = data[description_column])
    cluster.ax_heatmap.set_title(description)
    cluster.ax_heatmap.set_xlabel('Replicate')
    cluster.ax_heatmap.set_ylabel(description_column)
    return cluster
    
''' TODO add option for use arbitrary column to plot valid rows'''
def visualize_invidual(inputfile,quantified_entity,entrylist,quantcolumns,conditions,description_column,max_individual_ANOVA,min_individual_difference,neighbors_knn_rowcluster):

    datasource = pd.read_csv(inputfile, sep = '\t')
    datasource = datasource[pd.isna(datasource[quantified_entity]) == False]
    #datasource[quantcolumns] = datasource[quantcolumns].apply(np.log2)
    datasource_imputed = impute(datasource,quantcolumns,quantified_entity,neighbors_knn_rowcluster)
    datasource = validate_rows(datasource,quantcolumns,conditions,max_individual_ANOVA,min_individual_difference)
    entrylist = take_items(entrylist)
    dirname = entrylist[0][0]
    wd = os.getcwd()
    os.mkdir(dirname)
    os.chdir(dirname)
    for entry in entrylist:
        chunk = takechunk(datasource,datasource_imputed,quantcolumns,entry[1],entry[0],quantified_entity,'Valid Row',description_column)
        cluster = clustermap_single(chunk[0],chunk[1],quantcolumns,chunk[2],quantified_entity,'Valid Row',description_column)
        filename = entry[1].replace(' ','_')
        if len(filename) > 50:
            filename = filename[0:49]
        filename = "".join( x for x in filename if not x in invalid_characters)
        filename = filename + '.png'
        cluster.savefig(filename)
    os.chdir(wd)