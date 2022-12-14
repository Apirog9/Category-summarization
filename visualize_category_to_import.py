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



invalid_characters = ';:,[]{}~-<>/\=+^%&*()'




def take_items(entrylist):
    data = open(entrylist,'r').read().splitlines()
    itemlist = []
    for line in data:
        line = line.split("\t")
        itemlist.append(line)
    return itemlist

def validate_rows(datasource,quantcolumns,groups,max_ANOVA,mindiff):
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
    
    datasource['Valid Row'] = datasource.apply(validate_anova,args = [groups,max_ANOVA,mindiff],axis =1)
    return datasource


def impute(dataframe,columnset,indexcolumn,n):
    '''KNN imputation wth n neighbors'''
    print(dataframe.shape)
    proteintest_df = copy.copy(dataframe)
    proteintest_df = proteintest_df.reset_index()
    print(proteintest_df.shape)
    cols = list(proteintest_df.columns)
    proteintest_quant = proteintest_df[columnset + [indexcolumn]] #take quantitative set
    #proteintest_df = proteintest_df.drop_duplicates(subset = 'Protein.Group')
    print(proteintest_quant.shape)
    rest_cols = [x for x in cols if not x in columnset]
    rest_cols_frame = proteintest_df[rest_cols]
    print(rest_cols_frame.shape)                           #take rest of data
    rest_cols_frame = rest_cols_frame.set_index(indexcolumn)
    print(rest_cols_frame.shape)                #make identical index
    proteintest_quant = proteintest_quant.set_index(indexcolumn)          #make identical index
    print(proteintest_quant.shape) 
    imputer = KNNImputer(n_neighbors=n,weights="distance")
    imputed_proteintest_quant = imputer.fit_transform(proteintest_quant)
    imputed_proteintest_quant = pd.DataFrame(imputed_proteintest_quant,columns = proteintest_quant.columns,index = proteintest_quant.index)
    print(imputed_proteintest_quant.shape)
    imputed_data = imputed_proteintest_quant.join(rest_cols_frame)
    print(imputed_data.shape)
    imputed_data = imputed_data.reset_index()
    print(imputed_data.shape)
    return imputed_data


def check_existence_single(string,substrings):
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
def takechunk(datasource,datasource_imputed,quantcolumns,item,column,subitem_column,valid_column,description_column):
     print(item)
     splititem = item.split(';')
     datasource['lookup'] = datasource[column].apply(check_existence_single, args = [splititem])
     datachunk = datasource[datasource['lookup'] == True]
     cols = quantcolumns+[subitem_column,valid_column,description_column]
     datachunk = datachunk[cols]
     print(datasource_imputed.columns)
     datasource_imputed['lookup'] = datasource_imputed[column].apply(check_existence_single, args = [splititem])
     datachunk_imputed = datasource_imputed[datasource_imputed['lookup'] == True]
     columns = quantcolumns + [subitem_column]
     datachunk_imputed = datachunk_imputed[columns]
     datachunk = datachunk.set_index(subitem_column)
     datachunk_imputed = datachunk_imputed.set_index(subitem_column)
     return [datachunk,datachunk_imputed,item]
     


def clustermap_single(data,data_imputed,quantcolumns,description,subitem_column,valid_data,description_column):
    numrows = data.shape[0]
    print(description)



    transform = lambda x: x[0:70] if len(x)>70 else x
    description = transform(description)
    cols = list(data.columns)
    cols.remove(valid_data)
    cols.remove(description_column)
    #data = data.set_index(subitem_column)
    for_clustering = data[cols]
    matrix = data_imputed[quantcolumns]
    matrix = np.array(matrix)
    dist_matrix = distance(matrix)
    link = linkage(dist_matrix)
    seaborn.plotting_context(rc={'xtick.labelsize': 10,'ytick.labelsize': 10,"font.size" :10})
    cluster = seaborn.clustermap(for_clustering,cmap="coolwarm",z_score=0,metric="correlation",figsize = (15,(numrows/4)+2),col_cluster=False,\
    dendrogram_ratio=(0,0.1),colors_ratio = (0.05),cbar_pos = (0,1,0.05,0.2),tree_kws={"linewidths":0},\
    row_colors = data[valid_data],row_linkage = link,yticklabels = data[description_column])
    cluster.ax_heatmap.set_title(description)
    cluster.ax_heatmap.set_xlabel('Replicate')
    cluster.ax_heatmap.set_ylabel(description_column)
    return cluster
    
    
 













def visualize_invidual(inputfile,quantified_entity,entrylist,quantcolumns,conditions,description_column):

    datasource = pd.read_csv(inputfile, sep = '\t')
    datasource = datasource[pd.isna(datasource[quantified_entity]) == False]
    datasource[quantcolumns] = datasource[quantcolumns].apply(np.log2)
    datasource_imputed = impute(datasource,quantcolumns,quantified_entity,3)
    datasource = validate_rows(datasource,quantcolumns,conditions,0.01,0.7)
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