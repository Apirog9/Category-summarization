# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:22:51 2022

@author: Proccesing_PC2
"""


import pandas as pd
import copy
import scipy.stats as sts
import numpy as np
import seaborn
from sklearn.impute import KNNImputer
import os
import matplotlib.pyplot as plt


def prepare_data_cl(dataframe,quantcolumns,conditions,quantified_entity,minobs,mingroups,n):
    
    '''
    Will perform log2 transform, missing value filtering (drop rows with too many missing values) 
    and KNN imputation.
    dataframe - pandas dataframe, can contain NaN, not log2 transformed, properly normalized
    quantcolumns - all columns used for quantitative analysis
    conditions - condition:columns dictionary
    quantified_entity - actually qiantified entity in data. Most commonly, protein group
    minobs - minimum valid values in group for group to be considered valid
    migroups - minimum valid groups
    n- number of nearest neighbors for knn imputation
    Returns - dataframe without missing values
    
    '''
    def missing_filter_nobs_ngroups(serieslike,conditions,minobs,mingroups):
        validconditions=0
        for condition in conditions.keys():
            values = list(serieslike[conditions[condition]])
            valid = np.count_nonzero(~np.isnan(values))
            if valid>=minobs:
                validconditions+=1
        if validconditions >= mingroups:
            returnval = True
        else:
            returnval=False
        return returnval

    def impute(dataframe,columnset,quantified_entity,n):
        proteintest_df = copy.copy(dataframe)
        cols = list(proteintest_df.columns)
        proteintest_df = proteintest_df[proteintest_df["valid"] == True]
        proteintest_df = proteintest_df.drop("valid",axis=1)                  #filter valid
        proteintest_quant = proteintest_df[columnset + [quantified_entity]] #take quantitative set
        rest_cols = [x for x in cols if not x in columnset+['valid']]
        rest_cols_frame = proteintest_df[rest_cols]                           #take rest of data
        rest_cols_frame = rest_cols_frame.set_index(quantified_entity)              #make identical index
        proteintest_quant = proteintest_quant.set_index(quantified_entity)          #make identical index
        imputer = KNNImputer(n_neighbors=n,weights="distance")
        imputed_proteintest_quant = imputer.fit_transform(proteintest_quant)
        imputed_proteintest_quant = pd.DataFrame(imputed_proteintest_quant,columns = proteintest_quant.columns,index = proteintest_quant.index)
        imputed_data = imputed_proteintest_quant.join(rest_cols_frame)
        return imputed_data
    
    dataframe["valid"] = dataframe.apply(missing_filter_nobs_ngroups,args = [conditions,minobs,mingroups],axis=1) # filter valid entries
    dataframe[quantcolumns] = dataframe[quantcolumns].apply(np.log2) #make log2 transform
    ready_data = impute(dataframe,quantcolumns,quantified_entity,n) #impute
    ready_data = ready_data.reset_index() 
    return ready_data

#for now assume quantified entity one for row, all rows
def group_categories_simple(dataframe,quantcolumns,conditions,numcolumns,quantified_entity,max_valid_anova,textcolumns,categorycolumn,weigths = None,zsc = False,normalize_rows = False,deconcatenate_categories = True,characterizers = None):
    '''attepmt to perform merging quantifications for whole category
    Dataframe must not contain any NaN
    items in merged text columns are separated by | to diversify from orginal dataframe contents
    arguments:
    dataframe - data to operate on. Pandas dataframe
    quantcolumns - data containing quantifications. will be merged as average or weighted average. List of strings
    conditions - condition:columns dictionary
    numcolumns - numerical values, will be merged as average. List of strings
    quantified_entity - actually qiantified entity in data. Most commonly, protein group
    max_valid_anova -  maximum ANOVA p value for ANOVa limiter, to define informative entities, e.g for later use in merging
    textcolumns - text columns, will be merged with |. List of strings
    categorycolumn - column containing categorical vale to merge on. rows with no value are removed. String
    weights - column containing optional weights for weighted average. String or None. Default None
    zsc - whether to z-score or not this will seriously affect all results. Bool, default False
    normalize_rows - whether to perform mean normalization of rows. Bool
    deconcatenate_categories -if categories in rows are separated by ; make multiple rows with only one category per row. Bool
    caharacterizers - experimental option to run specific functions on individual entities for better characterization
    Returns - grouped dataframe'''
    
    
    def ANOVA_characterizer(grouplike):
        ''' Return string with new column name and 
        string with p values for all quantified entities in group '''
        def calc_anova(serieslike,conditions):
            arr = []
            for group in conditions.keys():
                groupdata = list(serieslike[conditions[group]])
                arr.append(groupdata)
            stat = sts.f_oneway(*arr)
            stat =stat.pvalue
            return stat
        grouplike["Anova_"+quantified_entity] = grouplike.apply(calc_anova,args = [conditions],axis =1)
        stats = list(grouplike["Anova_"+quantified_entity])
        stats = "|".join([str(x) for x in stats])
        return ("Anova_"+quantified_entity,stats)
    
    def ANOVA_limiter(grouplike):
        ''' Return string with new column name and string with all entities
        with p ANOVA < min_valid_anova'''
        def calc_anova(serieslike,conditions):
            arr = []
            for group in conditions.keys():
                groupdata = list(serieslike[conditions[group]])
                arr.append(groupdata)
            stat = sts.f_oneway(*arr)
            stat =stat.pvalue
            return stat
        grouplike["Anova_"+quantified_entity] = grouplike.apply(calc_anova,args = [conditions],axis =1)
        stats = list(grouplike["Anova_"+quantified_entity])
        names = list(grouplike[quantified_entity])
        valid = []
        for stat,name in zip(stats,names):
            if stat < max_valid_anova:
                valid.append(name)
        valid = "|".join(valid)
        return ("Valid_ANOVA_"+quantified_entity,valid)
    
    funcdict = {"Anova_characterizer":ANOVA_characterizer,"Anova_limiter":ANOVA_limiter}  # for characerizers to work, it contains their string names together with respective functions
    
    # check for NaN.probably not allowed
    if dataframe[quantcolumns].isnull().values.any():
        print("remove NaN!!")
        return 1
    #check if parameters do not form nonsense combinations
    if zsc and normalize_rows:
        print("Zscore and row normalization make no sense")
        return 1
    # select only rows with item
    dataframe = dataframe[dataframe[categorycolumn].str.len() >0]   
    
    #ensure all rows will have the same mean value
    if normalize_rows:
        total_mean =  dataframe[quantcolumns].mean().mean()
        dataframe[quantcolumns] = dataframe[quantcolumns].sub(dataframe[quantcolumns].mean(axis=1),axis=0)
        dataframe[quantcolumns] = dataframe[quantcolumns] + total_mean
        
    # z-score transformation of rows    
    if zsc:
        dataframe[quantcolumns] = dataframe[quantcolumns].apply(sts.zscore,axis=1)
    if deconcatenate_categories:
        dataframe = dataframe.drop(categorycolumn,axis=1).join(dataframe[categorycolumn].str.split(";",expand=True).stack().reset_index(level=1, drop=True).rename(categorycolumn))
        dataframe = dataframe.drop_duplicates()
        
    dataframe = dataframe.groupby(by=categorycolumn)   # initialize gropuping to further make custom calculations on groups
    newdataframe = []
    textcolumns = copy.copy(textcolumns)
    textcolumns.remove(categorycolumn)

    for group in dataframe:
        newline = {}
        if characterizers ==None:
            pass
        else:
            newcolumns = []
            for func in characterizers:
                func = funcdict[func]
                newline[func(group[1])[0]] = func(group[1])[1]   # result in colmn name identical as characterizer name, and string characterizer output in it
                newcolumns.append(func(group[1])[0])
        newline[categorycolumn] = group[0]
        newline["Entity_Number"] = len(group[1][quantified_entity].unique()) #calculates number of individual entities in group
        for column in numcolumns:
            newline[column] = group[1][column].mean(axis=0)
        for column in textcolumns:
            newline[column] = "|".join([str(x) for x in list(group[1][column])])
            #category normalization
        if weigths == None:
            for column in quantcolumns:
                newline[column] = group[1][column].mean(axis=0)
        else:
            for column in quantcolumns:
                data = list(group[1][column])
                wgt = list(group[1][weigths])
                newline[column] = np.average(data,weights=wgt)
        newdataframe.append(newline)
    os.chdir('..')
    outdataframe = pd.DataFrame(newdataframe)
    return (outdataframe,newcolumns)


def merge_identical_groups(dataframe,categorycolumn,mergecolumn,textcolumns,numcolumns,quantcolumns):
    '''Attepmpt to merge exatly identical categories
    dataframe - data to operate on. Pandas dataframe
    categorycolumn - column with categories to merge. String
    mergecolumns - column containing items (separated by |) for checking identity. Preferentially,
                    these are items that undergone actual quantification e.g protein groups, or items relevant for 
                    category annotation, e.g Genes, however the latter may be tricky. Also good quality quantifications with
                    meaningful change may be used, as by Anova_limiter result
    textcolumns - text columns, will be merged with |. List of strings
    numcolumns - numerical values, will be merged as average. List of strings
    quantcolumns - data containing quantifications. will be merged as average or weighted average. List of strings
    Returns - dataframe with similar or identical categories merged into one'''
    
    aggregation_functions = {}
    dataframe[mergecolumn] = dataframe[mergecolumn].apply(lambda x: "|".join(list(set(x.split("|")))))
    dataframe[mergecolumn] = dataframe[mergecolumn].apply(lambda x: "|".join(sorted(x.split("|"))))
    for item in textcolumns:
        aggregation_functions[item] = "|".join
    for item in numcolumns:
        aggregation_functions[item] = "mean"
    for item in quantcolumns:
        aggregation_functions[item] = "mean"
    aggregation_functions[categorycolumn] = ";".join
    aggregation_functions["Entity_Number"] = "mean"
    dataframe = dataframe.groupby(dataframe[mergecolumn]).aggregate(aggregation_functions)
    
    for item in textcolumns:
        dataframe[item] = dataframe[item].apply(lambda x: "|".join(set(x.split("|"))))
    
    
    return dataframe


def calculate_ANOVA(dataframe,quantcolumns,conditions,suffix):
    '''calculate ANOVA p-value, return dataframe with additional column
    dataframe - data to operate on. Pandas dataframe
    quantcolumns - data containing quantifications. will be merged as average or weighted average. List of strings
    groups - groups of to calculate ANOVA. dict {"name":["col1","col2","col3"]}
    suffix - column will be named Anova+suffix. string
    Returns dataframe with new column with simple uncorrected p-value
    '''
    def calc_anova(serieslike,groups):
        arr = []
        for group in groups.keys():
            groupdata = list(serieslike[groups[group]])
            arr.append(groupdata)
        stat = sts.f_oneway(*arr)
        stat =stat.pvalue
        return stat
    dataframe["Anova"+suffix] = dataframe.apply(calc_anova,args = [conditions],axis =1)
    return dataframe
    
    
def filter_simple(dataframe,quantcolumns,conditions,anovacolumn,maxp,mindiff,minnum):
    
    '''
    Filter out categorical groups without meaningful regulation
    dataframe - dataframe with calulated per-category changes produced by group_categories_simple
    quantcolumns - all columns used for quantitative analysis
    conditions - condition:columns dictionary
    anovacolumn - column containing per-category ANOVA result
    maxp - maximum per-category p-value
    mindiff - minimum difference between groups
    minnum - minum entities per catogory
    Returns - filtered, category grouped dataframe ready for plotting
    '''
    def calc_maxdiff(serieslike,groups):
        means = []
        for group in groups.keys():
            groupdata = list(serieslike[groups[group]])
            mean = np.mean(groupdata)
            means.append(mean)
        maxdiff = max(means) - min(means)
        return maxdiff
    if anovacolumn != None:    
        dataframe = dataframe[dataframe[anovacolumn] < maxp]
    dataframe["Max_difference"] = dataframe[quantcolumns].apply(calc_maxdiff,args = [conditions],axis =1)
    dataframe = dataframe[dataframe["Max_difference"] > mindiff]
    if minnum > 0:
        dataframe = dataframe[dataframe["Entity_Number"] > minnum]
    else:
        pass
    
    return dataframe
    
def clustermap(data,quantcolumns,description,renamer,x_lab_s,y_lab_s,fig_x_s,fig_y_s):
    ''' 
    draws visualization
    data - dataframe to visualize
    quantcolumns - all columns used for quantitative analysis
    description - column containing row labels, e.g genes, protein names
    renamer - renamer dictionary to give category column types a neat name
    x_lab_s - x label font size
    y_lab_s - y label font size
    fig_x_s - figure x size
    fig_y_s - figure y size
    returns seaborn.ClusterGrid instance
    
    '''
    data = data.rename(renamer,axis=1)
    data[description] = data[description].apply(lambda x: x[0:70] if len(x)>70 else x )
    for_clustering = data[quantcolumns + [description]]
    for_clustering = for_clustering.set_index(description)
    with seaborn.plotting_context(rc={'xtick.labelsize': x_lab_s,'ytick.labelsize': y_lab_s,"font.size" :10}):
        cluster = seaborn.clustermap(for_clustering,cmap="bwr",z_score=0,metric="correlation",figsize = (fig_x_s,fig_y_s),col_cluster=False,\
        dendrogram_ratio=(0.3,0.1),colors_ratio = (0.05),cbar_pos = None,tree_kws={"linewidths":2.5})
    return cluster


''' TODO add opition for miobs mingroups for row filtering
    for now change in line 311 -2 and -1 argument'''     
def perform_grouping(inputfile,categorycolumn,quantcolumns,numcolumns,textcolumns,conditions,comparisons,\
                     renamer,quantified_entity,max_valid_for_merging,weight_name,characterizer_list,\
                    group_merging_column,group_filtering_column,max_categorywise_anova,minimum_categorywise_difference,\
                       minimum_entities,fig_x_lab,fig_y_lab,fig_x_size,fig_y_size,minobs_grp,mingroups_grp,neighbors_knn_grp):
    ''' Arguments from previous functions and params.txt file
    Returns seaborn.ClusterGrid instance for plotting and list of finally used categories'''
    
    
    dataframe = pd.read_csv(inputfile, sep = "\t")  
    # impute and filter valid values
    test_frame = prepare_data_cl(dataframe,quantcolumns,conditions,quantified_entity,minobs_grp,mingroups_grp,neighbors_knn_grp)
    # calculate ANOVA per row in frame
    dataframe = calculate_ANOVA(test_frame,quantcolumns,conditions,"_separate")
    # transform ANOVA to pval to -log10
    dataframe["Anova_separate"] = dataframe["Anova_separate"].apply(lambda x: -1*np.log10(x))
    output = group_categories_simple(dataframe,quantcolumns,conditions,numcolumns,quantified_entity,max_valid_for_merging,textcolumns,categorycolumn,weigths=weight_name,characterizers=characterizer_list)
    outdata = output[0]
    textcolumns = textcolumns +output[1]
    outdata2 = merge_identical_groups(outdata,categorycolumn,group_merging_column,textcolumns,numcolumns,quantcolumns)
    outdata3 = calculate_ANOVA(outdata2,quantcolumns,conditions,"_grouped")
    outdata4 = filter_simple(outdata3,quantcolumns,conditions,group_filtering_column,max_categorywise_anova,minimum_categorywise_difference,minimum_entities)
    cluster_all = clustermap(outdata4,quantcolumns,renamer[categorycolumn],renamer,fig_x_lab,fig_y_lab,fig_x_size,fig_y_size)
    outdata4.to_csv("category_mean_prepared_1"+ "_".join(categorycolumn.split(" ")[1:])+".tsv",sep="\t",index=False)
    final_categories = list(outdata4[categorycolumn].unique())

    return cluster_all,final_categories
