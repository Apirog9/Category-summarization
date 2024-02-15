# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:00:59 2024

@author: Proccesing_PC2
"""
import os
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
from sklearn.impute import KNNImputer
from scipy.spatial.distance import squareform, pdist as distance
from scipy.cluster.hierarchy import linkage

import scipy.stats as sts




invalid_characters = ';:,[]{}~-<>/\=+^%&*()'


class GO_annotate:
    def __init__(self):
        
        return 'not implemented'
    

class MergingAnalysis:
    """
    Merge/average analysis of quantitative data annotated with categories (typically - Gene Ontology terms).
    
    Attributes
    ----------
    categorycolumns : list of str
        List of category columns to consider
    raw_quant_data : DataFrame
        Quantitation data read as is
    full_imputed_data : DataFrame
        completely imputed data (without prefiltering). Used only for individual category clutermap
        tree preparation
    quantcolumns : list of str
        List of column names containing quantitative data
    conditions : dict
        Dictionary of str:list of str form containing condition names and 
        corresponding quantitative columns names
    renamer : dict
        Dictionary of str:str of str form to rename category column names to pretty names
    quantified_entity : str
        Column name of actually quantified entity, typically Genes or Protein.Group
    max_valid_for_merging : float
        Maximum individual ANOVA p-value for entity to be cosidered as informative 
        and be used for merging the categories taking most of their
        regulation from the same proteins. Usually affect the result only slightly
    weight_name : str or None
        Weight type for weighted averaging. Currently support None for no weights or
        'Anova_separate' for ANOVA -log10(p-value) weighting
    group_merging_column : str
        Currently, use 'Valid_ANOVA_Gene' only
    group_filtering_column : str
        Currently, use 'Anova_grouped' only
    categorylist_file :  str or None
        If str, contain file name with selected categories to plot individually
    description_column : str
        Name of column containing entity names to be plotted on individual category plot
    max_individual_ANOVA : float
        Maximum ANOVA p-value for entity to be considered valid
    min_individual_difference : float
        Minimum difference for entity to be considered valid
    minobs_before_knn : int
        Minimum valid values in condition for condition to be considered valid
    mingroups_before_knn : int
        Minumim valid conditions for entity not to be removed before imputation
    n_neighbors_category_analysis : int
        N neighbors for imputation of global quantification table before grouping
    n_neighbors_row_clustering : int
        N neighbors for imputation for row clustering in individual category graph
        affects only visualization
    merged_frames : dict
        Dictionary of form category column : Dataframe storing category-aggregated dataframes
    filtered_frames : dict
        Dictionary of form category column : Dataframe storing category-aggregated and filtered dataframes
    full_imputed_data : DataFrame
        Raw_quant_data imputed without prefiltering. Useful only for clustering individual entities
    plotting_categories : dict
        Dict of Form {Category Type : List of categories currently selected for plotting on individual entity level}
    """

    def __init__(self,quantdata,paramfile = 'param.txt'):
        """
        Read old-style param file or initialize all necessary values as None or necessary default.
        
        Supports generating both global plots of selected category behaviour as well as 
        plotting individual categories on entity level (typically Gene,Protein)

        Parameters
        ----------
        quantdata : string
            filename of tab-separated file with quantitation data (GO-annotated)
        paramfile : string, optional
            DESCRIPTION. The default is 'param.txt'. Parameter file name

        Returns
        -------
        None.

        """
        self.raw_quant_data = pd.read_csv(quantdata,sep='\t')
        
        if paramfile:
            linedict = {}
            lines = open(paramfile,'r').read().splitlines()
            for line in lines:
                if line.startswith('#'):
                    pass
                else:
                    line = line.split('\t')
                    linedict[line[0]] = line[1]
                
            self.categorycolumns = linedict['categorycolumns'].split(';')
            self.quantcolumns = linedict['quantcolumns'].split(';')
            conditions = linedict['conditions'].split(';')
            cond_dict = {}
            for item in conditions:
                name = item.split(':')[0]
                conditions=item.split(':')[1]
                conditions = conditions.split(',')
                cond_dict[name] = conditions
            self.conditions = cond_dict
            renamer = linedict['renamer'].split(';')
            ren_dict = {}
            for item in renamer:
                ren_dict[item.split(':')[0]] = item.split(':')[1]
            self.renamer = ren_dict
            self.quantified_entity = linedict['quantified_entity']
            self.max_valid_for_merging = float(linedict['max_valid_for_merging'])
            self.weight_name = linedict['weigths_type_for_concatenation']
            if self.weight_name == 'None':
                self.weight_name = None
            self.group_merging_column = linedict['group_merging_column']
            self.group_filtering_column = linedict['group_filtering']
            self.categorylist_file = linedict['categorylist']
            if self.categorylist_file == 'None':
                self.categorylist_file = None
            self.description_column = linedict['description_column']
            self.max_individual_ANOVA = float(linedict['max_individual_ANOVA'])
            self.min_individual_difference = float(linedict['min_individual_difference'])
            self.minobs_before_knn = int(linedict['minobs_before_knn'])
            self.mingroups_before_knn = int(linedict['mingroups_before_knn'])
            self.n_neighbors_category_analysis = int(linedict['n_neighbors_category_analysis'])
            self.n_neighbors_row_clustering = int(linedict['n_neighbors_row clustering'])
            self.merged_frames = {}
            self.filtered_frames = {}
            self.full_imputed_data = None
            self.plotting_categories = {}
            self.raw_quant_data = self.raw_quant_data[self.quantcolumns + [x for x in self.raw_quant_data.columns if x not in self.quantcolumns]]
        else:
            self.categorycolumns = None
            self.quantcolumns = None
            self.conditions = None
            self.renamer = None
            self.quantified_entity = None
            self.max_valid_for_merging = None
            self.weight_name = None
            self.group_merging_column = 'Valid_ANOVA_Gene'
            self.group_filtering_column = 'Anova_grouped'
            self.categorylist = None
            self.description_column = None
            self.max_individual_ANOVA = None
            self.min_individual_difference = None
            self.minobs_before_knn = None
            self.mingroups_before_knn = None
            self.n_neighbors_category_analysis = None
            self.n_neighbors_row_clustering = None
            self.merged_frames = {}
            self.filtered_frames = {}
            self.full_imputed_data = None
            self.plotting_categories = {}
            
        if self.categorycolumns:
            if not all([x in self.raw_quant_data.columns for x in self.categorycolumns]):
                print('check category column names')
        if self.quantcolumns:
            if not all([x in self.raw_quant_data.columns for x in self.quantcolumns]):
                print('check quant column names')        
        if self.conditions:
            if not all([x in self.raw_quant_data.columns for x in list(it.chain.from_iterable(list(self.conditions.values())))]):
                print('check condition columns names')    
                
    def prepare_data(self,conditions=None,minobs_before_knn=None,mingroups_before_knn=None,n_neighbors_category_analysis=None):
        """
        Perform inital data processing.
        
        Filtering badly quantified entities by
        selecting ones with min valid values in min number of conditions and
        KNN imputation of filtered data.

        Parameters
        ----------
        conditions: dict
            Dictionary of str:list of str form containing condition names and
            corresponding quantitative columns names.
        minobs_before_knn: int
            Minimum valid values in condition for condition to be considered valid.
        mingroups_before_knn: int
            Minumim valid conditions for entity not to be removed before imputation.
        n_neighbors_category_analysis: int
            N neighbors for imputation of global quantification table before grouping.

        Returns
        -------
        None.

        """
        if conditions: self.conditions = conditions
        if minobs_before_knn: self.minobs_before_knn = minobs_before_knn
        if mingroups_before_knn: self.mingroups_before_knn = mingroups_before_knn
        if n_neighbors_category_analysis: self.n_neighbors_category_analysis = n_neighbors_category_analysis
        
        
        if not all([self.conditions,self.minobs_before_knn,self.mingroups_before_knn,self.n_neighbors_category_analysis]):
            print('check parameters!')
            return None
        
        self.prepared_data = copy.copy(self.raw_quant_data)
        self.prepared_data["valid"] = self.prepared_data.apply(MergingAnalysis.__missing_filter_nobs_ngroups,args = [self.conditions,self.minobs_before_knn,self.mingroups_before_knn],axis=1) # filter valid entries
        if self.prepared_data[self.quantcolumns].max(axis=None).max() > 100:
            self.prepared_data[self.quantcolumns] = self.prepared_data[self.quantcolumns].apply(np.log2) #make log2 transform
        ready_data = MergingAnalysis.__impute_partial(self.prepared_data,self.quantcolumns,self.quantified_entity,self.n_neighbors_category_analysis) #impute
        ready_data = ready_data.reset_index() 
        self.prepared_data = ready_data
    
    @staticmethod 
    def __missing_filter_nobs_ngroups(serieslike,conditions,minobs,mingroups):
        """
        Filter entities with too many missing values.

        Parameters
        ----------
        serieslike : Series
            Input Dataframe row
        conditions : dict
            Dictionary of str:list of str form containing condition names and 
            corresponding quantitative columns names
        minobs : int
            Minimum valid values in condition for condition to be considered valid
        mingroups : int
            Minumim valid conditions for entity not to be removed before imputation

        Returns
        -------
        returnval : bool
            True if valid and fulfill criteria, False otherwise.

        """
        validconditions=0
        for condition in conditions.keys():
            valid = serieslike[conditions[condition]].notna().sum()
            if valid>=minobs:
                validconditions+=1
        if validconditions >= mingroups:
            returnval = True
        else:
            returnval = False
       
        return returnval        
    @staticmethod
    def __calc_anova(serieslike,groups):
        """
        Calculate  ANOVA p-value.
            
        Parameters
        ----------
        serieslike : Series
            Input dataframe Series
        groups : dict
            Dictionary of str:list of str form containing condition names and 
            corresponding quantitative columns names

        Returns
        -------
        stat : float
            ANOVA p-value

        """
        arr = []
        for group in groups.keys():
            groupdata = list(serieslike[groups[group]])
            arr.append(groupdata)
        stat = sts.f_oneway(*arr)
        stat =stat.pvalue
        return stat
    
    @staticmethod
    def __impute_partial(dataframe,columnset,quantified_entity,n):
        """
        Perform kNN imputation of filtered data.

        Parameters
        ----------
        dataframe : Dataframe
            Dataframe for imputation of quantitative values
        columnset : list of str
            List of column names containing quantitative data
        quantified_entity : str
            Column name of actually quantified entity, typically Genes or Protein.Group
        n : int
            N neighbors for kNN imputation 

        Returns
        -------
        imputed_data : Dataframe
            Imputed dataframe with all original columns.

        """
        proteintest_df = copy.copy(dataframe)
        cols = list(proteintest_df.columns)
        proteintest_df = proteintest_df[proteintest_df["valid"] == True]
        proteintest_df = proteintest_df.drop("valid",axis=1)                  
        proteintest_quant = proteintest_df[columnset + [quantified_entity]] 
        rest_cols = [x for x in cols if not x in columnset+['valid']]
        rest_cols_frame = proteintest_df[rest_cols]                           
        rest_cols_frame = rest_cols_frame.set_index(quantified_entity)              
        proteintest_quant = proteintest_quant.set_index(quantified_entity)
        imputer = KNNImputer(n_neighbors=n,weights="distance")
        imputed_proteintest_quant = imputer.fit_transform(proteintest_quant)
        imputed_proteintest_quant = pd.DataFrame(imputed_proteintest_quant,columns = proteintest_quant.columns,index = proteintest_quant.index)
        imputed_data = imputed_proteintest_quant.join(rest_cols_frame)
        
        return imputed_data    
    
    @staticmethod
    def __validate_anova(serieslike,groups,max_ANOVA,mindiff):
        """
        Mark valid row by simple ANOVA with arbitrary cutoffs.
        
        Parameters
        ----------
        serieslike : Series
            Dataframe line for validation.
        groups : dict
            Dictionary of str:list of str form containing condition names and 
            corresponding quantitative columns names
        max_ANOVA : float
            Maximum ANOVA p-value (uncorrected!) for entity to be considered valid 
            (having reasonable inter-group change).
        mindiff : float
            Minimum difference for entity to be considered valid 
            (having reasonable inter-group change).

        Returns
        -------
        returnval : list of ints
            Returns directly a RGB code for coloring an additional column of heatmap

        """
        stat = MergingAnalysis.__calc_anova(serieslike, groups)
        if stat <= max_ANOVA:
            validstat = True
        else:
            validstat = False
        means = [serieslike[groups[group]].mean() for group in groups.keys()]
        comparisons = it.combinations(means,2)
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
    
    @staticmethod
    def __validate_list(serieslike,proteincolumn,proteinlist):
        """
        Mark valid row using an external list of valid entities.

        Parameters
        ----------
        serieslike : Series
            Dataframe line for validation.
        proteincolumn : str
            Column name of quantified entity
        proteinlist : list
            List of entities considered valid

        Returns
        -------
        returnval : list of onts
            Returns directly a RGB code for coloring an additional column of heatmap.

        """
        valid = serieslike[proteincolumn] in proteinlist
        if valid:
            returnval = [0,1,0]
        else:
            returnval =[1,0,0]
        return returnval
    
    @staticmethod
    def __ANOVA_characterizer(grouplike,quantified_entity,conditions):
        """
        Calculate condensed ANOVA results for one category.

        Parameters
        ----------
        grouplike : DataFrame groupby element
            Groupby element to process
        quantified_entity : str
            Column name of actually quantified entity, typically Genes or Protein.Group
        conditions : dict
            Dictionary of str:list of str form containing condition names and 
            corresponding quantitative columns names

        Returns
        -------
        colname : str
            aggregated ANOVA column name
        stats : str
            aggregated ANOVA per entity string

        """
        grouplike["Anova_"+quantified_entity] = grouplike.apply(MergingAnalysis.__calc_anova,args = [conditions],axis =1)
        stats = list(grouplike["Anova_"+quantified_entity])
        stats = "|".join([str(x) for x in stats])
        colname = "Anova_"+quantified_entity
        return (colname,stats)    
    
    
    def calculate_ANOVA_separate(self):
        """Calculate ANOVA one-way to assess quantification quality of individual entities, -log10 result."""
        self.prepared_data["Anova_separate"] = self.prepared_data.apply(self.__calc_anova,args = [self.conditions],axis =1)
        self.prepared_data["Anova_separate"] = self.prepared_data["Anova_separate"].apply(lambda x: -1*np.log10(x))

    def calculate_ANOVA_group(self,selected_category):
        """
        Calculate ANOVA one-way to assess quantification quality of grouped entities.

        Parameters
        ----------
        selected_category : str
            category column name containing deconcatenated category strings, one per line.

        Returns
        -------
        None.

        """
        self.merged_frames[selected_category]["Anova_grouped"] = self.merged_frames[selected_category].apply(self.__calc_anova,args = [self.conditions],axis =1)


    @staticmethod
    def __ANOVA_limiter(grouplike,quantified_entity,conditions,max_valid_anova):
        """
        Select informative entities relying on -log10(p-value) limit.

        Parameters
        ----------
        grouplike : DataFrame groupby element
            Groupby element to process
        quantified_entity : str
            Column name of actually quantified entity, typically Genes or Protein.Group
        conditions : dict
            Dictionary of str:list of str form containing condition names and 
            corresponding quantitative columns names.
        max_valid_anova : float
            Maximum individual ANOVA p-value for entity to be cosidered as informative 
            and be used for merging the categories taking most of their
            regulation from the same proteins. Usually affect the result only slightly

        Returns
        -------
        colname : str
            Column name for aggregated validated entities - for merging only
        valid : str
            Aggregated validated entities - for merging only

        """
        #grouplike["Anova_"+quantified_entity] = grouplike.apply(MergingAnalysis.__calc_anova,args = [conditions],axis =1)
        stats = list(grouplike["Anova_"+quantified_entity])
        names = list(grouplike[quantified_entity])
        valid = []
        for stat,name in zip(stats,names):
            if stat < max_valid_anova:
                valid.append(name)
        valid = "|".join(valid)
        colname = "Valid_ANOVA_"+quantified_entity
        return (colname,valid)
    
    @staticmethod
    def __takechunk(self,category,category_type):
        """
        Extract chunks of data necessary for preparing single category heatmap.

        Parameters
        ----------
        category : str
            Category to extract.
        category_type : str
            Column name for searching for category.

        Returns
        -------
        dict
            Dict of form category:[raw_part_for_plotting,imputed_part_for_clustering].
            Imputed part is used only to nicly produce row clusters on heatmap.

        """
        datasource = copy.copy(self.raw_quant_data)
        datasource_imputed = copy.copy(self.full_imputed_data)
        datasource['lookup'] = datasource[category_type].apply(lambda x: category in str(x).split(';'))
        datachunk = datasource[datasource['lookup']]
        cols = self.quantcolumns+list(set([self.quantified_entity,'Valid Row',self.description_column]))
        datachunk = datachunk[cols]
        datasource_imputed['lookup'] = datasource_imputed[category_type].apply(lambda x: category in str(x).split(';'))
        datachunk_imputed = datasource_imputed[datasource_imputed['lookup']]
        cols = self.quantcolumns + [self.quantified_entity]
        datachunk_imputed = datachunk_imputed[cols]
        datachunk = datachunk.set_index(self.quantified_entity)
        datachunk_imputed = datachunk_imputed.set_index(self.quantified_entity)
        if self.description_column == self.quantified_entity:
            datachunk[self.description_column] = datachunk.index.values
            datachunk_imputed[self.description_column] = datachunk_imputed.index.values

        return {category:[datachunk,datachunk_imputed]}
    
    
    @staticmethod
    def __calc_maxdiff(serieslike,groups):
        """
        Calculate maximum between-group difference.

        Parameters
        ----------
        serieslike : Series
            Series to calculate maximum between-group difference
        groups : dict
            Dictionary of str:list of str form containing condition names and 
            corresponding quantitative columns names

        Returns
        -------
        maxdiff : float
            maximum between-group difference

        """
        means = []
        for group in groups.keys():
            groupdata = list(serieslike[groups[group]])
            mean = np.mean(groupdata)
            means.append(mean)
        maxdiff = max(means) - min(means)
        return maxdiff
     
    @staticmethod
    def __merge_identical_groups(dataframe,categorycolumn,mergecolumn,quantcolumns):
        """
        Merge very similar categories into one.
        
        Prevents proliferation of groups composed with almost identical informative (reasonable ANOVA) entries. 

        Parameters
        ----------
        dataframe : DataFrame
            Dataframe after aggregation into categories.
        categorycolumn : str
            Selected category from category columns.
        mergecolumn : str
            Column to check identity. Usually contains prefiltered entities (informative entities only)
        quantcolumns : list of str
            List of column names containing quantitative data.

        Returns
        -------
        dataframe : DataFrame
            Dataframe with merged almost identical categories.

        """
        aggregation_functions = {}
        dataframe[mergecolumn] = dataframe[mergecolumn].apply(lambda x: "|".join(list(set(x.split("|")))))
        dataframe[mergecolumn] = dataframe[mergecolumn].apply(lambda x: "|".join(sorted(x.split("|"))))
        for item in quantcolumns:
            aggregation_functions[item] = "mean"
        aggregation_functions[categorycolumn] = ";".join
        aggregation_functions["Entity_Number"] = "mean"
        aggregation_functions['Entities'] = ";".join
        dataframe = dataframe.groupby(dataframe[mergecolumn]).aggregate(aggregation_functions)
        
        return dataframe        
     
    @staticmethod    
    def __linebreaker(string,maxlength):
        """
        Break string into lines of length maxlength.
        
        Makes lines of exactly identical length for plot labels (space-filled)
        
        Parameters
        ----------
        string : TYPE
            DESCRIPTION.
        maxlength : int
            Length of line.

        Returns
        -------
        result : str
            String with newline characters for label.

        """
        #string = str(string[0])
        string = string.replace('/',' ')
        
        contents = string.split(' ')
        newstring = []
        
        i=0
        while i <len(contents):
            newstringpart = []
            while len(' '.join(newstringpart)) <= maxlength and i <len(contents):
                word = contents[i]
                newstringpart.append(word)
                i=i+1
            newstring = newstring+newstringpart+['\n']
        result = ' '.join(newstring)
        result = result.removesuffix('\n')
        if len(result) < maxlength:
            result = result + ' '*((maxlength-len(result))+10)
        return result
        
    def group_categories_simple(self,selected_category,weights = None,zsc = False,normalize_rows = False,deconcatenate_categories = True):
        """
        Perform grouping of entities into quantified categories.

        Parameters
        ----------
        selected_category : str
            Selected category from category columns
        weights : str on None
            Weights can be None or Anova_separate to use -log10(p-value)
            of individual entity ANOVA as weight. The default is None. 
        zsc : bool, optional
            Whether to Z-score before grouping. Substantially affect results. The default is False.
        normalize_rows : bool, optional
            whether to normalize row values to common mean. Affects results. The default is False.
        deconcatenate_categories : bool, optional
            Whether to deconcatenate ';' separated category strings from selected_category column.
            Usually necessary. The default is True.

        Returns
        -------
        int
            Return 1 if failed on parameter check

        """
        dataframe = self.prepared_data[self.quantcolumns+[selected_category,self.quantified_entity,'Anova_separate']]
        # check for NaN.probably not allowed
        if dataframe[self.quantcolumns].isnull().values.any():
            print("remove NaN!!")
            return 1
        #check if parameters do not form nonsense combinations
        if zsc and normalize_rows:
            print("Zscore and row normalization make no sense")
            return 1
        
        if weights not in [None,'Anova_separate']:
            print('weights can be None or Anova_separate')
        
        # select only rows with item
        dataframe = dataframe[dataframe[selected_category].str.len() >0]   
        
        #ensure all rows will have the same mean value
        if normalize_rows:
            total_mean =  dataframe[self.quantcolumns].mean().mean()
            dataframe[self.quantcolumns] = dataframe[self.quantcolumns].sub(dataframe[self.quantcolumns].mean(axis=1),axis=0)
            dataframe[self.quantcolumns] = dataframe[self.quantcolumns] + total_mean
            
        # z-score transformation of rows    
        if zsc:
            dataframe[self.quantcolumns] = dataframe[self.quantcolumns].apply(sts.zscore,axis=1)
        if deconcatenate_categories:
            dataframe = dataframe.drop(selected_category,axis=1).join(dataframe[selected_category].str.split(";",expand=True).stack().reset_index(level=1, drop=True).rename(selected_category))
            dataframe = dataframe.drop_duplicates()
            
        dataframe = dataframe.groupby(by=selected_category)   # initialize gropuping to further make custom calculations on groups
        
        newdataframe = []
        for group in dataframe:
            newline = {}
            anova_result = MergingAnalysis.__ANOVA_characterizer(group[1], self.quantified_entity, self.conditions)
            newline[anova_result[0]] = anova_result[1]
            anova_validate = MergingAnalysis.__ANOVA_limiter(group[1], self.quantified_entity, self.conditions, self.max_valid_for_merging)
            newline[anova_validate[0]] = anova_validate[1]
            newline[selected_category] = group[0]
            newline["Entity_Number"] = len(group[1][self.quantified_entity].unique()) #calculates number of individual entities in group
            newline["Entities"] = "|".join([str(x) for x in list(group[1][self.quantified_entity])])

            #category normalization
            if weights == None:
                for column in self.quantcolumns:
                    newline[column] = group[1][column].mean(axis=0)
            else:
                for column in self.quantcolumns:
                    data = list(group[1][column])
                    wgt = list(group[1][weights])
                    newline[column] = np.average(data,weights=wgt)
            newdataframe.append(newline)
            
        newdataframe = pd.DataFrame(newdataframe)
        newdataframe = MergingAnalysis.__merge_identical_groups(newdataframe,selected_category,anova_validate[0],self.quantcolumns)

        self.merged_frames[selected_category] = newdataframe

    def filter_simple(self,selected_category,maxp,mindiff,minnum):
        """
        Filter aggregated categories to retain meaningfully regulated for plotting.

        Parameters
        ----------
        selected_category : str
            Selected category from category columns
        maxp : float
            Maximum ANOVA p-value for group to be considered valid.
        mindiff : float
            Minimum difference for group to be considered valid.
        minnum : int
            Minimum entities in group to be considered valid.

        Returns
        -------
        None.

        """
        dataframe = copy.copy(self.merged_frames[selected_category])
        dataframe = dataframe[dataframe['Anova_grouped'] < maxp]
        dataframe["Max_difference"] = dataframe[self.quantcolumns].apply(MergingAnalysis.__calc_maxdiff,args = [self.conditions],axis =1)
        dataframe = dataframe[dataframe["Max_difference"] > mindiff]
        if minnum > 0:
            dataframe = dataframe[dataframe["Entity_Number"] > minnum]
        else:
            pass
        self.filtered_frames[selected_category] = dataframe
        
    def clustermap(self,selected_category,x_lab_s,y_lab_s,fig_x_s,fig_y_s,font_size=10,z_score = False,normalize_mean=True,**kwargs):
        """
        Draw heatmap from prepared category data.

        Parameters
        ----------
        selected_category : str
            Selected category from category columns.
        x_lab_s : float
            x label size.
        y_lab_s : float
            y label size.
        fig_x_s : float
            Figure x size.
        fig_y_s : float
            Figure y size.
        **kwargs key, value mappings
            Arguments passed to seaborn.clustermap

        Returns
        -------
        cluster : Seaborn ClusterGrid instance
            Resulting heatmap.

        """
        kwargs.setdefault('cmap', "bwr")
        kwargs.setdefault('metric', "correlation")
        kwargs.setdefault('col_cluster', False)
        kwargs.setdefault('dendrogram_ratio', (0,0))
        kwargs.setdefault('colors_ratio', (0.01))
        kwargs.setdefault('cbar_pos', None)
        kwargs.setdefault('tree_kws', {"linewidths":2.5})
        
        if z_score:
            kwargs.setdefault('z_score',0)
        else:
            kwargs.setdefault('z_score',None)
        data = copy.copy(self.filtered_frames[selected_category])
        data = data.rename(self.renamer,axis=1)
        description_column = self.renamer[selected_category]
        data[description_column] = data[description_column].apply(lambda x: x[0:70] if len(x)>70 else x )
        for_clustering = data[self.quantcolumns + [description_column]]
        if normalize_mean:
            for_clustering[self.quantcolumns] = for_clustering[self.quantcolumns].subtract(for_clustering[self.quantcolumns].mean(axis=1),axis=0)
        for_clustering = for_clustering.set_index(description_column)
        with sns.plotting_context(rc={'xtick.labelsize': x_lab_s,'ytick.labelsize': y_lab_s,"font.size" :10}):
            cluster = sns.clustermap(for_clustering,figsize = (fig_x_s,fig_y_s),**kwargs)
            plt.xticks(rotation=45)
            cluster.ax_heatmap.set_aspect(aspect=(fig_y_s/fig_x_s))
        return cluster
        


    def impute_full(self):
        """
        Perform full imputation of unfiletered quantitative data.
        
        Should be used only for creating clustering tree for individual entity visualization.

        Returns
        -------
        None.

        """
        protein_df = copy.copy(self.raw_quant_data)
        
        if protein_df[self.quantcolumns].max(axis=None).max() > 100:
            protein_df[self.quantcolumns] = protein_df[self.quantcolumns].apply(np.log2)
        protein_df = protein_df.reset_index()
        
        protein_quant = protein_df[self.quantcolumns + [self.quantified_entity]] #take quantitative set
        if self.description_column != self.quantified_entity:
            rest_cols_frame = protein_df[self.categorycolumns + [self.description_column,self.quantified_entity]]#take rest of data
        else:
            rest_cols_frame = protein_df[self.categorycolumns + [self.quantified_entity]]
        rest_cols_frame = rest_cols_frame.set_index(self.quantified_entity)#make identical index
        protein_quant = protein_quant.set_index(self.quantified_entity)          #make identical index
        
        imputer = KNNImputer(n_neighbors=self.n_neighbors_row_clustering,weights="distance")
        imputed_protein_quant = imputer.fit_transform(protein_quant)
        imputed_protein_quant = pd.DataFrame(imputed_protein_quant,columns = protein_quant.columns,index = protein_quant.index)
        imputed_data = imputed_protein_quant.join(rest_cols_frame)
        #print(list(imputed_protein_quant.index))
        
        self.full_imputed_data = imputed_data.reset_index()
        
    def validate_for_visualization_ANOVA(self,max_ANOVA,mindiff):
        """
        Mark valid rows by simple ANOVA with arbitrary cutoffs.

        Parameters
        ----------
        max_ANOVA : float
            Maximum ANOVA p-value (uncorrected!) for entity to be considered valid 
            (having reasonable inter-group change).
        mindiff : float
            Minimum difference for entity to be considered valid 
            (having reasonable inter-group change)..

        Returns
        -------
        None.

        """
        self.raw_quant_data['Valid Row'] = self.raw_quant_data.apply(MergingAnalysis.__validate_anova,args = [self.conditions,max_ANOVA,mindiff],axis =1)
   
    def validate_for_visualization_list(self,proteinlist):
        """
        Mark valid rows using an external list of valid entities.

        Parameters
        ----------
        proteinlist : str or list
            If str, a file with one valid entity per line is expected. 
            If list use as list of valid entities

        Returns
        -------
        None.

        """
        if isinstance(proteinlist,str):
            proteinlist = open(proteinlist,'r').read().splitlines()
            
        self.raw_quant_data['Valid Row'] = self.raw_quant_data.apply(MergingAnalysis.__validate_list,args = [self.quantified_entity,proteinlist],axis =1)

    def take_items(self,categories_to_plot,category_type):
        """
        Take list of categories of interest from filtered frame or file.

        Parameters
        ----------
        categories_to_plot : str
            If str is a key in MergingAnalysis.filtered_frames, will extract list of categories from filtered frame.
            Else, will attempt to open and read the file containing one category per line
        category_type : str
            Column name in which categories will be located.
       
        Returns
        -------
        None.
        """
        if categories_to_plot in self.filtered_frames:
            itemlist = list(self.filtered_frames[categories_to_plot][categories_to_plot])
            itemlist = [x.split(';') for x in itemlist]
            itemlist = list(it.chain.from_iterable(itemlist))
        else:
            itemlist = open(categories_to_plot,'r').read().splitlines()
        self.plotting_categories = {category_type:itemlist}
        
        
    def __clustermap_single(self,data,data_imputed,category,normalize_mean=True,z_score=False,plottitle = None,xlab_size=10,ylab_size=10,font_size=10,figsize=None,**kwargs):
        """
        Draw heatmap of single category in entity level.

        Parameters
        ----------
        data : DataFrame
            Slice of non-imuted dataframe.
        data_imputed : DataFrame
            Slice of non-imuted dataframe.
        category : str
            categroy name, will be used as default plot title.
        normalize_mean : bool, optional
            whether to normalize row means. The default is True.
        z_score : bool, optional
            whether to z-score rows. The default is False.
        plottitle : str, optional
            plot title if different than category. The default is None.
        xlab_size : float, optional
            x label size. The default is 10.
        ylab_size : float, optional
            y label size. The default is 10.
        font_size : float, optional
            font size. The default is 10.
        figsize : tuple (float,float), optional
            figure size. The default is None.
        **kwargs : key, value mappings
            Arguments passed to seaborn.clustermap.

        Returns
        -------
        cluster : Seaborn ClusterGrid instance
            Resulting heatmap.

        """
        kwargs.setdefault('cmap', "coolwarm")
        kwargs.setdefault('metric', "correlation")
        kwargs.setdefault('col_cluster', False)
        kwargs.setdefault('dendrogram_ratio', (0,0.1))
        kwargs.setdefault('colors_ratio', (0.05))
        kwargs.setdefault('cbar_pos', (0,1,0.05,0.2))
        kwargs.setdefault('tree_kws', {"linewidths":0})
        
        numrows = data.shape[0]
        if not plottitle:
            plottitle = category
        if not figsize:
            figsize = (16,(numrows/2)+3)

            
        
        transform = lambda x: x[0:70] if len(x)>70 else x
        plottitle = transform(plottitle)
        cols = list(data.columns)
        print(cols)
        cols.remove('Valid Row')
        if self.description_column == self.quantified_entity:
            cols.remove(self.quantified_entity)
        for_clustering = data[cols]
        if normalize_mean:
            for_clustering[self.quantcolumns] = for_clustering[self.quantcolumns].subtract(for_clustering[self.quantcolumns].mean(axis=1),axis=0)
        if z_score:
            kwargs.setdefault('z_score',0)
        else:
            kwargs.setdefault('z_score',None)
        matrix = data_imputed[self.quantcolumns]
        matrix = np.array(matrix)
        dist_matrix = distance(matrix)
        link = linkage(dist_matrix)
        data[self.description_column] = data[self.description_column].apply(MergingAnalysis.__linebreaker,args = [50])
        sns.plotting_context(rc={'xtick.labelsize': xlab_size,'ytick.labelsize': ylab_size,"font.size" :font_size})
        
        cluster = sns.clustermap(for_clustering,figsize = figsize,row_colors = data['Valid Row'],row_linkage = link,yticklabels = data[self.description_column],**kwargs)
        
        cluster.ax_heatmap.set_title(plottitle)
        cluster.ax_heatmap.set_xlabel('Replicate',fontweight='bold',fontsize=xlab_size)
        cluster.ax_heatmap.set_ylabel('Protein',fontweight='bold',fontsize=ylab_size)
        cluster.ax_heatmap.set_title(cluster.ax_heatmap.get_title(),fontweight='bold',fontsize=font_size)
        cluster.ax_heatmap.set_xticklabels(cluster.ax_heatmap.get_xticklabels(), rotation = 45,fontsize=font_size)
        cluster.ax_heatmap.set_yticklabels(cluster.ax_heatmap.get_yticklabels(), fontsize = font_size)
        cluster.ax_row_colors.set_xticklabels(cluster.ax_row_colors.get_xticklabels(), fontsize = font_size)
        return cluster
    
    
    def plot_single_categories(self,toplot=None,normalize_mean=True,z_score=False,plottitle = None,xlab_size=10,ylab_size=10,font_size=10,figsize=None,**kwargs):
        """
        Plot defined set of categories.

        Parameters
        ----------
        toplot : dict, optional
            dictionary of form category column name : list of categories. The default is None and 
            will use self.plotting_categories
        normalize_mean : bool, optional
            whether to normalize row meand. The default is True.
        z_score : bool, optional
            whether to z-score rows. The default is False.
        plottitle : str, optional
            plot title if different from names of categories. The default is None.
        xlab_size : float, optional
            x label size. The default is 10.
        ylab_size : float, optional
            y label size. The default is 10.
        font_size : float, optional
            font size. The default is 10.
        figsize : tuple (float,float), optional
            figure size. The default is None.
        **kwargs : key, value mappings
            Arguments passed to seaborn.clustermap.

        Returns
        -------
        None.

        """
        if not toplot:
            toplot = self.plotting_categories
        
        category_type = list(toplot.keys())[0]
        categories = toplot[category_type]
        try:
            os.mkdir(category_type)
        except FileExistsError:
            pass
        for category in categories:
            chunk = MergingAnalysis.__takechunk(self, category, category_type)
            category_name = list(chunk.keys())[0]
            data = chunk[category_name][0]
            data_imputed = chunk[category_name][1]
            single_map = MergingAnalysis.__clustermap_single(self, data, data_imputed, category_name, normalize_mean=True,
                                                             z_score=False,plottitle = None,xlab_size=10,ylab_size=10,font_size=10,
                                                             figsize=None,**kwargs)

            filename = category_name.replace(' ','_')
            if len(filename) > 50:
                filename = filename[0:49]
            filename = "".join( x for x in filename if not x in invalid_characters)
            filename = filename + '.png'
            single_map.savefig(filename)
     

    
            
            
        
        

        
