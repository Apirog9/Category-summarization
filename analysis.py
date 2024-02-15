# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:34:37 2024

@author: Proccesing_PC2


Short manual.
At first step, prepare quantitative data in subsequent format:
    - per-entity(protein group, gene) quantification columns, one per sample
    - column with entities(unique, quantified entities)
    - at least one column with category names annotated to entities, ';' separated
    - optionally column with verbose entity name (not need to be unique)
    - optionally a list of entities considered significantly regulated
    - param file as presented in example param.txt - or set all attributes manually (default param.txt stored in execution directory)


The tool is rather heuristic and aimed and data exploration, thus do not provide significance analysis of any kind.
Id is rather designed to improve searching for relevant terms/processes in the dataset, which may be omitted otherwise.

1.) create MergingAnalysis object by:
    analysis = goanalysis.MergingAnalysis('datafile','paramfile')
2.) run data preparation by:
    analysis.prepare_data()
    it will filter out very poorly quantified entities, perform log scale if necessary and
    finally perform kNN imputation of prepared data, stored as dataframe in analysis.prepared_data
    kNN defaults provided in param file should be reasonable. 
    minobs/mingroups parameters should be adjustet so majority of the groups should contain minimum
    2-3 values for imputation.
    All parameters are take by default from parameter file, but may be also passed to function call
3.) Calculate ANOVA for all entities by:
    analysis.calculate_ANOVA_separate()
    This will serve as an approximation of entity probability of being regulated.
4.) Perform grouping on selected category column by:
    analysis.group_categories_simple('category column name')
    On this step it is necessary to decide if to use weighting. 
    Using weights = 'Anova_separate' will heavily prefer regulated proteins to contribute to overall category quantification, and
    even single protein will change the quantification. It will preferentially show "where are regulated proteins". Without weights, the 
    category quantification is more balanced, yet less sensitive to changes in expression.
5.) Calculate groupwise ANOVA by:
    analysis.calculate_ANOVA_group('category column name')
    
    
    
    


"""

import goanalysis
#%%
analysis = goanalysis.MergingAnalysis('GO_PCA_sota_hurdle_msqrob.tsv')
analysis.prepare_data()
analysis.calculate_ANOVA_separate()
analysis.group_categories_simple('F_name',weights = 'Anova_separate')
analysis.calculate_ANOVA_group('F_name')
#%%
analysis.filter_simple('F_name', 0.0001, 0.7, 4)
clustermap = analysis.clustermap('F_name', 16, 16, 8, 8,cmap='plasma')
#%%
analysis.impute_full()
analysis.validate_for_visualization_ANOVA(0.00001, 0.6)
analysis.take_items('F_name','F_name')
analysis.plot_single_categories(cmap='plasma')