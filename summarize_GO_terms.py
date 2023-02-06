# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:38:42 2022


functional interface to Gene Ontology Category summarizer
A lot of stuff is suboptimal

@author: APirog
"""
import pandas as pd
import os
import category_merging_to_import
import visualize_category_to_import

paramfile = 'param.txt'

linedict = {}
lines = open(paramfile,'r').read().splitlines()
for line in lines:
    print(line)
    if line.startswith('#'):
        pass
    else:
        line = line.split('\t')
        linedict[line[0]] = line[1]
    
inputfile = linedict['inputfile']

categorycolumn = linedict['categorycolumn']

quantcolumns = linedict['quantcolumns'].split(';')

numcolumns = linedict['numcolumns'].split(';')

textcolumns = linedict['textcolumns'].split(';')

conditions = linedict['conditions'].split(';')
cond_dict = {}
for item in conditions:
    name = item.split(':')[0]
    conditions=item.split(':')[1]
    conditions = conditions.split(',')
    cond_dict[name] = conditions
conditions = cond_dict

comparisons = linedict['comparisons'].split(';')
comparisons = [x.split(',') for x in comparisons]

renamer = linedict['renamer'].split(';')
ren_dict = {}
for item in renamer:
    ren_dict[item.split(':')[0]] = item.split(':')[1]
renamer = ren_dict

quantified_entity = linedict['quantified_entity']

max_valid_for_merging = float(linedict['max_valid_for_merging'])

weight_name = linedict['weigths_type_for_concatenation']
if weight_name == 'None':
    weight_name = None

characterizer_list = linedict['characterizers'].split(';')
group_merging_column = linedict['group_merging_column']
group_filtering_column = linedict['group_filtering']
max_categorywise_anova = float(linedict['max_categorywise_anova'])
minimum_categorywise_difference = float(linedict['minimum_categorywise_difference'])
minimum_entities = int(linedict['minimum_entities_per_category'])
fig_x_lab= float(linedict['x_label_size'])
fig_y_lab=float(linedict['y_label_size'])
fig_x_size = float(linedict['x_figure_size'])
fig_y_size = float(linedict['y_figure_size'])

global_only = linedict['global_only']
if global_only == 'True':
    global_only = True
else:
    global_only = False
    
single_only = linedict['single_only']
if single_only == 'True':
    single_only = True
else:
    single_only = False
    
categorylist = linedict['categorylist']
if categorylist == 'None':
    categorylist = None
    
normalize_means = linedict['normalize_means_for_map']
if normalize_means == 'True':
    normalize_means = True
else:
    normalize_means = False
    
merge_groups = linedict['merge_groups']
if merge_groups == 'True':
    merge_groups = True
else:
    merge_groups = False

    
description_column = linedict['description_column']
max_individual_ANOVA = float(linedict['max_individual_ANOVA'])
min_individual_difference = float(linedict['min_individual_difference'])
minobs_before_knn = int(linedict['minobs_before_knn'])
mingroups_before_knn = int(linedict['mingroups_before_knn'])
n_neighbors_category_analysis = int(linedict['n_neighbors_category_analysis'])
n_neighbors_row_clustering = int(linedict['n_neighbors_row clustering'])

print(inputfile)
print(categorycolumn)
print(quantcolumns)
print(numcolumns)
print(textcolumns)
print(conditions)
print(comparisons)
print(renamer)
print(quantified_entity)
print(max_valid_for_merging)
print(weight_name)
print(characterizer_list)
print(group_merging_column)
print(group_filtering_column)
print(max_categorywise_anova)
print(minimum_categorywise_difference)
print(minimum_entities)
print(fig_x_lab)
print(fig_y_lab)
print(fig_x_size)
print(fig_y_size)

wd=os.getcwd()

listname = None
'''todo add minobs mingroups arugument for valid row filtering before imputation'''
"""why the hell double modifiedsequence???????"""
#Here filter for binder peptides
complete = pd.read_csv(inputfile, sep = '\t')
filtered  = complete[complete['Any_Weak_Binder'] == True]
filtered.to_csv('binders_only_' + inputfile, sep = '\t', index=False)
inputfile = 'binders_only_' + inputfile


if not single_only:

    plot,list_entries,dataframe_selected, dataframe_all = category_merging_to_import.perform_grouping(inputfile,categorycolumn,quantcolumns,numcolumns,textcolumns,conditions,comparisons,\
                     renamer,quantified_entity,max_valid_for_merging,weight_name,characterizer_list,\
                    group_merging_column,group_filtering_column,max_categorywise_anova,minimum_categorywise_difference,\
                       minimum_entities,fig_x_lab,fig_y_lab,fig_x_size,fig_y_size,minobs_before_knn,\
                           mingroups_before_knn,n_neighbors_category_analysis,normalize_means, merge_groups)
    os.chdir(wd)
    listname = categorycolumn+'_chosen_categories_.txt'
    with open(listname,'w') as output:
        for category in list_entries:
            output.write(categorycolumn+'\t'+category+'\n')
    plot.savefig(categorycolumn+'.png')
    dataframe_selected.to_csv('category_mean_selected_'+categorycolumn+'_.tsv',sep='\t',index=False)
    dataframe_all.to_csv('category_mean_all_'+categorycolumn+'_.tsv',sep='\t',index=False)

if not global_only:
    if listname == None and categorylist == None:
        print('No IDs to graph!')
    elif listname == None and categorylist != None:
        listname = categorylist
    elif listname !=None and categorylist != None:
        listname = categorylist
    print(listname)    
    visualize_category_to_import.visualize_invidual(inputfile,quantified_entity,listname,quantcolumns,\
                                                    conditions,description_column,max_individual_ANOVA,\
                                                    min_individual_difference,n_neighbors_row_clustering)
    
    




