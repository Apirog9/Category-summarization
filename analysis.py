# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:34:37 2024

@author: Proccesing_PC2
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
clustermap = analysis.clustermap('F_name', 16, 16, 8, 8)
#%%
analysis.impute_full()
analysis.validate_for_visualization_ANOVA(0.00001, 0.6)
analysis.take_items('F_name','F_name')
analysis.plot_single_categories()