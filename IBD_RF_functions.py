from sklearn import tree
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os

import seaborn as sns
from matplotlib import pyplot as plt

import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h

from os.path import exists, join
from os import makedirs

def make_directory_if_not_exist(directory_to_make):
	if exists(directory_to_make)==False:
		makedirs(directory_to_make)
		print("Created {}".format(directory_to_make))
	else:
		print("{} already exists.".format(directory_to_make))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_test_random_forest_model(data_no_target_column_df,
								   target_column_series,
								   train_proportion = 0.5,
								   params_dict = None,
								   #n_estimators = 200,
								   #max_depth = None,
								   random_state = 42,
								   n_jobs=12):
	#data_no_target_column_df = data[[c for c in data.columns if c != target_column]]
	X_train, X_test, y_train, y_test = train_test_split(data_no_target_column_df,
														target_column_series,
														random_state=random_state,
														train_size=train_proportion)
	rf = RandomForestClassifier()
	if params_dict is not None:
		if "n_jobs" not in params_dict:
			params_dict["n_jobs"] = n_jobs
		rf = RandomForestClassifier(**params_dict) 
	rf = rf.fit(X_train, y_train)
	model_score = rf.score(X_test, y_test)
	feature_importances_df = pd.DataFrame({"feature":list(data_no_target_column_df.columns),
										   "importance":rf.feature_importances_}).set_index("feature")
	feature_importances_df = feature_importances_df.sort_values(by="importance",ascending=False)
	return {"model_score":model_score,"feature_importances":feature_importances_df}
	


def repeatedly_train_test_random_forest_model(data_no_target_column_df,
											  target_column_series,
											  n_iterations=1,
											  train_proportion = 0.5,
											  params_dict = {"max_depth":None,"n_estimators":200},
											  n_jobs=12,
											  random_state=42
											 ):
	feature_importance_df_lst = []
	scores_lst = []
	for iteration_number in range(0,n_iterations):
		rf_results_dict = train_test_random_forest_model(data_no_target_column_df,
														 target_column_series,
														 train_proportion = train_proportion,
														 params_dict = params_dict,
														 #n_estimators = n_estimators,
														 #max_depth = max_depth,
														 n_jobs=n_jobs,
														 random_state=random_state)
		scores_lst.append(rf_results_dict["model_score"])
		feature_importance_df_lst.append(rf_results_dict["feature_importances"])
		print("Done with iteration {}/{}".format(iteration_number+1,n_iterations))
	scores_series= pd.Series(scores_lst)
	feature_importances_df = pd.concat(feature_importance_df_lst,axis=1)
	return scores_series, feature_importances_df
	
def train_test_logistic_regression_model(data_no_target_column_df,
								   target_column_series,
								   train_proportion = 0.5,
								   params_dict = None,
								   n_jobs=12,
								   #random_state=42
								   ):
	#data_no_target_column_df = data[[c for c in data.columns if c != target_column]]
	X_train, X_test, y_train, y_test = train_test_split(data_no_target_column_df,
														target_column_series,
														#random_state=random_state,
														train_size=train_proportion)
	lr = LogisticRegression()
	if params_dict is not None: 
		if "n_jobs" not in params_dict:
			params_dict["n_jobs"] = n_jobs
		lr = LogisticRegression(**params_dict) 
	lr = lr.fit(X_train, y_train)
	model_score = lr.score(X_test, y_test)
	feature_coeffs_df = pd.DataFrame({"feature":list(data_no_target_column_df.columns),
										"coeff":lr.coef_[0]}).set_index("feature")
	feature_coeffs_df = feature_coeffs_df.sort_values(by="coeff",ascending=False)
	return {"model_score":model_score,"feature_coeffs":feature_coeffs_df}
	
def repeatedly_train_test_logistic_regression_model(data_no_target_column_df,
													target_column_series,
													n_iterations=1,
													train_proportion = 0.5,
													params_dict = None,
													n_jobs=12,
													random_state=42
													):
	feature_coeff_df_lst = []
	scores_lst = []
	for iteration_number in range(0,n_iterations):
		rf_results_dict = train_test_logistic_regression_model(data_no_target_column_df,
														 target_column_series,
														 train_proportion = train_proportion,
														 params_dict = params_dict,
														 n_jobs=n_jobs,
														 #random_state=random_state
														 )
		scores_lst.append(rf_results_dict["model_score"])
		feature_coeff_df_lst.append(rf_results_dict["feature_coeffs"])
		print("Done with iteration {}/{}".format(iteration_number+1,n_iterations))
	scores_series= pd.Series(scores_lst)
	feature_coeff_df = pd.concat(feature_coeff_df_lst,axis=1)
	return scores_series, feature_coeff_df

'''def train_test_random_forest_different_n_estimators(data_no_target_column_df,
													 target_column_series,
													 max_n_iterations=1,
													 train_proportion = 0.5,
													 params_dict = {"max_depth":None},
													 max_n_estimators=10,
													 #max_depth = None,
													 n_jobs=12,
													 random_state=42):
	scores_per_n_estimators_df_dict = dict()
	lst_feature_importances = []
	
	iterate_over_this = []
	
	if isinstance(max_n_estimators, list):
		iterate_over_this = max_n_estimators
	else:
		iterate_over_this = range(1,max_n_estimators+1)
	for n_estimators in iterate_over_this:
		if n_estimators not in scores_per_n_estimators_df_dict:
			scores_per_n_estimators_df_dict[n_estimators] = []
		n_estimators_params_dict = {k:params_dict[k] for k in params_dict}
		#if 'n_estimators' not in n_estimators_params_dict:
		n_estimators_params_dict["n_estimators"] = n_estimators
		all_columns_score_series, all_columns_feature_importances_df = repeatedly_train_test_random_forest_model(
		data_no_target_column_df,
		target_column_series,
		n_iterations=max_n_iterations,
		train_proportion = train_proportion,
		#n_estimators = n_estimators,
		#max_depth = max_depth, 
		params_dict =  n_estimators_params_dict,
		n_jobs=n_jobs,
		random_state=random_state)
		scores_per_n_estimators_df_dict[n_estimators] = all_columns_score_series
		lst_feature_importances.append(all_columns_feature_importances_df)
		print("Done with {}/{} estimators.".format(n_estimators,max_n_estimators))
	scores_per_n_estimators_df = pd.DataFrame(scores_per_n_estimators_df_dict)
	return scores_per_n_estimators_df, lst_feature_importances

def plot_n_estimators_vs_score(score_df,title="Effect of Number of Estimators"):
	long_form_dict = {"n_estimators":[],"iteration_number":[],"score":[]}
	for c in score_df.columns:
		n_classifiers_scores_column_series = list(score_df[c])
		iteration_numbers = list(score_df[c].index)
		long_form_dict["n_estimators"].extend([c]*len(n_classifiers_scores_column_series))
		long_form_dict["iteration_number"].extend(iteration_numbers)
		long_form_dict["score"].extend(n_classifiers_scores_column_series)
	long_form_df = pd.DataFrame(long_form_dict)
	fig, ax = plt.subplots(figsize=(15,10))
	sns.lineplot(data=long_form_df,x="n_estimators",y="score")
	plt.title(title)
	plt.ylabel('Model Score')
	plt.xlabel('Number of Estimators')
	plt.show()'''