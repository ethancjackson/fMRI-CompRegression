import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from math import isnan
import pandas as pd
from allExpressions_NL_Loc1_TalROI_ALL_MODELS import getFuncs as funcs_Loc1
from allExpressions_NL_Loc2_TalROI_ALL_MODELS import getFuncs as funcs_Loc2
from allExpressions_NL_Rest_TalROI import getFuncs as funcs_Rest

import warnings
warnings.filterwarnings('ignore')

nested_dict = lambda: defaultdict(nested_dict)

data_root = '/Volumes/ETHAN-SCRAT/MainExpRawData/Main_Experiment'


subjects = ['AL98', 'AQ18', 'BT93', 'CA83', 'CH03', 'CU97', 'HN95', 'KP91',
            'QB90', 'QJ87']

valid_subjects = ['QQ92', 'RF86', 'RZ96', 'UQ89']

runs = ['Loc1', 'Loc2']

prt_root = '/Users/ethan/Dropbox/Share_Kevin_Ethan/Main/PRTs_NoButton'

data_avg = nested_dict()

csv_folder = './CSVs/'

# Load the localizer runs
for sub in subjects + valid_subjects:
    for run in runs:
        frame = pd.read_csv(csv_folder + '{}_{}_face_TalROI.csv'.format(sub,run), header=None)
        data_avg[sub][run] = frame.as_matrix().T[:48,:]

# Load the resting state run
rest_frame = pd.read_csv(csv_folder + 'IK82_Rest_rest_H-O.csv', header=None)
data_rest = rest_frame.as_matrix().T[:48,:]
data_avg['IK82']['Rest'] = data_rest

def generate_predictor_loc(run, body=1, face=1, hand=1, scrambled=1, constant=0):
    sum_list = []
    dm_file = open('./{} DM.sdm'.format(run), 'r')
    for line in dm_file:
        line = line.strip(' \n').split()
        line = list(map(lambda x: float(x), line))
        sum_list.append(line[0]*body + line[1]*face + line[2]*hand + line[3]*scrambled)
    dm_file.close()
    return np.array(sum_list)

# Create predictor time series
predictors = nested_dict()
predictors['Loc1']['face'] = generate_predictor_loc('Loc1',0,1,0,0)
predictors['Loc1']['all'] = generate_predictor_loc('Loc1',1,1,1,1)
predictors['Loc2']['face'] = generate_predictor_loc('Loc2',0,1,0,0)
predictors['Loc2']['all'] = generate_predictor_loc('Loc2',1,1,1,1)
predictors['Rest']['face'] = generate_predictor_loc('Loc2',0,1,0,0)
predictors['Rest']['all'] = generate_predictor_loc('Loc2',1,1,1,1)

scaler = MinMaxScaler()

args = nested_dict()
NL_funcs = nested_dict()

for sub in subjects:
    func_list_1 = funcs_Loc1()[subjects.index(sub)]
    for i in range(len(func_list_1)):
        NL_funcs[sub]['Loc1'][i] = func_list_1[i]

    func_list_2 = funcs_Loc2()[subjects.index(sub)]
    for i in range(len(func_list_2)):
        NL_funcs[sub]['Loc2'][i] = func_list_2[i]

for sub in subjects + valid_subjects:
    for run in runs:
        # Fill dictionary with variables for NL models
        for roi in range(50):
            if roi < 48:
                ts = data_avg[sub][run][roi]
                for t in range(340):
                    args[sub][run][t]['v{}'.format(roi)] = ts[t]
            else:
                for t in range(340):
                    args[sub][run][t]['v{}'.format(roi)] = 0.0

sub = 'IK82'
run = 'Rest'
# Fill dictionary with variables for NL models
for roi in range(50):
    if roi < 48:
        ts = data_avg[sub][run][roi]
        for t in range(340):
            args[sub][run][t]['v{}'.format(roi)] = ts[t]
    else:
        for t in range(340):
            args[sub][run][t]['v{}'.format(roi)] = 0.0
NL_funcs[sub]['Rest'] = funcs_Rest()[0][0]


def predict_NL(model_sub, data_sub, model_run, data_run, model_index):
    x = []
    for i in range(340):
        try:
            value = NL_funcs[model_sub][model_run][model_index](**args[data_sub][data_run][i])
            if not isnan(value):
                x.append(value)
            else:
                x.append(0)
        except OverflowError:
            x.append(1.0)
        except (ZeroDivisionError, ValueError):
            x.append(0.0)
    return np.array(x)

def predict_L(model_sub, data_sub, model_run, data_run):
    data = data_avg[model_sub][model_run]
    scaler = MinMaxScaler()
    x_transformed = scaler.fit_transform(data.T).T
    y_transformed = scaler.fit_transform(predictors[model_run]['all'].reshape(-1, 1)).T
    lin_regression = Ridge(alpha=1.0)
    lin_regression.fit(x_transformed.T, y_transformed[0].T)
    return lin_regression.predict(data_avg[data_sub][data_run].T)

###############################################################################################
###############################################################################################
###############################################################################################

# for each subject, find the NL model that had the best perf on its other run
NL_best_models = nested_dict()
best_nl_model_indices = nested_dict()
best_nl_scores = []

model_run = 'Loc1'
data_run = 'Loc2'
y = predictors[data_run]['face']
y_transformed = scaler.fit_transform(y.reshape(-1, 1))
for sub in subjects:
    model_scores = []
    for model_index in NL_funcs[sub][model_run]:
        predicted = predict_NL(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run,
                               model_index=model_index)
        score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
        if not isnan(score):
            model_scores.append(score)

    best_model_index = np.argmax(model_scores)
    best_nl_model_indices[sub][model_run][data_run] = best_model_index

    all_scores = np.array(model_scores)
    best_nl_scores.append(np.max(all_scores))


model_run = 'Loc2'
data_run = 'Loc1'
y = predictors[data_run]['face']
y_transformed = scaler.fit_transform(y.reshape(-1, 1))
for sub in subjects:
    model_scores = []
    for model_index in NL_funcs[sub][model_run]:
        predicted = predict_NL(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run,
                               model_index=model_index)
        score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
        if not isnan(score):
            model_scores.append(score)

    best_model_index = np.argmax(model_scores)
    best_nl_model_indices[sub][model_run][data_run] = best_model_index

    all_scores = np.array(model_scores)
    best_nl_scores.append(np.max(all_scores))




###############################################################################################
###############################################################################################
###############################################################################################
# Hold on, we're doing validation!

# get output from all main subjects models

# NONLINEAR
nl_valid_predictions_best = []
nl_valid_scores_best = []

# plt.plot(y)
y = scaler.fit_transform(predictors['Loc2']['face'].reshape(-1, 1))
y_transformed = scaler.fit_transform(y.reshape(-1, 1))
for sub_v in valid_subjects:
    avg_output = []
    for sub in subjects:
        predicted = predict_NL(sub, sub_v, 'Loc1', 'Loc2', best_nl_model_indices[sub]['Loc1']['Loc2'])
        if predicted.shape == (340,):
            avg_output.append(predicted)
    avg_output = np.average(np.array(avg_output), axis=0)
    nl_valid_predictions_best.append(avg_output)
    score = scipy.stats.pearsonr(avg_output, y_transformed[:, 0].T)[0]
    nl_valid_scores_best.append(score)
    plt.plot(avg_output, alpha=0.2)

y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1, 1))
y_transformed = scaler.fit_transform(y.reshape(-1, 1))
for sub_v in valid_subjects:
    avg_output = []
    for sub in subjects:
        predicted = predict_NL(sub, sub_v, 'Loc2', 'Loc1', best_nl_model_indices[sub]['Loc2']['Loc1'])
        if predicted.shape == (340,):
            avg_output.append(predicted)
    avg_output = np.average(np.array(avg_output), axis=0)
    nl_valid_predictions_best.append(avg_output)
    score = scipy.stats.pearsonr(avg_output, y_transformed[:, 0].T)[0]
    nl_valid_scores_best.append(score)
    plt.plot(avg_output, alpha=0.2)

plt.title('Average Model Validation - Non-Linear')
plt.show()
nl_valid_predictions_best = np.array(nl_valid_predictions_best)
nl_valid_scores_best = np.array(nl_valid_scores_best)








# Between Subject Gen Matrix - NONLINEAR
pearson_matrix_1 = np.zeros((len(subjects), len(valid_subjects)))

for i in range(len(subjects)):
    for j in range(len(valid_subjects)):

        y = predictors['Loc2']['face']
        y_transformed = scaler.fit_transform(y.reshape(-1, 1))

        predicted = predict_NL(model_sub=subjects[i], data_sub=valid_subjects[j], model_run='Loc2', data_run='Loc1',
                               model_index=best_nl_model_indices[subjects[i]]['Loc2']['Loc1'])
        # print(predicted)

        score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
        if not isnan(score):
            pearson_matrix_1[i, j] = score
        else:
            pearson_matrix_1[i, j] = 0.0

gen_ls = []
for i in range(len(subjects)):
    for j in range(len(valid_subjects)):
        gen_ls.append(pearson_matrix_1[i,j])

pearson_matrix_2 = np.zeros((len(subjects), len(valid_subjects)))

for i in range(len(subjects)):
    for j in range(len(valid_subjects)):

        y = predictors['Loc1']['face']
        y_transformed = scaler.fit_transform(y.reshape(-1, 1))

        predicted = predict_NL(model_sub=subjects[i], data_sub=valid_subjects[j], model_run='Loc1', data_run='Loc2',
                               model_index=best_nl_model_indices[subjects[i]]['Loc1']['Loc2'])
        # print(predicted)

        score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
        if not isnan(score):
            pearson_matrix_2[i, j] = score
        else:
            pearson_matrix_2[i, j] = 0.0

for i in range(len(subjects)):
    for j in range(len(valid_subjects)):
        gen_ls.append(pearson_matrix_2[i,j])

gen_array_nl_best = np.array(gen_ls)



















