import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from math import isnan
import pandas as pd
from allExpressions_NL_Loc1_HO_ALL_MODELS import getFuncs as funcs_Loc1
from allExpressions_NL_Loc2_HO_ALL_MODELS import getFuncs as funcs_Loc2
from allExpressions_NL_Rest_H_O_All import getFuncs as funcs_Rest

import warnings
warnings.filterwarnings('ignore')

nested_dict = lambda: defaultdict(nested_dict)

data_root = '/Volumes/ETHAN-SCRAT/MainExpRawData/Main_Experiment'


subjects = ['AL98', 'AQ18', 'BT93', 'CA83', 'CH03', 'CU97', 'HN95', 'KP91',
            'QB90', 'QJ87']

runs = ['Loc1', 'Loc2']

prt_root = '/Users/ethan/Dropbox/Share_Kevin_Ethan/Main/PRTs_NoButton'

data_avg = nested_dict()

csv_folder = './CSVs/'

# Load the localizer runs
for sub in subjects:
    for run in runs:
        frame = pd.read_csv(csv_folder + '{}_{}_all_H-O.csv'.format(sub,run), header=None)
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

    for run in runs:
        # data = []
        # for roi in data_avg[sub][run]:
        #     data.append(data_avg[sub][run][roi][0])
        #     print(roi, data_avg[sub][run][roi][0].shape)
        # data.append(predictors[run]['face'])
        # data = np.array(data)
        # print(data.shape)
        # data = scaler.fit_transform(data.T).T
        # data_frame = pd.DataFrame(data.T)
        # data_frame.to_csv('./CSVs/{}_{}_{}_{}.csv'.format(sub, run, 'face', 'H-O'), header=False, index=False)

        # Fix problem with missing ROIs

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
# Within Subject Gen Scores - NONLINEAR
best_nl_scores = []
for sub in subjects:
    gen_ls = []
    #NONLINEAR
    model_run = 'Loc1'
    data_run = 'Loc2'
    y = predictors[data_run]['all']
    y_transformed = scaler.fit_transform(y.reshape(-1, 1))

    model_scores = []
    for model_index in NL_funcs[sub]['Loc1']:
        predicted = predict_NL(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run,
                               model_index=model_index)
        score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
        if not isnan(score):
            model_scores.append(score)
    gen_ls.append(np.array(model_scores))
    all_scores = []
    for ls in gen_ls:
        for val in ls:
            all_scores.append(val)
    all_scores = np.array(all_scores)
    best_nl_scores.append(np.max(all_scores))
    plt.hist(all_scores, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # LINEAR
    model_run = 'Loc1'
    data_run = 'Loc2'
    y = predictors[data_run]['all']
    y_transformed = scaler.fit_transform(y.reshape(-1, 1))
    predicted = predict_L(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run)
    linear_score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]

    # plt.axvline(linear_score, color='black')
    # plt.title('Linear R^2: {:.2f}  Best Non-Linear R^2: {:.2f}'.format(linear_score, np.max(all_scores)))
    # plt.savefig('./WithinSubImages/{}-Loc1-Loc2-H-O.png'.format(sub))
    # plt.figure()
    # OTHER WAY AROUND NOW...

    gen_ls = []
    # NONLINEAR
    model_run = 'Loc2'
    data_run = 'Loc1'
    y = predictors[data_run]['all']
    y_transformed = scaler.fit_transform(y.reshape(-1, 1))

    model_scores = []
    for model_index in NL_funcs[sub]['Loc2']:
        predicted = predict_NL(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run,
                               model_index=model_index)
        score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
        if not isnan(score):
            model_scores.append(score)
    gen_ls.append(np.array(model_scores))
    all_scores = []
    for ls in gen_ls:
        for val in ls:
            all_scores.append(val)
    all_scores = np.array(all_scores)
    best_nl_scores.append(np.max(all_scores))
    plt.hist(all_scores, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # LINEAR
    model_run = 'Loc2'
    data_run = 'Loc1'
    y = predictors[data_run]['all']
    y_transformed = scaler.fit_transform(y.reshape(-1, 1))
    predicted = predict_L(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run)
    linear_score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]

    # plt.axvline(linear_score, color='black')
    # plt.title('Linear R^2: {:.2f}  Best Non-Linear R^2: {:.2f}'.format(linear_score, np.max(all_scores)))
    # plt.savefig('./WithinSubImages/{}-Loc2-Loc1-H-O.png'.format(sub))
    # plt.figure()




gen_array_nl = np.array(best_nl_scores)
gen_score = np.average(gen_array_nl)
gen_stdev = np.std(gen_array_nl)
print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))

###############################################################################################

###############################################################################################
# Within Subject Gen Scores - LINEAR
gen_ls = []

for sub in subjects:
    model_run = 'Loc1'
    data_run = 'Loc2'
    y = predictors[data_run]['all']
    y_transformed = scaler.fit_transform(y.reshape(-1, 1))
    predicted = predict_L(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run)
    gen_ls.append(scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0])
for sub in subjects:
    model_run = 'Loc2'
    data_run = 'Loc1'
    y = predictors[data_run]['all']
    y_transformed = scaler.fit_transform(y.reshape(-1, 1))
    predicted = predict_L(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run)
    gen_ls.append(scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0])
    # plt.plot(y_transformed)
    # plt.plot(predicted)
    # plt.show()

gen_array_l = np.array(gen_ls)
gen_score = np.average(gen_array_l)
gen_stdev = np.std(gen_array_l)
print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
###############################################################################################

# Are they different????
print(scipy.stats.ttest_ind(gen_array_l, gen_array_nl))




