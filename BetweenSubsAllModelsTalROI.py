import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from math import isnan
import pandas as pd
from allExpressions_NL_Loc1_TalROI import getFuncs as funcs_Loc1
from allExpressions_NL_Loc2_TalROI import getFuncs as funcs_Loc2
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
rest_frame = pd.read_csv(csv_folder + 'IK82_Rest_rest_TalROI.csv', header=None)
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
    # Fill dictionary with NL funcs
    NL_funcs[sub]['Loc1'] = funcs_Loc1()[subjects.index(sub)][0]
    NL_funcs[sub]['Loc2'] = funcs_Loc2()[subjects.index(sub)][0]

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


def predict_NL(model_sub, data_sub, model_run, data_run):
    x = []
    for i in range(340):
        try:
            value = NL_funcs[model_sub][model_run](**args[data_sub][data_run][i])
            if not isnan(value):
                if value > 1:
                    x.append(1.0)
                else:
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
    y_transformed = scaler.fit_transform(predictors[model_run]['face'].reshape(-1, 1)).T
    lin_regression = Ridge(alpha=1.0)
    lin_regression.fit(x_transformed.T, y_transformed[0].T)
    return lin_regression.predict(data_avg[data_sub][data_run].T)



# ###############################################################################################
# # Between Subject Gen Matrix - NONLINEAR
# pearson_matrix = np.zeros((len(subjects), len(subjects)))
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#
#         y = predictors['Loc2']['face']
#         y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#
#         predicted = predict_NL(model_sub=subjects[i], data_sub=subjects[j], model_run='Loc2', data_run='Loc2')
#         # print(predicted)
#
#         pearson_matrix[i, j] = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
#
# gen_ls = []
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#         if i != j:
#             gen_ls.append(pearson_matrix[i,j])
#
#
# # plt.matshow(pearson_matrix, vmin=0, vmax=1)
# # plt.title('Between Subject R^2 Scores\nLoc2, Non-Linear')
# # plt.colorbar()
# # plt.show()
#
# pearson_matrix = np.zeros((len(subjects), len(subjects)))
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#
#         y = predictors['Loc1']['face']
#         y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#
#         predicted = predict_NL(model_sub=subjects[i], data_sub=subjects[j], model_run='Loc1', data_run='Loc1')
#         # print(predicted)
#
#         score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
#         if isnan(score):
#             score = 0.0
#
#         pearson_matrix[i, j] = score
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#         if i != j:
#             gen_ls.append(pearson_matrix[i,j])
# gen_array_nl = np.array(gen_ls)
#
#
#
# gen_score = np.average(gen_array_nl)
# gen_stdev = np.std(gen_array_nl)
# print('Nonlinear: Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
#
# # plt.matshow(pearson_matrix, vmin=0, vmax=1)
# # plt.title('Between Subject R^2 Scores\nLoc1, Non-Linear')
# # plt.colorbar()
# # plt.show()
# ###############################################################################################
#
# ###############################################################################################
# # Between Subject Gen Matrix - LINEAR
# pearson_matrix = np.zeros((len(subjects), len(subjects)))
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#
#         y = predictors['Loc2']['face']
#         y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#
#         predicted = predict_L(model_sub=subjects[i], data_sub=subjects[j], model_run='Loc2', data_run='Loc2')
#         # print(predicted)
#
#         pearson_matrix[i, j] = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
#
# gen_ls = []
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#         if i != j:
#             gen_ls.append(pearson_matrix[i,j])
#
#
# # plt.matshow(pearson_matrix, vmin=0, vmax=1)
# # plt.title('Between Subject R^2 Scores\nLoc2, Linear')
# # plt.colorbar()
# # plt.show()
#
# pearson_matrix = np.zeros((len(subjects), len(subjects)))
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#
#         y = predictors['Loc1']['face']
#         y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#
#         predicted = predict_L(model_sub=subjects[i], data_sub=subjects[j], model_run='Loc1', data_run='Loc1')
#         # print(predicted)
#
#         pearson_matrix[i, j] = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#         if i != j:
#             gen_ls.append(pearson_matrix[i,j])
#
#
#
# gen_array_l = np.array(gen_ls)
#
# gen_score = np.average(gen_array_l)
# gen_stdev = np.std(gen_array_l)
# print('Linear Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
#
# # plt.matshow(pearson_matrix, vmin=0, vmax=1)
# # plt.title('Between Subject R^2 Scores\nLoc1, Linear')
# # plt.colorbar()
# # plt.show()
# ###############################################################################################
#
# # Are they different????
#
#
#

#
#
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import scipy
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import Ridge
# from math import isnan
# import pandas as pd
# from allExpressions_NL_Loc1_TalROI_ALL_MODELS import getFuncs as funcs_Loc1
# from allExpressions_NL_Loc2_TalROI_ALL_MODELS import getFuncs as funcs_Loc2
# from allExpressions_NL_Rest_TalROI import getFuncs as funcs_Rest
#
# import warnings
# warnings.filterwarnings('ignore')
#
# nested_dict = lambda: defaultdict(nested_dict)
#
# data_root = '/Volumes/ETHAN-SCRAT/MainExpRawData/Main_Experiment'
#
#
# subjects = ['AL98', 'AQ18', 'BT93', 'CA83', 'CH03', 'CU97', 'HN95', 'KP91',
#             'QB90', 'QJ87']
#
# runs = ['Loc1', 'Loc2']
#
# prt_root = '/Users/ethan/Dropbox/Share_Kevin_Ethan/Main/PRTs_NoButton'
#
# data_avg = nested_dict()
#
# csv_folder = './CSVs/'
# # Load the localizer runs
# for sub in subjects:
#     for run in runs:
#         frame = pd.read_csv(csv_folder + '{}_{}_face_TalROI.csv'.format(sub,run), header=None)
#         data_avg[sub][run] = frame.as_matrix().T[:48,:]
#
# # Load the resting state run
# rest_frame = pd.read_csv(csv_folder + 'IK82_Rest_rest_TalROI.csv', header=None)
# data_rest = rest_frame.as_matrix().T[:48,:]
# data_avg['IK82']['Rest'] = data_rest
#
# def generate_predictor_loc(run, body=1, face=1, hand=1, scrambled=1, constant=0):
#     sum_list = []
#     dm_file = open('./{} DM.sdm'.format(run), 'r')
#     for line in dm_file:
#         line = line.strip(' \n').split()
#         line = list(map(lambda x: float(x), line))
#         sum_list.append(line[0]*body + line[1]*face + line[2]*hand + line[3]*scrambled)
#     dm_file.close()
#     return np.array(sum_list)
#
# # Create predictor time series
# predictors = nested_dict()
# predictors['Loc1']['face'] = generate_predictor_loc('Loc1',0,1,0,0)
# predictors['Loc1']['all'] = generate_predictor_loc('Loc1',1,1,1,1)
# predictors['Loc2']['face'] = generate_predictor_loc('Loc2',0,1,0,0)
# predictors['Loc2']['all'] = generate_predictor_loc('Loc2',1,1,1,1)
# predictors['Rest']['face'] = generate_predictor_loc('Loc2',0,1,0,0)
# predictors['Rest']['all'] = generate_predictor_loc('Loc2',1,1,1,1)
#
# scaler = MinMaxScaler()
#
# args = nested_dict()
# NL_funcs = nested_dict()
#
# for sub in subjects:
#     func_list_1 = funcs_Loc1()[subjects.index(sub)]
#     for i in range(len(func_list_1)):
#         NL_funcs[sub]['Loc1'][i] = func_list_1[i]
#
#     func_list_2 = funcs_Loc2()[subjects.index(sub)]
#     for i in range(len(func_list_2)):
#         NL_funcs[sub]['Loc2'][i] = func_list_2[i]
#
#     for run in runs:
#         # data = []
#         # for roi in data_avg[sub][run]:
#         #     data.append(data_avg[sub][run][roi][0])
#         #     print(roi, data_avg[sub][run][roi][0].shape)
#         # data.append(predictors[run]['face'])
#         # data = np.array(data)
#         # print(data.shape)
#         # data = scaler.fit_transform(data.T).T
#         # data_frame = pd.DataFrame(data.T)
#         # data_frame.to_csv('./CSVs/{}_{}_{}_{}.csv'.format(sub, run, 'face', 'H-O'), header=False, index=False)
#
#         # Fix problem with missing ROIs
#
#         # Fill dictionary with variables for NL models
#         for roi in range(50):
#             if roi < 48:
#                 ts = data_avg[sub][run][roi]
#                 for t in range(340):
#                     args[sub][run][t]['v{}'.format(roi)] = ts[t]
#             else:
#                 for t in range(340):
#                     args[sub][run][t]['v{}'.format(roi)] = 0.0
#
# sub = 'IK82'
# run = 'Rest'
# # Fill dictionary with variables for NL models
# for roi in range(50):
#     if roi < 48:
#         ts = data_avg[sub][run][roi]
#         for t in range(340):
#             args[sub][run][t]['v{}'.format(roi)] = ts[t]
#     else:
#         for t in range(340):
#             args[sub][run][t]['v{}'.format(roi)] = 0.0
# NL_funcs[sub]['Rest'] = funcs_Rest()[0][0]
#
#
# def predict_NL(model_sub, data_sub, model_run, data_run, model_index):
#     x = []
#     for i in range(340):
#         try:
#             value = NL_funcs[model_sub][model_run][model_index](**args[data_sub][data_run][i])
#             if not isnan(value):
#                 x.append(value)
#             else:
#                 x.append(0)
#         except OverflowError:
#             x.append(1.0)
#         except (ZeroDivisionError, ValueError):
#             x.append(0.0)
#     return np.array(x)
#
# def predict_L(model_sub, data_sub, model_run, data_run):
#     data = data_avg[model_sub][model_run]
#     scaler = MinMaxScaler()
#     x_transformed = scaler.fit_transform(data.T).T
#     y_transformed = scaler.fit_transform(predictors[model_run]['all'].reshape(-1, 1)).T
#     lin_regression = Ridge(alpha=1.0)
#     lin_regression.fit(x_transformed.T, y_transformed[0].T)
#     return lin_regression.predict(data_avg[data_sub][data_run].T)
#
#
#
#
#
# ###############################################################################################
# # BETWEEN Subject Gen Scores - NONLINEAR
# best_nl_scores = []
#
# pearson_matrix_1 = np.zeros((len(subjects), len(subjects)))
# best_nl_model_indices = nested_dict() # keep track of which of i's models was the best on j
#
#
# #NONLINEAR
# model_run = 'Loc1'
# data_run = 'Loc1'
# y = predictors[data_run]['face']
# y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#
#         model_scores = []
#         for model_index in NL_funcs[subjects[i]][model_run]:
#             predicted = predict_NL(model_sub=subjects[i], data_sub=subjects[j], model_run=model_run, data_run=data_run,
#                                    model_index=model_index)
#             score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
#             if not isnan(score):
#                 model_scores.append(score)
#
#         best_model_index = np.argmax(model_scores)
#         best_nl_model_indices[model_run][subjects[i]][subjects[j]] = best_model_index
#
#         all_scores = np.array(model_scores)
#         best_nl_scores.append(np.max(all_scores))
#         pearson_matrix_1[i, j] = np.max(all_scores)
#
# gen_ls = []
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#         if i != j:
#             gen_ls.append(pearson_matrix_1[i, j])
#
#
# pearson_matrix_2 = np.zeros((len(subjects), len(subjects)))
#
# #NONLINEAR
# model_run = 'Loc2'
# data_run = 'Loc2'
# y = predictors[data_run]['face']
# y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#
#         model_scores = []
#         for model_index in NL_funcs[subjects[i]][model_run]:
#             predicted = predict_NL(model_sub=subjects[i], data_sub=subjects[j], model_run=model_run, data_run=data_run,
#                                    model_index=model_index)
#             score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
#             if not isnan(score):
#                 model_scores.append(score)
#
#         best_model_index = np.argmax(model_scores)
#         best_nl_model_indices[model_run][subjects[i]][subjects[j]] = best_model_index
#
#         all_scores = np.array(model_scores)
#         best_nl_scores.append(np.max(all_scores))
#         pearson_matrix_2[i, j] += np.max(all_scores)
#
# for i in range(len(subjects)):
#     for j in range(len(subjects)):
#         if i != j:
#             gen_ls.append(pearson_matrix_2[i, j])
#
# # plt.matshow(pearson_matrix / 2.0)
# # plt.colorbar()
# # plt.title('Between-Subject N-L (Best) R^2 Scores')
# # plt.savefig('./BetweenSubImages/H-O.png')
#
# gen_array_nl_best = np.array(best_nl_scores)
# gen_score = np.average(gen_array_nl_best)
# gen_stdev = np.std(gen_array_nl_best)
# print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
#
# ###############################################################################################
#
#
# print('Linear vs. nonlinear')
# print('Linear mean: {:.4f} std: {:.4f}'.format(np.mean(gen_array_l), np.std(gen_array_l)))
# print('Nonlinear mean: {:.4f} std: {:.4f}'.format(np.mean(gen_array_nl), np.std(gen_array_nl)))
# print('Stats ' + str(scipy.stats.ttest_ind(gen_array_l, gen_array_nl)))
# print()
# print('Nonlinear (best) vs. nonlinear')
# print('Nonlinear (best) mean: {:.4f} std: {:.4f}'.format(np.mean(gen_array_nl_best), np.std(gen_array_nl_best)))
# print('Nonlinear mean: {:.4f} std: {:.4f}'.format(np.mean(gen_array_nl), np.std(gen_array_nl)))
# print('Stats ' + str(scipy.stats.ttest_ind(gen_array_nl_best, gen_array_nl)))
# print()
# print('Nonlinear (best) vs. linear')
# print('Linear mean: {:.4f} std: {:.4f}'.format(np.mean(gen_array_l), np.std(gen_array_l)))
# print('Nonlinear (best) mean: {:.4f} std: {:.4f}'.format(np.mean(gen_array_nl_best), np.std(gen_array_nl_best)))
# print('Stats ' + str(scipy.stats.ttest_ind(gen_array_l, gen_array_nl_best)))
# print()
#
#
#
#
# # AVERAGE STUFF HERE - BEST NONLINEAR
# ###############################################################################################
# ###############################################################################################
# ###############################################################################################
# ###############################################################################################
# ###############################################################################################
# # Average Nonlinear Models
#
# gen_ls = []
# scaler = MinMaxScaler()
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_NL(subjects[i], subjects[j], 'Loc2', 'Loc2', best_nl_model_indices['Loc2'][subjects[i]][subjects[j]])
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc2']['face'].reshape(-1,1))
#     score = scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]
#     if not isnan(score):
#         gen_ls.append(score)
#         print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc2', 'TalROI', 'face', score))
#
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_NL(subjects[i], subjects[j], 'Loc1', 'Loc1', best_nl_model_indices['Loc1'][subjects[i]][subjects[j]])
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
#     score = scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]
#     if not isnan(score):
#         gen_ls.append(score)
#         print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc2', 'TalROI', 'face', score))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
#
# gen_array_nl_best = np.array(gen_ls)
# gen_score = np.average(gen_array_nl_best)
# gen_stdev = np.std(gen_array_nl_best)
# print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
# ###############################################################################################






###############################################################################################
###############################################################################################
###############################################################################################
# Hold on, we're doing validation!

# get output from all main subjects models

# NONLINEAR
nl_valid_predictions = []
nl_valid_scores = []
for run in runs:
    y = scaler.fit_transform(predictors[run]['face'].reshape(-1, 1))

    for sub_v in valid_subjects:
        avg_output = []
        for sub in subjects:
            predicted = predict_NL(sub, sub_v, run, run)
            if predicted.shape == (340,):
                avg_output.append(predicted)
        avg_output = np.average(np.array(avg_output), axis=0)
        nl_valid_predictions.append(avg_output)
        score = scipy.stats.pearsonr(avg_output, y[:, 0].T)[0]
        nl_valid_scores.append(score)
        # plt.plot(avg_output, alpha=0.2)
# plt.title('Average Model Validation - Non-Linear')
# plt.show()
nl_valid_predictions = np.array(nl_valid_predictions)
nl_valid_scores = np.array(nl_valid_scores)

# LINEAR
l_valid_predictions = []
l_valid_scores = []
for run in runs:
    y = scaler.fit_transform(predictors[run]['face'].reshape(-1, 1))
    for sub_v in valid_subjects:
        avg_output = []
        for sub in subjects:
            predicted = predict_L(sub, sub_v, run, run)
            if predicted.shape == (340,):
                avg_output.append(predicted)
        avg_output = np.average(np.array(avg_output), axis=0)
        l_valid_predictions.append(avg_output)
        score = scipy.stats.pearsonr(avg_output, y[:, 0].T)[0]
        l_valid_scores.append(score)
        # plt.plot(avg_output, alpha=0.2)
# plt.title('Average Model Validation - Linear')
# plt.show()
l_valid_predictions = np.array(l_valid_predictions)
l_valid_scores = np.array(l_valid_scores)


print('Linear vs. nonlinear')
print('Linear mean: {:.4f} std: {:.4f}'.format(np.mean(l_valid_scores), np.std(l_valid_scores)))
print('Nonlinear mean: {:.4f} std: {:.4f}'.format(np.mean(nl_valid_scores), np.std(nl_valid_scores)))
print('Stats ' + str(scipy.stats.ttest_ind(l_valid_scores, nl_valid_scores)))


###############################################################################################
###############################################################################################







###############################################################################################
# Between Subject Gen Matrix - NONLINEAR
pearson_matrix_1 = np.zeros((len(subjects), len(valid_subjects)))

for i in range(len(subjects)):
    for j in range(len(valid_subjects)):

        y = predictors['Loc2']['face']
        y_transformed = scaler.fit_transform(y.reshape(-1, 1))

        predicted = predict_NL(model_sub=subjects[i], data_sub=valid_subjects[j], model_run='Loc2', data_run='Loc2')
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

        predicted = predict_NL(model_sub=subjects[i], data_sub=valid_subjects[j], model_run='Loc1', data_run='Loc1')
        # print(predicted)

        score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
        if not isnan(score):
            pearson_matrix_2[i, j] = score
        else:
            pearson_matrix_2[i, j] = 0.0

for i in range(len(subjects)):
    for j in range(len(valid_subjects)):
        gen_ls.append(pearson_matrix_2[i,j])

gen_array_nl = np.array(gen_ls)

gen_score = np.average(gen_array_nl)
gen_stdev = np.std(gen_array_nl)
print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))

plt.matshow((pearson_matrix_1 + pearson_matrix_2) / 2, vmin=0, vmax=1)
# plt.title('Pairwise Between Subject Validation - Non-Linear')
plt.colorbar()
plt.show()
###############################################################################################

###############################################################################################
# Between Subject Gen Matrix - LINEAR
# Between Subject Gen Matrix - NONLINEAR
pearson_matrix_1 = np.zeros((len(subjects), len(valid_subjects)))

for i in range(len(subjects)):
    for j in range(len(valid_subjects)):

        y = predictors['Loc2']['face']
        y_transformed = scaler.fit_transform(y.reshape(-1, 1))

        predicted = predict_L(model_sub=subjects[i], data_sub=valid_subjects[j], model_run='Loc2', data_run='Loc2')
        # print(predicted)

        pearson_matrix_1[i, j] = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]

gen_ls = []
for i in range(len(subjects)):
    for j in range(len(valid_subjects)):
        gen_ls.append(pearson_matrix_1[i,j])

pearson_matrix_2 = np.zeros((len(subjects), len(valid_subjects)))

for i in range(len(subjects)):
    for j in range(len(valid_subjects)):

        y = predictors['Loc1']['face']
        y_transformed = scaler.fit_transform(y.reshape(-1, 1))

        predicted = predict_L(model_sub=subjects[i], data_sub=valid_subjects[j], model_run='Loc1', data_run='Loc1')
        # print(predicted)

        pearson_matrix_2[i, j] += scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]

for i in range(len(subjects)):
    for j in range(len(valid_subjects)):
        gen_ls.append(pearson_matrix_2[i,j])


gen_array_l = np.array(gen_ls)

gen_score = np.average(gen_array_l)
gen_stdev = np.std(gen_array_l)
print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))

plt.matshow((pearson_matrix_1 + pearson_matrix_2) / 2, vmin=0, vmax=1)
# plt.title('Pairwise Between Subject Validation - Non-Linear')
plt.colorbar()
plt.show()
###############################################################################################

# Are they different????
print(scipy.stats.ttest_ind(gen_array_l, gen_array_nl))
