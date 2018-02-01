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

runs = ['Loc1', 'Loc2']

prt_root = '/Users/ethan/Dropbox/Share_Kevin_Ethan/Main/PRTs_NoButton'

data_avg = nested_dict()

csv_folder = './CSVs/'

# Load the localizer runs
for sub in subjects:
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
predictors['Loc1']['body'] = generate_predictor_loc('Loc1',1,0,0,0)
predictors['Loc1']['hand'] = generate_predictor_loc('Loc1',0,0,1,0)
predictors['Loc1']['scram'] = generate_predictor_loc('Loc1',0,0,0,1)
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


def predict_NL(model_sub, data_sub, model_run, data_run):
    x = []
    for i in range(340):
        try:
            value = NL_funcs[model_sub][model_run](**args[data_sub][data_run][i])
            if not isnan(value):
                x.append(value)
            else:
                print(value)
                x.append(0.5)
        except OverflowError:
            pass
            #x.append(1.0)
        except (ZeroDivisionError, ValueError):
            x.append(1.0)
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
# # Average Nonlinear Models
#
# gen_ls = []
# scaler = MinMaxScaler()
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_NL(subjects[i], subjects[j], 'Loc2', 'Loc2')
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc2']['face'].reshape(-1,1))
#     gen_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#     print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc2', 'TalROI', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_NL(subjects[i], subjects[j], 'Loc1', 'Loc1')
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
#     gen_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#     print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc1', 'TalROI', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
#
# gen_array_nl = np.array(gen_ls)
# gen_score = np.average(gen_array_nl)
# gen_stdev = np.std(gen_array_nl)
# print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
# ###############################################################################################
#
#
# ###############################################################################################
# # Average LINEAR Models
#
# gen_ls = []
# scaler = MinMaxScaler()
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_L(subjects[i], subjects[j], 'Loc1', 'Loc1')
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
#     gen_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#     print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc1', 'H-O', 'all', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_L(subjects[i], subjects[j], 'Loc2', 'Loc2')
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc2']['face'].reshape(-1,1))
#     gen_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#     print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc2', 'H-O', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
#
# gen_array_l = np.array(gen_ls)
# gen_score = np.average(gen_array_l)
# gen_stdev = np.std(gen_array_l)
# print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
# ###############################################################################################
#
# # Are they different????
# print(scipy.stats.ttest_ind(gen_array_l, gen_array_nl))
#
#
#
#
#
#
#
#
#
#
#
#
# ###############################################################################################
# # Within Subject Gen Scores - NONLINEAR
# gen_ls = []
# for sub in subjects:
#     model_run = 'Loc1'
#     data_run = 'Loc2'
#     y = predictors[data_run]['face']
#     y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#     predicted = predict_NL(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run)
#     score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
#     if not isnan(score):
#         gen_ls.append(score)
# for sub in subjects:
#     model_run = 'Loc2'
#     data_run = 'Loc1'
#     y = predictors[data_run]['face']
#     y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#     predicted = predict_NL(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run)
#     score = scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0]
#     if not isnan(score):
#         gen_ls.append(score)
#         # plt.plot(y_transformed)
#     # plt.plot(predicted)
#     # plt.show()
#
# gen_array_nl = np.array(gen_ls)
# gen_score = np.average(gen_array_nl)
# gen_stdev = np.std(gen_array_nl)
# print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
#
# ###############################################################################################
#
# ###############################################################################################
# # Within Subject Gen Scores - LINEAR
# gen_ls = []
#
# for sub in subjects:
#     model_run = 'Loc1'
#     data_run = 'Loc2'
#     y = predictors[data_run]['face']
#     y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#     predicted = predict_L(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run)
#     gen_ls.append(scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0])
# for sub in subjects:
#     model_run = 'Loc2'
#     data_run = 'Loc1'
#     y = predictors[data_run]['face']
#     y_transformed = scaler.fit_transform(y.reshape(-1, 1))
#     predicted = predict_L(model_sub=sub, data_sub=sub, model_run=model_run, data_run=data_run)
#     gen_ls.append(scipy.stats.pearsonr(predicted, y_transformed[:, 0].T)[0])
#     # plt.plot(y_transformed)
#     # plt.plot(predicted)
#     # plt.show()
#
# gen_array_l = np.array(gen_ls)
# gen_score = np.average(gen_array_l)
# gen_stdev = np.std(gen_array_l)
# print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
# ###############################################################################################
#
# # Are they different????
# print(scipy.stats.ttest_ind(gen_array_l, gen_array_nl))
#
#
#
#
#
#
#
#
#
#
#
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
#             predicted = predict_NL(subjects[i], subjects[j], 'Loc2', 'Loc2')
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc2']['face'].reshape(-1,1))
#     gen_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#     # print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc2', 'TalROI', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
# gen_array_nl = np.array(gen_ls)
# gen_score = np.average(gen_array_nl)
# gen_stdev = np.std(gen_array_nl)
# print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
# ###############################################################################################
#
#
# ###############################################################################################
# # Average LINEAR Models
#
# gen_ls = []
# scaler = MinMaxScaler()
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_L(subjects[i], subjects[j], 'Loc1', 'Loc1')
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
#     gen_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#     print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc1', 'TalROI', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
# gen_array_l = np.array(gen_ls)
# gen_score = np.average(gen_array_l)
# gen_stdev = np.std(gen_array_l)
# print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
# ###############################################################################################
#
#
#
#
# ###############################################################################################
# # Averare NONLINEAR applied to resting state data
# gen_ls = []
# scaler = MinMaxScaler()
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_NL(subjects[i], subjects[j], 'Loc1', 'Loc1')
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
#     gen_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#     print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc1', 'TalROI', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
# gen_array = np.array(gen_ls)
# gen_score = np.average(gen_array)
# gen_stdev = np.std(gen_array)
# # print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
#
#
# rest_ls = []
# scaler = MinMaxScaler()
#
# all_predictions = []
# for j in range(len(subjects)):
#     predicted = predict_NL(model_sub=subjects[j], data_sub='IK82', model_run='Loc1', data_run='Rest')
#     all_predictions.append(predicted)
#     y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
#     # plt.plot(y)
#     # plt.plot(predicted)
#     # plt.show()
#
# all_predictions = np.array(all_predictions)
# avg_prediction = np.average(all_predictions, axis=0)
# y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
# rest_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
# # plt.plot(y)
# # plt.plot(avg_prediction)
# # print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc2', 'TalROI', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
# # plt.show()
# # if not file == None:
#     # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
# #print(scipy.stats.pearsonr(avg_prediction, y)[0])
# rest_array = np.array(rest_ls)
# rest_score = np.average(rest_ls)
# rest_stdev = np.std(rest_ls)
# # print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
#
# print('Mean R^2: {:.4f}'.format(gen_score))
# print('Rest R^2: {:.4f}'.format(rest_score))
# print('Different Dist: {}'.format(scipy.stats.ttest_1samp(a=gen_array, popmean=rest_score)))
# ###############################################################################################
#
# ###############################################################################################
# # Averare Linear applied to resting state data
# gen_ls = []
# scaler = MinMaxScaler()
#
# for i in range(len(subjects)):
#     all_predictions = []
#     for j in range(len(subjects)):
#         if not i == j:
#             predicted = predict_L(subjects[i], subjects[j], 'Loc1', 'Loc1')
#             all_predictions.append(predicted)
#             #plt.plot(y_transformed)
#             #plt.plot(predicted)
#             #plt.show()
#
#     all_predictions = np.array(all_predictions)
#     avg_prediction = np.average(all_predictions, axis=0)
#     y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
#     gen_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
#     # plt.plot(y)
#     # plt.plot(avg_prediction)
#     print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc1', 'TalROI', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     # plt.show()
#     # if not file == None:
#         # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
#     #print(scipy.stats.pearsonr(avg_prediction, y)[0])
# gen_array = np.array(gen_ls)
# gen_score = np.average(gen_array)
# gen_stdev = np.std(gen_array)
# # print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
#
#
# rest_ls = []
# scaler = MinMaxScaler()
#
# all_predictions = []
# for j in range(len(subjects)):
#     predicted = predict_L(model_sub=subjects[j], data_sub='IK82', model_run='Loc1', data_run='Rest')
#     all_predictions.append(predicted)
#     y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
#     # plt.plot(y)
#     # plt.plot(predicted)
#     # plt.show()
#
# all_predictions = np.array(all_predictions)
# avg_prediction = np.average(all_predictions, axis=0)
# y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1,1))
# rest_ls.append(scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0])
# # plt.plot(y)
# # plt.plot(avg_prediction)
# # print('{}\t{}\t{}\t{}\t{}'.format(subjects[i], 'Loc2', 'TalROI', 'face', scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
# # plt.show()
# # if not file == None:
#     # file.write('{},{},{},{},{}\n'.format(subjects[i], run, dd_name, predictor, scipy.stats.pearsonr(avg_prediction,y[:, 0].T)[0]))
# #print(scipy.stats.pearsonr(avg_prediction, y)[0])
# rest_array = np.array(rest_ls)
# rest_score = np.average(rest_ls)
# rest_stdev = np.std(rest_ls)
# # print('Gen mean: {:.4f}\tGen stdev: {:.4f}'.format(gen_score, gen_stdev))
#
# print('Mean R^2: {:.4f}'.format(gen_score))
# print('Rest R^2: {:.4f}'.format(rest_score))
# print('Different Dist: {}'.format(scipy.stats.ttest_1samp(a=gen_array, popmean=rest_score)))
# ###############################################################################################




###############################################################################################
# Fit linear model for all-predictor to resting state data. Compare to dist of fits (R^2) for linear.
ls = []
y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1, 1))
plt.plot(y)
for sub in subjects:
    x = predict_L(model_sub=sub, data_sub=sub, model_run='Loc1', data_run='Loc1')
    plt.plot(x, alpha=0.2)
    ls.append(scipy.stats.pearsonr(x,y[:, 0].T)[0])

ls = np.array(ls)

x = predict_L(model_sub='IK82', data_sub='IK82', model_run='Rest', data_run='Rest')
plt.plot(x)
obs = scipy.stats.pearsonr(x,y[:, 0].T)[0]
print(ls)
print('Mean R^2: {:.4f}'.format(np.mean(ls)))
print('Rest R^2: {:.4f}'.format(obs))
print('Different Dist: {}'.format(scipy.stats.ttest_1samp(a=ls, popmean=obs)))
plt.show()

###############################################################################################


###############################################################################################
# Fit model for all-predictor to resting state data. Compare to dist of fits (R^2) for NONLINEAR.
ls = []
y = scaler.fit_transform(predictors['Loc1']['face'].reshape(-1, 1))
plt.plot(y)
for sub in subjects:
    x = predict_NL(model_sub=sub, data_sub=sub, model_run='Loc1', data_run='Loc1')
    x = scaler.fit_transform(x.reshape(1,-1).T).T
    plt.plot(x.T, alpha=0.2)
    ls.append(scipy.stats.pearsonr(x[0],y[:, 0].T)[0])

ls = np.array(ls)

x = predict_NL(model_sub='IK82', data_sub='IK82', model_run='Rest', data_run='Rest')
plt.plot(x)
obs = scipy.stats.pearsonr(x,y[:, 0].T)[0]
print(ls)
print('Mean R^2: {:.4f}'.format(np.mean(ls)))
print('Rest R^2: {:.4f}'.format(obs))
print('Different Dist: {}'.format(scipy.stats.ttest_1samp(a=ls, popmean=obs)))

plt.show()

###############################################################################################