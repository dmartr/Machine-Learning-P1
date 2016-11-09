# Useful starting lines
import numpy as np
from functions import *
import itertools
from helpers import *
from proj1_helpers import *
from implementations import *


#Set parameters
DATA_TRAIN_PATH = 'train.csv'
DATA_TEST_PATH = 'test.csv'
OUTPUT_PATH = 'predictions1.csv'
lambda_ = 0.0001
degree = 11
threshold = 0.01


#Load the train data
print("Loading train set...")
DATA_TRAIN_PATH = 'train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#replace missing values by mean
tX, means_missing = handle_missing(tX)
#Get indexes of the features that have a correlation with labels bigger that the threshold
corr_idx = correlated(y, tX, threshold)
#Keep only correlated features
tX_comb = feature_combinations(tX, corr_idx)

#Filter uncorrelated features and concatenate with combinations
tX = tX[:,corr_idx]

#Concatenate with combinated features
tX = np.concatenate((tX, tX_comb), axis=1)

#Compute polynomial basis
tX = poly_basis(tX, degree)

# standardization
print('Standardization')
tX, means_tX, std_tX = standardize(tX)

#Train the model
print("Training the model...")
weights, loss = ridge_regression(y, tX, lambda_)
saveAsCSV(weights, 'bestWeightsRidge')


#Load test data
print("Loading the test set")
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
#R=========================================================remove for submission
limit = np.ceil(tX_test.shape[0]/2)
tX_test = tX_test[0:limit, :]
ids_test = ids_test[0:limit]



#Feature engineering
tX_test = handle_missing_test(tX_test, means_missing)
tX_test_comb = feature_combinations(tX_test, corr_idx)
tX_test = tX_test[:,corr_idx]
tX_test = np.concatenate((tX_test, tX_test_comb), axis=1)
tX_test = poly_basis(tX_test, degree)

tX_test, m, s = standardize(tX_test, means_tX, std_tX)

#Predict the label
print("Predicting the labels")
y_pred = predict_labels(weights, tX_test)


create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print("Done")





















