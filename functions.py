import numpy as np

from scipy.special import expit
from costs import * 
from implementations import *
from helpers import *
from proj1_helpers import *

# data split
def split_data(data_size, ratio):
    permuted = np.random.permutation(data_size)
    tr_size = (int)(data_size*ratio)
    tr_indices = permuted[0:tr_size]
    ts_indices = permuted[tr_size:]
    return tr_indices, ts_indices
    
# handle missing values (-999)
def handle_missing(data):
    means = []
    for i in range(data.shape[1]):
        missing = (data[:,i] == -999)
        no_missing = (data[:,i] != -999)
        mean_i = np.mean(data[no_missing,i])
        data[missing,i] = mean_i
        means.append(mean_i)
    return data, means

def handle_missing_test(data, means):
    for i in range(data.shape[1]):
        missing = (data[:,i] == -999)
        data[missing,i] = means[i]
    return data

def poly_basis(data, degree):
    if(degree == 1):
        return data
    else:
        data_org = np.copy(data)
        for i in range(2, degree):
            data = np.concatenate((data,data_org**(i)),axis=1)
        return data

def feature_combinations(tX, c_index):
    comb_tX = []
    for l in range(len(c_index)):
        for k in range(l+1, len(c_index)):
            m = 0
            m=tX[:, c_index[l]]*tX[:, c_index[k]]
            comb_tX.append(m)
    comb_tX = np.asarray(comb_tX)
    return comb_tX.T

def sigmoid(x):
    if(x.max() < 710):
        return (1 - (1/(1 + np.exp(x))))
    else:
        non_overflow = np.where(x < 710)
        overflow = np.where(x >= 710)
        result = np.zeros(x.shape)
        result[overflow] = 1
        result[non_overflow] = (1 - (1/(1 + np.exp(x[non_overflow]))))
        return result

def correlated(y, tx, threshold = 0):
    """ compute the correlation between the label y and each features of tx
    return the array of arg of the nth most correlated feature
    """
    #print('y shape', y.shape)
    cor = np.corrcoef(y.T, tx.T)
    y_xs_cor = cor[0,1:]
    y_xs_threshold = y_xs_cor[np.abs(y_xs_cor) >=threshold]
    arg_sorted = np.argsort(np.abs(y_xs_cor))[::-1] 
    #print('All agr', arg_sorted)
    #print('Arg with threshol',arg_sorted[:len(y_xs_threshold)])
    #print('Corr',np.sort(np.abs(y_xs_cor))[::-1])
    return arg_sorted[:len(y_xs_threshold)]


def test_model(test_y, prediction):
    "retrun the ratio of good answers"
    comparison = [test_y == prediction]
    return np.sum(comparison) / len(test_y)

def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x,k_indices, k_fold=3,gamma=0, lambda_=0.01, degree=11,  threshold_cor=0.01, max_iters=500, classifier='ridge'):
    """compute cross-validation."""
    loss_train =0
    loss_test =0
    accuracy = 0

    for i in range(k_fold):
        #put the first index at the end
        k_indices = np.r_[k_indices[1:],k_indices[0].reshape(1,k_indices.shape[1])]

        test_tx = x[k_indices[0]]
        test_y = y[k_indices[0]]
        train_tx = x[k_indices[1:].flatten()]
        train_y = y[k_indices[1:].flatten()]
         
        train_y = train_y.reshape([len(train_y), 1])
        test_y = test_y.reshape([len(test_y),1])


        #Handle the missing values
        train_tx, means = handle_missing(train_tx)
        test_tx = handle_missing_test(test_tx, means)

        
        #Get indexes of the features that have a correlation with labels bigger that the threshold
        corr_idx = correlated(train_y, train_tx, threshold_cor)
        
        #Keep only correlated features
        train_tx_comb = feature_combinations(train_tx, corr_idx)
        test_tx_comb = feature_combinations(test_tx, corr_idx)
        
        #Filter uncorrelated features and concatenate with combinations
        train_tx = train_tx[:,corr_idx]
        test_tx = test_tx[:,corr_idx]
        
        train_tx = np.concatenate((train_tx, train_tx_comb), axis=1)
        test_tx = np.concatenate((test_tx, test_tx_comb), axis=1)

        #Compute polynomial basis
        train_tx = poly_basis(train_tx, degree)
        test_tx = poly_basis(test_tx, degree)
        
        #Standardize
        train_tx, mean_train, std_train = standardize(train_tx)
        test_tx, m_test, std_test = standardize(test_tx, mean_train, std_train)
        initial_weights = np.zeros((train_tx.shape[1], 1))
        
        if(classifier == 'logistic'):
            train_y_logistic = np.copy(train_y)
            train_y_logistic = (train_y_logistic + 1) / 2
            test_y_logistic = np.copy(test_y)
            test_y_logistic = (test_y_logistic + 1) / 2
            weights, loss_tr = logistic_regression(train_y_logistic, train_tx, initial_weights, max_iters, gamma) 
            loss_te = compute_loss(test_y_logistic, test_tx, weights, 'logistic')
            prediction = sigmoid(test_tx @ weights)
            prediction[np.where(prediction <= 0.5)]= -1
            prediction[np.where(prediction > 0.5)]= 1
            
        elif(classifier == 'reg_logistic'):
            train_y_logistic = np.copy(train_y)
            train_y_logistic = (train_y_logistic + 1) / 2
            test_y_logistic = np.copy(test_y)
            test_y_logistic = (test_y_logistic + 1) / 2
            weights, loss_tr = reg_logistic_regression(train_y_logistic, train_tx, lambda_, initial_weights, max_iters, gamma) 
            loss_te = compute_loss(test_y_logistic, test_tx, weights, 'logistic', lambda_)
            prediction = sigmoid(test_tx @ weights)
            prediction[np.where(prediction <= 0.5)]= -1
            prediction[np.where(prediction > 0.5)]= 1
        
        elif(classifier == 'ls_normal'):
            weights, loss_tr  = least_squares(train_y, train_tx)
            loss_te = compute_loss(test_y, test_tx, weights, 'mse')
            prediction =  predict_labels(weights, test_tx)

        
        elif(classifier == 'ridge'):
            weights, loss_tr  = ridge_regression(train_y, train_tx, lambda_)
            loss_te = compute_loss(test_y, test_tx, weights, 'mse')
            prediction =  predict_labels(weights, test_tx)

        
        elif(classifier == 'ls_gd'):
            weights, loss_tr = least_squares_GD(train_y, train_tx, initial_weights, max_iters, gamma)
            loss_te = compute_loss(test_y, test_tx, weights, 'mse')
            prediction = predict_labels(weights, test_tx)
            
            
        elif(classifier == 'ls_sgd'):
            weights, loss_tr = least_squares_SGD(train_y, train_tx, initial_weights, max_iters, gamma)
            loss_te = compute_loss(test_y, test_tx, weights, 'mse')
            prediction = predict_labels(weights, test_tx)
        
        acc = test_model(test_y, prediction)

        loss_train +=loss_tr
        loss_test += loss_te
        accuracy += acc
    
    return accuracy/k_fold, loss_train/k_fold, loss_test/k_fold

def saveAsCSV(data, fileName):
    np.savetxt(
        fileName,           # file name
        data,                # array to save
        delimiter=',',          # column delimiter
        newline='\n',           # new line character
        footer='end of file',   # file footer
        comments='# ',          # character to use for comments
        header='accuracy, loss_tr, loss_te, g , lambda_,pb')      # file header


















