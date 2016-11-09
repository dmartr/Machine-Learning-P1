# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_logistic(y, tx, w, lambda_=0):
    sum_w = np.sum(np.power(w,2))
    loss = np.log(1 + np.exp(np.dot(tx,w))) - (y * (np.dot(tx,w)))
    return (np.sum(loss) + lambda_ * sum_w) / tx.shape[0]

def compute_loss(y, tx, w, loss='mse', lambda_=0):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    if(loss == 'logistic'):
        sum_w = np.sum(np.power(w,2))
        return calculate_logistic(y, tx, w, lambda_)
    elif(loss == 'mse'):
        e = y - tx.dot(w)
        return calculate_mse(e)
    elif(loss == 'mae'):
        e = y - tx.dot(w)
        return calculate_mae(e)
    
    return calculate_mse(e)
    # return calculate_mae(e)
