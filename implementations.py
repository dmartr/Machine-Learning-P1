import numpy as np
from costs import *
from functions import *

#============================
#  Logistic regression
#============================
def compute_gradient_logistic(y, tx, w, lambda_=0):
    sigma = sigmoid(np.dot(tx,w)) 
    e = sigma - y
    l = np.dot(tx.T,e)
    return (l + 2*(lambda_*w)) / tx.shape[0]
   
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    y = y.reshape([len(y),1])
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, 'logistic')
        grad = compute_gradient_logistic(y, tx, w)
        #print("Iteration: " + str(n_iter))
        #print("Loss: " + str(loss))
        #print("Gradient: " + str(grad))
        w = w - (gamma * grad)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    y = y.reshape([len(y),1])
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, 'logistic', lambda_)
        grad = compute_gradient_logistic(y,tx,w,lambda_)
        #print("Iteration: " + str(n_iter))
        #print("Loss: " + str(loss))
        #print("Gradient: " + str(grad))
        w = w - (gamma*grad)
    return w, loss

#============================
#  Linear regression
#============================
def compute_gradient(y, tx, w, regularizer='none',lambda_=0):
    """Compute the gradient."""
    N = len(y);
    e = y- tx.dot(w);
    gradient = -1/N * tx.T.dot(e);
    if(regularizer == 'none'):
        return gradient;
    elif(regularizer == 'l1'):
        return gradient + (lambda_ * np.sign(w))
    elif(regularizer == 'l2'):
        return gradient + (lambda_ * w)


def least_squares_GD (y, tx, initial_w, max_iters, gamma):    
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    last_loss = np.NaN          
    y = y.reshape([len(y),1])
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w);
        #if n_iter % 50 == 0:
        #    print(n_iter,' - Loss: ',loss)
        gradient = compute_gradient(y,tx,w);
        w = w - gamma * gradient;
        loss = compute_loss(y,tx,w,'mse');
        
        last_loss = loss

    return w, loss    

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    # ***************************************************
    np.random.seed(1)
    idx = np.random.randint(len(y))

    y_ = y[idx,:]
    N = 1
    tx_ = tx[idx,:].reshape([1,tx.shape[1]])

    e = y_- tx_.dot(w);
    return -1/N * tx_.T @ e;


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    y = y.reshape([len(y),1])
    w = initial_w
    batch_size = 1
    losses = []
    last_loss = -1
    ws = [w]          
    for n_iter in range(max_iters):
        gradient = compute_stoch_gradient(y,tx,w);
        w = w - gamma * gradient;
        # store w and loss
        ws.append(w)
        loss = compute_loss(y,tx,w,'mse');
        losses.append(loss)        
        last_loss = loss
    loss_argmin = np.argmin(losses[-50:])
    w = ws[loss_argmin]    
    loss = losses[loss_argmin]

    return w, loss    

#=================================
#  Linear regression - normal eqn.
#=================================

def least_squares(y, tx):
    """calculate the least squares solution."""
    y = y.reshape([len(y),1])
    a = tx.T @ tx
    b = tx.T @ y
    wopt = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, wopt, 'mse')
    return wopt, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    #lambp = lambda_*2*len(y)
    y = y.reshape([len(y),1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T,tx)+lambda_*np.identity(tx.shape[1])), tx.T), y)
    mse = compute_loss(y, tx, w)
    return w, mse 