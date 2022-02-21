from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        
        max_score = scores.max()
        scores -= max_score
        
        loss += -correct_class_score + max_score + np.log(np.exp(scores).sum())
        
        for j in range(num_classes):
            dW[:,j] += np.exp(scores[j]) / np.exp(scores).sum() * X[i,:]
        
        dW[:,y[i]] -= X[i,:]
        
    loss /= num_train
    loss += reg * np.sum(W*W)
    
    dW /= num_train
    dW += reg * np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    num_train = X.shape[0]
    correct_class_scores = scores[range(num_train),y]
    max_scores = np.max(scores, axis=1, keepdims=True) #predicted scores
    scores -= max_scores #avoid large exp
    
    loss = max_scores.sum() - correct_class_scores.sum() + np.log(np.sum(np.exp(scores), axis=1)).sum()
    
    loss /= num_train
    loss += reg*np.sum(W*W)
    
    
    softmax_deriv = (np.exp(scores)/np.exp(scores).sum(axis=1).reshape(-1,1))
    softmax_deriv[range(num_train),y] -= 1
    
    dW = X.T.dot(softmax_deriv)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
