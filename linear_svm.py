from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
      dw = loss function의 gradient = loss function을 W에 대해 미분하였다.
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # 10개
    num_train = X.shape[0] # 500개 minibatch
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # (1,3073) * (3073,10) = (10) 각 class 별 점수
        correct_class_score = scores[y[i]] # 정답 클래스 점수를 갖고옴
        for j in range(num_classes):
            if j == y[i]:
                continue
            # Sj - S(yi) +1
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            # Max(0,Sj - S(yi) +1)
            if margin > 0:
                loss += margin
                dW[:, y[i]] = dW[:, y[i]] - X[i] 
                dW[:,j] = dW[:,j] + X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss. 
    # L2 regularization to prevent overfitting
    loss += reg * np.sum(W * W)
    dW = dW + reg*2*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):

    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = y.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores=X.dot(W)
    y1=scores[np.arange(num_train),y]  #value of scores
    y1=np.reshape(y1,(num_train,1))
  
    loss1 = np.maximum(scores - y1 + 1,0) 
    loss=np.sum( np.sum(loss1,axis=1)-1 )
  
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    kernel = 1* ((scores-y1+1)>0)
    kernel1 =np.sum(kernel,axis=1)-1
    kernel[np.arange(num_train),y]=-kernel1
    dW=(X.T).dot(kernel)
  
    dW /= num_train
    dW += reg*W

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
 
  
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
