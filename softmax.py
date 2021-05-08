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
    dw = np.zeros_like(W)
    #if you are not careful here, it is easy to run into numeric instability. 라는 부분이 있다. 
    #주의하지 않으면 numeric instability에 빠질 수 있다. 라는 의미인데 cs231n의 lecture note를 보면 다음과 같은 설명을 찾을 수 있다

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    for i in range(X.shape[0]):
      scores = X[i].dot(W)
      scores -= np.max(scores) #normalization 벡터의 값들을 가장 큰 값이 0이되도록 shift해야 한다는 뜻

      scores_exp = np.sum(np.exp(scores))
      correct_exp = np.exp(scores[y[i]])

      loss -= np.log(correct_exp/scores_exp)

      for j in range(W.shape[1]):
        if j == y[i]: # 정답 클래스일경우 패스
          continue
        dw[:,j] += np.exp(scores[j])/scores_exp *X[i]
      dw[:,y[i]]-= (scores_exp - correct_exp) / scores_exp *X[i]

    #평균내주기
    loss/=X.shape[0]
    dw/=X.shape[0]

    loss += reg * np.sum(W*W)
    dw +=2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dw


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

    num_train = X.shape[0]
    num_class = W.shape[1]

    scores = X.dot(W)
    scores -=np.max(scores)

    scores_exp = np.exp(scores)
    scores_expsum = np.sum(scores_exp, axis = 1)
    correct_exp = scores_exp[range(num_train),y]

    loss = correct_exp /scores_expsum
    loss = -np.sum(np.log(loss))/num_train + reg*np.sum(W*W)
    #loss function 미분
    margin = scores_exp/ scores_expsum.reshape(num_train,1)
    margin[range(num_train),y] = -(scores_expsum - correct_exp)/scores_expsum
    dW = X.T.dot(margin)
    dW/=num_train
    dW+= 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
