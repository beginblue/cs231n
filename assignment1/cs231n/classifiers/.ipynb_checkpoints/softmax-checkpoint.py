import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]#c
  num_train = X.shape[0]#n
  for i in range(num_train):
    #scores we calculated
    scores = X[i].dot(W) # 1,d * d,c = 1,c # x[i] : 1,d
    scores -= np.max(scores)
    #scores we assumed right --- and we dont need it when using softmax
    #correct_class_score = scores[y[i]] # correct results 1,c
    #the exponent of the difference between above
    eScores = np.exp(scores)#-correct_class_score)
    #the sum of all exponents 
    eSum = np.sum(eScores)
    #negative log softmax
    pEach = -np.log(eScores[y[i]]/eSum)
    loss+=np.sum(pEach)
    dW[:, y[i]] -= X[i] # d,1 -=d,1
    for j in range(num_classes): #c
      #print(i,j,num_classes)
      dW[:, j] += eScores[j] / eSum * X[i] # (1d += 1/1 * 1d)*c
  loss /= num_train
  loss -= 0.5*reg*np.sum(W*W)
  dW = dW / num_train + reg * W
  #print(loss)
  #print(softmax.shape,z.shape)
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  #print(W)
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  c = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # n,c
  scores-= np.max(scores,axis=1).reshape(N,1)
  # delta  = np.repeat(np.max(scores,axis=1),10) #n,1
  # scores-= np.reshape(delta,(500,-1))# n,c
  eScores= np.exp(scores)# n,c
  eSum   = np.sum(eScores,axis=1) # n,1
  pEach  =-np.log(eScores[range(N),y]/eSum) # 
  loss  += (np.sum(pEach))/N
  loss  +=-0.5*reg*np.sum(W*W)
 

  counts = eScores / eSum.reshape(N,1)
  counts[range(N), y] -= 1
  dW = np.dot(X.T, counts)

  dW = dW / N + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

