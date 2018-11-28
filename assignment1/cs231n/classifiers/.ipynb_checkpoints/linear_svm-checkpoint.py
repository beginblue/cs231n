import numpy as np
from random import shuffle

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
  """
  dW = np.zeros(W.shape) # d,c initialize the gradient as zero
 
  # compute the loss and the gradient
  num_classes = W.shape[1]#c
  num_train = X.shape[0]#n
  loss = 0.0
  
  for i in range(num_train):
    # every loop is a sample with d data
    # compute the score which is the predicted
    # label of each data 
    scores = X[i].dot(W) # 1,d * d,c = 1,c # x[i] : 1,d
    # y[i] is the true label of X[i]
    # socres[y[i]] stands for we assume that in this sample
    # the y[i]th data is correct and use it as the standard
    # of this loop.
    correct_class_score = scores[y[i]] # Judge standard

    for j in range(num_classes): 
      '''
      Process each data in one sample and compute the loss of
      each data and then sum them as the loss of this sample
      '''
      if j == y[i]: # if this is the correct one assumed just doing nothing
        continue

      # if not, compute the difference between each data and the one we assumed correct 
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      # max(0,margin)
      if margin > 0:
        loss += margin
        #计算j不等于yi的行的梯度差
        #损失元的倒数
        #https://github.com/whyscience/CS231n-Note-Translation_CN/blob/master/CS231n%204.2%EF%BC%9A%E6%9C%80%E4%BC%98%E5%8C%96%E7%AC%94%E8%AE%B0%EF%BC%88%E4%B8%8B%EF%BC%89.md
        #微分分析计算的梯度 （ 不懂 不懂 .jpg
        dW[:, j] += X[i]
        #j=yi时的梯度
        dW[:, y[i]]+=(-X[i])
      


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train # average loss of every sample
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # compute every samples once 500,10
  loss = 0.0
  #print(y.reshape((scores.shape[0],-1)).shape)
  
  # margin = scores - scores[:num_train,y].reshape((500,-1)) + 1 
  # CONFUSED: what is the difference between np.arange() and [:]
  correct_class_score = scores[np.arange(num_train),y]
  correct_class_score = np.reshape(
    np.repeat(correct_class_score,10),
    (num_train,num_classes)
    )
  margin = scores - correct_class_score +1
  margin[np.arange(num_train),y]=0
  loss = np.sum(margin[margin>0])/num_train+0.5*reg * np.sum(W * W)
  # #print(loss)
  # num_train=X.shape[0]
  # num_classes = W.shape[1]
  # scores = X.dot(W)
  # #这里得到一个500*10的矩阵,表示500个image的ground truth
  # correct_class_score = scores[np.arange(num_train),y]
  # #重复10次,得到500*10的矩阵,才可以和scores相加相减
  # correct_class_score = np.reshape(np.repeat(correct_class_score,num_classes),(num_train,num_classes))
  # margin = scores-correct_class_score+1.0
  # margin[np.arange(num_train),y]=0

  # loss = (np.sum(margin[margin > 0]))/num_train
  # loss+=0.5*reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
   #gradient
  margin[margin>0]=1
  margin[margin<=0]=0

  row_sum = np.sum(margin, axis=1)                  # 1 by N
  margin[np.arange(num_train), y] = -row_sum
  dW += np.dot(X.T, margin)     # D by C
  # for xi in range(num_train):
  #   dW+=np.reshape(X[xi],(dW.shape[0],1))*\
  #       np.reshape(margin[xi],(1,dW.shape[1]))

  dW/=num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
