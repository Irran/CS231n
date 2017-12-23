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
  num_train, dim = X.shape
  num_class = W.shape[1]

  f = X.dot(W)
  for i in range(num_train):
    f_max = np.max(f[i])
    prob = np.exp(f[i] - f_max) / np.sum(np.exp(f[i] - f_max))
    for j in range(num_class):
        if(j == y[i]):
            loss += -np.log(prob[j])
            dW[:,j] += (prob[j] - 1) * X[i]
        else:
            dW[:,j] += prob[j] * X[i]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W

  # loss = 0.0
  # dW = np.zeros_like(W)    # 得到一个和W同样shape的矩阵
  # num_train, dim = X.shape
  # num_class = W.shape[1]
  # f = X.dot(W)    # N by C
  # f_max = np.max(f, axis=1, keepdims=True)   # 找到最大值然后减去，这样是为了防止后面的操作会出现数值上的一些偏差
  # prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True) # N by C
  # y_trueClass = np.zeros_like(prob)
  # y_trueClass[np.arange(num_train), y] = 1.0
  # for i in range(num_train):
  #   for j in range(num_class):
  #     loss += -(y_trueClass[i, j] * np.log(prob[i, j]))    # 损失函数的公式
  #     dW[:, j] += -(y_trueClass[i, j] - prob[i, j]) * X[i]#梯度的
  # loss /= num_train
  # loss += 0.5 * reg * np.sum(W * W)  # 加上正则
  # dW /= num_train
  # dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train, dim = X.shape
  num_class = W.shape[1]

  f = X.dot(W)
  f_max = np.max(f, axis=1, keepdims=True)
  prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
  loss += -np.sum(np.log(prob[np.arange(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  correct_class = np.zeros_like(prob)
  correct_class[np.arange(num_train), y] = 1
  dW += (prob - correct_class).T.dot(X).T
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

