import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

def get_data():
	print("Please wait while the data is loaded...")
	train = pd.read_csv('mnist_train.csv')
	test = pd.read_csv('mnist_test.csv')
	train = train.values
	test = test.values
	print("Data loading completed successfully!")
	print("Please wait while the data is pre-processed...")
	X_train = (np.reshape(train[:,1:], [train.shape[0], 28, 28])/255)
	X_train = np.expand_dims(X_train, 3)
	y_train = np.reshape(train[:,0], [train.shape[0],])
	X_train, y_train = shuffle(X_train, y_train)
	X_test = (np.reshape(test[:,1:], [test.shape[0], 28, 28])/255)
	X_test = np.expand_dims(X_test, 3)
	y_test = np.reshape(test[:,0], [test.shape[0],])
	del train
	del test
	y_train = tf.one_hot(y_train, 10)
	y_test = tf.one_hot(y_test, 10)
	with tf.Session() as sess:
		y_train, y_test = sess.run([y_train, y_test])
	print("Data pre-processing completed successfully!")
	return X_train, y_train, X_test, y_test

def convpool(X, W, b):
	conv_out = tf.nn.conv2d(X, W, 
		strides = [1, 1, 1, 1],
		padding = 'SAME')
	conv_out = tf.nn.bias_add(conv_out, b)
	pool_out = tf.nn.max_pool(conv_out,
		ksize = [1, 2, 2, 1],
		strides = [1, 2, 2, 1],
		padding = 'SAME')
	return tf.nn.relu(pool_out)

def init_filter(shape, poolsz):
	#w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
	w = np.random.randn(*shape) * np.sqrt(2.0/np.prod(shape[:-1]))
	return w.astype(np.float32)

X_train, y_train, X_test, y_test = get_data()
N = X_train.shape[0]
batchsz = 500
batch_sz = 500
epochs = 3
print_period = 10
n_batches = N // batchsz
M = 256
K = 10
poolsz = (2, 2)

W1_shape = (5, 5, 1, 20)
W1_init = init_filter(W1_shape, poolsz)
b1_init = np.zeros(W1_shape[-1], np.float32)

W2_shape = (5, 5, 20, 50)
W2_init = init_filter(W2_shape, poolsz)
b2_init = np.zeros(W2_shape[-1], np.float32)

W3_init = np.random.randn(W2_shape[-1]*7*7, M) / np.sqrt(W2_shape[-1]*7*7+M)
b3_init = np.zeros(M, np.float32)

W4_init = np.random.randn(M, K) / np.sqrt(M+K)
b4_init = np.zeros(K, np.float32)

X = tf.placeholder(shape = (batch_sz, 28, 28, 1), dtype = tf.float32)
y = tf.placeholder(shape = (batch_sz, 10), dtype = tf.float32)
X_p = tf.placeholder(shape = (1, 28, 28, 1), dtype = tf.float32)
y_p = tf.placeholder(shape = (1, 10), dtype = tf.float32)
W1 = tf.Variable(W1_init, dtype = tf.float32)
W2 = tf.Variable(W2_init, dtype = tf.float32)
W3 = tf.Variable(W3_init, dtype = tf.float32)
W4 = tf.Variable(W4_init, dtype = tf.float32)
b1 = tf.Variable(b1_init, dtype = tf.float32)
b2 = tf.Variable(b2_init, dtype = tf.float32)
b3 = tf.Variable(b3_init, dtype = tf.float32)
b4 = tf.Variable(b4_init, dtype = tf.float32)

Z1 = convpool(X, W1, b1)
Z2 = convpool(Z1, W2, b2)
Z2_shape = Z2.get_shape().as_list()
Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
Z3 = tf.nn.relu(tf.matmul(Z2r, W3) + b3)
y_logit = tf.matmul(Z3, W4) + b4

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits=y_logit))
train_op = tf.train.RMSPropOptimizer(0.0001).minimize(cost)
Z1a = convpool(X_p, W1, b1)
Z2a = convpool(Z1a, W2, b2)
Z2_shapea = Z2a.get_shape().as_list()
Z2ra = tf.reshape(Z2a, [Z2_shapea[0], np.prod(Z2_shapea[1:])])
Z3a = tf.nn.relu(tf.matmul(Z2ra, W3) + b3)
y_logita = tf.matmul(Z3a, W4) + b4
predict_op = tf.argmax(y_logita, 1)

t0 = datetime.now()
LL = []
W1_val = None
W2_val = None
saver = tf.train.Saver()
# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     for epoch in range(epochs):
#         for batch in range(n_batches):
#             Xbatch = X_train[batch*batch_sz:(batch*batch_sz + batch_sz),]
#             ybatch = y_train[batch*batch_sz:(batch*batch_sz + batch_sz),]
#             if len(Xbatch) == batch_sz:
#                 session.run(train_op, feed_dict={X: Xbatch, y: ybatch})
#                 if batch % print_period == 0:
#                     test_cost = 0
#                     prediction = np.zeros(len(X_test))
#                     for k in range(len(X_test) // batch_sz):
#                         Xtestbatch = X_test[k*batch_sz:(k*batch_sz + batch_sz),]
#                         ytestbatch = y_test[k*batch_sz:(k*batch_sz + batch_sz),]
#                         test_cost += session.run(cost, feed_dict={X: Xtestbatch, y: ytestbatch})
#                     print("Epoch: {} Batch: {} and Cost: {}".format(epoch, batch, test_cost))
#                     LL.append(test_cost)
#         saver.save(session, './weights.ckpt')
#     W1_val = W1.eval()
#     W2_val = W2.eval()

# print("Elapsed Time: ", (datetime.now() - t0))
# plt.plot(LL)
# plt.show()

Xa = np.expand_dims(X_test[1], 0)
plt.imshow(Xa[0,:,:,0], cmap='gray')
plt.show()

with tf.Session() as session:
	saver.restore(session, './weights.ckpt')
	prediction = session.run(predict_op, feed_dict={X_p:Xa})
print("The prediction for above digit is: ", prediction[0])