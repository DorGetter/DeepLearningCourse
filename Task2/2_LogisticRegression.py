import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
from keras.utils import to_categorical

tf.compat.v1.disable_eager_execution()

"""
Second try- Logistic Regression Cifar10:
``````````
* softmax
* loss = -tf.reduce_sum(y_*tf.log(y)).
* Adam optimizer
* learning rate = 0.001.

results:
-------
epoch: 30,
train_acc:  0.430039,
test_acc:  0.405000,
train_loss:  1.668675,
test_lost:  1.721887
"""


# Collecting data and prepare data for the model.

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test


nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_cifar10()

# Building the Logistic Regression model:

features = 3072
categories = nb_classes

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, categories])
W = tf.Variable(tf.zeros([features, categories]))
b = tf.Variable(tf.zeros([categories]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_,tf.matmul(x, W) + b) )

update = tf.train.AdamOptimizer(0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# for epoch in range(0, 2001):
#     sess.run(update, feed_dict={x: x_train, y_: y_train})  # BGD
#     train_acc, train_loss = sess.run([accuracy, loss] , feed_dict={x: x_train, y_: y_train})
#     if epoch % 100 == 0 and epoch != 0:
#         test_acc, test_loss = sess.run([accuracy, loss], feed_dict={x: x_test, y_: y_test})
#         print("epoch:{epochs: }, train_acc: {train_acc: .6f}, test_acc: {test_acc: .6f}, train_loss: {train_loss: .6f}, test_lost: {test_loss: .6f}".
#               format(epochs=epoch, train_acc=train_acc, test_acc=test_acc, train_loss=train_loss, test_loss=test_loss))


# run with batches:

for i in range(31):
    for j in range(64):
        batch_xs = x_train[j*780:(j+1)*780-1,:]
        batch_ys = y_train[j*780:(j+1)*780-1,:]
        sess.run(update, feed_dict={x: batch_xs, y_: batch_ys})  # BGD
        train_acc, train_loss = sess.run([accuracy, loss], feed_dict={x: batch_xs, y_: batch_ys})
    test_acc, test_loss = sess.run([accuracy, loss], feed_dict={x: x_test, y_: y_test})
    print("epoch:{epochs: }, train_acc: {train_acc: .6f}, test_acc: {test_acc: .6f}, train_loss: {train_loss: .6f}, test_lost: {test_loss: .6f}".
               format(epochs=i, train_acc=train_acc, test_acc=test_acc, train_loss=train_loss, test_loss=test_loss))


"""
OutPut: 
````````
(Adam 0.001)

epoch: 100, train_acc:   0.390720, test_acc:  0.387300, train_loss:  1.783852, test_lost:  1.792325
epoch: 200, train_acc:   0.409520, test_acc:  0.399900, train_loss:  1.734016, test_lost:  1.752294
epoch: 300, train_acc:   0.420800, test_acc:  0.402300, train_loss:  1.707271, test_lost:  1.735123
epoch: 400, train_acc:   0.427320, test_acc:  0.403000, train_loss:  1.689450, test_lost:  1.725634
epoch: 500, train_acc:   0.433340, test_acc:  0.404800, train_loss:  1.676413, test_lost:  1.720480
epoch: 600, train_acc:   0.437040, test_acc:  0.407800, train_loss:  1.666145, test_lost:  1.717407
epoch: 700, train_acc:   0.439360, test_acc:  0.407300, train_loss:  1.658645, test_lost:  1.716164
epoch: 800, train_acc:   0.444120, test_acc:  0.410700, train_loss:  1.650708, test_lost:  1.714642
epoch: 900, train_acc:   0.446840, test_acc:  0.410800, train_loss:  1.644599, test_lost:  1.714135
epoch: 1000, train_acc:  0.448840, test_acc:  0.410800, train_loss:  1.640162, test_lost:  1.715123
epoch: 1100, train_acc:  0.450500, test_acc:  0.411100, train_loss:  1.634955, test_lost:  1.714818
epoch: 1200, train_acc:  0.452320, test_acc:  0.411300, train_loss:  1.630217, test_lost:  1.714367
epoch: 1300, train_acc:  0.453260, test_acc:  0.409600, train_loss:  1.627117, test_lost:  1.715826
epoch: 1400, train_acc:  0.455480, test_acc:  0.411900, train_loss:  1.622661, test_lost:  1.715178
epoch: 1500, train_acc:  0.457020, test_acc:  0.410300, train_loss:  1.619308, test_lost:  1.715744
epoch: 1600, train_acc:  0.457840, test_acc:  0.409000, train_loss:  1.616271, test_lost:  1.716343
epoch: 1700, train_acc:  0.458740, test_acc:  0.409300, train_loss:  1.613532, test_lost:  1.717130
epoch: 1800, train_acc:  0.460360, test_acc:  0.409400, train_loss:  1.610515, test_lost:  1.717538
epoch: 1900, train_acc:  0.458920, test_acc:  0.406000, train_loss:  1.609263, test_lost:  1.719313
epoch: 2000, train_acc:  0.460320, test_acc:  0.408000, train_loss:  1.606134, test_lost:  1.719146
Process finished with exit code 0

batches: 
````````
epoch: 0, train_acc:   0.369705, test_acc:  0.348000, train_loss:  1.859305, test_lost:  1.879896
epoch: 1, train_acc:   0.400513, test_acc:  0.369600, train_loss:  1.802073, test_lost:  1.824074
epoch: 2, train_acc:   0.419769, test_acc:  0.379000, train_loss:  1.777094, test_lost:  1.798091
epoch: 3, train_acc:   0.417202, test_acc:  0.384600, train_loss:  1.762274, test_lost:  1.783322
epoch: 4, train_acc:   0.405648, test_acc:  0.388700, train_loss:  1.751315, test_lost:  1.772861
epoch: 5, train_acc:   0.408216, test_acc:  0.393600, train_loss:  1.742457, test_lost:  1.764929
epoch: 6, train_acc:   0.409499, test_acc:  0.395700, train_loss:  1.735082, test_lost:  1.758686
epoch: 7, train_acc:   0.409499, test_acc:  0.397600, train_loss:  1.728803, test_lost:  1.753636
epoch: 8, train_acc:   0.412067, test_acc:  0.399700, train_loss:  1.723342, test_lost:  1.749460
epoch: 9, train_acc:   0.417202, test_acc:  0.400800, train_loss:  1.718509, test_lost:  1.745945
epoch: 10, train_acc:  0.417202, test_acc:  0.402800, train_loss:  1.714171, test_lost:  1.742945
epoch: 11, train_acc:  0.415918, test_acc:  0.403500, train_loss:  1.710235, test_lost:  1.740357
epoch: 12, train_acc:  0.417202, test_acc:  0.402600, train_loss:  1.706632, test_lost:  1.738104
epoch: 13, train_acc:  0.414634, test_acc:  0.402800, train_loss:  1.703310, test_lost:  1.736130
epoch: 14, train_acc:  0.418485, test_acc:  0.403500, train_loss:  1.700228, test_lost:  1.734388
epoch: 15, train_acc:  0.422336, test_acc:  0.403900, train_loss:  1.697356, test_lost:  1.732844
epoch: 16, train_acc:  0.426187, test_acc:  0.404300, train_loss:  1.694667, test_lost:  1.731470
epoch: 17, train_acc:  0.426187, test_acc:  0.403900, train_loss:  1.692140, test_lost:  1.730242
epoch: 18, train_acc:  0.424904, test_acc:  0.403500, train_loss:  1.689759, test_lost:  1.729142
epoch: 19, train_acc:  0.424904, test_acc:  0.403500, train_loss:  1.687507, test_lost:  1.728153
epoch: 20, train_acc:  0.424904, test_acc:  0.404600, train_loss:  1.685373, test_lost:  1.727262
epoch: 21, train_acc:  0.423620, test_acc:  0.404200, train_loss:  1.683345, test_lost:  1.726457
epoch: 22, train_acc:  0.423620, test_acc:  0.404400, train_loss:  1.681415, test_lost:  1.725730
epoch: 23, train_acc:  0.423620, test_acc:  0.404600, train_loss:  1.679574, test_lost:  1.725072
epoch: 24, train_acc:  0.426187, test_acc:  0.404800, train_loss:  1.677816, test_lost:  1.724475
epoch: 25, train_acc:  0.427471, test_acc:  0.405100, train_loss:  1.676133, test_lost:  1.723933
epoch: 26, train_acc:  0.426187, test_acc:  0.404100, train_loss:  1.674520, test_lost:  1.723442
epoch: 27, train_acc:  0.426187, test_acc:  0.404000, train_loss:  1.672972, test_lost:  1.722996
epoch: 28, train_acc:  0.428755, test_acc:  0.404300, train_loss:  1.671485, test_lost:  1.722590
epoch: 29, train_acc:  0.428755, test_acc:  0.405100, train_loss:  1.670053, test_lost:  1.722221
epoch: 30, train_acc:  0.430039, test_acc:  0.405000, train_loss:  1.668675, test_lost:  1.721887

Process finished with exit code 0

"""
