import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
from keras.utils import to_categorical

tf.compat.v1.disable_eager_execution()


"""
First try- Logistic Regression Cifar10:
``````````
* softmax
* average cross entropy loss function
* Gradient Descent optimizer
* learning rate = 0.01

results: 
--------
epoch: 30, 
train_acc:  0.351733,
test_acc:  0.339900, 
train_loss:  0.192839, 
test_lost:  0.193772

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

loss = -tf.reduce_mean(y_ * tf.log(y))

update = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

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
#
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
epoch: 0, train_acc:   0.175866, test_acc:  0.171800, train_loss:  0.224892, test_lost:  0.225241
epoch: 1, train_acc:   0.229782, test_acc:  0.217700, train_loss:  0.221087, test_lost:  0.221456
epoch: 2, train_acc:   0.254172, test_acc:  0.236300, train_loss:  0.217960, test_lost:  0.218339
epoch: 3, train_acc:   0.263158, test_acc:  0.251200, train_loss:  0.215343, test_lost:  0.215729
epoch: 4, train_acc:   0.273427, test_acc:  0.263000, train_loss:  0.213117, test_lost:  0.213509
epoch: 5, train_acc:   0.284981, test_acc:  0.274800, train_loss:  0.211195, test_lost:  0.211595
epoch: 6, train_acc:   0.295250, test_acc:  0.281200, train_loss:  0.209514, test_lost:  0.209923
epoch: 7, train_acc:   0.302952, test_acc:  0.289800, train_loss:  0.208028, test_lost:  0.208448
epoch: 8, train_acc:   0.309371, test_acc:  0.294200, train_loss:  0.206699, test_lost:  0.207133
epoch: 9, train_acc:   0.318357, test_acc:  0.299800, train_loss:  0.205501, test_lost:  0.205951
epoch: 10, train_acc:  0.324775, test_acc:  0.303100, train_loss:  0.204413, test_lost:  0.204880
epoch: 11, train_acc:  0.327343, test_acc:  0.306500, train_loss:  0.203418, test_lost:  0.203904
epoch: 12, train_acc:  0.326059, test_acc:  0.310300, train_loss:  0.202502, test_lost:  0.203010
epoch: 13, train_acc:  0.328626, test_acc:  0.312800, train_loss:  0.201655, test_lost:  0.202185
epoch: 14, train_acc:  0.328626, test_acc:  0.315700, train_loss:  0.200868, test_lost:  0.201421
epoch: 15, train_acc:  0.333761, test_acc:  0.318500, train_loss:  0.200134, test_lost:  0.200711
epoch: 16, train_acc:  0.336329, test_acc:  0.320500, train_loss:  0.199446, test_lost:  0.200047
epoch: 17, train_acc:  0.337612, test_acc:  0.321700, train_loss:  0.198800, test_lost:  0.199426
epoch: 18, train_acc:  0.340180, test_acc:  0.323400, train_loss:  0.198191, test_lost:  0.198842
epoch: 19, train_acc:  0.342747, test_acc:  0.325700, train_loss:  0.197616, test_lost:  0.198292
epoch: 20, train_acc:  0.342747, test_acc:  0.328800, train_loss:  0.197071, test_lost:  0.197772
epoch: 21, train_acc:  0.346598, test_acc:  0.330800, train_loss:  0.196554, test_lost:  0.197280
epoch: 22, train_acc:  0.351733, test_acc:  0.332700, train_loss:  0.196062, test_lost:  0.196813
epoch: 23, train_acc:  0.347882, test_acc:  0.333500, train_loss:  0.195594, test_lost:  0.196369
epoch: 24, train_acc:  0.349166, test_acc:  0.334900, train_loss:  0.195147, test_lost:  0.195947
epoch: 25, train_acc:  0.349166, test_acc:  0.335800, train_loss:  0.194720, test_lost:  0.195543
epoch: 26, train_acc:  0.351733, test_acc:  0.336100, train_loss:  0.194312, test_lost:  0.195158
epoch: 27, train_acc:  0.351733, test_acc:  0.337000, train_loss:  0.193921, test_lost:  0.194789
epoch: 28, train_acc:  0.353017, test_acc:  0.338600, train_loss:  0.193545, test_lost:  0.194436
epoch: 29, train_acc:  0.351733, test_acc:  0.339600, train_loss:  0.193185, test_lost:  0.194097
epoch: 30, train_acc:  0.351733, test_acc:  0.339900, train_loss:  0.192839, test_lost:  0.193772

Process finished with exit code 0

"""


