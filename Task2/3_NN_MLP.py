import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical

tf.compat.v1.disable_eager_execution()

"""
NN Cifar10:
``````````
* Two layers NN. (512, 512)
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

features = 3072
categories = nb_classes

(hidden1_size, hidden2_size) = (512, 512)  # Two Hidden layers

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, categories])

# first layer
W1 = tf.Variable(tf.truncated_normal([features, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))

z1 = tf.nn.elu(tf.matmul(x, W1) + b1)

# second layer
W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))

z2 = tf.nn.elu(tf.matmul(z1, W2) + b2)

# third layer
W3 = tf.Variable(tf.truncated_normal([hidden2_size, 10], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[10]))

y = tf.nn.softmax(tf.matmul(z2, W3) + b3)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(y_, tf.matmul(z2, W3) + b3, axis=1)
cross_entropy = tf.reduce_mean(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# for epoch in range(0, 1001):
#     train_acc = sess.run(accuracy, feed_dict={x: x_train, y_: y_train})
#     _, loss_val1 = sess.run([train_step, cross_entropy],feed_dict={x: x_train, y_: y_train})
#
#     #if (epoch % 100 == 0):
#     #  print("epoch: %3d train_acc: %f loss1: %f" % (epoch, train_acc, loss_val1))
#     if(epoch % 50 == 0 and epoch != 0):
#         test_acc = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
#         _, loss_val2 = sess.run([train_step, cross_entropy], feed_dict={x: x_test, y_: y_test})
#
#         print("epoch: %3d train_acc: %f test_acc: %f train_loss1: %f test_loss: %f " % (epoch, train_acc, test_acc, loss_val1, loss_val2))

for i in range(51):
    for j in range(100):
        batch_xs = x_train[j * 500:(j + 1) * 500 - 1, :]
        batch_ys = y_train[j * 500:(j + 1) * 500 - 1, :]
        train_acc, _, train_loss = sess.run([accuracy, train_step, cross_entropy],
                                            feed_dict={x: batch_xs, y_: batch_ys})

    test_acc, _, test_loss = sess.run([accuracy, train_step, cross_entropy], feed_dict={x: x_test, y_: y_test})
    print("epoch: %3d train_acc: %f test_acc: %f train_loss1: %f test_loss: %f " % (
        i, train_acc, test_acc, train_loss, test_loss))

"""
Adam ( 0.001 )
(hidden1_size, hidden2_size) = (512, 512)
activation: relu

epoch:   0 train_acc: 0.340681 test_acc: 0.346300 train_loss1: 1.886697 test_loss: 1.848496 
epoch:   1 train_acc: 0.406814 test_acc: 0.387600 train_loss1: 1.749767 test_loss: 1.724008 
epoch:   2 train_acc: 0.438878 test_acc: 0.417600 train_loss1: 1.665179 test_loss: 1.656626 
epoch:   3 train_acc: 0.458918 test_acc: 0.435400 train_loss1: 1.585773 test_loss: 1.611818 
epoch:   4 train_acc: 0.488978 test_acc: 0.448800 train_loss1: 1.526593 test_loss: 1.575887 
epoch:   5 train_acc: 0.486974 test_acc: 0.456600 train_loss1: 1.497826 test_loss: 1.550367 
epoch:   6 train_acc: 0.498998 test_acc: 0.459500 train_loss1: 1.431433 test_loss: 1.539771 
epoch:   7 train_acc: 0.537074 test_acc: 0.481400 train_loss1: 1.385219 test_loss: 1.485014 
epoch:   8 train_acc: 0.541082 test_acc: 0.483300 train_loss1: 1.346718 test_loss: 1.484440 
epoch:   9 train_acc: 0.527054 test_acc: 0.490400 train_loss1: 1.334555 test_loss: 1.448758 
epoch:  10 train_acc: 0.537074 test_acc: 0.492900 train_loss1: 1.303872 test_loss: 1.438214 
epoch:  11 train_acc: 0.535070 test_acc: 0.496300 train_loss1: 1.301425 test_loss: 1.430930 
epoch:  12 train_acc: 0.557114 test_acc: 0.496500 train_loss1: 1.263085 test_loss: 1.429736 
epoch:  13 train_acc: 0.565130 test_acc: 0.502300 train_loss1: 1.247998 test_loss: 1.429688 
epoch:  14 train_acc: 0.585170 test_acc: 0.497500 train_loss1: 1.218870 test_loss: 1.431898 
epoch:  15 train_acc: 0.575150 test_acc: 0.494000 train_loss1: 1.217144 test_loss: 1.445559 
epoch:  16 train_acc: 0.591182 test_acc: 0.495600 train_loss1: 1.170306 test_loss: 1.444378 
epoch:  17 train_acc: 0.595190 test_acc: 0.494000 train_loss1: 1.143654 test_loss: 1.449620 
epoch:  18 train_acc: 0.611222 test_acc: 0.503800 train_loss1: 1.117916 test_loss: 1.432820 
epoch:  19 train_acc: 0.591182 test_acc: 0.501600 train_loss1: 1.095004 test_loss: 1.456624 
epoch:  20 train_acc: 0.599198 test_acc: 0.502700 train_loss1: 1.099928 test_loss: 1.456316 
epoch:  21 train_acc: 0.617234 test_acc: 0.508100 train_loss1: 1.085128 test_loss: 1.436423 
epoch:  22 train_acc: 0.627254 test_acc: 0.512400 train_loss1: 1.064150 test_loss: 1.428190 
epoch:  23 train_acc: 0.617234 test_acc: 0.510000 train_loss1: 1.059500 test_loss: 1.442518 
epoch:  24 train_acc: 0.617234 test_acc: 0.509800 train_loss1: 1.037862 test_loss: 1.446921 
epoch:  25 train_acc: 0.617234 test_acc: 0.509800 train_loss1: 1.011631 test_loss: 1.466501 
epoch:  26 train_acc: 0.643287 test_acc: 0.513700 train_loss1: 0.976908 test_loss: 1.459831 
epoch:  27 train_acc: 0.653307 test_acc: 0.521100 train_loss1: 0.991341 test_loss: 1.440614 
epoch:  28 train_acc: 0.643287 test_acc: 0.513000 train_loss1: 1.015728 test_loss: 1.463923 
epoch:  29 train_acc: 0.617234 test_acc: 0.500800 train_loss1: 0.991602 test_loss: 1.512645 
epoch:  30 train_acc: 0.653307 test_acc: 0.509700 train_loss1: 1.004565 test_loss: 1.480759 
epoch:  31 train_acc: 0.627254 test_acc: 0.504200 train_loss1: 0.997810 test_loss: 1.497196 
epoch:  32 train_acc: 0.661323 test_acc: 0.508900 train_loss1: 0.961682 test_loss: 1.501720 
epoch:  33 train_acc: 0.675351 test_acc: 0.501900 train_loss1: 0.938797 test_loss: 1.529199 
epoch:  34 train_acc: 0.677355 test_acc: 0.507400 train_loss1: 0.915100 test_loss: 1.506687 
epoch:  35 train_acc: 0.671343 test_acc: 0.511500 train_loss1: 0.924807 test_loss: 1.496607 
epoch:  36 train_acc: 0.673347 test_acc: 0.517100 train_loss1: 0.920988 test_loss: 1.477580 
epoch:  37 train_acc: 0.655311 test_acc: 0.511500 train_loss1: 0.927179 test_loss: 1.503760 
epoch:  38 train_acc: 0.677355 test_acc: 0.508700 train_loss1: 0.927875 test_loss: 1.528415 
epoch:  39 train_acc: 0.643287 test_acc: 0.514800 train_loss1: 0.952974 test_loss: 1.516359 
epoch:  40 train_acc: 0.661323 test_acc: 0.522100 train_loss1: 0.923640 test_loss: 1.517724 
epoch:  41 train_acc: 0.669339 test_acc: 0.518900 train_loss1: 0.927088 test_loss: 1.495250 
epoch:  42 train_acc: 0.689379 test_acc: 0.515200 train_loss1: 0.854152 test_loss: 1.513031 
epoch:  43 train_acc: 0.727455 test_acc: 0.521900 train_loss1: 0.815665 test_loss: 1.533466 
epoch:  44 train_acc: 0.697395 test_acc: 0.519800 train_loss1: 0.835820 test_loss: 1.545685 
epoch:  45 train_acc: 0.661323 test_acc: 0.525500 train_loss1: 0.857653 test_loss: 1.533215 
epoch:  46 train_acc: 0.693387 test_acc: 0.520000 train_loss1: 0.842126 test_loss: 1.535173 
epoch:  47 train_acc: 0.703407 test_acc: 0.521000 train_loss1: 0.806487 test_loss: 1.536048 
epoch:  48 train_acc: 0.713427 test_acc: 0.516400 train_loss1: 0.849747 test_loss: 1.525640 
epoch:  49 train_acc: 0.713427 test_acc: 0.516100 train_loss1: 0.853238 test_loss: 1.543972 
epoch:  50 train_acc: 0.701403 test_acc: 0.517300 train_loss1: 0.845708 test_loss: 1.547436 

Process finished with exit code 0


"""


"""
Adam ( 0.001 )
(hidden1_size, hidden2_size) = (512, 512)
activation: elu

epoch:   0 train_acc: 0.334669 test_acc: 0.341300 train_loss1: 1.910673 test_loss: 1.878451 
epoch:   1 train_acc: 0.384770 test_acc: 0.390400 train_loss1: 1.809605 test_loss: 1.734599 
epoch:   2 train_acc: 0.418838 test_acc: 0.420900 train_loss1: 1.717995 test_loss: 1.662077 
epoch:   3 train_acc: 0.442886 test_acc: 0.432500 train_loss1: 1.633156 test_loss: 1.627724 
epoch:   4 train_acc: 0.462926 test_acc: 0.441300 train_loss1: 1.559513 test_loss: 1.608062 
epoch:   5 train_acc: 0.478958 test_acc: 0.448800 train_loss1: 1.503449 test_loss: 1.584348 
epoch:   6 train_acc: 0.486974 test_acc: 0.449000 train_loss1: 1.454077 test_loss: 1.572353 
epoch:   7 train_acc: 0.498998 test_acc: 0.457600 train_loss1: 1.409694 test_loss: 1.545426 
epoch:   8 train_acc: 0.494990 test_acc: 0.466000 train_loss1: 1.376826 test_loss: 1.521306 
epoch:   9 train_acc: 0.513026 test_acc: 0.476800 train_loss1: 1.337950 test_loss: 1.492699 
epoch:  10 train_acc: 0.525050 test_acc: 0.480000 train_loss1: 1.312681 test_loss: 1.480509 
epoch:  11 train_acc: 0.533066 test_acc: 0.491400 train_loss1: 1.296153 test_loss: 1.460040 
epoch:  12 train_acc: 0.537074 test_acc: 0.502000 train_loss1: 1.285604 test_loss: 1.435085 
epoch:  13 train_acc: 0.549098 test_acc: 0.509000 train_loss1: 1.257542 test_loss: 1.424175 
epoch:  14 train_acc: 0.573146 test_acc: 0.509100 train_loss1: 1.243030 test_loss: 1.413647 
epoch:  15 train_acc: 0.569138 test_acc: 0.508000 train_loss1: 1.243962 test_loss: 1.413108 
epoch:  16 train_acc: 0.565130 test_acc: 0.503800 train_loss1: 1.200497 test_loss: 1.436882 
epoch:  17 train_acc: 0.583166 test_acc: 0.499800 train_loss1: 1.152090 test_loss: 1.435037 
epoch:  18 train_acc: 0.577154 test_acc: 0.506000 train_loss1: 1.130827 test_loss: 1.416680 
epoch:  19 train_acc: 0.611222 test_acc: 0.513900 train_loss1: 1.098616 test_loss: 1.400322 
epoch:  20 train_acc: 0.617234 test_acc: 0.520000 train_loss1: 1.063813 test_loss: 1.395797 
epoch:  21 train_acc: 0.633267 test_acc: 0.520400 train_loss1: 1.036345 test_loss: 1.400433 
epoch:  22 train_acc: 0.633267 test_acc: 0.526900 train_loss1: 1.026297 test_loss: 1.394304 
epoch:  23 train_acc: 0.641283 test_acc: 0.513300 train_loss1: 0.964767 test_loss: 1.460297 
epoch:  24 train_acc: 0.657315 test_acc: 0.504900 train_loss1: 0.940650 test_loss: 1.470470 
epoch:  25 train_acc: 0.665331 test_acc: 0.517000 train_loss1: 0.933166 test_loss: 1.451062 
epoch:  26 train_acc: 0.655311 test_acc: 0.520700 train_loss1: 0.920440 test_loss: 1.444265 
epoch:  27 train_acc: 0.661323 test_acc: 0.525400 train_loss1: 0.945834 test_loss: 1.431093 
epoch:  28 train_acc: 0.667335 test_acc: 0.514000 train_loss1: 0.946637 test_loss: 1.467386 
epoch:  29 train_acc: 0.657315 test_acc: 0.512900 train_loss1: 0.934208 test_loss: 1.473829 
epoch:  30 train_acc: 0.669339 test_acc: 0.517400 train_loss1: 0.916730 test_loss: 1.456837 
epoch:  31 train_acc: 0.679359 test_acc: 0.512000 train_loss1: 0.854138 test_loss: 1.501726 
epoch:  32 train_acc: 0.695391 test_acc: 0.512800 train_loss1: 0.843074 test_loss: 1.518347 
epoch:  33 train_acc: 0.701403 test_acc: 0.504800 train_loss1: 0.823465 test_loss: 1.561593 
epoch:  34 train_acc: 0.703407 test_acc: 0.518700 train_loss1: 0.805986 test_loss: 1.520694 
epoch:  35 train_acc: 0.707415 test_acc: 0.514300 train_loss1: 0.785823 test_loss: 1.553086 
epoch:  36 train_acc: 0.709419 test_acc: 0.522200 train_loss1: 0.774622 test_loss: 1.543430 
epoch:  37 train_acc: 0.683367 test_acc: 0.516500 train_loss1: 0.819684 test_loss: 1.562811 
epoch:  38 train_acc: 0.743487 test_acc: 0.505300 train_loss1: 0.761124 test_loss: 1.579249 
epoch:  39 train_acc: 0.715431 test_acc: 0.518600 train_loss1: 0.788130 test_loss: 1.553794 
epoch:  40 train_acc: 0.689379 test_acc: 0.529600 train_loss1: 0.788015 test_loss: 1.532653 
epoch:  41 train_acc: 0.723447 test_acc: 0.527100 train_loss1: 0.726888 test_loss: 1.564312 
epoch:  42 train_acc: 0.733467 test_acc: 0.512700 train_loss1: 0.752134 test_loss: 1.639098 
epoch:  43 train_acc: 0.725451 test_acc: 0.517400 train_loss1: 0.761825 test_loss: 1.622760 
epoch:  44 train_acc: 0.725451 test_acc: 0.519700 train_loss1: 0.745069 test_loss: 1.621219 
epoch:  45 train_acc: 0.747495 test_acc: 0.531000 train_loss1: 0.713821 test_loss: 1.583121 
epoch:  46 train_acc: 0.739479 test_acc: 0.535600 train_loss1: 0.749074 test_loss: 1.591335 
epoch:  47 train_acc: 0.751503 test_acc: 0.535500 train_loss1: 0.729273 test_loss: 1.607264 
epoch:  48 train_acc: 0.747495 test_acc: 0.535000 train_loss1: 0.698376 test_loss: 1.647036 
epoch:  49 train_acc: 0.745491 test_acc: 0.532100 train_loss1: 0.677824 test_loss: 1.679078 
epoch:  50 train_acc: 0.771543 test_acc: 0.531300 train_loss1: 0.653315 test_loss: 1.678059 

Process finished with exit code 0

"""