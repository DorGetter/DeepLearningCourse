import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical

tf.compat.v1.disable_eager_execution()


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

(hidden1_size, hidden2_size, hidden3_size, hidden4_size) = (512, 256, 128, 64) # 4 hidden layers

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, categories])
W1 = tf.Variable(tf.truncated_normal([features, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))

z1 = tf.nn.elu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))

z2 = tf.nn.elu(tf.matmul(z1, W2) + b2)


W3 = tf.Variable(tf.truncated_normal([hidden2_size, hidden3_size], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[hidden3_size]))

z3 = tf.nn.elu(tf.matmul(z2, W3) + b3)


W4 = tf.Variable(tf.truncated_normal([hidden3_size, hidden4_size], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[hidden4_size]))

z4 = tf.nn.elu(tf.matmul(z3, W4) + b4)

W5 = tf.Variable(tf.truncated_normal([hidden4_size, 10], stddev=0.1))
b5 = tf.Variable(tf.constant(0.1, shape=[10]))

y = tf.nn.softmax(tf.matmul(z4, W5) + b5)


loss = tf.nn.softmax_cross_entropy_with_logits_v2(y_, tf.matmul(z4, W5) + b5, axis=1)
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

for i in range (21):
    for j in range(100):
        batch_xs = x_train[j*500:(j+1)*500-1,:]
        batch_ys = y_train[j*500:(j+1)*500-1,:]
        train_acc, _, train_loss = sess.run([accuracy, train_step, cross_entropy],feed_dict={x: batch_xs, y_: batch_ys})

    test_acc, _, test_loss = sess.run([accuracy, train_step, cross_entropy],feed_dict={x: x_test, y_: y_test})

    print("epoch: %3d train_acc: %f test_acc: %f train_loss1: %f test_loss: %f " % (
    i, train_acc, test_acc, train_loss, test_loss))


"""
(hidden1_size, hidden2_size, hidden3_size, hidden4_size) = (512,256 ,128 ,64 )

epoch: 200 train_acc:  0.457380 test_acc: 0.457300 train_loss1: 1.546040 test_loss: 1.558456 
epoch: 400 train_acc:  0.523040 test_acc: 0.485200 train_loss1: 1.356413 test_loss: 1.446634 
epoch: 600 train_acc:  0.574380 test_acc: 0.512100 train_loss1: 1.214561 test_loss: 1.384030 
epoch: 800 train_acc:  0.617580 test_acc: 0.509600 train_loss1: 1.079059 test_loss: 1.407974 
epoch: 1000 train_acc: 0.651660 test_acc: 0.527000 train_loss1: 0.976566 test_loss: 1.401817 


batches: 
epoch:   0 train_acc: 0.338677 test_acc: 0.291400 train_loss1: 1.851309 test_loss: 1.918327 
epoch:   1 train_acc: 0.426854 test_acc: 0.412400 train_loss1: 1.698911 test_loss: 1.650582 
epoch:   2 train_acc: 0.430862 test_acc: 0.426900 train_loss1: 1.604165 test_loss: 1.600816 
epoch:   3 train_acc: 0.474950 test_acc: 0.446700 train_loss1: 1.538269 test_loss: 1.562348 
epoch:   4 train_acc: 0.484970 test_acc: 0.460700 train_loss1: 1.498440 test_loss: 1.524146 
epoch:   5 train_acc: 0.519038 test_acc: 0.475800 train_loss1: 1.409481 test_loss: 1.493421 
epoch:   6 train_acc: 0.543086 test_acc: 0.485300 train_loss1: 1.349111 test_loss: 1.462312 
epoch:   7 train_acc: 0.547094 test_acc: 0.488700 train_loss1: 1.336687 test_loss: 1.444235 
epoch:   8 train_acc: 0.529058 test_acc: 0.505000 train_loss1: 1.304458 test_loss: 1.400280 
epoch:   9 train_acc: 0.531062 test_acc: 0.500400 train_loss1: 1.299447 test_loss: 1.419435 
epoch:  10 train_acc: 0.537074 test_acc: 0.501200 train_loss1: 1.243132 test_loss: 1.410362 
epoch:  11 train_acc: 0.569138 test_acc: 0.513500 train_loss1: 1.205818 test_loss: 1.374455 
epoch:  12 train_acc: 0.575150 test_acc: 0.515300 train_loss1: 1.168585 test_loss: 1.378505 
epoch:  13 train_acc: 0.583166 test_acc: 0.514500 train_loss1: 1.158508 test_loss: 1.374440 
epoch:  14 train_acc: 0.587174 test_acc: 0.516600 train_loss1: 1.141322 test_loss: 1.379013 
epoch:  15 train_acc: 0.559118 test_acc: 0.511500 train_loss1: 1.140380 test_loss: 1.403964 
epoch:  16 train_acc: 0.575150 test_acc: 0.517400 train_loss1: 1.116380 test_loss: 1.375073 
epoch:  17 train_acc: 0.579158 test_acc: 0.514600 train_loss1: 1.092824 test_loss: 1.387553 
epoch:  18 train_acc: 0.589178 test_acc: 0.521400 train_loss1: 1.039294 test_loss: 1.368665 

Process finished with exit code 0

batches: (second try) 

epoch:   0 train_acc: 0.368737 test_acc: 0.345800 train_loss1: 1.873381 test_loss: 1.850723 
epoch:   1 train_acc: 0.404810 test_acc: 0.397700 train_loss1: 1.730209 test_loss: 1.694477 
epoch:   2 train_acc: 0.456914 test_acc: 0.427100 train_loss1: 1.631842 test_loss: 1.626000 
epoch:   3 train_acc: 0.452906 test_acc: 0.456800 train_loss1: 1.565364 test_loss: 1.536135 
epoch:   4 train_acc: 0.472946 test_acc: 0.451900 train_loss1: 1.504527 test_loss: 1.523404 
epoch:   5 train_acc: 0.492986 test_acc: 0.471500 train_loss1: 1.466792 test_loss: 1.484307 
epoch:   6 train_acc: 0.509018 test_acc: 0.477300 train_loss1: 1.423200 test_loss: 1.459376 
epoch:   7 train_acc: 0.527054 test_acc: 0.483400 train_loss1: 1.379737 test_loss: 1.442524 
epoch:   8 train_acc: 0.547094 test_acc: 0.486400 train_loss1: 1.322408 test_loss: 1.437115 
epoch:   9 train_acc: 0.549098 test_acc: 0.489900 train_loss1: 1.277171 test_loss: 1.429731 
epoch:  10 train_acc: 0.539078 test_acc: 0.500600 train_loss1: 1.256508 test_loss: 1.404081 
epoch:  11 train_acc: 0.563126 test_acc: 0.513500 train_loss1: 1.213115 test_loss: 1.376702 
epoch:  12 train_acc: 0.567134 test_acc: 0.506800 train_loss1: 1.234116 test_loss: 1.390145 
epoch:  13 train_acc: 0.561122 test_acc: 0.508300 train_loss1: 1.211678 test_loss: 1.390786 
epoch:  14 train_acc: 0.573146 test_acc: 0.514200 train_loss1: 1.159135 test_loss: 1.373450 
epoch:  15 train_acc: 0.599198 test_acc: 0.522300 train_loss1: 1.124177 test_loss: 1.363203 
epoch:  16 train_acc: 0.599198 test_acc: 0.518100 train_loss1: 1.119353 test_loss: 1.388747 
epoch:  17 train_acc: 0.597194 test_acc: 0.522800 train_loss1: 1.106771 test_loss: 1.364775 
 

Process finished with exit code 0


"""