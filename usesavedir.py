import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data




mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

def weight_variable(shape,name):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.name_scope('Input'):
    x= tf.placeholder(tf.float32,[None,784])
    y= tf.placeholder(tf.float32,[None,10])
    with tf.name_scope('x_image'):
       x_image = tf.reshape(x,[-1,28,28,1],name='x_image')
#first 卷积
with tf.name_scope('Conv1'):
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5,5,1,32],name='W_conv1')
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32],name='b_conv1')

    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image,W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

#second 卷积
with tf.name_scope('Conv2'):
    with tf.name_scope('W_conv2'):
        W_cow2 = weight_variable([5,5,32,64],name='W_conv2')
    with tf.name_scope('b_conv2'):
        b_cov2 = bias_variable([64],name='b_conv2')

    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1,W_cow2) + b_cov2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

#first 全连接
with tf.name_scope('fc1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7*7*64,1024],name='W_fc1')
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024],name='b_fc1')
    #把池化层2的输出扁平化为1
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64],name = 'h_pool2_flat')
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat,W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop =tf.nn.dropout(h_fc1,keep_prob , name = 'h_fc1_drop')

#second 全连接
with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024,10],name = 'W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10],name = 'b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=wx_plus_b2))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(wx_plus_b2,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 占用80%显存
session = tf.Session(config=config)

saver = tf.train.Saver()

with tf.Session() as sess:


    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test',sess.graph)

    for epoch in range(1001):

            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

            summary = sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
            train_writer.add_summary(summary,epoch)

            batch_xs, batch_ys = mnist.test.next_batch(batch_size)
            summary =sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
            test_writer.add_summary(summary,epoch)

            if epoch%50==0:
                test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.7})
                train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images[:10000],y:mnist.train.labels[:10000],keep_prob:1.0})
                print("iter " + str(epoch) + "testing Accuracy= " +str(test_acc) + "training Accuracy= " +str(train_acc))

    saver.save(sess,'net/my_net.ckpt')