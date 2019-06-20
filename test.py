import tensorflow as tf
from load_data import *
import scipy.io as sio
import kNN_cosine
from numpy import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def compute_accuracy(test_att, test_word, test_visual, test_id, img_list):
    global center_1
    print('Testing')
    pre = sess.run(center_1, feed_dict={att_features: test_att, word_features: test_word, train: False})
    test_id = np.squeeze(np.asarray(test_id))
    fobj = open('submit.txt', 'w')
    for a in range(11740):
        outputLabel = kNN_cosine.kNNClassify(test_visual[a, :], pre, test_id, 1)
        fobj.write(img_list[a] + '\t' + label_list_train[int(outputLabel)]+'\n')
        print(a)
    fobj.close
    return 1


# # data

validation_embeddings = sio.loadmat('./DatasetB_20180919/validation_embedding.mat')['validation_embedding']
word_embeddings = sio.loadmat('./DatasetB_20180919/word_embedding2.mat')['word_embedding2']
attributes = sio.loadmat('./DatasetB_20180919/full_attributes.mat')['full_attributes']
validation_attributes = sio.loadmat('./DatasetB_20180919/full_validation_attributes.mat')['full_validation_attributes']

# test_id = sio.loadmat('./DatasetA_train_20180813/labels.mat')['labels']

# test_label, x_test = load_test_data()
img_list, x_test = load_data_test()

iteration = 1364
batch_size = 64

# # Placeholder

# define placeholder for inputs to network
att_features = tf.placeholder(tf.float32, [None, 30])
word_features = tf.placeholder(tf.float32, [None, 300])
visual_features = tf.placeholder(tf.float32, [None, 2048])
train = tf.placeholder(tf.bool)

# # Network

W_left_w1_1 = weight_variable([300, 1600])
b_left_w1_1 = bias_variable([1600])
left_w1_1 = tf.nn.relu(tf.matmul(word_features, W_left_w1_1) + b_left_w1_1)
left_w1_1 = tf.layers.batch_normalization(left_w1_1, training=train)

W_left_a1_1 = weight_variable([30, 1600])
b_left_a1_1 = bias_variable([1600])
left_a1_1 = tf.nn.relu(tf.matmul(att_features, W_left_a1_1) + b_left_a1_1)
left_a1_1 = tf.layers.batch_normalization(left_a1_1, training=train)

multimodal = 4 * left_w1_1 + left_a1_1
# multimodal = left_a1_1
multimodal = tf.layers.dropout(multimodal, rate=0.2, training=train)
multimodal = tf.layers.batch_normalization(multimodal, training=train)

W_center_1 = weight_variable([1600, 2048])
b_center_1 = bias_variable([2048])
center_1 = tf.nn.relu((tf.matmul(multimodal, W_center_1) + b_center_1))

# # loss


loss = tf.reduce_mean(tf.square(center_1 - visual_features))

# L2 regularisation for the fully connected parameters.
regularisers_1 = tf.nn.l2_loss(W_left_a1_1) + tf.nn.l2_loss(
    b_left_a1_1)  # + tf.nn.l2_loss(W_left_a1_2) + tf.nn.l2_loss(b_left_a1_2)
regularisers_2 = tf.nn.l2_loss(W_left_w1_1) + tf.nn.l2_loss(
    b_left_w1_1)  # + tf.nn.l2_loss(W_left_w1_2) + tf.nn.l2_loss(b_left_w1_2)
regularisers_3 = tf.nn.l2_loss(W_center_1) + tf.nn.l2_loss(b_center_1)

regularisers = 1e-3 * regularisers_2 + 1e-2 * regularisers_3 + 1e-2 * regularisers_1
# regularisers = 1e-2 * regularisers_3 + 1e-2 * regularisers_1
# Add the regularization term to the loss.
loss += regularisers

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)



saver = tf.train.Saver(tf.global_variables())

sess = tf.Session()
# sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state('./zsl_model1')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    # print("load sucess")
else:
    sess.run(tf.global_variables_initializer())
# # Run
# iter_ = data_iterator()
test_id = np.arange(0, 285)
for i in range(1):
    c = compute_accuracy(attributes, word_embeddings, x_test, test_id, img_list)