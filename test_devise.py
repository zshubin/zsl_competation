from __future__ import division, print_function, absolute_import
import tensorflow as tf
from load_data_devise import *
import numpy as np
import math
from kNN_cosine import *
import scipy.io as sio
from tensorflow.python.framework import graph_util
# from word2vec import *

embedding_vector = sio.loadmat('./DatasetB_20180919/word_embedding2.mat')['word_embedding2']

# label_index = np.argmax(Y_test_1, axis=0)
true_class_old = []
batch_size = 64
epochs = 1
iteration = 566
validation_iter = 184
#evaluate loss that DeViSE provided.
def get_loss(prediction, length, ys):
    with tf.name_scope('evaluate_loss'):
        global true_class_old
        label_size = 285
        prediction = tf.reshape(prediction,(-1, 300))
        logit_old = tf.matmul(embedding_vector ,tf.transpose(tf.nn.l2_normalize(prediction, dim=1)))
        logit = tf.transpose(logit_old)

        true_class_old = tf.argmax(ys, axis = 1)
        logit_flat = tf.reshape(logit, [-1])
        number = label_size * np.arange(batch_size, dtype=np.int64).reshape(-1)
        true_class = true_class_old + number
        t_label = tf.reshape(tf.gather(logit_flat, true_class), [1, batch_size])
        martix_t_label_one = tf.ones(shape = [label_size, 1], dtype = tf.float32)

        martix_t_label = tf.matmul(martix_t_label_one, t_label)

        margin = 0.1
        margin_one = tf.ones(shape=[label_size, batch_size], dtype = tf.float32)
        matrix_margin = margin * margin_one
        matrix_logit = tf.transpose(logit)
        loss = (tf.reduce_sum(tf.nn.relu(matrix_margin - martix_t_label + matrix_logit)) - batch_size*margin) / batch_size
        return loss


frozen_graph = "Visual_model.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

inputs = graph.get_tensor_by_name("pool_5:0")
samples = graph.get_tensor_by_name("input_holder:0")

with tf.Session(graph=graph) as sess:
    ys = tf.placeholder(tf.float32, shape=[None, 285], name="labels")
    prediction = tf.contrib.layers.fully_connected(
    inputs=inputs,
    num_outputs=300,
    weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
    scope="transformation"
    )
    # length = tf.placeholder(tf.float32, shape=[1,1],name="length")
    loss = get_loss(prediction, batch_size, ys)
    Trainloss = tf.summary.scalar('Trainloss', loss)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # sess.run(init)
    saver.restore(sess, './devise_model/devise.ckpt')
    writer = tf.summary.FileWriter("logs/", sess.graph)

    fobj = open('submit.txt', 'w')
    # for epoch in range(epochs):
    #     if epoch % 1 == 0:
    v_index = 0
    v_img_name = os.listdir('./DatasetB_20180919/test/')
    for num in range(1, validation_iter + 1):
        v_true_num = 0
        if v_index + batch_size < 11740:
            v_input_x = v_img_name[v_index: v_index + batch_size]
        else:
            v_input_x = v_img_name[v_index:]
        v_index += batch_size
        v_batch_x = load_test_data(v_input_x, len(v_input_x))
        # v_batch_x = color_preprocessing(v_batch_x)
        feed_dicts = {samples: v_batch_x}
        pred = sess.run(prediction, feed_dict=feed_dicts)
        pred = tf.reshape(pred, (-1, 300))
        result = tf.matmul(embedding_vector, tf.transpose(tf.nn.l2_normalize(pred, dim=1)))
        result = tf.argmax(result).eval()
        for a in range(len(v_input_x)):
            fobj.write('\n'+v_input_x[a][0:32]+'.jpg' + '\t' + label_list[int(result[a])])
        print(num)
    fobj.close()
