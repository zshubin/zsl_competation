from __future__ import division, print_function, absolute_import
import tensorflow as tf
from load_data_devise import *
import numpy as np
import math
from kNN_cosine import *
import scipy.io as sio
from tensorflow.python.framework import graph_util

embedding_vector = sio.loadmat('./DatasetB_20180919/word_embedding2.mat')['word_embedding2']

true_class_old = []
batch_size = 64
epochs = 100
iteration = 1081
validation_iter = 283
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
    loss = get_loss(prediction, batch_size, ys)
    Trainloss = tf.summary.scalar('Trainloss', loss)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # sess.run(init)
    saver.restore(sess, './devise_model/devise.ckpt')
    writer = tf.summary.FileWriter("logs/", sess.graph)

    for epoch in range(epochs):
        train_dic = {}
        with open('./DatasetB_20180919/train_list.txt') as fp:
            for line in fp:
                wn, name = line.split('\t')
                train_dic[wn] = name.strip()
        train_img_label = list()
        train_img_name = list()
        for f_name, l_name in train_dic.items():
            train_img_name.append(str(f_name))
            train_img_label.append(str(l_name))
        train_img_name = np.array(train_img_name)
        train_img_label = np.array(train_img_label)
        indices = np.random.permutation(len(train_img_label))
        train_img_label = train_img_label[indices]
        train_img_name = train_img_name[indices]
        pre_index = 0
        epoch_loss = 0.0
        cou = 0
        for step in range(1, iteration + 1):
            true_num = 0
            if pre_index + batch_size < 69163:
                input_x = train_img_name[pre_index: pre_index + batch_size]
                input_y = train_img_label[pre_index: pre_index + batch_size]
            else:
                input_x = train_img_name[pre_index:]
                input_y = train_img_label[pre_index:]
                break
            epoch_x, epoch_y = load_batch_data(input_x, input_y, len(input_x))
            # epoch_x = color_preprocessing(epoch_x)
            epoch_x = data_augmentation(epoch_x)
            _, loss_, l = sess.run([train_step, Trainloss, loss], feed_dict={samples: epoch_x, ys:epoch_y})
            pre_index += batch_size
            epoch_loss += l
            writer.add_summary(loss_, epoch)
        print("epoch:%d loss: %4f" % (epoch, epoch_loss))
        saver.save(sess,'./devise_model/devise.ckpt')
        a = 0
        b = 100
        acc = 0
        v_index = 0
        if epoch % 1 == 0:
            v_dic = {}
            with open('./DatasetB_20180919/validation_list.txt') as fp:
                for line in fp:
                    v_wn, v_name = line.split('\t')
                    v_dic[v_wn] = v_name.strip()
            v_img_label = list()
            v_img_name = list()
            for v_f_name, v_l_name in v_dic.items():
                v_img_name.append(str(v_f_name))
                v_img_label.append(str(v_l_name))
            v_img_name = np.array(v_img_name)
            v_img_label = np.array(v_img_label)
            v_indices = np.random.permutation(len(v_img_label))
            v_img_label = v_img_label[v_indices]
            v_img_name = v_img_name[v_indices]
            v_cou = 0
            acc = 0
            for num in range(1, validation_iter + 1):
                v_true_num = 0
                if v_index + batch_size < 18086:
                    v_input_x = v_img_name[v_index: v_index + batch_size]
                    v_input_y = v_img_label[v_index: v_index + batch_size]
                else:
                    v_input_x = v_img_name[v_index:]
                    v_input_y = v_img_label[v_index:]
                v_index += batch_size
                v_batch_x, v_batch_y = load_validation_data(v_input_x, v_input_y, len(v_input_x))
                # v_batch_x = color_preprocessing(v_batch_x)
                feed_dicts = {samples: v_batch_x, ys: v_batch_y}
                pred,label_index = sess.run([prediction, true_class_old], feed_dict=feed_dicts)
                pred = tf.reshape(pred, (-1, 300))
                result = tf.matmul(embedding_vector, tf.transpose(tf.nn.l2_normalize(pred, dim=1)))
                result = tf.transpose(result)
                outputLabel = tf.argmax(result, axis=1)
                outputLabel = outputLabel.eval()
                for a in range(len(v_input_x)):
                    if outputLabel[a] == label_index[a]:
                        acc += 1
                        v_true_num += 1
                print("vali_batch:%d,acc:%d/%d" % (num, v_true_num, len(v_input_x)))
            print("vali_acc:%d/%d" % (acc, 18086))




