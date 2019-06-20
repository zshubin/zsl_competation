from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from cifar10 import *
import data_processing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import os
import os.path
import time

weight_decay = 0.0005
total_epochs = 100
learning_rate = 0.0001
iteration = 682
validation_iter = 63
batch_size = 128





def get_accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, [None, 64, 64, 3])
    labels = tf.placeholder(tf.float32, [None, len(data_processing.IMG_CLASSES)])
    keep_prob = tf.placeholder(tf.float32)
    logits, _ = nets.resnet_v1.resnet_v1_50(inputs=images, num_classes=2,  is_training=True)
    logits = tf.reshape(logits, [-1, 2])
    with tf.name_scope('cross_entropy'):
         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    tf.summary.scalar('cross_entropy', loss)
    output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v1_50/logits')
    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=output_vars)
    with tf.name_scope('accuracy'):
         accuracy = get_accuracy(logits, labels)
    tf.summary.scalar('accuracy', accuracy)
    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(TRAIN_LOG_DIR)
        ckpt = tf.train.get_checkpoint_state('./pretraincheck_point')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        step = 0

        global_step = 0
        valid_step = 0
        for epoch in range(1, total_epochs + 1):
            train_dic = {}
            v_dic = {}
            with open('./DatasetB_20180919/train_list.txt') as fp:
                for line in fp:
                    wn, name = line.split('\t')
                    train_dic[wn] = name.strip()
            with open('./DatasetB_20180919/validation_list.txt') as fp:
                for line in fp:
                    v_wn, v_name = line.split('\t')
                    v_dic[v_wn] = v_name.strip()
            train_img_label = list()
            train_img_name = list()
            v_img_label = list()
            v_img_name = list()
            for f_name, l_name in train_dic.items():
                train_img_name.append(str(f_name))
                train_img_label.append(str(l_name))
            for v_f_name, v_l_name in v_dic.items():
                v_img_name.append(str(v_f_name))
                v_img_label.append(str(v_l_name))
            train_img_name = np.array(train_img_name)
            train_img_label = np.array(train_img_label)
            indices = np.random.permutation(len(train_img_label))
            train_img_label = train_img_label[indices]
            train_img_name = train_img_name[indices]
            v_img_name = np.array(v_img_name)
            v_img_label = np.array(v_img_label)
            v_indices = np.random.permutation(len(v_img_label))
            v_img_label = v_img_label[v_indices]
            v_img_name = v_img_name[v_indices]

            pre_index = 0
            train_loss = 0.0
            cou = 0
            all_acc = 0
            for step in range(1, iteration + 1):
                true_num = 0
                if pre_index + batch_size < 69163:
                    input_x = train_img_name[pre_index: pre_index + batch_size]
                    input_y = train_img_label[pre_index: pre_index + batch_size]
                else:
                    input_x = train_img_name[pre_index:]
                    input_y = train_img_label[pre_index:]
                batch_x, batch_y = load_batch_data(input_x, input_y, len(input_x))
                # batch_x = color_preprocessing(batch_x)
                batch_x = data_augmentation(batch_x)
                feed_dicts = {images: batch_x, labels: batch_y}
                _, ls_, l, out, acc = sess.run([optimizer, merged, loss, logits, accuracy], feed_dict=feed_dicts)
                train_loss += l
                pre_index += batch_size
                # for j in range(len(input_x)):
                #     if np.argmax(out[, :]) == np.argmax(batch_y[j, :]):
                #         cou += 1
                #         true_num += 1
                all_acc += acc
                train_writer.add_summary(ls_, global_step)
                print("train epoch:%3d, idx:%4d, loss: %10.6f batch_acc: %4f" % (
                epoch, step, l, acc))
                v_index = 0
                v_cou = 0
                v_all_acc = 0
                if step % 321 == 0:
                    for num in range(1, validation_iter + 1):
                        v_true_num = 0
                        if v_index + batch_size < 8000:
                            v_input_x = v_img_name[v_index: v_index + batch_size]
                            v_input_y = v_img_label[v_index: v_index + batch_size]
                        else:
                            v_input_x = v_img_name[v_index:]
                            v_input_y = v_img_label[v_index:]
                        v_index += batch_size
                        v_batch_x, v_batch_y = load_validation_data(v_input_x, v_input_y, len(v_input_x))
                        # v_batch_x = color_preprocessing(v_batch_x)
                        v_batch_x = data_augmentation(v_batch_x)
                        feed_dicts = {images: v_batch_x, labels: v_batch_y}
                        _, v_l, out, acc = sess.run([optimizer, loss, logits, accuracy], feed_dict=feed_dicts)
                        # for count in range(len(v_input_x)):
                        #     if np.argmax(v_fc_16[count, :]) == np.argmax(v_batch_y[count, :]):
                        #         v_true_num += 1
                        #         v_cou += 1
                        v_all_acc += acc
                        print("validation_idx:%4d, loss: %10.6f batch_acc: %4f" % (num, v_l, acc))
                    print("validation_acc: %4f" % (v_all_acc/123))
                    saver.save(sess, './ckpt/model.ckpt')
            print("epoch:%d, train avg_loss:%10.6f,train_acc:%4f" % (epoch, train_loss / 1364, all_acc/123))
            saver.save(sess, './ckpt/model.ckpt')
