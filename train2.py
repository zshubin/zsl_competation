import tensorflow as tf
import sys
from resnet2 import ResNet
from cifar10 import *
import numpy as np
# 0.01 in ckpt_1

weight_decay = 0.0005
total_epochs = 100
learning_rate = 0.0001
iteration = 682
validation_iter = 63
batch_size = 128

net = ResNet([64, 64], 205)
net.build()

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
loss = net.loss()
# print(tf.global_variables())
ckpt_path = './ckpt/model.ckpt'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
# sess = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss + l2_loss * weight_decay)
saver = tf.train.Saver()

ls = tf.summary.scalar('loss', loss)

train_writer = tf.summary.FileWriter('./log_train', sess.graph)
valid_writer = tf.summary.FileWriter('./log_valid', sess.graph)



ckpt = tf.train.get_checkpoint_state('./ckpt')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())


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
     # if epoch % 30 == 0:
     #     epoch_learning_rate = epoch_learning_rate / 10

    pre_index = 0
    train_loss = 0.0
    cou = 0
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
        feed_dicts = {net.inputs: batch_x, net.ground_truth: batch_y}
        _, ls_, l, fc_16 = sess.run([optimizer, ls, loss, net.result], feed_dict=feed_dicts)
        train_loss += l
        pre_index += batch_size
        for j in range(len(input_x)):
            if np.argmax(fc_16[j, :]) == np.argmax(batch_y[j, :]):
                cou += 1
                true_num += 1
        train_writer.add_summary(ls_, global_step)
        print("train epoch:%3d, idx:%4d, loss: %10.6f batch_acc: %4d/%4d" % (epoch, step, l, true_num, len(input_x)))
        v_index = 0
        v_cou = 0
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
                  feed_dicts = {net.inputs: v_batch_x, net.ground_truth: v_batch_y}
                  _, v_l, v_fc_16 = sess.run([optimizer,loss, net.result], feed_dict=feed_dicts)
                  for count in range(len(v_input_x)):
                      if np.argmax(v_fc_16[count, :]) == np.argmax(v_batch_y[count, :]):
                          v_true_num += 1
                          v_cou += 1
                  print("validation_idx:%4d, loss: %10.6f batch_acc: %d/%d" % (num, l, v_true_num, len(v_input_x)))
              print("validation_acc: %d/%d" % (v_cou, 8000))
              saver.save(sess, './ckpt/model.ckpt')
    print("epoch:%d, train avg_loss:%10.6f,train_acc:%d/%d" % (epoch, train_loss/1364, cou, 87249))
    saver.save(sess,  './ckpt/model.ckpt')


