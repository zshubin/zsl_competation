import tensorflow as tf
import sys
from resnet2 import ResNet
from cifar10 import *
import numpy as np
import scipy.io as sio
from kNN_cosine import kNNClassify

weight_decay = 0.0005
total_epochs = 1
learning_rate = 0.0001
iteration = 92
validation_iter = 63
batch_size = 128

net = ResNet([64, 64], 205)
net.build()

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
loss = net.loss()
ckpt_path = './ckpt/model.ckpt'

label_list_full = ['ZJL1', 'ZJL10', 'ZJL100', 'ZJL101', 'ZJL102', 'ZJL103', 'ZJL104', 'ZJL105', 'ZJL106', 'ZJL107', 'ZJL108', 'ZJL109', 'ZJL11', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117', 'ZJL118', 'ZJL119', 'ZJL12', 'ZJL120', 'ZJL121', 'ZJL122', 'ZJL123', 'ZJL124', 'ZJL125', 'ZJL126', 'ZJL127', 'ZJL128', 'ZJL129', 'ZJL13', 'ZJL130', 'ZJL131', 'ZJL132', 'ZJL133', 'ZJL135', 'ZJL137', 'ZJL138', 'ZJL139', 'ZJL14', 'ZJL140', 'ZJL141', 'ZJL142', 'ZJL143', 'ZJL144', 'ZJL145', 'ZJL146', 'ZJL147', 'ZJL149', 'ZJL15', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL154', 'ZJL156', 'ZJL157', 'ZJL158', 'ZJL159', 'ZJL16', 'ZJL160', 'ZJL161', 'ZJL162', 'ZJL163', 'ZJL164', 'ZJL165', 'ZJL166', 'ZJL167', 'ZJL168', 'ZJL169', 'ZJL170', 'ZJL171', 'ZJL172', 'ZJL173', 'ZJL174', 'ZJL175', 'ZJL176', 'ZJL177', 'ZJL178', 'ZJL179', 'ZJL18', 'ZJL180', 'ZJL181', 'ZJL182', 'ZJL183', 'ZJL184', 'ZJL185', 'ZJL186', 'ZJL187', 'ZJL188', 'ZJL189', 'ZJL19', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194', 'ZJL195', 'ZJL196', 'ZJL197', 'ZJL198', 'ZJL199', 'ZJL2', 'ZJL200', 'ZJL21', 'ZJL22', 'ZJL23', 'ZJL24', 'ZJL25', 'ZJL26', 'ZJL28', 'ZJL29', 'ZJL3', 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL38', 'ZJL39', 'ZJL4', 'ZJL40', 'ZJL41', 'ZJL42', 'ZJL43', 'ZJL44', 'ZJL45', 'ZJL46', 'ZJL47', 'ZJL48', 'ZJL49', 'ZJL5', 'ZJL50', 'ZJL51', 'ZJL52', 'ZJL53', 'ZJL54', 'ZJL55', 'ZJL56', 'ZJL57', 'ZJL58', 'ZJL59', 'ZJL6', 'ZJL60', 'ZJL61', 'ZJL62', 'ZJL63', 'ZJL64', 'ZJL65', 'ZJL66', 'ZJL67', 'ZJL68', 'ZJL69', 'ZJL7', 'ZJL70', 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL8', 'ZJL80', 'ZJL81', 'ZJL82', 'ZJL83', 'ZJL84', 'ZJL85', 'ZJL86', 'ZJL87', 'ZJL88', 'ZJL89', 'ZJL9', 'ZJL90', 'ZJL91', 'ZJL92', 'ZJL93', 'ZJL94', 'ZJL95', 'ZJL96', 'ZJL97', 'ZJL98', 'ZJL99', 'ZJL201', 'ZJL202', 'ZJL203', 'ZJL204', 'ZJL205', 'ZJL206', 'ZJL207', 'ZJL208', 'ZJL209', 'ZJL210', 'ZJL211', 'ZJL212', 'ZJL213', 'ZJL214', 'ZJL215', 'ZJL216', 'ZJL217', 'ZJL218', 'ZJL219', 'ZJL220', 'ZJL221', 'ZJL222', 'ZJL223', 'ZJL224', 'ZJL225', 'ZJL226', 'ZJL227', 'ZJL228', 'ZJL229', 'ZJL230', 'ZJL231', 'ZJL232', 'ZJL233', 'ZJL234', 'ZJL235', 'ZJL236', 'ZJL237', 'ZJL238', 'ZJL239', 'ZJL240', 'ZJL241', 'ZJL242', 'ZJL243', 'ZJL244', 'ZJL245', 'ZJL246', 'ZJL247', 'ZJL248', 'ZJL249', 'ZJL250', 'ZJL251', 'ZJL252', 'ZJL253', 'ZJL254', 'ZJL255', 'ZJL256', 'ZJL257', 'ZJL258', 'ZJL259', 'ZJL260', 'ZJL261', 'ZJL262', 'ZJL263', 'ZJL264', 'ZJL265', 'ZJL266', 'ZJL267', 'ZJL268', 'ZJL269', 'ZJL270', 'ZJL271', 'ZJL272', 'ZJL273', 'ZJL274', 'ZJL275', 'ZJL276', 'ZJL277', 'ZJL278', 'ZJL279', 'ZJL280', 'ZJL281', 'ZJL282', 'ZJL283', 'ZJL284', 'ZJL285', 'ZJL286', 'ZJL287', 'ZJL288', 'ZJL289', 'ZJL290', 'ZJL291', 'ZJL292', 'ZJL293', 'ZJL294', 'ZJL295']
validation_list = ['ZJL154', 'ZJL157', 'ZJL273', 'ZJL77', 'ZJL194', 'ZJL159', 'ZJL114', 'ZJL116', 'ZJL118', 'ZJL119', 'ZJL176', 'ZJL80', 'ZJL83', 'ZJL139', 'ZJL29', 'ZJL42', 'ZJL46', 'ZJL187', 'ZJL5', 'ZJL138', 'ZJL3', 'ZJL166', 'ZJL182', 'ZJL141', 'ZJL144', 'ZJL72', 'ZJL265', 'ZJL121', 'ZJL120', 'ZJL76', 'ZJL108', 'ZJL128', 'ZJL101', 'ZJL91', 'ZJL15', 'ZJL13', 'ZJL31', 'ZJL32', 'ZJL55', 'ZJL73']

word_embedding = sio.loadmat('./DatasetB_20180919/word_embedding2.mat')['word_embedding2']
embedding_vector = sio.loadmat('./DatasetB_20180919/train_embedding.mat')['train_embedding']
validation_embedding = sio.loadmat('./DatasetB_20180919/validation_embedding.mat')['validation_embedding']
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
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
fobj = open('submit.txt', 'w')
for epoch in range(1, total_epochs + 1):
    # train_dic = {}
    # with open('./DatasetB_20180919/train_list.txt') as fp:
    #     for line in fp:
    #         wn, name = line.split('\t')
    #         train_dic[wn] = name.strip()
    # train_img_label = list()
    # train_img_name = list()
    # for f_name, l_name in train_dic.items():
    #     train_img_name.append(str(f_name))
    #     train_img_label.append(str(l_name))
    # train_img_name = np.array(train_img_name)
    # train_img_label = np.array(train_img_label)
    # indices = np.random.permutation(len(train_img_label))
    # train_img_label = train_img_label[indices]
    train_img_name = os.listdir('./DatasetB_20180919/test/')

    pre_index = 0
    train_loss = 0.0
    cou = 0
    for step in range(1, iteration + 1):
        true_num = 0
        if pre_index + batch_size < 11740:
            input_x = train_img_name[pre_index: pre_index + batch_size]
        else:
            input_x = train_img_name[pre_index:]
        batch_x = load_test_data(input_x, len(input_x))
        feed_dicts = {net.inputs: batch_x}
        fc_16 = sess.run(net.result, feed_dict=feed_dicts)
        sorted_indices = np.argsort(fc_16, axis=1)
        # embedding = np.zeros([1, 300],dtype="float32")
        # for i in range(10):
        #     embedding += fc
        embedding = np.dot(fc_16, embedding_vector)
        add_s = np.sum(fc_16, axis=1)
        add_s = np.reshape(add_s, (len(input_x), 1))
        add_s = np.tile(add_s, (1, 300))
        embedding = embedding / add_s
        # train_loss += l
        pre_index += batch_size
        labels = np.arange(0, 285)
        for j in range(len(input_x)):
            label = kNNClassify(embedding[j], word_embedding, labels, 1)
            fobj.write(input_x[j] + '\t' + label_list_full[int(label)] + '\n')
        # train_writer.add_summary(ls_, global_step)
        print("train epoch:%3d, idx:%4d,  batch_acc: %4d/%4d" % (epoch, step, true_num, len(input_x)))
    print("epoch:%d, train avg_loss:%10.6f,train_acc:%d/%d" % (epoch, train_loss/1364, cou, 87249))
    saver.save(sess,  './ckpt/model.ckpt')


