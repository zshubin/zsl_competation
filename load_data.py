import os
import numpy as np
import scipy.io as sio

label_list_train = ['ZJL1', 'ZJL10', 'ZJL100', 'ZJL101', 'ZJL102', 'ZJL103', 'ZJL104', 'ZJL105', 'ZJL106', 'ZJL107', 'ZJL108', 'ZJL109', 'ZJL11', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117', 'ZJL118', 'ZJL119', 'ZJL12', 'ZJL120', 'ZJL121', 'ZJL122', 'ZJL123', 'ZJL124', 'ZJL125', 'ZJL126', 'ZJL127', 'ZJL128', 'ZJL129', 'ZJL13', 'ZJL130', 'ZJL131', 'ZJL132', 'ZJL133', 'ZJL135', 'ZJL137', 'ZJL138', 'ZJL139', 'ZJL14', 'ZJL140', 'ZJL141', 'ZJL142', 'ZJL143', 'ZJL144', 'ZJL145', 'ZJL146', 'ZJL147', 'ZJL149', 'ZJL15', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL154', 'ZJL156', 'ZJL157', 'ZJL158', 'ZJL159', 'ZJL16', 'ZJL160', 'ZJL161', 'ZJL162', 'ZJL163', 'ZJL164', 'ZJL165', 'ZJL166', 'ZJL167', 'ZJL168', 'ZJL169', 'ZJL170', 'ZJL171', 'ZJL172', 'ZJL173', 'ZJL174', 'ZJL175', 'ZJL176', 'ZJL177', 'ZJL178', 'ZJL179', 'ZJL18', 'ZJL180', 'ZJL181', 'ZJL182', 'ZJL183', 'ZJL184', 'ZJL185', 'ZJL186', 'ZJL187', 'ZJL188', 'ZJL189', 'ZJL19', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194', 'ZJL195', 'ZJL196', 'ZJL197', 'ZJL198', 'ZJL199', 'ZJL2', 'ZJL200', 'ZJL21', 'ZJL22', 'ZJL23', 'ZJL24', 'ZJL25', 'ZJL26', 'ZJL28', 'ZJL29', 'ZJL3', 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL38', 'ZJL39', 'ZJL4', 'ZJL40', 'ZJL41', 'ZJL42', 'ZJL43', 'ZJL44', 'ZJL45', 'ZJL46', 'ZJL47', 'ZJL48', 'ZJL49', 'ZJL5', 'ZJL50', 'ZJL51', 'ZJL52', 'ZJL53', 'ZJL54', 'ZJL55', 'ZJL56', 'ZJL57', 'ZJL58', 'ZJL59', 'ZJL6', 'ZJL60', 'ZJL61', 'ZJL62', 'ZJL63', 'ZJL64', 'ZJL65', 'ZJL66', 'ZJL67', 'ZJL68', 'ZJL69', 'ZJL7', 'ZJL70', 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL8', 'ZJL80', 'ZJL81', 'ZJL82', 'ZJL83', 'ZJL84', 'ZJL85', 'ZJL86', 'ZJL87', 'ZJL88', 'ZJL89', 'ZJL9', 'ZJL90', 'ZJL91', 'ZJL92', 'ZJL93', 'ZJL94', 'ZJL95', 'ZJL96', 'ZJL97', 'ZJL98', 'ZJL99', 'ZJL201', 'ZJL202', 'ZJL203', 'ZJL204', 'ZJL205', 'ZJL206', 'ZJL207', 'ZJL208', 'ZJL209', 'ZJL210', 'ZJL211', 'ZJL212', 'ZJL213', 'ZJL214', 'ZJL215', 'ZJL216', 'ZJL217', 'ZJL218', 'ZJL219', 'ZJL220', 'ZJL221', 'ZJL222', 'ZJL223', 'ZJL224', 'ZJL225', 'ZJL226', 'ZJL227', 'ZJL228', 'ZJL229', 'ZJL230', 'ZJL231', 'ZJL232', 'ZJL233', 'ZJL234', 'ZJL235', 'ZJL236', 'ZJL237', 'ZJL238', 'ZJL239', 'ZJL240', 'ZJL241', 'ZJL242', 'ZJL243', 'ZJL244', 'ZJL245', 'ZJL246', 'ZJL247', 'ZJL248', 'ZJL249', 'ZJL250', 'ZJL251', 'ZJL252', 'ZJL253', 'ZJL254', 'ZJL255', 'ZJL256', 'ZJL257', 'ZJL258', 'ZJL259', 'ZJL260', 'ZJL261', 'ZJL262', 'ZJL263', 'ZJL264', 'ZJL265', 'ZJL266', 'ZJL267', 'ZJL268', 'ZJL269', 'ZJL270', 'ZJL271', 'ZJL272', 'ZJL273', 'ZJL274', 'ZJL275', 'ZJL276', 'ZJL277', 'ZJL278', 'ZJL279', 'ZJL280', 'ZJL281', 'ZJL282', 'ZJL283', 'ZJL284', 'ZJL285', 'ZJL286', 'ZJL287', 'ZJL288', 'ZJL289', 'ZJL290', 'ZJL291', 'ZJL292', 'ZJL293', 'ZJL294', 'ZJL295']
label_list_validation = ['ZJL154', 'ZJL157', 'ZJL273', 'ZJL77', 'ZJL194', 'ZJL159', 'ZJL114', 'ZJL116', 'ZJL118', 'ZJL119', 'ZJL176', 'ZJL80', 'ZJL83', 'ZJL139', 'ZJL29', 'ZJL42', 'ZJL46', 'ZJL187', 'ZJL5', 'ZJL138', 'ZJL3', 'ZJL166', 'ZJL182', 'ZJL141', 'ZJL144', 'ZJL72', 'ZJL265', 'ZJL121', 'ZJL120', 'ZJL76', 'ZJL108', 'ZJL128', 'ZJL101', 'ZJL91', 'ZJL15', 'ZJL13', 'ZJL31', 'ZJL32', 'ZJL55', 'ZJL73']

def load_test_data():
    train_dic = {}
    with open('./DatasetB_20180919/validation_list.txt') as fp:
        for line in fp:
            wn, name = line.split('\t')
            train_dic[wn] = name.strip()
    label_vector = np.zeros((24086, 1), dtype="float32")
    visual_vector = np.zeros((24086, 2048), dtype="float32")
    print('Loading test data')
    i = 0
    for k, value in train_dic.items():
        # label = train_dic.get(img_list[i])
        index = label_list_train.index(value)
        label_vector[i] = index
        visual_vector[i] = sio.loadmat('./DatasetB_20180919/validation_mat/'+k[0:32]+'.mat')['visual_feature']
        i += 1
    print('Test data loaded over')
    return label_vector, visual_vector

# a,b = load_test_data()
# print(a)
def load_train_data(img_list,attributes, batch_size, word_embeddings):#, attributes):
    train_visual = np.zeros((batch_size, 2048))
    train_att = np.zeros((batch_size, 30))
    train_word = np.zeros((batch_size, 300))
    train_dic = {}
    img_list = list(img_list)
    with open('./DatasetB_20180919/train.txt') as fp:
        for line in fp:
            wn, name = line.split('\t')
            train_dic[wn] = name.strip()
    for i in range(len(img_list)):
        train_visual[i] = sio.loadmat('./DatasetB_20180919/train_mat/'+img_list[i][0:32]+'.mat')['visual_feature']
        label = train_dic.get(img_list[i])
        index = label_list_train.index(label)
        train_att[i] = attributes[index]
        train_word[i] = word_embeddings[index]
    return train_visual, train_word, train_att

def load_data_test():
    img_list = os.listdir('./DatasetB_20180919/test/')
    visual_vector = np.zeros((11740, 2048))
    print('Loading test data')
    for i in range(len(img_list)):
        visual_vector[i] = sio.loadmat('./DatasetB_20180919/test_mat/' + img_list[i][0:32]+'.mat')['visual_feature']
    print('Test data loaded over')
    return img_list, visual_vector