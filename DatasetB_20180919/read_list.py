import numpy as np
import scipy.io as sio
# labels = ['ZJL198', 'ZJL199', 'ZJL211', 'ZJL210', 'ZJL217', 'ZJL216', 'ZJL215', 'ZJL212', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194', 'ZJL195', 'ZJL196', 'ZJL197', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117', 'ZJL118', 'ZJL119', 'ZJL23', 'ZJL22', 'ZJL21', 'ZJL26', 'ZJL25', 'ZJL24', 'ZJL29', 'ZJL28', 'ZJL219', 'ZJL222', 'ZJL218', 'ZJL109', 'ZJL108', 'ZJL107', 'ZJL106', 'ZJL105', 'ZJL104', 'ZJL103', 'ZJL102', 'ZJL101', 'ZJL100', 'ZJL38', 'ZJL39', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL213', 'ZJL237', 'ZJL178', 'ZJL179', 'ZJL172', 'ZJL173', 'ZJL170', 'ZJL171', 'ZJL176', 'ZJL177', 'ZJL174', 'ZJL175', 'ZJL49', 'ZJL48', 'ZJL236', 'ZJL41', 'ZJL40', 'ZJL43', 'ZJL42', 'ZJL45', 'ZJL44', 'ZJL47', 'ZJL46', 'ZJL187', 'ZJL186', 'ZJL185', 'ZJL184', 'ZJL183', 'ZJL182', 'ZJL181', 'ZJL180', 'ZJL98', 'ZJL189', 'ZJL188', 'ZJL169', 'ZJL168', 'ZJL161', 'ZJL160', 'ZJL163', 'ZJL162', 'ZJL165', 'ZJL164', 'ZJL167', 'ZJL166', 'ZJL240', 'ZJL58', 'ZJL59', 'ZJL52', 'ZJL53', 'ZJL50', 'ZJL51', 'ZJL56', 'ZJL57', 'ZJL54', 'ZJL55', 'ZJL214', 'ZJL154', 'ZJL156', 'ZJL157', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL99', 'ZJL158', 'ZJL159', 'ZJL67', 'ZJL66', 'ZJL65', 'ZJL64', 'ZJL63', 'ZJL62', 'ZJL61', 'ZJL60', 'ZJL69', 'ZJL68', 'ZJL92', 'ZJL93', 'ZJL4', 'ZJL5', 'ZJL6', 'ZJL7', 'ZJL1', 'ZJL2', 'ZJL3', 'ZJL8', 'ZJL9', 'ZJL143', 'ZJL142', 'ZJL141', 'ZJL140', 'ZJL147', 'ZJL146', 'ZJL145', 'ZJL144', 'ZJL149', 'ZJL70', 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL220', 'ZJL221', 'ZJL226', 'ZJL227', 'ZJL224', 'ZJL225', 'ZJL238', 'ZJL89', 'ZJL88', 'ZJL223', 'ZJL239', 'ZJL137', 'ZJL135', 'ZJL132', 'ZJL133', 'ZJL130', 'ZJL131', 'ZJL85', 'ZJL84', 'ZJL87', 'ZJL86', 'ZJL81', 'ZJL80', 'ZJL138', 'ZJL139', 'ZJL232', 'ZJL235', 'ZJL231', 'ZJL234', 'ZJL83', 'ZJL233', 'ZJL82', 'ZJL204', 'ZJL205', 'ZJL206', 'ZJL207', 'ZJL200', 'ZJL201', 'ZJL202', 'ZJL203', 'ZJL208', 'ZJL209', 'ZJL125', 'ZJL124', 'ZJL127', 'ZJL126', 'ZJL121', 'ZJL120', 'ZJL123', 'ZJL122', 'ZJL96', 'ZJL97', 'ZJL94', 'ZJL95', 'ZJL129', 'ZJL128', 'ZJL90', 'ZJL91', 'ZJL16', 'ZJL14', 'ZJL15', 'ZJL12', 'ZJL13', 'ZJL10', 'ZJL11', 'ZJL228', 'ZJL18', 'ZJL19', 'ZJL230', 'ZJL229']
labels = ['ZJL1', 'ZJL10', 'ZJL100', 'ZJL101', 'ZJL102', 'ZJL103', 'ZJL104', 'ZJL105', 'ZJL106', 'ZJL107', 'ZJL108', 'ZJL109', 'ZJL11', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117', 'ZJL118', 'ZJL119', 'ZJL12', 'ZJL120', 'ZJL121', 'ZJL122', 'ZJL123', 'ZJL124', 'ZJL125', 'ZJL126', 'ZJL127', 'ZJL128', 'ZJL129', 'ZJL13', 'ZJL130', 'ZJL131', 'ZJL132', 'ZJL133', 'ZJL135', 'ZJL137', 'ZJL138', 'ZJL139', 'ZJL14', 'ZJL140', 'ZJL141', 'ZJL142', 'ZJL143', 'ZJL144', 'ZJL145', 'ZJL146', 'ZJL147', 'ZJL149', 'ZJL15', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL154', 'ZJL156', 'ZJL157', 'ZJL158', 'ZJL159', 'ZJL16', 'ZJL160', 'ZJL161', 'ZJL162', 'ZJL163', 'ZJL164', 'ZJL165', 'ZJL166', 'ZJL167', 'ZJL168', 'ZJL169', 'ZJL170', 'ZJL171', 'ZJL172', 'ZJL173', 'ZJL174', 'ZJL175', 'ZJL176', 'ZJL177', 'ZJL178', 'ZJL179', 'ZJL18', 'ZJL180', 'ZJL181', 'ZJL182', 'ZJL183', 'ZJL184', 'ZJL185', 'ZJL186', 'ZJL187', 'ZJL188', 'ZJL189', 'ZJL19', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194', 'ZJL195', 'ZJL196', 'ZJL197', 'ZJL198', 'ZJL199', 'ZJL2', 'ZJL200', 'ZJL21', 'ZJL22', 'ZJL23', 'ZJL24', 'ZJL25', 'ZJL26', 'ZJL28', 'ZJL29', 'ZJL3', 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL38', 'ZJL39', 'ZJL4', 'ZJL40', 'ZJL41', 'ZJL42', 'ZJL43', 'ZJL44', 'ZJL45', 'ZJL46', 'ZJL47', 'ZJL48', 'ZJL49', 'ZJL5', 'ZJL50', 'ZJL51', 'ZJL52', 'ZJL53', 'ZJL54', 'ZJL55', 'ZJL56', 'ZJL57', 'ZJL58', 'ZJL59', 'ZJL6', 'ZJL60', 'ZJL61', 'ZJL62', 'ZJL63', 'ZJL64', 'ZJL65', 'ZJL66', 'ZJL67', 'ZJL68', 'ZJL69', 'ZJL7', 'ZJL70', 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL8', 'ZJL80', 'ZJL81', 'ZJL82', 'ZJL83', 'ZJL84', 'ZJL85', 'ZJL86', 'ZJL87', 'ZJL88', 'ZJL89', 'ZJL9', 'ZJL90', 'ZJL91', 'ZJL92', 'ZJL93', 'ZJL94', 'ZJL95', 'ZJL96', 'ZJL97', 'ZJL98', 'ZJL99', 'ZJL201', 'ZJL202', 'ZJL203', 'ZJL204', 'ZJL205', 'ZJL206', 'ZJL207', 'ZJL208', 'ZJL209', 'ZJL210', 'ZJL211', 'ZJL212', 'ZJL213', 'ZJL214', 'ZJL215', 'ZJL216', 'ZJL217', 'ZJL218', 'ZJL219', 'ZJL220', 'ZJL221', 'ZJL222', 'ZJL223', 'ZJL224', 'ZJL225', 'ZJL226', 'ZJL227', 'ZJL228', 'ZJL229', 'ZJL230', 'ZJL231', 'ZJL232', 'ZJL233', 'ZJL234', 'ZJL235', 'ZJL236', 'ZJL237', 'ZJL238', 'ZJL239', 'ZJL240', 'ZJL241', 'ZJL242', 'ZJL243', 'ZJL244', 'ZJL245', 'ZJL246', 'ZJL247', 'ZJL248', 'ZJL249', 'ZJL250', 'ZJL251', 'ZJL252', 'ZJL253', 'ZJL254', 'ZJL255', 'ZJL256', 'ZJL257', 'ZJL258', 'ZJL259', 'ZJL260', 'ZJL261', 'ZJL262', 'ZJL263', 'ZJL264', 'ZJL265', 'ZJL266', 'ZJL267', 'ZJL268', 'ZJL269', 'ZJL270', 'ZJL271', 'ZJL272', 'ZJL273', 'ZJL274', 'ZJL275', 'ZJL276', 'ZJL277', 'ZJL278', 'ZJL279', 'ZJL280', 'ZJL281', 'ZJL282', 'ZJL283', 'ZJL284', 'ZJL285', 'ZJL286', 'ZJL287', 'ZJL288', 'ZJL289', 'ZJL290', 'ZJL291', 'ZJL292', 'ZJL293', 'ZJL294', 'ZJL295']
label_list = ['ZJL198', 'ZJL199', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194', 'ZJL195', 'ZJL196', 'ZJL197', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117', 'ZJL118', 'ZJL119', 'ZJL23', 'ZJL22', 'ZJL21', 'ZJL26', 'ZJL25', 'ZJL24', 'ZJL29', 'ZJL28', 'ZJL268', 'ZJL267', 'ZJL264', 'ZJL265', 'ZJL263', 'ZJL109', 'ZJL108', 'ZJL107', 'ZJL106', 'ZJL105', 'ZJL104', 'ZJL103', 'ZJL102', 'ZJL101', 'ZJL100', 'ZJL38', 'ZJL39', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL275', 'ZJL274', 'ZJL178', 'ZJL179', 'ZJL273', 'ZJL172', 'ZJL173', 'ZJL170', 'ZJL171', 'ZJL161', 'ZJL177', 'ZJL174', 'ZJL175', 'ZJL49', 'ZJL48', 'ZJL41', 'ZJL40', 'ZJL43', 'ZJL42', 'ZJL45', 'ZJL44', 'ZJL47', 'ZJL46', 'ZJL187', 'ZJL186', 'ZJL185', 'ZJL184', 'ZJL183', 'ZJL182', 'ZJL181', 'ZJL180', 'ZJL123', 'ZJL189', 'ZJL188', 'ZJL169', 'ZJL168', 'ZJL244', 'ZJL247', 'ZJL248', 'ZJL160', 'ZJL163', 'ZJL162', 'ZJL165', 'ZJL164', 'ZJL167', 'ZJL166', 'ZJL58', 'ZJL59', 'ZJL52', 'ZJL53', 'ZJL50', 'ZJL51', 'ZJL56', 'ZJL57', 'ZJL54', 'ZJL55', 'ZJL154', 'ZJL156', 'ZJL157', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL99', 'ZJL158', 'ZJL159', 'ZJL67', 'ZJL66', 'ZJL65', 'ZJL64', 'ZJL63', 'ZJL62', 'ZJL61', 'ZJL60', 'ZJL256', 'ZJL254', 'ZJL69', 'ZJL68', 'ZJL129', 'ZJL128', 'ZJL4', 'ZJL5', 'ZJL6', 'ZJL7', 'ZJL1', 'ZJL2', 'ZJL3', 'ZJL8', 'ZJL9', 'ZJL143', 'ZJL142', 'ZJL141', 'ZJL140', 'ZJL147', 'ZJL146', 'ZJL145', 'ZJL144', 'ZJL149', 'ZJL70', 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL276', 'ZJL89', 'ZJL88', 'ZJL137', 'ZJL135', 'ZJL132', 'ZJL133', 'ZJL130', 'ZJL131', 'ZJL85', 'ZJL84', 'ZJL87', 'ZJL86', 'ZJL81', 'ZJL80', 'ZJL83', 'ZJL139', 'ZJL176', 'ZJL261', 'ZJL138', 'ZJL82', 'ZJL200', 'ZJL125', 'ZJL124', 'ZJL127', 'ZJL126', 'ZJL121', 'ZJL120', 'ZJL98', 'ZJL122', 'ZJL96', 'ZJL97', 'ZJL94', 'ZJL95', 'ZJL92', 'ZJL93', 'ZJL90', 'ZJL91', 'ZJL16', 'ZJL14', 'ZJL15', 'ZJL12', 'ZJL13', 'ZJL10', 'ZJL11', 'ZJL18', 'ZJL19']
label_list_validation = ['ZJL154', 'ZJL157', 'ZJL273', 'ZJL77', 'ZJL194', 'ZJL159', 'ZJL114', 'ZJL116', 'ZJL118', 'ZJL119', 'ZJL176', 'ZJL80', 'ZJL83', 'ZJL139', 'ZJL29', 'ZJL42', 'ZJL46', 'ZJL187', 'ZJL5', 'ZJL138', 'ZJL3', 'ZJL166', 'ZJL182', 'ZJL141', 'ZJL144', 'ZJL72', 'ZJL265', 'ZJL121', 'ZJL120', 'ZJL76', 'ZJL108', 'ZJL128', 'ZJL101', 'ZJL91', 'ZJL15', 'ZJL13', 'ZJL31', 'ZJL32', 'ZJL55', 'ZJL73']

label_name = []
transports = np.zeros((285, 30), dtype='float32')
with open('./attributes_per_class.txt') as fp:
    count = 0
    for line in fp:
        for i in range(19):
            if i == 0:
                label_name.append(line.split('\t')[0].strip())
            else:
                transports[count, i-1] = line.split('\t')[i].strip()
        count += 1
attributes = np.zeros((40, 30), dtype='float32')
for i in range(len(label_list_validation)):
    attributes[i] = transports[label_name.index(label_list_validation[i])]

#
# transport = np.zeros((285, 300), dtype='float32')
# name = []
# with open('./class_wordembeddings.txt') as fp:
#     count = 0
#     for line in fp:
#         for i in range(301):
#             if i == 0:
#                 name.append(line.split(' ')[0].strip())
#             else:
#                 transport[count, i-1] = line.split(' ')[i].strip()
#         count += 1
# train_dic = {}
# with open('./train.txt') as fp:
#     for line in fp:
#         wn, n = line.split('\t')
#         train_dic[wn] = n.strip()
# img_name = []
# img_label = []
# for f_name, l_name in train_dic.items():
#     img_label.append(str(f_name))
#     img_name.append(str(l_name))
# img_name = set(img_name)
# word_embedding2 = np.zeros((285, 300), dtype='float32')
# for i in range(285):
#     name_index = img_name.index(name[i])
#     find_label = img_label[name_index]
#     label_index = labels.index(find_label)
#     word_embedding2[label_index] = transport[i]
# #
# #         # length = len(line.split('\t'))
# # label_name = np.array(label_name)
# # sio.savemat('C:/Users/Administrator/Desktop/1/DatasetA_train_20180813/attributes.mat', {'attributes': attributes})
# label_list = []
# t_dic = {}
# with open('./train.txt') as fp:
#     for line in fp:
#         wn, n = line.split('\t')
#         t_dic[wn] = n.strip()
# img_nt = []
# img_lt = []
# for f_name, l_name in t_dic.items():
#     img_lt.append(str(f_name))
#     img_nt.append(str(l_name))
# for i in range(len(img_nt)):
#     if img_nt[i] in img_label:
#         label_list.append(img_nt[i])
# # label_list = set(label_list)
# print(img_label)
# print(len(img_label))
# sio.savemat('C:/Users/Administrator/Desktop/1/DatasetB_20180919/word_embedding2.mat', {'word_embedding2': word_embedding2})
# print(attributes)

# print('labels')
#
# labels = np.zeros((230, 1), dtype='float32')
# for i in range(230):
#     index = label_list.index(label_name[i])
#     labels[i] = index
# # # print(y_input[3])
sio.savemat('full_validation_attributes.mat', {'full_validation_attributes': attributes})
#
#
# test = sio.loadmat('word_embedding2.mat')['word_embedding2']
#
# print(img_name)
# print(len(img_name))
