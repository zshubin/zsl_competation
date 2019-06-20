
# import shutil
# import numpy as np
# import os
# label_list = ['ZJL198', 'ZJL199', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194', 'ZJL195', 'ZJL196', 'ZJL197', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117', 'ZJL118', 'ZJL119', 'ZJL23', 'ZJL22', 'ZJL21', 'ZJL26', 'ZJL25', 'ZJL24', 'ZJL29', 'ZJL28', 'ZJL268', 'ZJL267', 'ZJL264', 'ZJL265', 'ZJL263', 'ZJL109', 'ZJL108', 'ZJL107', 'ZJL106', 'ZJL105', 'ZJL104', 'ZJL103', 'ZJL102', 'ZJL101', 'ZJL100', 'ZJL38', 'ZJL39', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL275', 'ZJL274', 'ZJL178', 'ZJL179', 'ZJL273', 'ZJL172', 'ZJL173', 'ZJL170', 'ZJL171', 'ZJL161', 'ZJL177', 'ZJL174', 'ZJL175', 'ZJL49', 'ZJL48', 'ZJL41', 'ZJL40', 'ZJL43', 'ZJL42', 'ZJL45', 'ZJL44', 'ZJL47', 'ZJL46', 'ZJL187', 'ZJL186', 'ZJL185', 'ZJL184', 'ZJL183', 'ZJL182', 'ZJL181', 'ZJL180', 'ZJL123', 'ZJL189', 'ZJL188', 'ZJL169', 'ZJL168', 'ZJL244', 'ZJL247', 'ZJL248', 'ZJL160', 'ZJL163', 'ZJL162', 'ZJL165', 'ZJL164', 'ZJL167', 'ZJL166', 'ZJL58', 'ZJL59', 'ZJL52', 'ZJL53', 'ZJL50', 'ZJL51', 'ZJL56', 'ZJL57', 'ZJL54', 'ZJL55', 'ZJL154', 'ZJL156', 'ZJL157', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL99', 'ZJL158', 'ZJL159', 'ZJL67', 'ZJL66', 'ZJL65', 'ZJL64', 'ZJL63', 'ZJL62', 'ZJL61', 'ZJL60', 'ZJL256', 'ZJL254', 'ZJL69', 'ZJL68', 'ZJL129', 'ZJL128', 'ZJL4', 'ZJL5', 'ZJL6', 'ZJL7', 'ZJL1', 'ZJL2', 'ZJL3', 'ZJL8', 'ZJL9', 'ZJL143', 'ZJL142', 'ZJL141', 'ZJL140', 'ZJL147', 'ZJL146', 'ZJL145', 'ZJL144', 'ZJL149', 'ZJL70', 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL276', 'ZJL89', 'ZJL88', 'ZJL137', 'ZJL135', 'ZJL132', 'ZJL133', 'ZJL130', 'ZJL131', 'ZJL85', 'ZJL84', 'ZJL87', 'ZJL86', 'ZJL81', 'ZJL80', 'ZJL83', 'ZJL139', 'ZJL176', 'ZJL261', 'ZJL138', 'ZJL82', 'ZJL200', 'ZJL125', 'ZJL124', 'ZJL127', 'ZJL126', 'ZJL121', 'ZJL120', 'ZJL98', 'ZJL122', 'ZJL96', 'ZJL97', 'ZJL94', 'ZJL95', 'ZJL92', 'ZJL93', 'ZJL90', 'ZJL91', 'ZJL16', 'ZJL14', 'ZJL15', 'ZJL12', 'ZJL13', 'ZJL10', 'ZJL11', 'ZJL18', 'ZJL19']
#
# label_list = np.array(label_list)
# indices = np.random.permutation(len(label_list))
# label_list = label_list[indices]
#
# validation_label = label_list[0:40]
# train_list = label_list[40:205]
#
#
# train_dic = {}
# with open('./train.txt') as fp:
#     for line in fp:
#         wn, name = line.split('\t')
#         train_dic[wn] = name.strip()
# img_label = list()
# img_name = list()
# for f_name, l_name in train_dic.items():
#     img_name.append(str(f_name))
#     img_label.append(str(l_name))
#
# for i in range(len(validation_label)):
#     idx = [count for count,x in enumerate(img_label) if x == validation_label[i]]
#     for j in range(len(idx)):
#         shutil.move('./train/{}'.format(img_name[idx[j]]), './validation_img')
#
#
#
#
# # img_label = np.array(img_label)
# # indices = np.random.permutation(len(img_label))
# # img_label = img_label[indices]
# # img_name = img_name[indices]
# #
# # for i in range(8000):
# #     shutil.move('./train/{}'.format(img_name[i]), './validation_img')
# #     # shutil.move('./mat/{}.mat'.format(img_name[i][0:31]), './validation_mat')
# #
# img_label = list(img_label)
# img_name = list(img_name)
# train_list = os.listdir('./train')
# validation_list = os.listdir('./validation_img')
# train_txt = open('train_list.txt', 'w')
# validation_txt = open('validation_list.txt', 'w')
#
# for i in range(len(train_list)):
#     index = img_name.index(train_list[i])
#     label = img_label[index]
#     train_txt.write(train_list[i] + '\t' + label + '\n')
# train_txt.close()
#
# for i in range(len(validation_list)):
#     index = img_name.index(validation_list[i])
#     label = img_label[index]
#     validation_txt.write(validation_list[i] + '\t' + label + '\n')
# validation_txt.close()

import shutil
import numpy as np
import os


train_dic = {}
with open('./train.txt') as fp:
    for line in fp:
        wn, name = line.split('\t')
        train_dic[wn] = name.strip()
img_label = list()
img_name = list()
for f_name, l_name in train_dic.items():
    img_name.append(str(f_name))
    img_label.append(str(l_name))
img_name = np.array(img_name)
img_label = np.array(img_label)
indices = np.random.permutation(len(img_label))
img_label = img_label[indices]
img_name = img_name[indices]

# for i in range(6000):
#     shutil.move('./train/{}'.format(img_name[i]), './validation_img')
#     shutil.move('./train_mat/{}.mat'.format(img_name[i][0:32]), './validation_mat')
    # shutil.move('./mat/{}.mat'.format(img_name[i][0:31]), './validation_mat')

img_label = list(img_label)
img_name = list(img_name)
train_list = os.listdir('./train')
validation_list = os.listdir('./validation_img')
train_txt = open('train_list.txt', 'w')
validation_txt = open('validation_list.txt', 'w')

for i in range(len(train_list)):
    index = img_name.index(train_list[i])
    label = img_label[index]
    train_txt.write('\n' + train_list[i] + '\t' + label)
train_txt.close()

for i in range(len(validation_list)):
    index = img_name.index(validation_list[i])
    label = img_label[index]
    validation_txt.write('\n' + validation_list[i] + '\t' + label)
validation_txt.close()


# print(id1)