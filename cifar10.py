# -*- coding:utf-8 -*-

import os
import sys
import time
import pickle
import cv2
import random
import numpy as np

class_num = 205
image_size = 64
img_channels = 3
# label_list = ['ZJL198', 'ZJL199', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194', 'ZJL195', 'ZJL196', 'ZJL197', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117', 'ZJL118', 'ZJL119', 'ZJL23', 'ZJL22', 'ZJL21', 'ZJL26', 'ZJL25', 'ZJL24', 'ZJL29', 'ZJL28', 'ZJL109', 'ZJL108', 'ZJL107', 'ZJL106', 'ZJL105', 'ZJL104', 'ZJL103', 'ZJL102', 'ZJL101', 'ZJL100', 'ZJL38', 'ZJL39', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL178', 'ZJL179', 'ZJL172', 'ZJL173', 'ZJL170', 'ZJL171', 'ZJL176', 'ZJL177', 'ZJL174', 'ZJL175', 'ZJL49', 'ZJL48', 'ZJL41', 'ZJL40', 'ZJL43', 'ZJL42', 'ZJL45', 'ZJL44', 'ZJL47', 'ZJL46', 'ZJL187', 'ZJL186', 'ZJL185', 'ZJL184', 'ZJL183', 'ZJL182', 'ZJL181', 'ZJL180', 'ZJL123', 'ZJL189', 'ZJL188', 'ZJL169', 'ZJL168', 'ZJL161', 'ZJL160', 'ZJL163', 'ZJL162', 'ZJL165', 'ZJL164', 'ZJL167', 'ZJL166', 'ZJL58', 'ZJL59', 'ZJL52', 'ZJL53', 'ZJL50', 'ZJL51', 'ZJL56', 'ZJL57', 'ZJL54', 'ZJL55', 'ZJL154', 'ZJL156', 'ZJL157', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL122', 'ZJL158', 'ZJL159', 'ZJL67', 'ZJL66', 'ZJL65', 'ZJL64', 'ZJL63', 'ZJL62', 'ZJL61', 'ZJL60', 'ZJL69', 'ZJL68', 'ZJL92', 'ZJL128', 'ZJL4', 'ZJL5', 'ZJL6', 'ZJL7', 'ZJL1', 'ZJL2', 'ZJL3', 'ZJL8', 'ZJL9', 'ZJL143', 'ZJL142', 'ZJL141', 'ZJL140', 'ZJL147', 'ZJL146', 'ZJL145', 'ZJL144', 'ZJL149', 'ZJL70', 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL89', 'ZJL133', 'ZJL137', 'ZJL135', 'ZJL132', 'ZJL88', 'ZJL130', 'ZJL131', 'ZJL85', 'ZJL84', 'ZJL87', 'ZJL86', 'ZJL81', 'ZJL80', 'ZJL83', 'ZJL139', 'ZJL138', 'ZJL82', 'ZJL200', 'ZJL125', 'ZJL124', 'ZJL127', 'ZJL126', 'ZJL121', 'ZJL120', 'ZJL98', 'ZJL99', 'ZJL96', 'ZJL97', 'ZJL94', 'ZJL95', 'ZJL129', 'ZJL93', 'ZJL90', 'ZJL91', 'ZJL16', 'ZJL14', 'ZJL15', 'ZJL12', 'ZJL13', 'ZJL10', 'ZJL11', 'ZJL18', 'ZJL19']
# label_list = ['ZJL97', 'ZJL26', 'ZJL98', 'ZJL150', 'ZJL247', 'ZJL85', 'ZJL73', 'ZJL92', 'ZJL88', 'ZJL80', 'ZJL265', 'ZJL23', 'ZJL116', 'ZJL144', 'ZJL3', 'ZJL75', 'ZJL49', 'ZJL261', 'ZJL273', 'ZJL71', 'ZJL50', 'ZJL70', 'ZJL61', 'ZJL154', 'ZJL14', 'ZJL37', 'ZJL131', 'ZJL51', 'ZJL19', 'ZJL67', 'ZJL5', 'ZJL72', 'ZJL143', 'ZJL86', 'ZJL274', 'ZJL30', 'ZJL256', 'ZJL124', 'ZJL48', 'ZJL55', 'ZJL7', 'ZJL84', 'ZJL46', 'ZJL91', 'ZJL275', 'ZJL139', 'ZJL25', 'ZJL47', 'ZJL69', 'ZJL135', 'ZJL45', 'ZJL130', 'ZJL101', 'ZJL151', 'ZJL82', 'ZJL121', 'ZJL267', 'ZJL158', 'ZJL111', 'ZJL149', 'ZJL141', 'ZJL140', 'ZJL34', 'ZJL32', 'ZJL142', 'ZJL43', 'ZJL129', 'ZJL40', 'ZJL254', 'ZJL146', 'ZJL102', 'ZJL63', 'ZJL56', 'ZJL36', 'ZJL118', 'ZJL127', 'ZJL119', 'ZJL268', 'ZJL147', 'ZJL138', 'ZJL115', 'ZJL128', 'ZJL93', 'ZJL120', 'ZJL64', 'ZJL104', 'ZJL38', 'ZJL28', 'ZJL10', 'ZJL110', 'ZJL65', 'ZJL90', 'ZJL35', 'ZJL78', 'ZJL263', 'ZJL96', 'ZJL24', 'ZJL117', 'ZJL11', 'ZJL99', 'ZJL153', 'ZJL107', 'ZJL89', 'ZJL103', 'ZJL114', 'ZJL248', 'ZJL152', 'ZJL105', 'ZJL29', 'ZJL159', 'ZJL106', 'ZJL81', 'ZJL125', 'ZJL31', 'ZJL41', 'ZJL66', 'ZJL276', 'ZJL126', 'ZJL16', 'ZJL83', 'ZJL15', 'ZJL57', 'ZJL87', 'ZJL77', 'ZJL264', 'ZJL132', 'ZJL44', 'ZJL21', 'ZJL53', 'ZJL123', 'ZJL9', 'ZJL6', 'ZJL54', 'ZJL133', 'ZJL156', 'ZJL18', 'ZJL68', 'ZJL58', 'ZJL12', 'ZJL145', 'ZJL8', 'ZJL100', 'ZJL22', 'ZJL122', 'ZJL113', 'ZJL244', 'ZJL157', 'ZJL95', 'ZJL39', 'ZJL52', 'ZJL13', 'ZJL62', 'ZJL60', 'ZJL1', 'ZJL76', 'ZJL109', 'ZJL2', 'ZJL137', 'ZJL59', 'ZJL79', 'ZJL94', 'ZJL42', 'ZJL4', 'ZJL108']
label_list = ['ZJL198', 'ZJL199', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194', 'ZJL195', 'ZJL196', 'ZJL197', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117', 'ZJL118', 'ZJL119', 'ZJL23', 'ZJL22', 'ZJL21', 'ZJL26', 'ZJL25', 'ZJL24', 'ZJL29', 'ZJL28', 'ZJL268', 'ZJL267', 'ZJL264', 'ZJL265', 'ZJL263', 'ZJL109', 'ZJL108', 'ZJL107', 'ZJL106', 'ZJL105', 'ZJL104', 'ZJL103', 'ZJL102', 'ZJL101', 'ZJL100', 'ZJL38', 'ZJL39', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL275', 'ZJL274', 'ZJL178', 'ZJL179', 'ZJL273', 'ZJL172', 'ZJL173', 'ZJL170', 'ZJL171', 'ZJL161', 'ZJL177', 'ZJL174', 'ZJL175', 'ZJL49', 'ZJL48', 'ZJL41', 'ZJL40', 'ZJL43', 'ZJL42', 'ZJL45', 'ZJL44', 'ZJL47', 'ZJL46', 'ZJL187', 'ZJL186', 'ZJL185', 'ZJL184', 'ZJL183', 'ZJL182', 'ZJL181', 'ZJL180', 'ZJL123', 'ZJL189', 'ZJL188', 'ZJL169', 'ZJL168', 'ZJL244', 'ZJL247', 'ZJL248', 'ZJL160', 'ZJL163', 'ZJL162', 'ZJL165', 'ZJL164', 'ZJL167', 'ZJL166', 'ZJL58', 'ZJL59', 'ZJL52', 'ZJL53', 'ZJL50', 'ZJL51', 'ZJL56', 'ZJL57', 'ZJL54', 'ZJL55', 'ZJL154', 'ZJL156', 'ZJL157', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL99', 'ZJL158', 'ZJL159', 'ZJL67', 'ZJL66', 'ZJL65', 'ZJL64', 'ZJL63', 'ZJL62', 'ZJL61', 'ZJL60', 'ZJL256', 'ZJL254', 'ZJL69', 'ZJL68', 'ZJL129', 'ZJL128', 'ZJL4', 'ZJL5', 'ZJL6', 'ZJL7', 'ZJL1', 'ZJL2', 'ZJL3', 'ZJL8', 'ZJL9', 'ZJL143', 'ZJL142', 'ZJL141', 'ZJL140', 'ZJL147', 'ZJL146', 'ZJL145', 'ZJL144', 'ZJL149', 'ZJL70', 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL276', 'ZJL89', 'ZJL88', 'ZJL137', 'ZJL135', 'ZJL132', 'ZJL133', 'ZJL130', 'ZJL131', 'ZJL85', 'ZJL84', 'ZJL87', 'ZJL86', 'ZJL81', 'ZJL80', 'ZJL83', 'ZJL139', 'ZJL176', 'ZJL261', 'ZJL138', 'ZJL82', 'ZJL200', 'ZJL125', 'ZJL124', 'ZJL127', 'ZJL126', 'ZJL121', 'ZJL120', 'ZJL98', 'ZJL122', 'ZJL96', 'ZJL97', 'ZJL94', 'ZJL95', 'ZJL92', 'ZJL93', 'ZJL90', 'ZJL91', 'ZJL16', 'ZJL14', 'ZJL15', 'ZJL12', 'ZJL13', 'ZJL10', 'ZJL11', 'ZJL18', 'ZJL19']



# print(len(label_list))


def load_test_data(img_name,batch_size):
    x_input = np.empty((batch_size, 64, 64, 3), dtype='float32')
    img = cv2.imread('./DatasetB_20180919/test/'+img_name[0])
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    x_input[0] = img
    return x_input



def load_batch_data(img_name, img_label, batch_size):
    global image_size, img_channels
    x_input = np.empty((batch_size, 64, 64, 3), dtype='float32')
    y_input = np.empty((batch_size, 205), dtype='float32')
    for i in range(len(img_label)):
        img = cv2.imread('./DatasetB_20180919/train/'+img_name[i])
        # img = cv2.resize(img, (64, 64))
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        # img = img.transpose([2, 1, 0])
        x_input[i] = img2
        index = label_list.index(img_label[i])
        label = np.zeros((1, 205))
        label[0, index] = 1
        y_input[i] = label
    return x_input, y_input

def load_validation_data(img_name, img_label, batch_size):
    global image_size, img_channels
    x_input = np.empty((batch_size, 64, 64, 3), dtype='float32')
    y_input = np.empty((batch_size, 205), dtype='float32')
    for i in range(len(img_label)):
        img = cv2.imread('./DatasetB_20180919/validation_img/'+img_name[i])
        # img = cv2.resize(img, (64, 64))
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        # img = img.transpose([2, 1, 0])
        x_input[i] = img2
        index = label_list.index(img_label[i])
        label = np.zeros((1, 205))
        label[0, index] = 1
        y_input[i] = label
    return x_input, y_input




# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# ========================================================== #

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def color_preprocessing(x_train):
    x_train = x_train.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])
    return x_train


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [64, 64], 4)
    return batch
