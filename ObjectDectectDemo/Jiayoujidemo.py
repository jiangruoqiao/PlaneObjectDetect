# ecoding:utf-8
import Tkinter as tk
import tkMessageBox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import cv2
import numpy as np
import nms
import multiprocessing
import tensorflow as tf
from PIL import Image
from skimage import io
import rot_mnist12K_model
import sys
import numpy as np
import os
import random

from scipy.ndimage.interpolation import rotate


def image_to_matrix(image):
    image_data = image.getdata()
    image_data = np.matrix(image_data, dtype='float') / 255.0
    new_image_matrix = np.resize(image_data, (32, 32, 3))
    return new_image_matrix


def matrix_to_image(matrix):
    matrix = matrix * 255
    new = Image.fromarray(matrix.astype(np.uint8))
    return new


def IOU(Reframe, GTframe):
    label = GTframe[4]
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)
    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)
    if width <= 0 or height <= 0:
        ratio = 0  # 重叠率为 0
    else:
        Area = width * height  # 两矩形相交面积
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio, Reframe, GTframe, label


def region_from_selective_search(img, value_scale, value_sigma):
    img_lbl, regions = selectivesearch.selective_search(img, scale=value_scale, sigma=value_sigma, min_size=100)
    return regions


def make_corr(line):
    resultLine = []
    numMat = eval(line)
    x1 = numMat[0][0]
    y1 = numMat[0][1]
    x2 = numMat[1][0]
    y2 = numMat[1][1]
    label = numMat[2]
    resultLine.append([x1, y1, x2, y2, label])
    return resultLine


def make_negative_region(imageFileNamePath, groundTruthFileNamePath, image_index):
    img = io.imread(imageFileNamePath)
    image = Image.open(imageFileNamePath)
    loadGrondTruth = open(groundTruthFileNamePath, 'r')
    labelResutlAll = []
    numberOfLineInImage = 0
    for line in loadGrondTruth.readlines():
        resultOfLabel = make_corr(line)
        labelResutlAll.extend(resultOfLabel)
        numberOfLineInImage = numberOfLineInImage + 1
    corrdGruondTruth = np.zeros((len(labelResutlAll), 5))  # 用来存放GroundTruth的坐标和标签
    number1 = 0
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    for x, y, w, h, label in labelResutlAll:
        corrdGruondTruth[number1, :] = ([int(x), int(y), int(w), int(h), int(label)])
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red')
        ax1.add_patch(rect)
        number1 = number1 + 1
    reg_1 = []
    reg_2 = []
    reg_3 = []
    reg_4 = []
    reg_5 = []
    reg_6 = []
    pool = multiprocessing.Pool(processes=4)
    reg_1.append(pool.apply_async(region_from_selective_search, (img, 1000, 0.9)))
    reg_2.append(pool.apply_async(region_from_selective_search, (img, 1000, 0.8)))
    reg_3.append(pool.apply_async(region_from_selective_search, (img, 1000, 0.7)))
    reg_4.append(pool.apply_async(region_from_selective_search, (img, 800, 0.9)))
    reg_5.append(pool.apply_async(region_from_selective_search, (img, 800, 0.7)))
    reg_6.append(pool.apply_async(region_from_selective_search, (img, 800, 0.6)))
    pool.close()
    pool.join()
    regions_1 = []
    regions_2 = []
    regions_3 = []
    regions_4 = []
    regions_5 = []
    regions_6 = []
    for res_1 in reg_1:
        regions_1 = res_1.get()
    for res_2 in reg_2:
        regions_2 = res_2.get()
    for res_3 in reg_3:
        regions_3 = res_3.get()
    for res_4 in reg_4:
        regions_4 = res_4.get()
    for res_5 in reg_5:
        regions_5 = res_5.get()
    for res_6 in reg_6:
        regions_6 = res_6.get()
    regions = regions_1 + regions_2 + regions_3 + regions_4 + regions_5 + regions_6
    candidates = set()  # 创建一个无序不重复元素集，可进行关系测试，可以删除重复数据，还可以计算交集等
    for r in regions:
        if r['rect'] in candidates:
            print(r['rect'])
            continue
        if r["size"] < 100:
            continue
        if r["size"] > 2000:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        if w / h > 1.5 or h / w > 1.5:
            continue
        candidates.add(r['rect'])
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    corrd = np.zeros((len(candidates), 4))
    number = 0
    for x, y, w, h in candidates:
        corrd[number, :] = ([int(x), int(y), int(w + x), int(h + y)])
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red')
        ax.add_patch(rect)
        number = number + 1
    return corrd, image


def _int_labels_to_one_hot(int_labels, number_of_classes):
    offsets = np.arange(2) * number_of_classes
    one_hot_labels = np.zeros((2, number_of_classes))
    flat_iterator = one_hot_labels.flat
    for index in xrange(2):
        flat_iterator[offsets[index] + int(int_labels[index])] = 1
    return one_hot_labels


def _transform(padded, number_of_transformations):
    tiled = np.tile(np.expand_dims(padded, 4), [number_of_transformations])
    for transformation_index in range(number_of_transformations):
        angle = 360.0 * transformation_index / float(number_of_transformations)
        tiled[:, :, :, :, transformation_index] = rotate(
            tiled[:, :, :, :, transformation_index],
            angle,
            axes=[1, 2],
            reshape=False)
    return tiled


'''
top = tk.Tk()
##添加一个Label控件
tk.Label(top, text = "图片加载路径:").grid(row = 0)
e = tk.Entry(top)
e.grid(row = 0,  column = 1)
e.delete(0, tk.END)
e.insert(0, "添加图片路径")
print e.get()
B = tk.Button(top, text="加油机识别", command=helloCallBack)
B.grid(row = 4, column = 1)
top.mainloop()
'''

LOADED_SIZE = 28
DESIRED_SIZE = 32
# model constants
NUMBER_OF_CLASSES = 11
NUMBER_OF_FILTERS = 40
NUMBER_OF_FC_FEATURES = 5120
NUMBER_OF_TRANSFORMATIONS = 8
# optimization constants
BATCH_SIZE = 64
# set seeds
np.random.seed(100)
tf.set_random_seed(100)
# set up training graph
x = tf.placeholder(tf.float32, shape=[None,
                                      DESIRED_SIZE,
                                      DESIRED_SIZE,
                                      3,
                                      NUMBER_OF_TRANSFORMATIONS])
y_gt = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])
keep_prob = tf.placeholder(tf.float32)
logits = rot_mnist12K_model.define_model(x,
                                         keep_prob,
                                         NUMBER_OF_CLASSES,
                                         NUMBER_OF_FILTERS,
                                         NUMBER_OF_FC_FEATURES)
# run training
fileName = '/home/yx/桌面/IMG_3460.JPG'
# 求ground truth
groundTruthFile = '/home/yx/桌面/NWPU VHR-10 dataset/ground truth/ 26.txt'
ang, img = make_negative_region(fileName, groundTruthFile, 26)
image = cv2.imread(fileName)
session = tf.Session()
saver = tf.train.Saver()
saver.restore(session, "ckpt/tipooling.ckpt")
result_set = []
for index in range(0, len(ang), 1):
    y1 = int(ang[index][1])
    y2 = int(ang[index][3])
    x1 = int(ang[index][0])
    x2 = int(ang[index][2])
    label = 1
    if y2 - y1 > x2 - x1:
        cropped = img.crop((x1, y1, x1 + x2 - x1, y1 + x2 - x1))
    else:
        cropped = img.crop((x1, y1, x1 + y2 - y1, y1 + y2 - y1))
    newImage = cropped.resize((32, 32))
    newMatrix = image_to_matrix(newImage)
    newMatrix = np.resize(newMatrix, (32, 32, 3))
    tem = np.zeros([2, 32, 32, 3])
    tem[1, :, :, :] = newMatrix
    inputx = _transform(tem, 8)
    inputx = inputx[1, :, :, :, :]
    inputx = np.resize(inputx, (1, 32, 32, 3, 8))
    temlabel = np.zeros([2])
    temlabel[0] = label
    inputlabal = _int_labels_to_one_hot(temlabel, 11)
    inputlabal = inputlabal[0]
    inputlabal = np.resize(inputx, (1, 11))
    train_accuracy = session.run(tf.argmax(logits, 1), feed_dict={x: inputx,
                                                                  y_gt: inputlabal,
                                                                  keep_prob: 1.0})
    if train_accuracy[0] == 1:
        result_set.append([int(x1), int(y1), int(x2), int(y2)])
    print train_accuracy
    sys.stdout.flush()

length = len(result_set)
corrd = np.zeros((len(result_set), 4))
for index2 in range(0, length):
    corrd[index2, 0] = result_set[index2][0]
    corrd[index2, 1] = result_set[index2][1]
    corrd[index2, 2] = result_set[index2][2]
    corrd[index2, 3] = result_set[index2][3]

orig = image.copy()
for (startX, startY, endX, endY) in corrd:
    cv2.rectangle(orig, (int(startX), int(startY)), (int(endX), int(endY)),
                  (int(random.uniform(0, 255)), int(random.uniform(0, 255)), int(random.uniform(0, 255))
                   ), 2)
pick = nms.nonMaximumSuppression(corrd, 0.4)
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)

cv2.imshow("Original", orig)
cv2.imshow("After NMS", image)
cv2.waitKey(1000000)
