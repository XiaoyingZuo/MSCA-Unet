#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lsd time :2022/5/17

from PIL import Image
import os
import nibabel as nib
import numpy as np
import re
import scipy.ndimage
import xlwt
# %%
def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


Pre="./evaluation_chatfull_prepress/test_result"#原文件夹路径
Lab="./evaluation_chatfull_prepress/test_result"#目标文件夹路径
regex=re.compile(r'\d+')
# 创建一个workbook 设置编码
workbook = xlwt.Workbook(encoding='utf-8')
# 创建一个worksheet
worksheet = workbook.add_sheet('My Worksheet')

# 写入excel
# 参数对应 行, 列, 值
worksheet.write(0, 0, label='图片名')
worksheet.write(0, 1, label='dice')
worksheet.write(0, 2, label='square')
i=1
for file in os.listdir(Pre):

    #遍历原文件夹中的文件
    if 'true' in file:
        pre = os.path.join(Pre, file)
        num = int(max(regex.findall(pre)))
        lab = os.path.join(Lab, '{}.png'.format(num))
        pre = np.array(Image.open(pre))
        lab = np.array(Image.open(lab))
        pre[np.where(pre >= 125)] = 1
        lab[np.where(lab >= 125)] = 1
        square = lab.sum()
        if pre.shape == lab.shape :
             dice=compute_dice_coefficient(lab, pre)
             print("这是图像： {},volumetric dice:          {},square  {}".format(num, dice, square))
        worksheet.write(i, 0, num)
        worksheet.write(i, 1, dice)
        worksheet.write(i, 2, int(square))
        i=i+1
workbook.save('Excel_test1.xls')
