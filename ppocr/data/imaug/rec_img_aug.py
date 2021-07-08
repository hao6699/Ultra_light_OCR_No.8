# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random

import cv2
import numpy as np
from imgaug import augmenters as iaa

from .text_image_aug import tia_perspective, tia_stretch, tia_distort


class RecAug(object):
    def __init__(self, use_tia=True, aug_prob=0.8, **kwargs):
        self.use_tia = use_tia
        self.aug_prob = aug_prob

    def __call__(self, data):
        img = data['image']
        img = warp(img, 10, self.use_tia, self.aug_prob)
        data['image'] = img
        return data


class RecAugRotateWord(object):
    def __init__(self, use_tia=True, aug_prob=0.8, **kwargs):
        self.use_tia = use_tia
        self.aug_prob = aug_prob
        sometimes5 = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([
            sometimes5(iaa.Lambda(img_block, keypoint_func)),
            # sigma在0~0.5间随机高斯模糊，且每张图纸生效的概率是0.5
            sometimes5(iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 0.3)),
                iaa.AverageBlur(k=(2, 3)),
                iaa.MedianBlur(k=(1, 3))
            ])),
            sometimes5(iaa.Cutout(nb_iterations=(2, 4), size=(0.2, 0.5))),
            # 高斯噪点
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255), per_channel=0.5),
            # 给每个像素乘上0.8-1.2之间的数来使图片变暗或变亮
            # 20%的图片在每个channel上乘以不同的因子
            iaa.Multiply((0.9, 1.1), per_channel=0.2),
            # 对每张图片进行仿射变换，包括缩放、平移、旋转、修剪等
            iaa.Affine(
                # 加入warp_img时不需要缩放和平移图像
                translate_percent={"x": (-0.03, 0.04), "y": (-0.04, 0.04)},
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                rotate=(-3, 3),
            )
        ], random_order=True)

    def __call__(self, data):
        img = data['image']
        ctc_pred = data['ctc_pred']
        if ctc_pred is not None:
            img = random_rotate_word_image(img, ctc_pred, 0.1)
        if random.random() < self.aug_prob:
            img = self.seq.augment_image(img)
        data['image'] = img
        return data


class ClsResizeImg(object):
    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']
        norm_img = resize_norm_img(img, self.image_shape)
        data['image'] = norm_img
        return data


class RecResizeImg(object):
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 character_type='ch',
                 **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_type = character_type

    def __call__(self, data):
        img = data['image']
        if self.infer_mode and self.character_type == "ch":
            norm_img = resize_norm_img_chinese(img, self.image_shape)
        else:
            norm_img = resize_norm_img(img, self.image_shape)
        data['image'] = norm_img
        return data


class SRNRecResizeImg(object):
    def __init__(self, image_shape, num_heads, max_text_length, **kwargs):
        self.image_shape = image_shape
        self.num_heads = num_heads
        self.max_text_length = max_text_length

    def __call__(self, data):
        img = data['image']
        norm_img = resize_norm_img_srn(img, self.image_shape)
        data['image'] = norm_img
        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            srn_other_inputs(self.image_shape, self.num_heads, self.max_text_length)

        data['encoder_word_pos'] = encoder_word_pos
        data['gsrm_word_pos'] = gsrm_word_pos
        data['gsrm_slf_attn_bias1'] = gsrm_slf_attn_bias1
        data['gsrm_slf_attn_bias2'] = gsrm_slf_attn_bias2
        return data


def resize_norm_img(img, image_shape):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32 * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    # print(resized_image.shape)
    return padding_im


def resize_norm_img_srn(img, image_shape):
    imgC, imgH, imgW = image_shape

    img_black = np.zeros((imgH, imgW))
    im_hei = img.shape[0]
    im_wid = img.shape[1]

    if im_wid <= im_hei * 1:
        img_new = cv2.resize(img, (imgH * 1, imgH))
    elif im_wid <= im_hei * 2:
        img_new = cv2.resize(img, (imgH * 2, imgH))
    elif im_wid <= im_hei * 3:
        img_new = cv2.resize(img, (imgH * 3, imgH))
    else:
        img_new = cv2.resize(img, (imgW, imgH))

    img_np = np.asarray(img_new)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_black[:, 0:img_np.shape[1]] = img_np
    img_black = img_black[:, :, np.newaxis]

    row, col, c = img_black.shape
    c = 1

    return np.reshape(img_black, (c, row, col)).astype(np.float32)


def srn_other_inputs(image_shape, num_heads, max_text_length):
    imgC, imgH, imgW = image_shape
    feature_dim = int((imgH / 8) * (imgW / 8))

    encoder_word_pos = np.array(range(0, feature_dim)).reshape(
        (feature_dim, 1)).astype('int64')
    gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
        (max_text_length, 1)).astype('int64')

    gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
    gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
        [1, max_text_length, max_text_length])
    gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1,
                                  [num_heads, 1, 1]) * [-1e9]

    gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
        [1, max_text_length, max_text_length])
    gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2,
                                  [num_heads, 1, 1]) * [-1e9]

    return [
        encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
        gsrm_slf_attn_bias2
    ]


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def cvtColor(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


class Config:
    """
    Config
    """

    def __init__(self, use_tia):
        self.anglex = random.random() * 30
        self.angley = random.random() * 15
        self.anglez = random.random() * 10
        self.fov = 42
        self.r = 0
        self.shearx = random.random() * 0.3
        self.sheary = random.random() * 0.05
        self.borderMode = cv2.BORDER_REPLICATE
        self.use_tia = use_tia

    def make(self, w, h, ang):
        """
        make
        """
        self.anglex = random.random() * 5 * flag()
        self.angley = random.random() * 5 * flag()
        self.anglez = -1 * random.random() * int(ang) * flag()
        self.fov = 42
        self.r = 0
        self.shearx = 0
        self.sheary = 0
        self.borderMode = cv2.BORDER_REPLICATE
        self.w = w
        self.h = h

        self.perspective = self.use_tia
        self.stretch = self.use_tia
        self.distort = self.use_tia

        self.crop = True
        self.affine = False
        self.reverse = True
        self.noise = True
        self.jitter = True
        self.blur = True
        self.color = True
        self.down_sample = True


def rad(x):
    """
    rad
    """
    return x * np.pi / 180


def get_warpR(config):
    """
    get_warpR
    """
    anglex, angley, anglez, fov, w, h, r = \
        config.anglex, config.angley, config.anglez, config.fov, config.w, config.h, config.r
    if w > 69 and w < 112:
        anglex = anglex * 1.5

    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # Homogeneous coordinate transformation matrix
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0], [
                       0,
                       -np.sin(rad(anglex)),
                       np.cos(rad(anglex)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0], [
                       -np.sin(rad(angley)),
                       0,
                       np.cos(rad(angley)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    r = rx.dot(ry).dot(rz)
    # generate 4 points
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = np.array([dst1, dst2, dst3, dst4])
    org = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    dst = np.zeros((4, 2), np.float32)
    # Project onto the image plane
    dst[:, 0] = list_dst[:, 0] * z / (z - list_dst[:, 2]) + pcenter[0]
    dst[:, 1] = list_dst[:, 1] * z / (z - list_dst[:, 2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    dst1, dst2, dst3, dst4 = dst
    r1 = int(min(dst1[1], dst2[1]))
    r2 = int(max(dst3[1], dst4[1]))
    c1 = int(min(dst1[0], dst3[0]))
    c2 = int(max(dst2[0], dst4[0]))

    try:
        ratio = min(1.0 * h / (r2 - r1), 1.0 * w / (c2 - c1))

        dx = -c1
        dy = -r1
        T1 = np.float32([[1., 0, dx], [0, 1., dy], [0, 0, 1.0 / ratio]])
        ret = T1.dot(warpR)
    except:
        ratio = 1.0
        T1 = np.float32([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        ret = T1
    return ret, (-r1, -c1), ratio, dst


def get_warpAffine(config):
    """
    get_warpAffine
    """
    anglez = config.anglez
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    return rz


def down_sample(x, h, w):
    new_h = h // 2
    new_w = w // 2
    new_img = cv2.resize(x, (new_w, new_h))
    return new_img


def warp(img, ang, use_tia=True, prob=1):
    """
    warp
    """
    h, w, _ = img.shape
    config = Config(use_tia=use_tia)
    config.make(w, h, ang)
    new_img = img
    if config.down_sample:
        if random.random() <= prob and h >= 15 and w >= 80:
            new_img = down_sample(img, h, w)
    if config.distort:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_distort(new_img, random.randint(3, 6))

    if config.stretch:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_stretch(new_img, random.randint(3, 6))

    if config.perspective:
        if random.random() <= prob:
            new_img = tia_perspective(new_img)

    if config.crop:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = get_crop(new_img)

    if config.blur:
        if random.random() <= prob:
            new_img = blur(new_img)
    if config.color:
        if random.random() <= prob:
            new_img = cvtColor(new_img)
    if config.jitter:
        new_img = jitter(new_img)
    if config.noise:
        if random.random() <= prob:
            new_img = add_gasuss_noise(new_img)
    if config.reverse:
        if random.random() <= prob:
            new_img = 255 - new_img
    return new_img


def random_rotate_word_image(img, ctc_pred, prob=0.1):
    '''
    字符旋转增强方法的实现，这里只实现了为所有字符随机挑选一种逆时针90°或顺时针90°旋转
    '''
    if random.random() <= prob:
        word_images = []
        non_zeros_index = []
        # 把ctc预测为非连接符的位置找出来，且连续相同的预测只取第一个预测位置
        for index in range(len(ctc_pred)):
            if ctc_pred[index] != 0 and ctc_pred[index] != ctc_pred[index - 1]:
                non_zeros_index.append(index)
        non_zeros_index.append(len(ctc_pred))
        # 计算预测的总字符数，可用于随机抽取
        total_word_nums = len(non_zeros_index) - 1
        # select_nums = random.randint(1, total_word_nums)
        select_nums = total_word_nums
        select_indexes = sorted(random.sample([x for x in range(total_word_nums)], select_nums))
        h, w, _ = img.shape
        # 随机选取逆时针90°或顺时针90°旋转
        method = random.choice([cv2.cv2.ROTATE_90_CLOCKWISE, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE])
        for index in range(1, len(non_zeros_index)):
            top_left_x = int(non_zeros_index[index - 1] / len(ctc_pred) * w)
            bottom_right_x = int(non_zeros_index[index] / len(ctc_pred) * w)
            crop_img = img[:, top_left_x:bottom_right_x, :]
            if (index - 1) in select_indexes:
                ori_crop_h, ori_crop_w, _ = crop_img.shape
                crop_img = cv2.rotate(crop_img, method)
                if ori_crop_h < ori_crop_w:
                    new_crop_h = ori_crop_h
                    new_crop_w = int(ori_crop_h / (ori_crop_w / ori_crop_h))
                    crop_img = cv2.resize(crop_img, (new_crop_w, new_crop_h))
                elif ori_crop_h > ori_crop_w:
                    padded_img = np.ones([ori_crop_h, ori_crop_h, 3]) * crop_img[0, 0]
                    start_index = int(ori_crop_h / 2 - ori_crop_w / 2)
                    end_index = start_index + ori_crop_w
                    padded_img[start_index:end_index] = crop_img
                    crop_img = padded_img.astype(np.uint8)
            word_images.append(crop_img)
        new_img = np.concatenate(word_images, 1)
    else:
        new_img = img
    return new_img


def img_block(images, random_state, parents, hooks):
    '''
    为图像增加横竖条纹
    '''
    for img in images:
        random_num = random.randint(0, 255)
        if random.random() < 0.5:
            img[::6] = random_num
        else:
            img[:, ::6] = random_num
    return images


def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images
