#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import timeit
import scipy.misc


def getTexture(path, dimension):
    texture = np.array(cv2.imread(path))
    texture = cv2.resize(texture, dimension, interpolation=cv2.INTER_LANCZOS4)
    # result = np.zeros((texture[:, :, 0].flatten().shape[0], 3))
    # result[:, 0] = texture[:, :, 0].flatten()
    # result[:, 1] = texture[:, :, 1].flatten()
    # result[:, 2] = texture[:, :, 2].flatten()
    # texture = np.unique(texture)
    return texture


def alpha_blending(src, mask):
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 0)
    # return src * (mask_blur / 255)
    return mask_blur


def applyMask(src, mask):
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_blur = mask_blur.astype('float16')
    return [src * (mask_blur / 255), mask_blur]
# parsing artistic image to get the textures


def vis_parsing_textures(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):

    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(
        vis_im, cv2.COLOR_RGB2BGR), 0.6, vis_parsing_anno_color, 0.4, 0)

    for pi in range(1, 2):
        index = np.where(vis_parsing_anno == pi)
        skin = np.zeros_like((index[0], index[1], 3))
        skin = vis_im[index[0], index[1], :]
        skin = cv2.resize(skin, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_path[:-4] + "skin.png", skin)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def vis_parsing_maps(im, style, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):

    style = int(style)
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    # texture_hair = getTexture('/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_parsing/textures/' + _styleArr[style], (512, 512))
    texture = getTexture('/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_parsing/textures/btexture' + str(style) + '.jpg', (512, 512))
    texture = texture[:,:,::-1]
    lips_eyes = np.ones_like(im).astype(np.uint8)*255
    kernel = np.ones((5,5), np.uint8)
    # skin 1, nose 10, upper_lip 12, lower-lip 13
    raw_textured = im.copy().astype("uint8")[:, :, ::-1]
    for pi in [1, 2, 3, 10, 11, 12, 13]:
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1],
                               :] = texture[index[0], index[1], :]
        lips_eyes[index[0], index[1], :] = 0
        raw_textured[index[0], index[1], :] = texture[index[0], index[1], :]

    lips_eyes = cv2.dilate(lips_eyes, kernel, iterations=2)
    for pi in [2, 3, 11, 12, 13]:
        index = np.where(vis_parsing_anno == pi)
        lips_eyes[index[0], index[1], :] = 255

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(
        vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # src2 processing
    src2_mask = np.ones_like(im).astype('uint8')*200
    src2 = texture.copy().astype('uint8')
    for pi in [1, 10]:
        index = np.where(vis_parsing_anno == pi)
        src2_mask[index[0], index[1], :] = 255
        src2[index[0], index[1], :] = vis_im[index[0], index[1], :]

    cv2.imwrite(save_path[:-4]+"_src2_texture.jpg", src2,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    src2, src2_mask = applyMask(src2, src2_mask)

    # first mask
    first_mask = im.copy().astype("uint8")[:, :, ::-1]
    # first_mask *= 255
    for pi in [1, 10, 2, 3, 17]:
        index = np.where(vis_parsing_anno == pi)
        first_mask[index[0], index[1], :] = vis_im[index[0], index[1], :]
    for pi in [4, 5, 7, 8, 9, 11, 12, 13, 17]:
        index = np.where(vis_parsing_anno == pi)
        first_mask[index[0], index[1], :] = im[:,
                                               :, :: -1][index[0], index[1], :]

    vis_im = cv2.addWeighted(cv2.cvtColor(
        vis_im, cv2.COLOR_RGB2BGR), 0, first_mask, 1, 0)

    lips_eyes = alpha_blending(vis_im, lips_eyes)
    print(lips_eyes.shape)
    mask1 = lips_eyes
    src1 = im.copy().astype('uint8')[:, :, ::-1]

    mask1 = mask1.astype('float16')
    mask1 = mask1 / 255
    # src2 = src2.astype('float16')
    dst = src1 * mask1 + src2 * (1 - mask1)
    # dst = src2 
    dst = dst.astype('uint8')

    # lips_eyes_applyMask = applyMask(lips_eyes)
    result = vis_im.copy()
    # result[lips_eyes != 0] = lips_eyes[lips_eyes != 0]

    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno)
        # cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(save_path[:-4]+"_src2.jpg", src2,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(save_path[:-4]+"_src2_before.jpg", result,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(save_path[:-4]+"_src2_mask.jpg", src2_mask,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(save_path[:-4]+"_src1.jpg", src1,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(save_path[:-4]+"_mask.jpg", lips_eyes,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(save_path[:-4]+"_mask2.jpg", dst,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return dst

    # return vis_im


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(
        './res/cp', cp)
    net.load_state_dict(torch.load(
        save_pth, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):

            start = timeit.default_timer()

            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True,
                             save_path=osp.join(respth, image_path))

            stop = timeit.default_timer()
            print('Time: ', stop - start)


def exportImgAPI(img, style, dspth='/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_parsing/test-img', cp='model_final_diss.pth'):
    # img = Image.open(osp.join(dspth, image_path))
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(
        '/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_parsing/res/cp', cp)
    net.load_state_dict(torch.load(
        save_pth, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        
        img = Image.fromarray(img)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        print(np.unique(parsing))
        

        return vis_parsing_maps(image, style, parsing, stride=1, save_im=True,
                                save_path='')


# if __name__ == "__main__":

#     evaluate(dspth='./test-img', cp='79999_iter.pth')
