import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pyplot as plt 
from detect_lm import detect_landmark, crop_img
import time
import os

def Int(x):
    temp = float(x)
    temp = int(temp)
    return temp


def load_landmark(file):
    landmark_points = []
    
    f = open(file, "r")
    data = f.read().split('{')
    data = data[1].split("}")
    data = data[0].splitlines()
    data = data[1:]
    for n in range(0, 68):
        temp = data[n].split()
        x = Int(temp[0])
        y = Int(temp[1])
        landmark_points.append((x,y))

    f.close()
    
    return landmark_points

def img_geom(img, lm, lma):
    images = img.copy()
    tform = PiecewiseAffineTransform()
    tform.estimate(lma, lm)
    images = img.copy()
    out = warp(images, tform, output_shape=(256, 256))
    return out

def option_landmarks(landmarks):
    part_ids = np.arange(0, 17)
    np_array = np.array([[0, 0], [0, 256], [256,256], [256,0]])
    
    np_a = np.append(landmarks[part_ids, :], np_array, axis = 0)
    return np_a

def geo(image, style):
    img = image.copy()
    styles = ['9', '32', '51', '55', '56', '58', '108', '140', '150', '153', '154', '155', '156']
    crop_img(img)
    image = cv2.imread('/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/output/input.png')
    landmark_points_img = load_landmark('/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/output/' + 'input' + '.pts')
    lm = np.array(landmark_points_img)
    landmark_points_art_img = load_landmark('/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/img/art/'+ styles[style-1] + '.pts')
    lma = np.array(landmark_points_art_img)
    landmark_art = option_landmarks(lma)
    landmark_face = option_landmarks(lm)
    image_geo = img_geom(image, landmark_face, landmark_art)
    image_geo*=255
    image_geo = image_geo.astype('uint8')
    return image_geo
