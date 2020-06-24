import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pyplot as plt 
from detect_lm import detect_landmark 
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

def geo(image):
    out_dir = 'out_img_to_geo_1'
    out_path = os.path.join('img\\', out_dir)
    img = image.copy()
    landmark_points_img = detect_landmark(img)
    lm = np.array(landmark_points_img)
    landmark_points_art_img = load_landmark('img\\art\\'+ '9' + '.pts')
    lma = np.array(landmark_points_art_img)
    # name = facialArr[numFace]+'_'+ artArr[num]
    # name = 'facial' + '_9'
    landmark_art = option_landmarks(lma)
    landmark_face = option_landmarks(lm)
    image_geo = img_geom(img, landmark_face, landmark_art)
    # if os.path.exists(os.path.join('img/' + out_dir, name + '.png')):
    #     continue
    # cv2.imwrite(os.path.join(out_path, name + '.png'), image*255)
    return image_geo

# img_detected = 'face/'
# path = 'face_2.png'
# img = cv2.imread(os.path.join(img_detected, path))


# start_time = time.time()
# image = geo(img)
# cv2.imshow('asdsad', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# end_time = time.time()
# print('total run-time: %f ms' % ((end_time - start_time)))