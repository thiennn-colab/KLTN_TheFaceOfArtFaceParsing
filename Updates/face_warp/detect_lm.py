import dlib
import cv2
import menpo.io as mio
import os
from menpo.shape.pointcloud import PointCloud
from skimage.io import imsave
import time
import numpy as np


def detect_landmark(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks_points = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/shape_predictor_68_face_landmarks.dat")
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

    # mio.export_landmark_file(PointCloud(landmarks_points), os.path.join(img_detected, path.split('.')[0] +'.pts'),
    #                  overwrite=True)

    return landmarks_points


# start_time = time.time()
# lm = detect_landmark(img)
# print(np.array(lm))
# end_time = time.time()
# print('total run-time: %f ms' % ((end_time - start_time)))