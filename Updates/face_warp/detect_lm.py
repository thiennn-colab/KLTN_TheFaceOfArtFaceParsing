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

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/shape_predictor_68_face_landmarks.dat")
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((y, x))
    cv2.imwrite('/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/input/input.png', img)
    mio.export_landmark_file(PointCloud(landmarks_points), '/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/input/input.pts',
                                                                                                                        overwrite=True)


def center_margin_bb(bb, img_bounds, margin=0.25):


    bb_size = ([bb[0, 2] - bb[0, 0], bb[0, 3] - bb[0, 1]])
    margins = (np.max(bb_size) * (1 + margin) - bb_size) / 2
    bb_new = np.zeros_like(bb)
    bb_new[0, 0] = np.maximum(bb[0, 0] - margins[0], 0)
    bb_new[0, 2] = np.minimum(bb[0, 2] + margins[0], img_bounds[1])
    bb_new[0, 1] = np.maximum(bb[0, 1] - margins[1] - margins[1] - margins[1] , 0)
    bb_new[0, 3] = np.minimum(bb[0, 3] + margins[1], img_bounds[0])
#     bb_new[0, 3] = np.minimum(bb[0, 3] , img_bounds[0])
    return bb_new

def crop_img_facial(img, margin=0.5):
    img_bounds = img.bounds()[1]
    grp_name = img.landmarks.group_labels[0]
    bb_menpo = img.landmarks[grp_name].bounding_box().points
    bb = np.array([[bb_menpo[0, 1], bb_menpo[0, 0], bb_menpo[2, 1], bb_menpo[2, 0]]])

    bb = center_margin_bb(bb, img_bounds, margin=margin)
    bb_pointcloud = PointCloud(np.array([[bb[0, 1], bb[0, 0]],
                                         [bb[0, 3], bb[0, 0]],
                                         [bb[0, 3], bb[0, 2]],
                                         [bb[0, 1], bb[0, 2]]]))

    face_crop = img.crop_to_pointcloud(bb_pointcloud)

    face_crop = face_crop.resize([256, 256])

    
    return face_crop

def save_image(img):
    im_pixels = np.rollaxis(img.pixels, 0, 3)

    outdir = 'output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    imsave('/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/output/input.png', im_pixels)
    mio.export_landmark_file(img.landmarks['PTS'], '/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/output/input.pts', overwrite=True)


def crop_img(img):
    detect_landmark(img)
    out_image_list = mio.import_images('/home/KLTN_TheFaceOfArtFaceParsing/Updates/face_warp/input/', verbose='true')
    image = crop_img_facial(out_image_list[0])
    save_image(image)

