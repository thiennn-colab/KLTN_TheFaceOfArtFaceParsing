import cv2
import numpy as np


def getTexture(path='./textures/water.jpg'):
    texture = np.array(cv2.imread(path))
    result = np.zeros((texture[:, :, 0].flatten().shape[0], 3))
    result[:, 0] = texture[:, :, 0].flatten()
    result[:, 1] = texture[:, :, 1].flatten()
    result[:, 2] = texture[:, :, 2].flatten()
    # texture = np.unique(texture)
    return result


a = getTexture()
print(a.shape)

index = np.random.choice(a.shape[0], (10))
print(a[index, :])
