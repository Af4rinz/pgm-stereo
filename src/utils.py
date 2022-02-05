import cv2
import numpy as np
from matplotlib import cm
import os

def pair2grey(left, right):
    left_grey = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_grey = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return left_grey, right_grey

# load the images from dataset-2005 subdirectories
def load_image(path, left_suffix='iew1.png', right_suffix='iew5.png'):
    # load images
    left_images = {}
    right_images = {}
    for folder in os.listdir(path):
            for file in os.listdir(path + folder + '/'):
                if file.endswith(left_suffix):
                    left_img = cv2.imread(path + folder+ '/' + file)
                    left_images[folder] = left_img
                elif file.endswith(right_suffix):
                    right_img = cv2.imread(path + folder + '/' + file)
                    right_images[folder] = right_img
    return left_images, right_images


def view_images(left_images, right_images):
    for k in left_images.keys():
        cv2.imshow('left', left_images[k])
        cv2.imshow('right', right_images[k])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def disp2grey(disp):
    image = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
    occluded = (disp < 0)
    image[:] = np.where(occluded, 0, 255 * disp / disp.max())[:, :, np.newaxis]
    image[occluded] = [0, 0, 0]
    return image

def disp2jet(disp):
    cm_jet = cm.ScalarMappable(cmap='jet')
    occluded = (disp < 0)
    jet = cm_jet.to_rgba(np.where(occluded, 0, disp), bytes=True)[:, :, :3]
    jet[occluded] = 0
    return jet
