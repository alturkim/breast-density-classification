# Credit: Some parts of this code are due to https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
import utilities as utils
import csv
import numpy as np
from skimage import feature
import argparse
import cv2 as cv2


argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--training", required=False, default="..\\IRMA_Patchers",
                       help="path to the training images")
args = vars(argparser.parse_args())


def get_lbp(img):
    descriptor = utils.LocalBinaryPatterns(numPoints=24, radius=8)
    hist = descriptor.describe(img)
    return hist


def get_rot_lbp(img):
    descriptor = utils.LocalBinaryPatterns(numPoints=24, radius=8, method="default")
    hist = descriptor.describe(img)
    return hist


def get_hog(img):
    hist, _ = feature.hog(image=img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          transform_sqrt=True, visualize=True, block_norm="L1")
    return hist


if __name__ == '__main__':
    # to hold feature vector for all images
    features = []
    total_images = 0
    # save all feature vectors + label in a csv file
    with open("..\\features.csv", 'w', newline="") as f:
        writer = csv.writer(f)

        # loop over the training images
        images_paths, labels = utils.read_dataset(args["training"])
        for image_path in images_paths:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            lbp_hist = get_lbp(gray)
            rot_lbp_hist = get_rot_lbp(gray)
            hog_hist = get_hog(gray)
            label = image_path.split("\\")[-2]
            features.append(np.concatenate((lbp_hist, rot_lbp_hist, hog_hist, label), axis=None))
            writer.writerows(features)
            features = []
            total_images = total_images + 1

    # log a brief details about the dataset
    with open("..\\data.txt", 'w',) as f:
        f.write("Number of images: " + str(total_images) + "\n")
        f.write("LBP feature vector length: " + str(np.shape(lbp_hist)[0]) + "\n")
        f.write("Rotational LBP feature vector length: " + str(np.shape(rot_lbp_hist)[0]) + "\n")
        f.write("HOG feature vector length: " + str(np.shape(hog_hist)[0]) + "\n")
        f.write("Number of total features: " + str(np.shape(lbp_hist)[0] + np.shape(rot_lbp_hist)[0] + np.shape(hog_hist)[0]) + "\n")
