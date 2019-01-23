import utilities as utils
import numpy as np
import argparse
import cv2 as cv2

# argparser = argparse.ArgumentParser()
# argparser.add_argument("-t", "--training", required=True,
#                        help="path to the training images")
# argparser.add_argument("-e", "--testing", required=False,
#                        help="path to the testing images")
# args = vars(argparser.parse_args())


def get_lbp():
    descriptor = utils.LocalBinaryPatterns(numPoints=24, radius=8)
    features = []
    labels = []
    images_paths, _ = utils.read_dataset("..\\IRMA_Patchers")

    # loop over the training images
    for image_path in images_paths:
        # load the image, covert it to garyscale, and describe it
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = descriptor.describe(gray)
        # print(np.shape(hist))

        # extract the label from the image path, then update the
        # label and data lists
        labels.append(image_path.split("\\")[-2])
        features.append(hist)
        # print(image_path.split("\\")[-2])
    return features, labels



