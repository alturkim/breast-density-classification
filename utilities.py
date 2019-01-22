# Credits: Local Binary Patterns code is from the following tutorial:
# https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/

from skimage import feature
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of point and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use it to build the histogram of patterns
        # "uniform" method indicates the rotation and grayscale invariant
        # form of LBPs
        # Note: returned lbp is a 2D array with the same width and height as image
        # each value in lbp array ranges from [0, numPoints + 2]
        lbp = feature.local_binary_pattern(image=image, P=self.numPoints,
                                           R=self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist


def split(path, test_size=0.3, random_state=42):
    imgs, labels = [], []
    training_txt_file_path = "../txt_files/training_set.txt"
    testing_txt_file_path = "../txt_files/testing_set.txt"
    with open(path, "r") as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

    for record in content:
        imgs.append(record.split()[0])
        labels.append(record.split()[1])

    train_set, test_set = train_test_split(content, test_size=test_size, random_state=random_state, stratify=labels)

    with open (training_txt_file_path, "w") as f:
        for i in range(0, len(train_set)):
            f.write(str(train_set[i]) + '\n')
    with open (testing_txt_file_path, "w") as f:
        for i in range(0, len(test_set)):
            f.write(str(train_set[i]) + '\n')

    return training_txt_file_path, testing_txt_file_path


def dump_image_file_names(path):
    output_path = "../txt_files/image_paths.txt"
    with open(output_path, "w") as f:
        for image_path in paths.list_images(path):
            f.write(str(image_path) + ' ' + image_path.split("\\")[-2] + '\n')
    return output_path






