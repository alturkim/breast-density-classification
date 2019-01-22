from utilities import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2 as cv2

argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--training", required=True,
                       help="path to the training images")
argparser.add_argument("-e", "--testing", required=False,
                       help="path to the testing images")
args = vars(argparser.parse_args())

descriptor = LocalBinaryPatterns(numPoints=24, radius=8)
data = []
labels = []


# loop over the training images
# TODO update the loop
for image_path in paths.list_images(args["training"]):
    # load the image, covert it to garyscale, and describe it
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = descriptor.describe(gray)
    #print (hist)

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(image_path.split("\\")[-2])
    data.append(hist)
    print(image_path.split("\\")[-2])
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)



