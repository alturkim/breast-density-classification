# Density Classification

The goal of this project is to classify tissue density into four classes giving an image.
The project uses two types of image descriptors: <br/>
1. Local Binary Pattern (LBP):  <br/>
   1. Rotational LBP. <br/>
   1. Rotational-invariant LBP. <br/>
1. Histogram of Oriented Gradients (HOG) <br/>
    
Giving the feature extracted from the method above, multiple classifiers are trained using the entire dataset.
## Descriptor Parameters
  ### Local Binary Pattern: 
    numPoints=24
    radius=8 

   ### Histogram of Oriented Gradients:
    orientations=9
    pixels_per_cell=(8, 8)
    cells_per_block=(2, 2)
    transform_sqrt=True
    visualize=True
    block_norm="L1"


## Feature Extraction
You can extract the features by running the following command: <br/>
`python features_extraction.py -t "path to the image files"`   
 
## Summary of the data
Number of images: 2779 <br/>
LBP feature vector length: 26 <br/>
Rotational LBP feature vector length: 26 <br/>
HOG feature vector length: 8100 <br/>
Number of total features: 8152  <br/>
