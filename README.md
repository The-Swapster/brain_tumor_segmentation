# brain_tumor_segmentation

## Dataset
The dataset used was from kaggle
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

## Preprocessing
* Brain tumor images are cropped to only the brain region
* Then they are plotted to see the output
* Since the size of the dataset is small, the images were augmented
  1. Rotation range is 10
  2. Width sift range is 0.1
  3. Height shift range is 0.1
  4. Shear range is 0.1
  5. Brightness range is 0.3 - 1.0
  6. Flipping horizontally
  7. Flipping vertically
* We printed the data summary
  1. Number of samples
  2. Percentage of positve examples
  3. Percentage of negative examples

## Loading Data
* Images are read using opencv
* Plotting some images
* Splitting data into train, test, and validation
* Printing the shape of the datasets

## Building the model
* Function for F1 Score
* Building Model
  1. Addig zero padding
  2. Convolutinal layer with 32 neurons and filter size of 7,7
  3. Batch normalization
  4. Maxpooling filter size 4,4
  5. Maxpooling filter size 4,4
  6. Flatten layer
  7. Dense layer with one neuron, output layer
* Print model summary
* Compile model
* Train model for 24 epochs
* Plot the training and validation accuracy and loss
* Send positively classified image to matlab for segmentation

## Segmentation
* Resizing the image
* Applying threshold
* Applying morphological operations
* Apply bounding box
* Inserting the tumor are in red colour
* Display the tumor images
