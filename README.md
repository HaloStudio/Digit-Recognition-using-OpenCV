# Digit-Recognition-using-OpenCV
This project uses OpenCV, sklearn and Python. The steps to detect handwritten digits uses Basic Image Processing techniques along with Support Vector Machine algorithm. Initially we create a database of Handwritten digits and then extract the HOG features from them, train a Linear SVM and then use a classifier to detect digits. We use MNIST database for handwritten digits in order to train the machine.For training a classifier we initially load the sklearn and numpy modules, load the classifier and then read the input image. Next we convert the image to Gray-scale in order to apply Gaussian Filtering. Then the threshold of the image is done, which means it is converted into foreground and background. The contours i.e boundaries of the image is identified and next we make Rectangles to contain each contour and HOG features for each of the rectangle is calculated. For testing a Classifier we load the data-set and extract the features and labels, next we create a multi class linear SVM object as there are multiple digits and then perform the training using the classifier by classifying on the basis of HOG features and Labels.

