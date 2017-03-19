# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as num
from collections import Counter

# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
features = num.array(dataset.data, 'int16') 
labels = num.array(dataset.target, 'int')

# Extract the HOG features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualise=True)
    list_hog_fd.append(fd)
hog_features = num.array(list_hog_fd, 'float64')

print "The Number of digits: ", Counter(labels)

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)
