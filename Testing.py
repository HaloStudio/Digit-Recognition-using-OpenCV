# Import the required modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as num
from collections import Counter

# Load the datasets
dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels and storing it in the Numpy Array
features = num.array(dataset.data, 'int16') 
labels = num.array(dataset.target, 'int')

# Extract the HOG features by creating a list of features and storing it
list_hog_fd = []
for ftre in features:
    featureDescriptor = hog(ftre.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualise=True)
    list_hog_fd.append(featureDescriptor)
hog_features = num.array(list_hog_fd, 'float64')

print "The Number of digits: ", Counter(labels)

# Create a linear SVM object
clf = LinearSVC()

# Performing the training using the Classifier
clf.fit(hog_features, labels)

# Saving the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)
