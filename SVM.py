import numpy
import pandas
import os
import joblib
import pickle

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from skimage import io
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data_dir = "archive/myData"
images = []
labels = pandas.read_csv("archive/labels.csv").iloc[:, 1]
label_codes = []

for sign_type in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, sign_type)):
        for img_name in os.listdir(os.path.join(data_dir, sign_type)):
            img_path = os.path.join(data_dir, sign_type, img_name)
            image = io.imread(img_path)
            images.append(image)
            label_codes.append(int(sign_type))

images = numpy.array(images)
labels = numpy.array(label_codes)

# Flatten the images and scale them
ct  = 0
images_flat = images.reshape(images.shape[0], -1)


# Split the data for testing and training
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=0, shuffle=True)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = None
model_file_path = "svm_model.pkl"
if os.path.isfile(model_file_path):
    # The file exists, so we can proceed to load the model
    with open(model_file_path, 'rb') as file:
        svm = joblib.load(file)
    print("SVM Model loaded successfully, file location:", model_file_path)
else:
    # Train the SVM model
    svm = SVC(random_state=1, kernel='rbf')
    svm.fit(X_train, y_train)
    # Save the SVM model to a pickle file
    joblib.dump(svm, 'svm_model.pkl')

# Evaluate the model
y_prediction = svm.predict(X_test)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_prediction)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prediction, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
