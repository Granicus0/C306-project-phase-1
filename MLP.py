import pickle
import numpy
import pandas
import os
import joblib
import seaborn
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from skimage import io
from sklearn.neural_network import MLPClassifier



data_dir = "archive/myData"
images = []
# Extracting the labels for the signs. Index corresponds with the folder name (e.g. folder '0' corresponds to label
# "Speed limit (20km/h)"). Can be used for mapping label_codes to the actual labels if we ever want to display images
# to the user
labels = pandas.read_csv("archive/labels.csv").iloc[:, 1]
# Array of integers that will specify the classes, corresponding to the flattened images in images_flat
label_codes = []

# Read in each of the images from each of the folders inside archive/myData/x, where 0 <= x <= 42 as a string
for sign_type in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, sign_type)):
        for img_name in os.listdir(os.path.join(data_dir, sign_type)):
            img_path = os.path.join(data_dir, sign_type, img_name)
            image = io.imread(img_path)
            images.append(image)
            label_codes.append(int(sign_type))

# Make numpy arrays
images = numpy.array(images)
labels = numpy.array(labels)

# We want to make sure we flatten the images from a 3D input (2D image with RGB color space = 3D) to a 1D output
# so that we can actually feed it into our MLP model.
images_flat = images.reshape(images.shape[0], -1)
# Split the data for testing and training, using a standard 20-80 split
X_train, X_test, y_train, y_test = train_test_split(images_flat, label_codes, test_size=0.2, random_state=0,shuffle=True)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# To avoid lengthy waits with training, load the previously trained model from a pickle file. If you want to
# train the model again with new parameters, just delete the pickle file before running the code you modified.
mlp = None
model_file_path = "mlp_model.pkl"
if os.path.isfile(model_file_path):
    # The file exists, so we can proceed to load the model
    with open(model_file_path, 'rb') as file:
        mlp = joblib.load(file)
    print("MLP Model loaded successfully, file location:", model_file_path)
else:
    # Limit the utilization to four threads because PyCharm sucks on Linux and locks up
    mlp = MLPClassifier(random_state=1, learning_rate_init=0.0001, max_iter=10000)
    mlp.fit(X_train, y_train)
    # Dump the model into a pickle file, so we don't have to constantly
    # re-train it if we want to debug or re-check values
    joblib.dump(mlp, 'mlp_model.pkl')




# We will find the accuracy of the model first
y_prediction = mlp.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_prediction) 
conf_matrix_row_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, numpy.newaxis]

plt.figure(figsize=(10, 8))
seaborn.heatmap(conf_matrix_row_normalized, annot=False, fmt='.2f', cmap='Blues')

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# output_file_path = 'confusion_matrix.csv'
# with open(output_file_path, 'w') as file:
#     file.write("Confusion Matrix:\n")
#     numpy.savetxt(output_file_path, conf_matrix_row_normalized, delimiter=',', fmt='%f')

#######################################################
# accuracy = accuracy_score(y_test, y_prediction)
# print("Accuracy:", accuracy)
# # Then find our precision, f1, and accuracy scores
# precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prediction, average='weighted')
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1_score)