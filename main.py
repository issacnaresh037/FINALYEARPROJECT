import numpy as np 
import matplotlib.pyplot as plt
import glob
import activate opencv-env
import conda install spyder
import cv2
import os
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)



print('\n')
print('\t\t','********************   READ DATA   ********************')
print('\n')


# Read input images and assign labels based on folder names
print(os.listdir("Dataset/"))

SIZE = 65  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("Dataset/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels into respective lists

test_images = []
test_labels = [] 
for directory_path in glob.glob("Dataset/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#********************************************************************************************************************#
print('\n')
print('\t\t','********************  LABEL ENCODING ********************')
print('\n')
#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
print("Test Label encoding",test_labels_encoded,"\n")
print("Train Label encoding",train_labels_encoded)

#***************************************************************************************************#
##############SPLITTING DATASET####################

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#***************************************************************************************************#

print('\n')
print('\t\t','************* SUPPORT VECTOR MACHINE (SVM) ***************','\n')

x_train1=x_train[:,:,1,1]
x_test1=x_test[:,:,1,1]

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

clf = SVC()
clf.fit(x_train1, y_train) 
svm_pred=clf.predict(x_test1)

svm_accu=accuracy_score(y_test,svm_pred)*100
print("\n"," SVM Accuracy : ",svm_accu,'%',"\n")


svm_cr=classification_report(y_test,svm_pred)
print("Classification Report: \n",svm_cr)

svm_cm = confusion_matrix(y_test, svm_pred)
  
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(svm_cm, cmap=plt.cm.BrBG, alpha=0.3)
for i in range(svm_cm.shape[0]):
    for j in range(svm_cm.shape[1]):
        ax.text(x=j, y=i,s=svm_cm[i, j], va='center', ha='center', size='xx-large')
print('Confusion Matrix :')
plt.show()

#***************************************************************************************************#

print('\n')
print('\t\t','*******  CONVOLUTION NEURAL NETWORK (CNN)  *******')
print('\n')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, 2, activation="relu", input_shape=(65,65,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 2, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(4, activation = 'sigmoid'))
model.compile(loss = 'categorical_crossentropy',optimizer = "adam",metrics = ['accuracy'])
model.summary()


model.fit(x_train, y_train_one_hot, batch_size=1,epochs=3, verbose=1)


#Y_pred=model.predict(x_test)
Y_p=model.predict_classes(x_test)
#Y_pd=np.round(abs(Y_p))
cm = confusion_matrix(y_test, Y_p)

acc = model.evaluate(x_train, y_train_one_hot)[1]*100
print('\n'," ACCURACY : ", acc,'%')

cr=classification_report(y_test,Y_p)
print(cr,'\n')

print('\t',' CONFUSION MATRIX :\n')
ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Oranges')
plt.show()

#***************************************************************************************************#
print('\n')
print('\t\t','********** ACCURACY COMPARISON  **********')
print('\n')

fig = plt.figure(figsize=(5.5, 3.5))
ax = fig.add_axes([0,0,1,1])
yaxis = [svm_accu,acc]
xaxis=['SUPPORT VECTOR MACHINE','CONVOLUTIONAL NEURAL NETWORK']
sns.barplot(xaxis,yaxis)
plt.show()

#***************************************************************************************************#

n=38 #Select the index of image to be loaded for testing
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
#input_img_features=feature_extractor.predict(input_img)
prediction_RF = model.predict_classes(input_img)[0]
prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("\n","The prediction for this image is: ", prediction_RF,"\n")
print("The actual label for this image is: ", test_labels[n])


#********************************************************************************************************************#
