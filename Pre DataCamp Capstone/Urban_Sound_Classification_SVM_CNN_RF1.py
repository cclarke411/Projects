import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.utils  import to_categorical
from keras.layers import  Dense, Conv2D,Flatten,MaxPooling2D,Dropout
from sklearn.svm import NuSVC
from sklearn.svm import SVC

y = []
X = []; 
yp = []; 
new_X = []
new_X_mel =[]
yp_mel=[]

path = 'C:/Users/clyde/Documents/Thinkful/Pre Data Science Bootcamp/Sound Classification Datbase/Train/'
df = pd.read_csv('C:/Users/clyde/Documents/Thinkful/Pre Data Science Bootcamp/Sound Classification Datbase/train.csv')
class_label = df.Class.unique().tolist()

def extract_features(window_size):

    for i in df.ID:    
        try:
            [y,sr] = sf.read('C:/Users/clyde/Documents/Thinkful/Pre Data Science Bootcamp/Sound Classification Datbase/Train/%d.wav'%i)
            if y.ndim ==2 and len(y) < window_size:
                y = np.append(y[:,1].T[0:len(y)],np.zeros(window_size-len(y)))
            elif len(y)<window_size & y.ndim ==1:
               y = np.append(y,np.zeros(window_size-len(y)))
            elif y.ndim == 2 and len(y)>=window_size:
               y= y[:,1].T[0:window_size]    
            #print(len(y1),y1.shape,cnt)
            elif y.ndim == 1 and len(y)<window_size:
               y = np.append(y[0:len(y)],np.zeros(window_size-len(y)))
            else:
               y = y[0:window_size]
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=200).T, axis = 0)
            mfccs1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=512).T
            idx = class_label.index(df[df.ID == i]['Class'].tolist()[0])
            yp.append(idx)
            yp_mel.append(idx)
            new_X_mel.append(mfccs1)
            new_X.append(mfccs)
        except:
            print('%d.wav'%i)
    return new_X, yp,new_X_mel,yp_mel

def extract_mel(window_size):

    for i in df.ID:    
        try:
            [y,sr] = sf.read('C:/Users/clyde/Documents/Thinkful/Pre Data Science Bootcamp/Sound Classification Datbase/Train/%d.wav'%i)
            if y.ndim ==2 and len(y) < window_size:
                y = np.append(y[:,1].T[0:len(y)],np.zeros(window_size-len(y)))
            elif len(y)<window_size & y.ndim ==1:
               y = np.append(y,np.zeros(window_size-len(y)))
            elif y.ndim == 2 and len(y)>=window_size:
               y= y[:,1].T[0:window_size]    
            #print(len(y1),y1.shape,cnt)
            elif y.ndim == 1 and len(y)<window_size:
               y = np.append(y[0:len(y)],np.zeros(window_size-len(y)))
            else:
               y = y[0:window_size]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=512).T
            idx = class_label.index(df[df.ID == i]['Class'].tolist()[0])
            yp_mel.append(idx)
            new_X_mel.append(mfccs)
        except:
            print('%d.wav'%i)
    return new_X_mel, yp_mel

NewX, y,new_X_mel, yp_mel = extract_features(14900)

NewX_train, NewX_test, Newy_train, Newy_test = train_test_split(NewX, y, test_size=0.2, shuffle = True)
NewMel_train, NewMel_test, NewMely_train, NewMely_test = train_test_split(new_X_mel,yp_mel,test_size=0.2,shuffle=True)

clf = RandomForestClassifier(n_estimators = 100)
clf1 = OneVsRestClassifier(RandomForestClassifier(n_estimators = 500, max_depth=20, min_samples_leaf=30))

clf.fit(NewX_train, Newy_train)
clf1.fit(NewX_train, Newy_train)

prediction = clf.predict(NewX_test)
prediction1 = clf.predict(NewX_test)

print(classification_report(prediction, Newy_test))

#print ('fit to train new: ', clf.score(NewX_train, Newy_train))
#print ('fit to test: ', clf.score(NewX_test, Newy_test))
#print ('fit to train new: ', clf1.score(NewX_train, Newy_train))
#print ('fit to test: ', clf1.score(NewX_test, Newy_test))

matrix = confusion_matrix(Newy_test, prediction)
plt.figure(figsize=[10,10])
plt.imshow(matrix, cmap='hot', interpolation='nearest',  vmin=0, vmax=200)
plt.colorbar()
plt.title('Random Forest Confusion Map', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.xlabel('Predicted', fontsize=18)
plt.grid(b=False)
plt.yticks(range(10), class_label, fontsize=10)
plt.xticks(range(10), class_label, fontsize=10, rotation='vertical')
plt.show()

matrix = confusion_matrix(Newy_test, prediction1)
plt.figure(figsize=[10,10])
plt.imshow(matrix, cmap='hot', interpolation='nearest',  vmin=0, vmax=200)
plt.colorbar()
plt.title('Random Forest Confusion Map', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.xlabel('Predicted', fontsize=18)
plt.grid(b=False)
plt.yticks(range(10), class_label, fontsize=14)
plt.xticks(range(10), class_label, fontsize=14, rotation='vertical')
plt.show()



svm = OneVsRestClassifier(NuSVC(nu=.008,gamma='scale', kernel='poly', decision_function_shape='ovr'))
svmmodel = svm.fit(NewX_train, Newy_train)

svc_prediction = svmmodel.predict(NewX_test)

print ('fit to train new: ', svm.score(NewX_train, Newy_train))
print ('fit to test: ', svm.score(NewX_test, Newy_test))

matrix = confusion_matrix(Newy_test, svc_prediction)
plt.figure(figsize=[10,10])
plt.imshow(matrix, cmap='hot', interpolation='nearest',  vmin=0, vmax=200)
plt.colorbar()
plt.title('Support Vector Machine Confusion Map', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.xlabel('Predicted', fontsize=18)
plt.grid(b=False)
plt.yticks(range(10), class_label, fontsize=14)
plt.xticks(range(10), class_label, fontsize=14, rotation='vertical')
plt.show()

# ConvNet
NM_test = np.zeros(shape=(1087,30,128))
NM_train = np.zeros(shape=(4348,30,128))
#NMy_test = np.zeros(shape=(1087,1))
#NMy_train = np.zeros(shape=(4348,1))

for i in range(0,1087):
    NM_test[i][:][:]=NewMel_test[i][:]
    
for i in range(0,4348):
    NM_train[i][:][:]=NewMel_train[i][:]
    
NM_test   = NM_test.reshape(1087,30,128,1)
NM_train  = NM_train.reshape(4348,30,128,1)
NMy_test1 = to_categorical(NewMely_test)
NMy_train1= to_categorical(NewMely_train)

model = Sequential()
#add model layers
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(30, 128,1)))
model.add(Conv2D(32, kernel_size=3, activation= 'relu'))
model.add(Conv2D(16, kernel_size=3, activation= 'relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train Data
history_vanilla = model.fit(NM_train, NMy_train1, validation_data=(NM_test, NMy_test1), epochs=10)

#predict first 4 images in the test set
predNM_test = model.predict(NM_test)

pred_X_mel1 = np.argmax(predNM_test, axis=1)
print(classification_report(pred_X_mel1, NewMely_test))


vgg = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
vgg.add(Conv2D(32, (3, 3), activation='relu',input_shape=(30, 128, 1)))
vgg.add(Conv2D(32, (3, 3), activation='relu'))
vgg.add(MaxPooling2D(pool_size=(2, 2)))
vgg.add(Dropout(0.25))

vgg.add(Conv2D(64, (3, 3), activation='relu'))
vgg.add(Conv2D(64, (3, 3), activation='relu'))
vgg.add(MaxPooling2D(pool_size=(2, 2)))
vgg.add(Dropout(0.25))

vgg.add(Flatten())
vgg.add(Dense(256, activation='relu'))
vgg.add(Dropout(0.5))
vgg.add(Dense(10, activation='softmax'))
vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_vgg = vgg.fit(NM_train, NMy_train1, validation_data=(NM_test, NMy_test1), epochs=100)

vgg.summary()

predNMvgg_test = model.predict(NM_test)
predNMvgg_test1 = np.argmax(predNMvgg_test, axis=1)

print(classification_report(predNMvgg_test1, NewMely_test))


history_vgg = vgg.fit(NM_train, NMy_train1, validation_data=(NM_test, NMy_test1),batch_size=32, epochs=100)
history_vanilla = model.fit(NM_train, NMy_train1, validation_data=(NM_test, NMy_test1),batch_size=32, epochs=100)

# summarize history for accuracy
plt.plot(history_vgg.history['acc'])
plt.plot(history_vgg.history['val_acc'])
plt.title('Architecture 2 model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_vgg.history['loss'])
plt.plot(history_vgg.history['val_loss'])
plt.title('Architecture 2 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history_vanilla.history['acc'])
plt.plot(history_vanilla.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_vanilla.history['loss'])
plt.plot(history_vanilla.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()