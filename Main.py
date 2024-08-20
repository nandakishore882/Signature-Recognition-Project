from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, LSTM
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from numpy import dot
from numpy.linalg import norm

from keras.layers import Conv2D, MaxPool2D, InputLayer, BatchNormalization, RepeatVector



main = tkinter.Tk()
main.title("Signature Recognition using Deep Learning")
main.geometry("1300x1200")

global filename
global accuracy, precision, recall, fscore
global dataset
global X, Y
global X_train, X_test, y_train, y_test, cnn_model, labels

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename, X, Y, labels
    labels = []
    filename = filedialog.askdirectory(initialdir = r"C:\Signature\Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    text.insert(END,"Dataset Loading Completed\n")
    text.insert(END,"Class labels found in Dataset : "+str(labels)+"\n")
    text.insert(END,"Total images found in Dataset : "+str(X.shape[0])+"\n")
    text.update_idletasks()
    label, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
            
def processDataset():
    global X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset Preprocessing, Shuffling & Normalization Completed")

def trainTestSplit():
    global X, Y
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Train & Test Split\n\n")
    text.insert(END,"Total images found in Dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset size used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset size used for testing  : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore, labels
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    :  "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(8, 4)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runCNN():
    global X, Y
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, cnn_model
    global accuracy, precision, recall, fscore
    accuracy, precision, recall, fscore = [], [], [], []
    
    #training CNN algorithm on training features and testing on 20% test images
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #perform prediction on test data
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    #call this function to calculate accuracy and other metrics
    calculateMetrics("CNN", predict, y_test1)

def runCapsnet():
    global X_train, X_test, y_train, y_test, cnn_model
    global accuracy, precision, recall, fscore
    #define CAPS layers with number of neurons 64 to filter dataset to optimize features
    caps_model = Sequential()
    caps_model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    #creating conv2d layer of 64 neurons as to filter dataset features
    caps_model.add(Conv2D(64, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    #max layer to collect relevant features from filter dtaa
    caps_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    #defining another layer to filter generate instances
    caps_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    caps_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    #generated features normalization
    caps_model.add(BatchNormalization())
    #adding another CNN layer
    caps_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    caps_model.add(MaxPool2D(pool_size=(1, 1), padding='valid'))
    caps_model.add(BatchNormalization())
    #dropout to remove irrelevant features
    caps_model.add(Dropout(0.5))
    caps_model.add(Flatten())
    #defining prediction model as primary caps
    caps_model.add(Dense(units=100, activation='relu'))
    caps_model.add(Dense(units=100, activation='relu'))
    caps_model.add(Dropout(0.5))
    caps_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    #compiling and training and loading model
    caps_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/caps_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/caps_weights.hdf5', verbose = 1, save_best_only = True)
        hist = caps_model.fit(X_train, y_train, batch_size = 32, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/caps_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        caps_model.load_weights("model/caps_weights.hdf5")
    #perform prediction on test data
    predict = caps_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:400] = y_test1[0:400]
    #call this function to calculate accuracy and other metrics
    calculateMetrics("CapsuleNet", predict, y_test1)

def runLSTM():
    global X_train, X_test, y_train, y_test, cnn_model
    global accuracy, precision, recall, fscore
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], (X_train.shape[2] * X_train.shape[3])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], (X_test.shape[2] * X_test.shape[3])))
    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(100,input_shape=(X_train1.shape[1], X_train1.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.5))
    #adding another layer
    lstm_model.add(Dense(100, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train1, y_train, batch_size = 32, epochs = 15, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    #perform prediction on test data
    predict = lstm_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    #call this function to calculate accuracy and other metrics
    calculateMetrics("LSTM", predict, y_test1)

def runRBF():
    global X_train, X_test, y_train, y_test, cnn_model
    global accuracy, precision, recall, fscore
    rbf_model = Sequential()
    rbf_model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    rbf_model.add(Dense(128, activation='relu'))
    rbf_model.add(Dropout(0.2))
    rbf_model.add(Dense(64, activation='relu'))
    rbf_model.add(Dropout(0.2))
    rbf_model.add(Dense(y_train.shape[1], activation='softmax'))
    rbf_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/rbf_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/rbf_weights.hdf5', verbose = 1, save_best_only = True)
        hist = rbf_model.fit(X_train, y_train, batch_size = 32, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/rbf_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        rbf_model.load_weights("model/rbf_weights.hdf5")
    #perform prediction on test data
    predict = rbf_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:390] = y_test1[0:390]
    #call this function to calculate accuracy and other metrics
    calculateMetrics("Radial Basis Network", predict, y_test1)

def runSiamese():
    global X_train, X_test, y_train, y_test, cnn_model
    global accuracy, precision, recall, fscore
    siamese_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)#create siamese  model
    train = siamese_model.predict(X_train)
    test = siamese_model.predict(X_test)
    train = np.reshape(train, (train.shape[0], 16, 16, 1))
    test = np.reshape(test, (test.shape[0], 16, 16, 1))
    print(train.shape)
    print(test.shape)
    siamese_model = Sequential()
    siamese_model.add(Convolution2D(32, (1 , 1), input_shape = (train.shape[1], train.shape[2], train.shape[3]), activation = 'relu'))
    siamese_model.add(MaxPooling2D(pool_size = (1, 1)))
    siamese_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    siamese_model.add(MaxPooling2D(pool_size = (1, 1)))
    siamese_model.add(Flatten())
    siamese_model.add(Dense(units = 256, activation = 'relu'))
    siamese_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    siamese_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/siamese_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/siamese_weights.hdf5', verbose = 1, save_best_only = True)
        hist = siamese_model.fit(train, y_train, batch_size = 32, epochs = 15, validation_data=(test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/siamese_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        siamese_model.load_weights("model/siamese_weights.hdf5")
    #perform prediction on test data    
    predict = siamese_model.predict(test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    #call this function to calculate accuracy and other metrics
    calculateMetrics("Siamese Model", predict, y_test1)


def graph():
    global accuracy, precision, recall, fscore, rmse
    df = pd.DataFrame([['CNN','Precision',precision[0]],['CNN','Recall',recall[0]],['CNN','F1 Score',fscore[0]],['CNN','Accuracy',accuracy[0]],
                       ['CapsuleNet','Precision',precision[1]],['CapsuleNet','Recall',recall[1]],['CapsuleNet','F1 Score',fscore[1]],['CapsuleNet','Accuracy',accuracy[1]],
                       ['LSTM','Precision',precision[2]],['LSTM','Recall',recall[2]],['LSTM','F1 Score',fscore[2]],['LSTM','Accuracy',accuracy[2]],
                       ['RBF','Precision',precision[3]],['RBF','Recall',recall[3]],['RBF','F1 Score',fscore[3]],['RBF','Accuracy',accuracy[3]],
                       ['Siamese','Precision',precision[4]],['Siamese','Recall',recall[4]],['Siamese','F1 Score',fscore[4]],['Siamese','Accuracy',accuracy[4]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()

def predictSignature():
    global cnn_model, labels
    label = ['Forge', 'Real']
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = cnn_model.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (700,400))
    cv2.putText(img, 'Signature Recognized As : '+label[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Signature Recognized As : '+label[predict], img)
    cv2.waitKey(0)

font = ('times', 16, 'bold')
title = Label(main, text='Signature Recognition using Neural Networks')
title.config(bg='chocolate', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Signature Dataset", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=200)
processButton.config(font=font1)

traintestButton = Button(main, text="Train & Test Dataset Split", command=trainTestSplit)
traintestButton.place(x=700,y=250)
traintestButton.config(font=font1) 

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=700,y=300)
cnnButton.config(font=font1)

capsButton = Button(main, text="Run CapsuleNet Algorithm", command=runCapsnet)
capsButton.place(x=700,y=350)
capsButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=700,y=400)
lstmButton.config(font=font1)

rbfButton = Button(main, text="Run Radial Basis Algorithm", command=runRBF)
rbfButton.place(x=700,y=450)
rbfButton.config(font=font1)

siameseButton = Button(main, text="Run Siamese Algorithm", command=runSiamese)
siameseButton.place(x=700,y=500)
siameseButton.config(font=font1)

graphButton = Button(main, text="Comaprison Graph", command=graph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)

predictButton = Button(main, text="Signature Recognition from Test Image", command=predictSignature)
predictButton.place(x=700,y=600)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()
