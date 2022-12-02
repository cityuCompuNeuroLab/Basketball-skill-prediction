import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
#from keras.optimizers import RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from scipy import signal
from scipy.signal import butter, filtfilt
import pandas as pd
from Time2Freq import valuematrix
from keras import regularizers
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import interpolate
import tensorflow as tf
from keras.callbacks import EarlyStopping


def create_model():
    tf.random.set_seed(0)
    # create model
    # Model
    model = Sequential()

    model.add(Conv2D(64, 2, 2, padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(64, 2, 2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, 2, 2, padding='same', activation='relu'))
    model.add(Conv2D(64, 2, 2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))


    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(0.0006, decay=0.001 / 10)


    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    MODEL_SUMMARY_BALANCE = r'./MODEL_SUMMARY_2type.txt'

    with open(MODEL_SUMMARY_BALANCE, "w") as fh:
        model.summary(print_fn=lambda line: fh.write(line + "\n"))

    return model


result_df = pd.DataFrame(columns=['loss', 'accuracy', 'R2', 'AUC', 'sensitivity', 'specificity', 'precision'])


for j in range(100):

    imgtrain, imgtest, train_lab, test_lab = valuematrix()
    imgtrain = imgtrain.reshape(len(imgtrain), 21, 33, 3)    # dimension of input figure
    imgtest = imgtest.reshape(len(imgtest), 21, 33, 3)


    trainNum = int(len(imgtrain)*0.8)
    trainX = imgtrain[:trainNum]
    trainY = train_lab[:trainNum]
    #trainY = to_categorical(train_lab[:trainNum])
    print("trainimg length" + str(len(trainX)))
    print("trainlabel length" + str(len(trainY)))

    validX = imgtrain[trainNum:]
    validY = train_lab[trainNum:]
    #validY = to_categorical(train_lab[trainNum:])
    print("validimg length" + str(len(validX)))
    print("validlabel length" + str(len(validY)))

    testX = imgtest
    #testY = to_categorical(test_lab)
    testY = test_lab
    print("testimg length" + str(len(testX)))
    print("testlabel length" + str(len(testY)))
    print("end loading dataset")

    # Hyperparams
    xH, xW, xD = imgtrain[0].shape
    print(imgtrain[0].shape)
    input_shape = (xH, xW, xD)
    print(xH)
    print(xW)
    print(xD)

    #input_shape = imgtrain[0].shape

    # Model
    model = create_model()

    # Training
    # fit network
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, min_delta=0.)]
    history = model.fit(trainX, trainY, epochs=200, batch_size=128, validation_data=(validX, validY), verbose=2, callbacks=callbacks)

    accuracy = model.evaluate(testX, testY)
    print('The loss and accuracy is' + str(accuracy))

    #plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('binary cross entropy loss', fontsize=16)
    plt.legend()
    #plt.show()
    plt.savefig('myfig.jpg', dpi=300)
    # save model
    MODEL_SUMMARY_FILE = "model_summary.txt"
    MODEL_FILE = "CNNP.h5"
    print("start saving the model")
    model.save(MODEL_FILE)
    print("end saving the model")

    # testing
    predY = model.predict(testX)
    predY_class = predY.copy()
    predY_class[predY_class >= 0.5] = 1
    predY_class[predY_class < 0.5] = 0

    R2 = r2_score(testY, predY)
    print(R2)

    # plot AUC
    # Learn to predict each class against the other
    n_classes = 2  # number of class

    fpr, tpr, _ = roc_curve(testY, predY)
    roc_auc = auc(fpr, tpr)
    print('AUC is' + str(roc_auc))


    tn, fp, fn, tp = confusion_matrix(testY, predY_class).ravel()


    sensitivity = tp / (fn + tp)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)

    result_df.loc[j, 'loss'] = accuracy[0]
    result_df.loc[j, 'accuracy'] = accuracy[1]
    result_df.loc[j, 'R2'] = R2
    result_df.loc[j, 'AUC'] = roc_auc
    result_df.loc[j, 'sensitivity'] = sensitivity
    result_df.loc[j, 'specificity'] = specificity
    result_df.loc[j, 'precision'] = precision


result_df.to_csv('CNN_axazgy.csv')