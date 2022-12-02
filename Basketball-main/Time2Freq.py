import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.utils import shuffle
from scipy import signal
from scipy.signal import butter, filtfilt
import pandas as pd
from pandas import DataFrame, Series
import random


def valuematrix():
    def pd2matrix(filename):
        returnmat = pd.read_csv(filename, header=None, delimiter=',')
        dataset = returnmat.iloc[1:len(returnmat) - 1, 1:]
        # print(type(dataset))
        # print(dataset)
        return dataset

    def pd2matrix1(filename):
        returnmat = pd.read_csv(filename, header=None, delimiter=',')
        dataset = returnmat.iloc[2:len(returnmat) - 2, :]
        #print(type(dataset))
        #print(dataset)
        return dataset

        ##### Calibration ##############

    def static(path1, path2, k):
        def to_float_list(arr):
            return [float(c) for c in arr]

        az1 = []
        az2 = []
        with open(path1, 'r') as f_p:
            for line in f_p:
                x1 = line.strip().split(' ')
                az1.append(x1[k])

            az1 = az1[1: 1500]
            az1 = to_float_list(az1)

        with open(path2, 'r') as f_g:
            for line in f_g:
                x2 = line.strip().split(' ')
                az2.append(x2[k])

            az2 = az2[1: 1500]
            az2 = to_float_list(az2)

        sensitivity = np.mean((np.array(az2) - np.array(az1)) / 2)
        print(sensitivity)
        print(type(sensitivity))
        mean = np.mean((np.array(az1) + np.array(az2)) / 2)
        print(mean)

        return sensitivity, mean

    def static_sydney(path1, path2, k):
        def to_float_list(arr):
            return [float(c) for c in arr]

        az1 = []
        az2 = []
        with open(path1, 'r') as f_p:
            for line in f_p:
                x1 = line.strip().split('\t')
                az1.append(x1[k])
            az1 = az1[1: 500]
            az1 = to_float_list(az1)

        with open(path2, 'r') as f_g:
            for line in f_g:
                x2 = line.strip().split('\t')
                az2.append(x2[k])

            az2 = az2[1: 500]
            az2 = to_float_list(az2)

        sensitivity = np.mean((np.array(az2) - np.array(az1)) / 2)
        print(sensitivity)
        mean = np.mean((np.array(az1) + np.array(az2)) / 2)
        print(mean)

        return sensitivity, mean

    def calibration(raw_data, sensitivity, mean):
        cal_data = sensitivity * raw_data + mean

        return cal_data

        ################ moving average filter ####################

    def mov_aver_filter(series, w):
        # Define mask and store as an array
        mask = np.ones((1, w)) / w
        mask = mask[0, :]
        # Convolve the mask with the raw data
        convolved_data = np.convolve(series, mask, 'valid')
        # Change series to data frame and add convolved data as a new column
        #series = Series.to_frame(series)
        return convolved_data

        ################ highpass_filter ####################

    def loading_data(path, path_x1, path_x2, path_y1, path_y2, path_z1, path_z2, label_athlete, duration = 320,
            overlap = 160,hk_g = 978.5):
        label = []

        value_ax = []
        value_ay = []
        value_az = []
        value_gx = []
        value_gy = []
        value_gz = []

        files = sorted(os.listdir(path))
        # get static data
        if label_athlete == 1:
            senX, meanX = static_sydney(path_x1, path_x2, 0)
            senY, meanY = static_sydney(path_y1, path_y2, 1)
            senZ, meanZ = static_sydney(path_z1, path_z2, 2)
            #print('sensitivity is', senX)
        else:
            senX, meanX = static(path_x1, path_x2, 0)
            senY, meanY = static(path_y1, path_y2, 1)
            senZ, meanZ = static(path_z1, path_z2, 2)
            #print('sensitivity is',senX)

        for file in files:
            f = os.path.basename(file)
            fileName = os.path.splitext(file)[0]
            print(f)
            dataset = pd2matrix1(path + os.path.basename(file))
            #print(dataset)
            #print(type(dataset))

            for i in range(6):
                a = dataset.iloc[:, i]
                a = a.values
                a = a.astype(np.float)
                #print('data is' + str(a))
                #print(type(a))
                if i == 0:
                    channel = 'ax'
                    a = calibration(a, senX, meanX)
                    #print('Calibration ax is' + str(a))
                    a = a * hk_g
                    c = mov_aver_filter(a, 15)
                    d = c
                elif i == 1:
                    channel = 'ay'
                    a = calibration(a, senY, meanY)
                    a = a * hk_g
                    c = mov_aver_filter(a, 15)
                    d = c
                    # print('HKG ay is' + str(a))
                elif i == 2:
                    channel = 'az'
                    a = calibration(a, senZ, meanZ)
                    a = a * hk_g
                    c = mov_aver_filter(a, 15)
                    d = c
                elif i == 3:
                    channel = 'gx'
                    c = mov_aver_filter(a, 15)
                    d = c
                elif i == 4:
                    channel = 'gy'
                    c = mov_aver_filter(a, 15)
                    d = c
                elif i == 5:
                    channel = 'gz'
                    c = mov_aver_filter(a, 15)
                    d = c

                for j in range((len(d) // overlap) - 1):
                    segg = d[overlap * j + 1:(overlap * j + duration)]
                    sig_nor = (segg - np.mean(segg)) / np.std(segg)
                    f, t, zxx = signal.stft(sig_nor, 50, window='hann', nperseg=40, noverlap=30, nfft=40)
                    absZ = np.abs(zxx)
                    absZ = list(absZ)
                        #print('absZ shape is ', absZ.shape)
                    if i == 0:
                        value_ax.append(absZ)
                        label.append(label_athlete)
                    elif i == 1:
                        value_ay.append(absZ)
                    elif i == 2:
                        value_az.append(absZ)
                    elif i == 3:
                        value_gx.append(absZ)
                    elif i == 4:
                        value_gy.append(absZ)
                    elif i == 5:
                        value_gz.append(absZ)

        return value_ax, value_ay, value_az, value_gx, value_gy, value_gz, label

    label = []

    allmatrix = []
    value_ax = []
    value_ay = []
    value_az = []
    value_gx = []
    value_gy = []
    value_gz = []


###################### loading Recreational data
    path = path_Rrecreational               # path of Rrecreational input on your computer

    path_x1 = 'path of SensorStaticHK\\top.txt'          # baseline of sensor stationary in top direction
    path_x2 = 'path of SensorStaticHK\\down.txt'         # baseline of sensor stationary in down direction
    path_y1 = 'path of SensorStaticHK\\left.txt'         # baseline of sensor stationary in left direction
    path_y2 = 'path of SensorStaticHK\\right.txt'        # baseline of sensor stationary in right direction
    path_z1 = 'path of SensorStaticHK\\backside.txt'          # baseline of sensor stationary in backside direction
    path_z2 = 'path of SensorStaticHK\\frontal.txt'           # baseline of sensor stationary in frontal direction

    value_ax_r, value_ay_r, value_az_r, value_gx_r, value_gy_r, value_gz_r, label_re = loading_data(path, path_x1, path_x2,
    path_y1, path_y2, path_z1, path_z2, label_athlete=0, duration=320, overlap=160, hk_g = 978.5)

    print('recreational image shape is' + str(len(value_ax_r)))




    # load professional data
    pathPro = path_Professional              # path of professional input on your computer

    path11 = 'path of SensorStaticSydney\\top.txt'             # baseline of sensor stationary in top direction
    path12 = 'path of SensorStaticSydney\\bottom.txt'          # baseline of sensor stationary in bottom direction
    path13 = 'path of SensorStaticSydney\\left.txt'            # baseline of sensor stationary in left direction
    path14 = 'path of SensorStaticSydney\\right.txt'           # baseline of sensor stationary in right direction
    path15 = 'path of SensorStaticSydney\\back.txt'            # baseline of sensor stationary in back direction
    path16 = 'path of SensorStaticSydney\\front.txt'           # baseline of sensor stationary in front direction

    value_ax_p, value_ay_p, value_az_p, value_gx_p, value_gy_p, value_gz_p, label_pro = loading_data(pathPro, path11, path12,
       path13, path14,path15, path16, label_athlete=1, duration=320, overlap=160, hk_g=978)



    value_ax = np.concatenate((value_ax_r, value_ax_p), axis=0)
    value_ax = np.array(value_ax)
    value_ay = np.concatenate((value_ay_r, value_ay_p), axis=0)
    value_ay = np.array(value_ay)
    value_az = np.concatenate((value_az_r, value_az_p), axis=0)
    value_az = np.array(value_az)

    value_gx = np.concatenate((value_gx_r, value_gx_p), axis=0)
    value_gx = np.array(value_gx)
    value_gy = np.concatenate((value_gy_r, value_gy_p), axis=0)
    value_gy = np.array(value_gy)
    value_gz = np.concatenate((value_gz_r, value_gz_p), axis=0)
    value_gz = np.array(value_gz)

    label = np.concatenate((label_re, label_pro))

    # input to CNN model
    allmatrix = np.stack((value_ax, value_az, value_gy), axis=3)  #  (value_ax[k], value_ay[k], value_az[k], value_gx[k], value_gy[k], value_gz[k]) (26, 19, 6)

    #allmatrix = value_gz

    print("Each img shape is" + str(value_ax[0].shape))
    print("All img shape is" + str(allmatrix.shape))


    (allmatrix, label) = shuffle(allmatrix, label)
    print('num of all images is' + str(len(allmatrix)))
    print('The label shape is' + str(label.shape))


    num_train = int(len(allmatrix) * 0.8)
    num_test = int(len(allmatrix) - num_train)
    print("num of trainset: " + str(num_train))
    print("num of testset: " + str(num_test))

    train_lab = label[0:num_train]
    test_lab = label[num_train:]
    imgtrain = allmatrix[0:num_train]
    imgtest = allmatrix[num_train:]

    print("imgtrain shape is: " + str(imgtrain.shape))
    print("imgtest shape is: " + str(imgtest.shape))
    print("imgtrain length is: " + str(len(imgtrain)))

    return imgtrain, imgtest, train_lab, test_lab


if __name__ == "__main__":

    imgtrain, imgtest, train_lab, test_lab = valuematrix()


# print(imgtrain)
# print(imgtest)
# print(train_lab)
# print(test_lab)
