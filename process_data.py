import pickle
import numpy as np
from PIL import Image

X_train=np.zeros((10000,32,32,3))

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_encoding(labels,n):
    y=np.zeros((len(labels),n))
    for i,x in enumerate(labels):
        y[i][x]=1

    return y



def get_train_data_into_array():
    input_data={}
    output_labels_encoded={}
    for i in range(1,6):
        file='./cifar-10-batches-py/data_batch_'+str(i)
        batch=unpickle(file)
        input_data[str(i)]=batch[b'data'].reshape(len(batch[b'data']),3,32,32).transpose(0,2,3,1)

        output_labels= batch[b'labels']
        output_labels_encoded[str(i)]=one_hot_encoding(output_labels,10)


    X_train=np.concatenate((input_data['1'],input_data['2'],input_data['3'],input_data['4'],input_data['5']))

    Y_train = np.concatenate((output_labels_encoded['1'], output_labels_encoded['2'], output_labels_encoded['3'], output_labels_encoded['4'], output_labels_encoded['5']))



    return X_train, Y_train

def get_test_data_into_array():


    file = './cifar-10-batches-py/test_batch'
    batch = unpickle(file)
    X_test= batch[b'data'].reshape(len(batch[b'data']), 3, 32, 32).transpose(0, 2, 3, 1)

    output_labels = batch[b'labels']
    Y_test = one_hot_encoding(output_labels, 10)

    return X_test, Y_test


def fetch_data():
    X_train,Y_train=get_train_data_into_array()

    print(X_train.shape)
    print(Y_train.shape)

    X_test,Y_test=get_test_data_into_array()
    print(X_test.shape)
    print(Y_test.shape)


    # img = Image.fromarray(X_train[100], 'RGB')
    # print(Y_train[100])
    # img.save('my.png')
    # img.show()

    X_train=X_train/255
    X_test=X_test/255

    return X_train,X_test,Y_train,Y_test

fetch_data()

