


import numpy as np
import os

from deeplearning.tp20_kerasTensorflowOrga.A_dataDealer import oneEpoch

"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=2,linewidth=5000,suppress=True)

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.datasets import mnist




class Hat_TwoConvLayers_28:

    def __init__(self,freezeConv):
        self.freezeConv=freezeConv



    def __call__(self,X:tf.Tensor):

        with tf.name_scope("Block1"):

            Y = Conv2D(filters=32, kernel_size=5, activation='relu', padding='same', name='conv1')(X)
            print(Y.name, "\t\t\t", Y.shape)

            Y = MaxPooling2D(pool_size=(2, 2), name='pool')(Y)
            print(Y.name, "\t\t\t", Y.shape)



        with tf.name_scope("Block2"):

            Y = Conv2D(filters=64, kernel_size=5, activation='relu', padding='same', name='conv1')(Y)
            print(Y.name, "\t\t\t", Y.shape)


            Y = MaxPooling2D(pool_size=(2, 2), name='pool')(Y)
            print(Y.name, "\t\t\t", Y.shape)


        if self.freezeConv: Y=tf.stop_gradient(Y)


        with tf.name_scope("Dense1"):

            Y = tf.reshape(Y,[-1,7*7*64])
            Y = Dense(units=1024,activation='relu')(Y)
            print(Y.name, "\t", Y.shape)



        with tf.name_scope("Dense2"):
            Y = Dense(units=10,activation='softmax')(Y)
            print(Y.name, "\t", Y.shape)


        return Y



""" test """
def step1():
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()

    """on récupère une donnée d'entrée """
    x=None
    for xx,yy in oneEpoch(x_train, y_train,batch_size=3,nb_batches=1):
        x=xx


    """conversion en tf"""
    _X = tf.constant(x)
    """ estimation de la sortie"""
    hat_y=Hat_TwoConvLayers_28(False)(_X)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(hat_y))



if __name__=="__main__":
    step1()

