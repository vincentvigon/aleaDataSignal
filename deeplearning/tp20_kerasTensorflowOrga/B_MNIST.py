

import numpy as np
import os
from keras.datasets import mnist

from deeplearning.tp20_kerasTensorflowOrga.A_dataDealer import oneEpoch
from deeplearning.tp20_kerasTensorflowOrga.TwoConvLayer_28 import Hat_TwoConvLayers_28

"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=2,linewidth=5000,suppress=True)

import tensorflow as tf
import matplotlib.pyplot as plt


"""  
On entraîne le modèle à reconnaître les digit de MNIST. 

Le coeur du modèle est dans la classe Hat_TwoConvLayer_28 qui transforme l'input x_ext en un vecteur de probabilités.


Remarquez la possibilité de "geler" les poids des noyaux de convolutions.
"""





def step0():



    _X = tf.placeholder(name="x_ext", dtype=tf.float32, shape=[None, 28, 28, 1])
    _Y = tf.placeholder(name="Y", dtype=tf.int32, shape=[None])
    _Y_proba=tf.one_hot(_Y,10)

    _lr = tf.get_variable("learningRate", initializer=1e-4, trainable=False)

    _hat_Y_proba = Hat_TwoConvLayers_28(freezeConv=True)(_X)
    _loss = - tf.reduce_mean(_Y_proba*tf.log(_hat_Y_proba+1e-10))


    _hat_Y = tf.cast(tf.argmax(_hat_Y_proba, axis=1),tf.int32)
    _accuracy = tf.reduce_mean(tf.cast(tf.equal(_hat_Y, _Y), tf.float32))

    _opt = tf.train.AdamOptimizer(_lr).minimize(_loss)


    losses_valid=[]
    itrs_valid=[]
    losses_train=[]
    itrs_train=[]


    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()

    (x_valid, y_valid)=(x_valid[:100,:,:,np.newaxis],y_valid[:100])



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        itr=-1

        for epoch in range(1):
            for x,y in oneEpoch(x_train,y_train,batch_size=50,nb_batches=100):
                itr += 1


                doValidation=  itr%20==0 and itr>0

                if not doValidation:
                    loss, accuracy,_ = sess.run([_loss, _accuracy,_opt], feed_dict={_X:x,_Y:y})
                    print("loss: %.3f \t accuracy: %.2f" % (loss, accuracy))
                    losses_train.append(loss)
                    itrs_train.append(itr)
                else:
                    loss, accuracy = sess.run([_loss, _accuracy], feed_dict={_X:x_valid,_Y:y_valid})
                    print("VALIDATION=>")
                    print("loss: %.3f \t accuracy: %.2f" % (loss, accuracy))
                    print("=" * 100 + "\n")
                    losses_valid.append(loss)
                    itrs_valid.append(itr)




    plt.plot(itrs_train,losses_train,label="train")
    plt.plot(itrs_valid,losses_valid,label="valid")
    plt.legend()
    plt.show()












