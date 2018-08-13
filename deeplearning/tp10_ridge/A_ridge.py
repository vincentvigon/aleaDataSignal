

import matplotlib

matplotlib.use('TkAgg')  # truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn.linear_model
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"


def genereData(nbData):
    """génération des données:
       Y= a_vrai x_ext + b_vrai + bruit
        """
    dataX = np.random.normal(size=(nbData, 2))
    a_vrai = np.array([3.,4.], dtype=np.float32).reshape((2,1))
    b_vrai = 1
    dataY = dataX @ a_vrai + b_vrai + np.random.normal(loc=0, scale=0.1, size=(nbData,1))

    return dataX,dataY



def step0():
    x,y=genereData(10)
    print(x.shape)
    print(y.shape)
    print("x",x)
    print("y",y)



"""modèle linéaire avec sklearn"""
def step1():
    dataX, dataY=genereData(100)
    model=sklearn.linear_model.LinearRegression()
    model.fit(dataX,dataY)

    print(model.coef_)
    print(model.intercept_)



""" modèle linéaire avec tensorflow"""
def step2():


    """ ON PREPARE LES CALCULS  """
    nbData=100

    tf.reset_default_graph()

    """les 2 placeholder correspondant à l'entrée et à la sortie"""
    _X = tf.placeholder(name="x_ext", shape=(nbData,2), dtype=tf.float32)
    _Y = tf.placeholder(name="Y", shape=(nbData,1), dtype=tf.float32)

    """ 2 variables trainable """
    _a = tf.get_variable(name="hat_a", initializer=[[3.],[4.]])
    _b = tf.get_variable(name="hat_b", initializer=1.)

    _Y_hat =  tf.matmul(_X,_a)  + _b


    learning_rate=0.1

    """ pénalisation (ici prendre wei_decay=0, c'est le mieux)"""
    wei_decay=0.0

    _loss = tf.reduce_mean( tf.square( _Y - _Y_hat))  + wei_decay *  tf.reduce_sum(_a ** 2)
    _trainProcess = tf.train.AdamOptimizer(learning_rate).minimize(_loss)



    losses = []


    dataX,dataY=genereData(nbData)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for itr in range(1000):
            _, loss ,hat_a,hat_b =  sess.run([_trainProcess, _loss, _a,_b], {_X: dataX, _Y: dataY})
            print("---------")
            print(loss)
            print(hat_a)
            print(hat_b)
            losses.append(loss)

    plt.plot(losses)
    plt.show()







""" modèle  ridge : on rajoute une pénalisation  """
def step3():


    """ ON PREPARE LES CALCULS  """
    nbData=100

    tf.reset_default_graph()

    """les 2 placeholder correspondant à l'entrée et à la sortie"""
    _X = tf.placeholder(name="x_ext", shape=(nbData,2), dtype=tf.float32)
    _Y = tf.placeholder(name="Y", shape=(nbData,1), dtype=tf.float32)

    """ 2 variables trainable """
    _a = tf.get_variable(name="hat_a", initializer=[[3.],[4.]])
    _b = tf.get_variable(name="hat_b", initializer=1.)

    _Y_hat =  tf.matmul(_X,_a)  + _b


    learning_rate=0.1

    """ pénalisation (ici prendre wei_decay=0, c'est le mieux)"""
    wei_decay=0.0

    _loss = tf.reduce_mean( tf.square( _Y - _Y_hat))  + wei_decay *  tf.reduce_sum(_a ** 2)
    _trainProcess = tf.train.AdamOptimizer(learning_rate).minimize(_loss)


    losses = []


    dataX,dataY=genereData(nbData)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for itr in range(1000):
            _, loss ,hat_a,hat_b =  sess.run([_trainProcess, _loss, _a,_b], {_X: dataX, _Y: dataY})
            print("---------")
            print(loss)
            print(hat_a)
            print(hat_b)
            losses.append(loss)

    plt.plot(losses)
    plt.show()









