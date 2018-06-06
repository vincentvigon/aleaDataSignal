
import numpy as np
from keras.datasets import mnist
import os

"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=2,linewidth=5000,suppress=True)




""" on observe les données MNIST. On créer notre distributeur de données """



"""  observons les données brutes """
def step0():
    """ la première fois cela prend du temps car keras télécharge les données, et les range dans  ~/.keras/datasets/ """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    print(x_train.shape,y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_train.dtype,y_train.dtype)

    print(x_train[0])
    print("cat:",y_train[0])





"""  distributeur de donnée par batch.  
  Avec le défaut nb_batches=None, toutes les données passent en une 'epoch' """

def oneEpoch(x,y,batch_size,nb_batches=None):

    if nb_batches is None: nb_batches=len(x)//batch_size

    assert batch_size*nb_batches<len(x), "pas assez de données"

    x=x.astype(np.float32)
    x/=255

    shuffle_index=np.random.permutation(len(x))

    x=x[shuffle_index]
    x=x[:,:,:,np.newaxis]

    y=y[shuffle_index]


    for i in range(nb_batches):
        yield x[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size]


"""test"""
def step1():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    for x,y in oneEpoch(x_train, y_train,batch_size=2,nb_batches=5):
        print("x.shape:",x.shape)
        print("y.shape",y.shape)
        print("y:",y)
        print("")


if __name__=="__main__":
    step0()