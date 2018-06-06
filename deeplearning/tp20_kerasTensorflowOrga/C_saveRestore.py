import matplotlib

from deeplearning.tp20_kerasTensorflowOrga.A_dataDealer import oneEpoch
from deeplearning.tp20_kerasTensorflowOrga.Hat_TwoConvLayer_28 import Hat_TwoConvLayers_28

matplotlib.use('TkAgg')  # truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.datasets import mnist



""" 
Même programme que précédemment, sauf que :
* on enferme toute la partie tf dans une classe "Modèle" avec les 3 fameuses méthodes .fit() .validate() .predict()
* on s'autorise de restorer le modèle depuis un apprentissage passé. 

"""



class Model_TwoConv_28:


    def __init__(self,savedir:str=None,freezeConv=False):

        self._X = tf.placeholder(name="x_ext", dtype=tf.float32, shape=[None, 28, 28, 1])
        self._Y = tf.placeholder(name="Y", dtype=tf.int32, shape=[None])
        self._Y_proba = tf.one_hot(self._Y, 10)

        self.lr = tf.get_variable("learningRate", initializer=1e-4, trainable=False)

        self._hat_Y_proba = Hat_TwoConvLayers_28(freezeConv)(self._X)
        self._loss = - tf.reduce_mean(self._Y_proba*tf.log(self._hat_Y_proba+1e-10))

        self._hat_Y = tf.cast(tf.argmax(self._hat_Y_proba, axis=1), tf.int32)
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._hat_Y, self._Y), tf.float32))

        self._opt = tf.train.AdamOptimizer(self.lr).minimize(self._loss)



        """pour sauver les variables du modèle"""
        self._saver = tf.train.Saver()
        """la session tensorflow"""
        self.sess = tf.Session()


        if savedir is None: self.sess.run(tf.global_variables_initializer())
        else : self._saver.restore(self.sess, savedir)

        self.verbose=True



    def saveAndClose(self,savedir:str):
        save_path = self._saver.save(self.sess, savedir)
        print("Model saved in file: %s" % save_path)
        self.close()


    def close(self):
        self.sess.close()
        """au cas où l'on lancerait plusieurs fois cette classe"""
        tf.reset_default_graph()


    def fit(self, X_train, Y_train):
        self.loss, self.accuracy, _ = self.sess.run([self._loss, self._accuracy, self._opt],
                                          feed_dict={self._X: X_train, self._Y: Y_train})

        if self.verbose:
            print("loss: %.3f \t accuracy: %.2f" % (self.loss, self.accuracy))

        return self.loss, self.accuracy




    def validate(self, X_valid, Y_valid):

        self.loss, self.accuracy = self.sess.run([self._loss,self. _accuracy], feed_dict={self._X:X_valid,self._Y:Y_valid})

        if self.verbose:
            print("VALIDATION=>")
            print("loss: %.3f \t accuracy: %.2f" % (self.loss, self.accuracy))
            print("=" * 100 + "\n")

        return self.loss, self.accuracy




    def predict(self, X_test):
        return self.sess.run(self._hat_Y, feed_dict={self._X: X_test})





""" programme principal """
def step0():


    saveDir="~/.savedNN"

    losses_valid = []
    itrs_valid = []
    losses_train = []
    itrs_train = []

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()

    (x_valid, y_valid) = (x_valid[:100, :, :, np.newaxis], y_valid[:100])


    restore=False

    if not restore:
        model = Model_TwoConv_28()
    else:
        model = Model_TwoConv_28(saveDir,freezeConv=False)

    """on règle le learning rate"""
    model.sess.run(model.lr.assign(1e-3))


    try:
        itr = -1
        for epoch in range(2):

            for x, y in oneEpoch(x_train, y_train, batch_size=50, nb_batches=20):
                itr += 1

                """tous les 10 itérations, on diminue le learning rate """
                if itr==10: model.sess.run(model.lr.assign(model.lr/2))

                print("lr:",model.sess.run(model.lr))


                doValidation = itr % 20 == 0 and itr > 0

                if not doValidation:
                    loss, accuracy = model.fit(x,y)
                    losses_train.append(loss)
                    itrs_train.append(itr)
                else:
                    loss, accuracy = model.validate(x_valid,y_valid)
                    losses_valid.append(loss)
                    itrs_valid.append(itr)




    except  KeyboardInterrupt:
        print("on a stoppé")


    model.saveAndClose(saveDir)

    plt.plot(itrs_train, losses_train, label="train")
    plt.plot(itrs_valid, losses_valid, label="valid")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    step0()


