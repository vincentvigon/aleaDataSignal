import keras
import matplotlib

from deeplearning.tp50_wordDetection.A_extract_data import filter_data, train_test_spliting, get_data, batch_generator, \
    paths_to_spectrograms

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import time

import numpy as np



from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten


np.set_printoptions(precision=2,linewidth=5000)


proportion_unknown_kept=0

def get_model(shape,num_classes):
    """"""
    '''on crée le modèle keras.'''

    inputLayer = Input(shape=shape)

    model = BatchNormalization()(inputLayer)
    model = Conv2D(16, (3, 3), activation='elu')(model)

    model = Dropout(0.5)(model)
    model = MaxPooling2D((2, 2))(model)

    model = Flatten()(model)
    """ on utilise une fonction d'activation elu (voir graph dans le fichier elu.png ci-joint)"""
    model = Dense(32, activation='elu')(model)
    model = Dropout(0.25)(model)

    """ il y a 10 ou 11 catégories avec ou sans 'unknown' """
    model = Dense(num_classes, activation='sigmoid')(model)

    model = keras.models.Model(inputs=inputLayer, outputs=model)

    return model



"""on entraîne :"""
def train():
    X, Y, class_names = filter_data(get_data(),proportion_unknown_kept)
    X_train, X_test, Y_train, Y_test=train_test_spliting(X,Y)


    model = get_model((129, 124,1),len(class_names))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


    batch_size=64


    train_gen = batch_generator(X_train,Y_train,len(class_names), batch_size=batch_size)
    valid_gen = batch_generator(X_test, Y_test, len(class_names), batch_size=batch_size)


    """les callbacks: ils sont lancées régulièrement durant le fit"""

    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=batch_size,write_images=True)

    """un schedule au hasard. La valeur par défaut de l'Adam est de 0.001. 
     On va un peu plus vite au début et on ralentit (c'est l'annealing). 
     Mais cela n'améliore pas vraiment les résultats.  """
    def schedule(epoch):
        if epoch <= 5: return 0.002
        if epoch <= 10: return 0.001
        if epoch <= 20: return 0.0005
    change_lr =keras.callbacks.LearningRateScheduler(schedule)

    """autre possibilité intéressante: quand la loss stagne, pendant 'patience'-epoch, on réluit le lr de 'factor' """
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=2, min_lr=0.001,verbose=1)



    """ici les epochs ne correspondent pas au passage d'un jeu de données"""
    model.fit_generator(
        generator=train_gen,
        epochs=50,
        steps_per_epoch=  50,
        validation_data= valid_gen,
        validation_steps= 20,
        callbacks=[tensorboard,reduce_lr])


    model.save_weights("weights")




"""
Ce programme est surtout intéressant que lorsque l'on s'interesse au 'unknown'.
Nous calculons l'accuracy séparément pour les unknown et les commandes vocales """

def test():
    X, Y, class_names = filter_data(get_data(), proportion_unknown_kept)
    """on garde le même spliting"""
    X_train, X_test, Y_train, Y_test = train_test_spliting(X, Y)

    model = get_model((129, 124, 1), len(class_names))
    model.load_weights("weights")
    print("model loaded")


    X_test_spec=np.expand_dims(paths_to_spectrograms(X_test), axis=3)

    probas=model.predict_on_batch(X_test_spec)
    print('probas.shape',probas.shape)
    Y_hat = np.argmax(probas,axis=1)


    verbose=True
    if verbose:
        print("true categories/estimation:")
        print(np.stack([Y_test[:100],Y_hat[:100]],axis=1).T)


    command_indices=(Y_test<10)

    print("accuracy entre commandes vocales")
    print(np.sum(Y_test[command_indices]==Y_hat[command_indices])/len(command_indices))

    if len(class_names)==11:
        unknown_indices=(Y_test==10)
        print("accuracy entre unknown")
        print(np.sum(Y_test[unknown_indices] == Y_hat[unknown_indices]) / len(unknown_indices))

        print("accuracy totale")
        print(np.sum(Y_test == Y_hat / len(Y_test)))



#train()
#test()


"""
A vous : 
Essayer cette exemple. Et si vous rajoutiez une ou deux couche de neurone ?
Quels sont les avantages de la fonction d'activation 'elu' ?

Essayez maintenant de rajouter les sons 'unknown'... Si vous y arrivez vous gagnez peut-être les 25 000$ du contours 
kaggle ! 

Pourquoi est-ce important de rajouter les 'unkonwn' dans le cadre de cette classification ? 




"""




