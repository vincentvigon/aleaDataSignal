
import matplotlib
from sklearn.cross_validation import train_test_split

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
from pathlib import Path
import time

from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal



"""
Récupérez les sons dans le data-store de ma page web. 
Ils sont mis à disposition dans le cadre d'un concours 'kaggle'
Il s'agit de classifier les sons qui seront utiliser comme commande vocale pour un appareil 
(d'une boite assez connu dont le nom commence par G)

"""


path_perso='/Users/vigon/GoogleDrive/permanent/public/data_store/small_words/train/audio/'
np.set_printoptions(precision=2,linewidth=5000)


def get_data():
    datadir = Path(path_perso)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])
    return df



"""on ne garde que les catégories qui correspondent à des commandes vocales. 
Les autres n'intéresse pas la machine. On les mets dans la catégorie 'unknown' 
On ne garde qu'une proportion de unknown
"""
def filter_data(df,proportion_unknown_kept):

    commands_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    all_words = df.word.unique().tolist()

    silence = ['_background_noise_']
    unknown = [w for w in all_words if w not in silence + commands_words]

    """ there are only 6 silence-files. Mark them as unknown too. """
    df.loc[df.word.isin(silence), 'word'] = 'unknown'
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'


    df_control = df[df['word'] != 'unknown']
    X_control = df_control['path'].values
    Y_control = df_control['word'].values

    """transformation des nombres en numéros"""
    class_names, Y_control = np.unique(Y_control, return_inverse=True)


    if proportion_unknown_kept==0:
        n = len(X_control)
        shuffle = np.random.permutation(n)
        X = X_control[shuffle]
        Y = Y_control[shuffle]
        return X,Y,class_names
    else:

        """ RAJOUT ÉVENTUEL DES UNKNOWN lorsque proportion_unknown_kept>0"""
        class_names=np.concatenate([class_names,["unknown"]])

        df_unknown=df[df.word=='unknown']
        X_unknown = df_unknown['path'].values
        Y_unknown = df_unknown['word'].values

        _,Y_unknown=np.unique(Y_unknown, return_inverse=True)


        """on n'en garde qu'un partie aléatoire"""
        if proportion_unknown_kept<1:
            n=len(X_unknown)
            shuffle=np.random.permutation(n)[:int(proportion_unknown_kept*n)]
            X_unknown=X_unknown[shuffle]
            Y_unknown=Y_unknown[shuffle]

        """on colle le tout"""
        X=np.concatenate([X_control,X_unknown])
        Y=np.concatenate([Y_control,Y_unknown])
        n=len(X)
        shuffle=np.random.permutation(n)
        X=X[shuffle]
        Y=Y[shuffle]

        return X,Y,class_names



def step0():
    df=get_data()
    X, Y, class_names=filter_data(df,proportion_unknown_kept=0)

    print("X.shape:",X.shape)
    print("X[:10]\n",X[:10])
    print("Y[:10]:\n",Y[:10])
    print("class_names:",class_names)

    print("nombre de son par classes:")
    X_series=pd.Series(Y)
    print(X_series.value_counts(dropna=False))






"""on fixe les paramètre du train_test_split. On fixe notamment la seed du générateur aléatoire"""
def train_test_spliting(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y,random_state=1231235)
    return X_train, X_test, Y_train, Y_test




""" Given list of paths, return specgrams  """
def paths_to_spectrograms(paths, nbSoundSamples=16000):


    # read the wav files
    wavs = [wavfile.read(x)[1] for x in paths]

    # zero pad the shorter samples and cut off the long ones.
    data = []
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (nbSoundSamples - wav.size, 0), mode='constant')
        else:
            d = wav[0:nbSoundSamples]
        data.append(d)

    # get the specgram
    specgram = [signal.spectrogram(d, nperseg=256, noverlap=128)[2] for d in data]

    return specgram




"""  
Critiquer ce programme (il s'agit d'un premier jet)
Améliorez-le et adapter-le pour qu'on puisse comparer 3 mots différents (autre que 'yes' et 'right' ). Vous n'êtes
pas obligé de vous limiter au 11 mots qui sont des commandes vocales. 

Pour la visualisation, on a appliqué un log au spectrogrammes, est-ce une bonne idée. 
Quelle autre genre de transformation conseillerez-vous ?
 """
def step1():

    X, Y, class_names = filter_data(get_data(), 0)

    twoWords={
        'yes':[],
        'right':[]
    }

    i=0
    while len(twoWords['yes'])<5 or len(twoWords['right'])<5 :

        path=X[i]
        word=class_names[Y[i]]

        print("word ",i," : " ,word)

        if word=='yes':
            twoWords['yes'].append(path)
        elif word=='right':
            twoWords['right'].append(path)

        i+=1


    specgramsWord1=paths_to_spectrograms(twoWords['yes'])
    print(len(specgramsWord1))
    specgramsWord2=paths_to_spectrograms(twoWords['right'])
    print(len(specgramsWord2))

    print("one specgram shape",specgramsWord1[0].shape)

    for i in range(10):

        if i<5:
            specgram = specgramsWord1[i]
            word="yes"
        else:
            specgram = specgramsWord2[i-5]
            word = "right"


        plt.subplot(2,5,i+1)
        plt.imshow(np.log(1+specgram),origin="down")
        plt.title(word)

    plt.show()




#from keras.utils.np_utils import to_categorical

"""  
 Remarquons que les transformation des données qui prennent de la place ne sont effectuées que dans le générateur.
 ça serait idiot de stocker tous les  specgrams.
 """
def batch_generator(X, Y,num_classes,batch_size):

    while True:
        shuffle = np.random.randint(0, X.shape[0], batch_size)
        X_sh = X[shuffle]
        Y_sh = Y[shuffle]

        """expand_dims car les convNet s'attendent à des images à plusieurs canaux"""
        X_specgram = np.expand_dims(paths_to_spectrograms(X_sh), axis=3)
        """ souvent on transforme les variables catégorielles comme ceci: """
        #Y_categorical = to_categorical(Y_sh, num_classes=num_classes)
        """ mais avec keras ce n'est pas nécessaire si l'on utilise comme loss la sparse_categorial_entropy """

        yield X_specgram,Y_sh


# TODO Ce générateur doit être améliorer pour passer tous les sons le même nombre de fois.


def step2():

    X, Y, class_names = filter_data(get_data(), 0)
    X_train, X_test, Y_train, Y_test = train_test_spliting(X, Y)

    batch_size = 4



    count=0
    for Xi,Yi in batch_generator(X_train, Y_train, len(class_names), batch_size=batch_size):

        print("batch numéro:",count)
        print("Xi.shape:",Xi.shape)
        print("Yi.shape:",Yi.shape)
        count+=1

        if (count>3): break




if __name__=="__main__":
    step2()


