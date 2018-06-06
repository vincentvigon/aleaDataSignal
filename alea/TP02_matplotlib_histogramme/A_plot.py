import numpy as np
np.set_printoptions(precision=2,suppress=True)

"""voici le second import à connaître par coeur :
 il permet de faire des graphiques. """
import matplotlib.pyplot as plt



""" observez : le graphe d'une fonction n'est qu'une ligne brisée """
def step0():
    nbPoint=10
    """des points régulièrement répartis entre 0 et 2 pi"""
    x=np.linspace(0,2*np.pi,nbPoint)
    y=np.sin(x)
    print("x",x)
    print("y",y)

    plt.plot(x,y)
    """idem, mais sans relier les points"""
    plt.plot(x,y,'o')


    """ a ne pas oublier: cela ouvre la fenêtre graphique """
    plt.show()
    """     ATTENTION : l'ouverture de la fenêtre graphique bloque l'excécussion du programme.
    La ligne suivante n'est pas exécutée:"""
    print("toto")

"""
Modifiez le programme précédent : augmenter le nombre de point pour donner une meilleurs représentation du
graphe d'une fonction.
"""



def step1():
    nbPoints=10
    x=np.linspace(0,2*np.pi,nbPoints)
    y=np.sin(x)
    z=np.cos(x)

    """ on fait une grille 2*3 de fenêtres. """
    plt.subplot(2,3,1)   #on remplit la fenêtre 1
    plt.plot(x,y)
    plt.subplot(2, 3, 2) #on remplit la fenêtre 2
    plt.plot(x, y,"o-")
    plt.subplot(2, 3, 3) # etc.
    plt.plot(x, y,".",color="red")

    """superposons"""
    plt.subplot(2,3,4)
    plt.plot(x,y,"o")
    plt.plot(x,z,color="blue")

    """ on comprend ici l'intérêt du show() : on ouvre la fenêtre uniquement quand tout est fini. """
    plt.show()



def step2():

    nbPoints=100
    x=np.linspace(0,5,nbPoints)

    """attention 'lambda' est un mot clef réservé de python. du coup j'utilise lamb"""
    for lamb in [0.1,0.3,1,2]:
        y=np.exp(lamb*x)
        plt.plot(x,y,label="lambda:"+str(lamb))

    plt.title("comparaison d'exponentielles")

    """on limite les ordonnées. Pourquoi est-ce important?"""
    plt.ylim([0,5])

    plt.legend()
    plt.show()



"""
EXO : dresser le graphe de la fonction x-> 1/x  sur l'intervalle [-1,1]
Aide : superposez deux graphes.
"""