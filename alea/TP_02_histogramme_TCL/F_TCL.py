import numpy as np
np.set_printoptions(precision=2,suppress=True)
import matplotlib.pyplot as plt
import scipy.stats as stats

"""
Considérons x_ext[0],x_ext[1],... une suite iid de va. Définissons S[n]=x_ext[0]+...x_ext[n-1].
On peut imaginer que les x_ext[i] sont des erreurs élémentaires, et donc S[n] une erreur globale.
Le théorème centrale limite (TCL) nous indique que, lorsque n grandis, la loi de S[n] "se rapproche" d'une gaussienne.
"""
def step0():
    nbEssaies=3000
    n = 100

    collect=np.zeros(nbEssaies)
    for i in range(nbEssaies):
        ech=np.random.exponential(1,size=n)
        collect[i]=(np.sum(ech)-n)/np.sqrt(n)

    plt.hist(collect,20,rwidth=0.95,normed=True)

    x=np.linspace(-4,4,1000)
    plt.plot(x,stats.norm.pdf(x))

    plt.show()

step0()

"""
la loi de S[n] "se rapproche" d'une gaussienne ... pas très précis cela.
Traduisons mathématiquement :  Quand n tend vers l'infini, la version centrée-réduite de S[n] converge
en loi vers une gaussienne (gaussienne centrée-réduite of course).

Modifiez le step0 pour illustrer cela. Superposez votre histogramme avec la densité de la gaussienne

"""






"""
Quand on a un peu l'habitude, on peut faire le même programme que le précédent sans boucle for ;
c'est plus rapide mais ce n'est pas une obligation.
"""

def step2():
    nbEssaies = 3000
    n=100
    """on tire une matrice ech[i,j] de va gaussiennes"""
    ech=np.random.normal(size=[nbEssaies,n])
    """ axis=1 signifie que l'on somme sur l'indice 1 (j):
    collect[i]=sum_j ech[i,j]  """
    collect=np.sum(ech, axis=1)
    """vérifions qu'on a un vecteur de la shape voulue."""
    print(collect.shape)
    plt.hist(collect, 20, rwidth=0.95)

    plt.show()



