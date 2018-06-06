import numpy as np
np.set_printoptions(linewidth=500,precision=2,suppress=True)
import matplotlib.pyplot as plt



""" histogramme: """
def step0():
    X=np.random.normal(0,1,size=1000)
    """
    x_ext : l'échantillon à observer
    bins=10 : on découpe l'intervalle [min(x_ext),max(x_ext)] en dix sous-intervalles.
    normed=True: la hauteur des batons est normalisée pour que cela ressemble à une densité
    rwidth=0.9: la largeur de chaque baton occupe 90% de chaque sous-intervalle.
    """
    plt.hist(X,10,color='blue',rwidth=0.9,normed=True)
    plt.show()



""" Mais parfois il est préférable de préciser nous même les sous-intervalles (=la base des batons) """
def step1():
    X=np.random.normal(0,1,size=1000)
    plt.hist(X, [-2,0.5,1,5],   rwidth=0.9,normed=True)
    plt.show()




""" Attention, pour les lois discrètes il faut obligatoirement préciser le découpage.
Pour voir une catastrophe, remplacez bins par 11 dans plt.hist(). Expliquez le phénomène."""
def step2():
    n=10
    X=np.random.binomial(n,0.5,size=3000)

    """attention np.arange(0,n+2,1) donne l'intervalle discret [0,n+2[= [0,n+1].
     on lui soustrait ensuite 0.5 pour avoir chaque entier de [0,n] dans un sous-intervalle"""
    bins=np.arange(0,n+2,1)-0.5

    plt.hist(X,bins=bins, histtype='bar', color='blue', rwidth=0.6)
    """on précise les graduations en x"""
    plt.xticks(np.arange(0,n+1,1))
    plt.show()



"""comparons des lois betas"""
def step3():
    nbData=10000
    X1=np.random.beta(3,1,size=nbData)
    X2=np.random.beta(2,3,size=nbData)
    X3=np.random.beta(1,0.5,size=nbData)


    plt.hist([X1,X2,X3],bins=30,label=["a=3,b=1","a=2,b=3","a=1,b=0.5"])

    plt.legend()
    plt.show()

step3()






""" 

La variété des formes possible d'une loi la rend très pratique en modélisation. 

Choisissez des lois bêta bien choisies (dilatée par une constante), pour modéliser les variables x_ext suivantes:
  x_ext : quantité chocolat consommée par les français  (sachant que plus on en mange, et plus on a envie d'en manger)
  x_ext : durée de vie des français 
  x_ext : durée de vie des grenouilles (forte mortalité infantile)
Dressez les histogrammes


Connaissez-vous d'autre loi pour des durées de vie ?
"""












