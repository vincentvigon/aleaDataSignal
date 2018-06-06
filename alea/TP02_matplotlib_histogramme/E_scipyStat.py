import numpy as np
np.set_printoptions(precision=2,suppress=True)
import matplotlib.pyplot as plt

"""Voici le troisième et dernier import qu'il faut connaître :"""
import scipy.stats as stats
""" scipy= sci-entific py-thon """



"""
4 mots clefs à retenir (qui permette aussi de s'améliorer en anglais scientifique) :

pdf -> Probability density function.  -> densité (prend des réels en argument)
pmf -> Probability mass function -> densité discrète (prend des entiers en argument)

cdf -> Cumulative density function.   -> fonction de répartition
ppf -> Percent point function (inverse of cdf ) ->  fonction quantile (ou percentile)

rvs -> Random variates. -> simulation d'un échantillon de va ayant la loi donnée


A ce point du TP, vous vous dites qu'il y a vraiment trop de choses à retenir. Mais nous
allons les pratiquez très souvent : cela rentrera tout seul.
"""




def step0():
    """"""

    """simulons un échantillon"""
    simus=stats.norm.rvs(loc=0, scale=1, size=1000)
    """
    c'est tout à fait identique à :

    simus=np.random.normal(loc=0,scale=1,size=1000)

    cependant scipy.stats contient encore plus de loi que numpy.random.
    On aura notamment besoin de la loi t de student pour les stats.
    """

    x=np.linspace(-3,3,100)
    pdf=stats.norm.pdf(x, loc=0, scale=1)
    cdf=stats.norm.cdf(x, loc=0, scale=1)
    ppf=stats.norm.ppf(x, loc=0, scale=1)

    plt.subplot(1,2,1)
    plt.hist(simus,20,normed=True,label="simus")
    plt.plot(x,pdf,label="pdf")
    plt.plot(x,cdf,label="cdf")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x, cdf,label="cdf")
    plt.plot(x, ppf,label="ppf")
    plt.plot(x, x)
    plt.legend()


    plt.show()

"""
Dans l'appelle:
simus=stat.norm.rvs(loc=0,scale=1,size=1000)

Tous les arguments sont facultatifs.
Les valeurs par défaut sont logiquement loc=0,scale=1,size=1


On peut écrire par exemple =

     simus=stat.norm.rvs(size=1000)
pour
     simus=stat.norm.rvs(loc=0,scale=1,size=1000)


par contre si on écrit :
     simus=stat.norm.rvs(1000)
cela donne
     simus=stat.norm.rvs(loc=1000)
ce qui est sans doute un bug.

Je vous conseille d'écrire le plus souvent le nom des arguments pour éviter ce genre de confusion.


"""

"""
Dans le step0 : Le tracé de la ppf n'est pas très joli, on a l'impression qu'il est incomplet.
Changez cela. Si vous n'avez pas d'idée : la solution est donnée tout en bas de ce fichier.
"""




""" avec une loi discrète """
def step1():
    """"""

    """simulons un échantillon"""
    n=10
    p=0.5
    simus=stats.binom.rvs(n, p, size=1000)


    x=np.arange(0,n+1)
    bins=np.arange(0,n+2)-0.5
    pdf=stats.binom.pmf(x, n, p)
    cdf=stats.binom.cdf(x, n, p)


    plt.hist(simus, bins, normed=True, label="simus")
    plt.plot(x, pdf,'o' , label="pdf")
    plt.plot(x, cdf,'o', label="cdf")
    plt.title("loi binomiale")
    plt.legend()


    plt.show()











step0()

































































"""
Solution : pourquoi le tracé du ppf est incomplet:
car on n'a pas mis assez de points : changez
x=np.linspace(-3,3,100)
en
x=np.linspace(-3,3,1000)

"""
