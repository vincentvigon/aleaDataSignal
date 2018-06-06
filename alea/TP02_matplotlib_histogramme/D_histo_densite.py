import numpy as np
np.set_printoptions(precision=2,suppress=True)
import matplotlib.pyplot as plt


"""
NB: Par abus de langage, nous dirons :
"dressez l'histrogramme de la loi machin"
pour dire
"dresser l'histogramme d'un échantillon de variables aléatoires ayant la loi machin
"""


"""superposons l'histogramme d'un échantillon gaussien avec la densité de la loi gaussienne"""
def step0():
    nbSimu=1000
    Simu=np.random.normal(size=nbSimu)

    """formule à emmener partout avec soi"""
    def density(x):
        return 1/(np.sqrt(2*np.pi))*np.exp(-0.5*x**2)

    bins=np.linspace(-3,3,10)
    plt.hist(Simu,bins,rwidth=0.9)

    x=np.linspace(-3,3,200)
    plt.plot(x,density(x))

    plt.show()

""" heue : l'histogramme et la densité ne se superpose pas. Qu'ai-je oublié ?"""







"""
IMPORTANT 
Il faut bien choisir ses bins. 
Exemple de mauvais choix:
bins=np.linspace(-30,30,10) -> trop large. 
bins=np.linspace(-3,3,200)  -> beaucoup trop de baton par rapport au nombre de simu (1000) -> l'histogramme est irrégulier
bins=np.linspace(-3,3,3)    -> pas assez de batons. Cela reflète mal la distribution des données.


A vous : modifier la seconde ligne en
    Simu=np.random.normal(loc=2,scale=3,size=nbSimu)

Changer le bins pour avoir toujours un bel histogramme. Et surtout changer la densité. 
AIDE pour la densité :

Proposition : si f(x) est la densité d'une va x_ext, alors
        1/sigma  f( (x-mu) / sigma )
est la densité de la va   sigma x_ext + mu


Cette astuce, vous permet de vérifiez que vous ne vous tromper pas dans le ou les paramètres des lois :  
Loi exponentielle : 
    exp(-x)  ->    1/sigma exp(-x/sigma)       [version anglaise: le paramètre est le paramètre d'échelle (=cst* écart-type) ]
             ->    lambda exp(-lambda x)       [version française: le paramètre est un taux (l'inverse d'un paramètre d'échelle)]
Cauchy :
    1/pi * 1/(1+x^2) -> ...
Normale : 
    ...   


Démo de la proposition:
Considérons phi fonction teste et x_ext une va de densité f.
        E[phi( sigma x_ext + mu )] =   int phi( sigma x + mu) f(x) dx 
on fait le changement de variable   sigma x + mu = y 
                               =   int phi( y) f( (x-mu)/sigma) dy/sigma 
On en déduit que la densité de sigma x_ext + mu est ... 

Il est important de savoir faire ce genre de démonstration rapidement. 
Ci-dessous, je vous propose une autre démo sur le même thème.
"""





"""
Observez. 
Que représente une distribution log-normale ?
Réponse : c'est la distribution de f(x_ext) avec f ... et x_ext ...
Vérifiez cela en superposant deux histogrammes. 
"""
def step1():
    size=1000
    ech=np.random.normal(size=size)
    exp_ech=np.exp(ech)

    ech2=np.random.lognormal(size=size)

    bins=np.linspace(0,10,20)
    plt.hist([exp_ech,ech2],bins=bins,normed=True)
    plt.show()




"""
EXO théorique (à la maison). 
Complétez le calcul de la densité de la log-normale :

Considérons phi fonction teste et x_ext de loi normale.
        E[phi( exp(x_ext) )] =  cst  int phi( exp(x) ) e^{-1/2* x^2 } dx 
on fait le changement de variable exp(x) = y ...
   ...
donc la densité de exp(x_ext) est ...


superposez cette densité avec les histogramme pour vérifiez votre calcul.

"""



















