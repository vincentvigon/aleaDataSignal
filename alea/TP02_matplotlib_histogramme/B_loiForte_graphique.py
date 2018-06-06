import numpy as np
import matplotlib.pyplot as plt




"""
"""

def step0():

    """"""
    """ np.arange c'est comme range() sauf qu'on peut préciser le pas."""
    ns=np.arange(0,10000,40)
    means=[]
    for n in ns:
        simus=np.random.random(size=n)
        means.append(np.mean(simus))

    plt.plot(ns,means)
    plt.show()

"""
Mais j'ai volontairement fait une erreur dans ce programme, une erreur que font beaucoup d'étudiant chaque année : 
Écrivez précisément la loi des grands nombre, comparez cet énoncé avec le programme ci-dessus.
Trouvez le bug mathématique.
"""














