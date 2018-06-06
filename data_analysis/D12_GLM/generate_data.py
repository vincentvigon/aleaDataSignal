
import numpy as np
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=50000)
import scipy.stats #as stats



def createData_gauss(nbData: int):
    x = np.random.random(nbData) * 2
    w = 10
    b = 5

    y = w * x + b + np.random.normal(0, 10, size=[nbData])

    np.savetxt("data/data0_x.csv",x,fmt="%.2f")
    np.savetxt("data/data0_y.csv",y,fmt="%.2f")




def createData_gamma(nbData: int):
    x = np.random.random(nbData) * 2
    w = 10
    b = 5
    ''' y[i]= sum_j x[ij] w[j] '''

    mu= w * x + b

    y=np.zeros(shape=[nbData])

    k=1.
    for i  in range (nbData):
        y[i]=np.random.gamma( shape=k, scale=mu[i]/k, size=1)

    np.savetxt("data/data1_x.csv",x,fmt="%.2f")
    np.savetxt("data/data1_y.csv",y,fmt="%.2f")




def createData_gamma_exp(nbData: int):
    x = np.random.random(nbData) * 2
    w = 2
    b = 3

    mu= np.exp(w * x + b)

    y=np.zeros(shape=[nbData])

    k=1
    for i  in range (nbData):
        y[i]=np.random.gamma( shape=k, scale=mu[i]/k, size=1)

    np.savetxt("data/data2_x.csv",x,fmt="%.2f")
    np.savetxt("data/data2_y.csv",y,fmt="%.2f")


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

"""
Voici un jeu de donnée fabriqué qui colle exactement au modèle logistique. 
Chaque ligne représente une bactérie.     
    #  y :  1-> alive, 0-> dead 
    #  x1 : food
    #  x2 : oxygen
"""

def create_data_bernoulli():

    nb_sample = 1000

    """ paramètres cachés """
    w0 = -15
    w1 = 4
    w2 = 2

    """ descripteurs = variables explicatives """
    x1= np.random.uniform(low=0.0, high=5.0, size=nb_sample)
    x2= np.random.uniform(low=0.0, high=5.0, size=nb_sample)


    mu = sigmoid(w0 + w1 * x1 + w2 * x2)

    y=np.zeros(shape=nb_sample,dtype=np.int64)
    for i in range(nb_sample):
        y[i]=np.random.binomial(n=1,p=mu[i],size=1)

    x=np.array([x1,x2]).T

    np.savetxt("data/bacteria_alone_x.csv",x,fmt="%.2f",header="food, oxygen")
    np.savetxt("data/bacteria_alone_y.csv",y,fmt="%d",header="1-> alive, 0-> dead ")



def create_data_binomiale():

    n_sample = 1000
    b0 = -15
    b1 = 4
    b2 = 2
    x0= np.ones(shape=n_sample)

    x1= np.random.uniform(low=0.0, high=5.0, size=n_sample)
    x2= np.random.uniform(low=0.0, high=5.0, size=n_sample)

    effectif=np.random.randint(low=5, high=11, size=n_sample)


    probs = sigmoid(b0 + b1 * x1 + b2 * x2)

    y1=np.zeros(shape=n_sample,dtype=np.int64)
    for i in range(n_sample):
        y1[i]=np.random.binomial(n=effectif[i],p=probs[i],size=1)

    y2 = effectif - y1
    x=np.array([x1,x2]).T
    y=np.array([y1,y2]).T

    np.savetxt("data/bacteria_grouped_x.csv", x, fmt="%.2f", header="food, oxygen")
    np.savetxt("data/bacteria_grouped_y.csv", y, fmt="%d", header="1-> alive, 0-> dead ")






"""
# y : nombre d'accident
# x : indice fangio du conducteur
"""

def create_data_poisson():
    nb_sample = 1000
    w0 = -4
    w1 = 5

    """ descripteurs = variables explicatives """
    x = np.random.beta(a=0.5,b=1.5, size=nb_sample)

    mu = np.exp(w0 + w1 * x)

    y=np.random.poisson(lam=mu,size=nb_sample)

    #print(x)
    #print(y)


    np.savetxt("data/accident_x.csv", x, fmt="%.2f")
    np.savetxt("data/accident_y.csv", y, fmt="%d")




def create_data_poisson_exposure():
    nb_sample = 1000
    w0 = -4
    w1 = 5

    """ descripteurs = variables explicatives """
    x0 = np.random.beta(a=0.5,b=1.5, size=nb_sample)
    x1 = np.random.randint(30,365*3,size=nb_sample)

    mu = np.exp(w0 + w1 * x0) * x1/365

    y=np.random.poisson(lam=mu,size=nb_sample)

    #print(x)
    #print(y)


    np.savetxt("data/accident_exposure_x.csv", np.stack([x0,x1]).T, fmt="%.2f %d")
    np.savetxt("data/accident_exposure_y.csv", y, fmt="%d")



create_data_poisson_exposure()



