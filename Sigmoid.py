import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcm
import pandas as pd
from sklearn import datasets 
from sklearn.cross_validation import train_test_split as tts 
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

if __name__=='__main__':
    z = np.arange(-7,7,0.1)
    phi_z = sigmoid(z)
    plt.plot(z,phi_z)
    plt.axvline(0.0,color='k')
    plt.axhspan(0.0,1.0,facecolor ='1.0',alpha=1.0,ls='dotted')
    plt.axhline(y=0.5,ls='dotted',color='k')
    plt.yticks([0.0,0.5,1.0])
    plt.ylim(-0.1,1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()
    

'''
