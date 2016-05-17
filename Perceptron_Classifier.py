import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcm
import pandas as pd
from sklearn import datasets 
from sklearn.cross_validation import train_test_split as tts 
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def plot_decision_regions(X,y,classifier,
                          test_idx =None, resolution=0.02):

    #setting up marker generator and colormap
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    #unique to get the different class labels stored in y 
    cmap = lcm (colors[:len(np.unique(y))])

    #ploting the decision surface
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max() +1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max() +1

    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,aplha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())


    #plot all samples
    x_test,y_test = X[test_idx,:],y[test_idx]
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx],label=cl)

    #highlight test samples

    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='',
                    alpha=1.0,linewidth=1,marker='o',
                    s=55,label='test set')

    

if __name__=='__main__':

    iris = datasets.load_iris()
    X = iris.data[:,[2,3]]
    y = iris.target
    
    #spliting the data for test(30%) and training(70%) using tts 
    X_train,X_test,y_train, y_test = \
            tts(X,y,test_size=0.3, random_state=0)    


    #Standardising the feature (feature scaling) using ss 
    sc =ss()
    #Using fit to estimate 'sample mean','standard deviation' to do feature scaling 
    #for each feature dimension using training data 
    sc.fit(X_train)
    #tranform is used to standardize the trainig data (TrDS) and test data(TsDS)
    #Note: we have used same parameter for feature scaling 
    X_train_std = sc.transform(X_train)
    X_test_std  = sc.transform(X_test)


    #n_iter:-  Number of Epochs(passes over the TrDS set)
    #eta0/eta:-learning rate
    #reproducibility of initial shuffling of TrDS after each epoch  
    ppn = Perceptron(n_iter=40,eta0=0.1, random_state=0)
    #training using fit 
    ppn.fit(X_train_std,y_train)

    #predicting 
    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d'%(y_test !=y_pred).sum())
    
    '''
    OUTPUT
    >>>  
    Misclassified samples: 4
    i.e 4 out of 89, 4/89=8.9% misclassification error
    Also,
    Misclassification is coversely used with respect to Accuracy
    thus accuracy is 91.1% 
    '''

    print('Accuracy: %.2f' % accuracy_score(y_test,y_pred))
    '''Accuracy: 0.91'''


    #Plotting decision regions and visualize seperate flower models
    X_combined_std = np.vstack((X_train_std,X_test_std))
    y_combined = np.hstack((y_train,y_test))

    plot_decision_regions(X=X_combined_std,
                          y=y_combined,
                          classifier=ppn,
                          test_idx=range(105,150))

    plt.xlabel('petal length[standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    

'''
Perceptron for classification.
Dataset is Irsis and flowers are Iris-Setosa, Iris-Versicolor, Iris-Virginica.

Note:- Although a perceptron algorithm is easy for ML but it is not recommended
       to use for a classification because if data is not perfectly linearnly
       seperable, algorithm will never converge. Even if no. of epocs are increased
       or learning rate is changed.

       So it is better to use Logistic Regression.
'''    
       






