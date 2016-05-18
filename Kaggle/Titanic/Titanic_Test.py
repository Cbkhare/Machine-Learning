import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcm
import pandas as pd
from sklearn import datasets 
from sklearn.cross_validation import train_test_split as tts, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import Perceptron, LinearRegression,LogisticRegression  
from sklearn.metrics import accuracy_score


if __name__=='__main__':

    titanic = pd.read_csv(r'C:\Users\Anu\Downloads\A Python\Machine Learning\Kaggle\Titanic\test.csv')
    titanic_train = pd.read_csv(r'C:\Users\Anu\Downloads\A Python\Machine Learning\Kaggle\Titanic\train.csv')

    #describing the dataset
    #print(titanic.describe())

    #Balancing Age
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())

    #Removed non-numeric value, converted non-numeric to numeric
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 0
    titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 1
    
    #Filled empty value in embarked with S and then converted them to numeric terms
    titanic["Embarked"] = titanic["Embarked"].fillna('S')
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    titanic_train["Embarked"] = titanic_train["Embarked"].fillna('S')
    titanic_train.loc[titanic_train["Embarked"] == "S", "Embarked"] = 0
    titanic_train.loc[titanic_train["Embarked"] == "C", "Embarked"] = 1
    titanic_train.loc[titanic_train["Embarked"] == "Q", "Embarked"] = 2

    #Balancing Fare
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    titanic_train["Fare"] = titanic_train["Fare"].fillna(titanic_train["Fare"].median())
    
    # Initialize the algorithm class
    alg = LogisticRegression(random_state=1)
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    # Train the algorithm using all the training data
    alg.fit(titanic_train[predictors], titanic_train["Survived"])

    # Make predictions using the test set.
    predictions = alg.predict(titanic[predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "PassengerId": titanic["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv(r'C:\Users\Anu\Downloads\A Python\Machine Learning\Kaggle\Titanic\kaggle.csv', index=False)

    '''
    #Step:-7-8 Applying liner regression and solving overfitting with KFold
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    alg = LinearRegression()    #Intializing Algo
    #Applying cross-validation(returns row indices for given train and test data)
    #set random_state to ensure same splits every time
    kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

    predictions = []

    for train,test in kf:
        train_predictors = (titanic[predictors].iloc[train,:])
        train_predictors = (titanic[predictors].iloc[train,:])
        # The target we're using to train the algorithm.
        train_target = titanic["Survived"].iloc[train]
        # Training the algorithm using the predictors and target.
        alg.fit(train_predictors, train_target)
        # We can now make predictions on the test fold
        test_predictions = alg.predict(titanic[predictors].iloc[test,:])
        predictions.append(test_predictions)

    #Step:-10 Error Metric 
    predictions = np.concatenate(predictions, axis=0)
    # Map predictions to outcomes (only possible outcomes are 1 and 0)
    predictions[predictions > .5] = 1
    predictions[predictions <=.5] = 0
    accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
    #Note (titanic.describe()["Survived"]["count"]) == len(predictions)


    #Step:-11 Logistic Regression
    alg = LogisticRegression(random_state=1)
    # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
    scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
    # Take the mean of the scores (because we have one for each fold)
    print(scores.mean())
    '''
    
















    
    

