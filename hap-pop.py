import seaborn as sns
from sklearn.neural_network import MLPClassifier
from subprocess import call
from sklearn.tree import export_graphviz
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import pickle
import statsmodels.api as sm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC  # "Support vector classifier"
import eli5  # for purmutation importance
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split

#import the necessary libraries for data manipulation, modeling, and visualization

def get(country):
    df = pd.read_csv("input.csv")
    df = df.loc[df['Entity'] == country]
    X = np.array(df.drop('Population', 1))
    y = np.array(df['Population'])
    df.columns = ['Entity', 'Year','Hapiness','Population', 'Pop Rate of change', 'Suicide Rates', 'Income inequality', 'Food per person']
    df = df.drop('Year',1)
    df = df.drop('Population',1)
    return(df)

#define a function to pull a single country from the dataset into a pandas dataframe

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
df = pd.read_csv("input.csv")
print(df.Entity.unique())

#Read data into dataframe and print available country names

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

#print training and testing shapes

rf_class = RandomForestClassifier(n_estimators=5)
log_class = LogisticRegression()
svm_class = SVC(kernel='rbf', C=1E11, verbose=False)

#Assign 3 models (random forest classifier, support vector machine, and logistic regression classifier) to variables

def run(model, model_name='this model', trainX=trainX, trainY=trainY, testX=testX, testY=testY):
    # print(cross_val_score(model, trainX, trainY, scoring='accuracy', cv=10))
    accuracy = cross_val_score(model, trainX, trainY,
                               scoring='accuracy', cv=2).mean() * 100
    model.fit(trainX, trainY)
    testAccuracy = model.score(testX, testY)
    print("Training accuracy of "+model_name+" is: ", accuracy)
    print("Testing accuracy of "+model_name+" is: ", testAccuracy*100)
    print('\n')

#create function to run and assess models

def heatmap(df):
  corr = df.corr()
  sns.heatmap(corr,
              xticklabels=corr.columns.values,
              yticklabels=corr.columns.values)

#create function to create a correlation matrix for a given dataset

df =  get('Italy')

heatmap(df)

#create and display correlation matrix


#OPTIONALLY, THE CODE BELOW CAN BE UNCOMMENTED TO USE ADVANCED MODELING TO SHOW COMPLEX CORRELATIONS


# model = log_class
# model.fit(trainX,trainY)

# perm = PermutationImportance(model, random_state=1).fit(testX, testY)
# eli5.show_weights(perm, feature_names=feature_names)
