# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Data import
raw_data = pd.read_csv('data/train.csv')
raw_test = pd.read_csv('data/test.csv')

# Transform data in something more 'readable' for ML algorithms
train = pd.DataFrame()
train['PassengerId'] = raw_data['PassengerId']
train['Pclass'] = raw_data['Pclass']
train['NameLen'] = raw_data['Name'].apply(len)
train['Sex'] = raw_data['Sex'].apply(lambda x: 0 if x =='female' else 1)
train['Age'] = raw_data['Age'].apply(lambda x: 0 if math.isnan(x) else int(x))
train['Cabin'] = raw_data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
train['SibSp'] = raw_data['SibSp']
train['Parch'] = raw_data['Parch']
raw_data['Fare'] = raw_data['Fare'].fillna(value=raw_data['Fare'].median())
train['Fare'] = raw_data['Fare'].round().apply(int)
raw_data['Embarked'] = raw_data['Embarked'].fillna(value='S')
train['Embarked'] = raw_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train['Family'] = 0
for index, row in raw_data.iterrows():
    train['Family'][index] = row['SibSp'] + row['Parch'] + 1
train['Title'] = 0
for index, row in raw_data.iterrows():
    if 'Mr' in row['Name']:
        train['Title'][index] = 1
    elif 'Miss' in row['Name']:
        train['Title'][index] = 2
    elif 'Mrs' in row['Name']:
        train['Title'][index] = 3
    elif 'Master' in row['Name']:
        train['Title'][index] = 4
    else:
        train['Title'][index] = 5
train['Survived'] = raw_data['Survived']


# Same thing on test sample
test = pd.DataFrame()
test['PassengerId'] = raw_test['PassengerId']
test['Pclass'] = raw_test['Pclass']
test['NameLen'] = raw_test['Name'].apply(len)
test['Sex'] = raw_test['Sex'].apply(lambda x: 0 if x =='female' else 1)
test['Age'] = raw_test['Age'].apply(lambda x: 0 if math.isnan(x) else int(x))
test['Cabin'] = raw_test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['SibSp'] = raw_test['SibSp']
test['Parch'] = raw_test['Parch']
raw_test['Fare'] = raw_test['Fare'].fillna(value=raw_test['Fare'].median())
test['Fare'] = raw_test['Fare'].round().apply(int)
raw_test['Embarked'] = raw_test['Embarked'].fillna(value='S')
test['Embarked'] = raw_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Title'] = 0
test['Family'] = 0
for index, row in raw_test.iterrows():
    test['Family'][index] = row['SibSp'] + row['Parch'] + 1
for index, row in raw_test.iterrows():
    if 'Mr' in row['Name']:
        test['Title'][index] = 1
    elif 'Miss' in row['Name']:
        test['Title'][index] = 2
    elif 'Mrs' in row['Name']:
        test['Title'][index] = 3
    elif 'Master' in row['Name']:
        test['Title'][index] = 4
    else:
        test['Title'][index] = 5


# Lets combine the results of all the classifiers models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Gaussian Process',
         'Decision Tree', 'Random Forest', 'Neural Net', 'AdaBoost',
         'Naive Bayes', 'QDA'
        ]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel='linear'),
    SVC(kernel='rbf'),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=1000),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

# Select the most importants features
X_train = train[['Pclass', 'NameLen', 'Sex', 'Cabin', 'Embarked', 'Family', 'Title']]
y_train = np.ravel(train['Survived'])
X_test = test[['Pclass', 'NameLen', 'Sex', 'Cabin', 'Embarked', 'Family', 'Title']]

# Iterate over classifiers
result = pd.DataFrame()
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    result[name]= clf.predict(X_test)

sub = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':result.mean(axis=1).round().apply(int)})
sub.to_csv('Submission.csv', index=False)