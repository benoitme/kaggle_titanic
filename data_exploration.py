# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Data import
raw_data = pd.read_csv('data/train.csv')
raw_test = pd.read_csv("data/test.csv")

# Let's start by looking at the data
print(raw_data.head())
print('---------------------------------------------')
print(raw_data.info())
print('---------------------------------------------')
print(raw_data.describe())
print('---------------------------------------------')

# General analysis : number of Nan in the data
print(raw_data.isnull().sum())

# Visual data analysis
# Let's display plots of the different characteristics
# Transform data in something more 'readable' for ML algorithms
train = pd.DataFrame()
train["Pclass"] = raw_data["Pclass"]
train["NameLen"] = raw_data["Name"].apply(len)
train["Sex"] = raw_data["Sex"].apply(lambda x: 0 if x =="female" else 1)
train["Age"] = raw_data["Age"].apply(lambda x: 0 if math.isnan(x) else int(x))
train["Cabin"] = raw_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
train["SibSp"] = raw_data["SibSp"]
train["Parch"] = raw_data["Parch"]
train['Fare'] = raw_data['Fare']
raw_data['Embarked'] = raw_data['Embarked'].fillna(value='S')
train["Embarked"] = raw_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train["Survived"] = raw_data["Survived"]
train.index = raw_data["PassengerId"]

# Same thing on test sample
test = pd.DataFrame()
test["PassengerId"] = raw_test["PassengerId"]
test["Pclass"] = raw_test["Pclass"]
test["NameLen"] = raw_test["Name"].apply(len)
test["Sex"] = raw_test["Sex"].apply(lambda x: 0 if x =="female" else 1)
test["Age"] = raw_test["Age"].apply(lambda x: 0 if math.isnan(x) else int(x))
test["Cabin"] = raw_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test["SibSp"] = raw_test["SibSp"]
test["Parch"] = raw_test["Parch"]
test['Fare'] = raw_test['Fare']
raw_test['Embarked'] = raw_test['Embarked'].fillna(value='S')
train["Embarked"] = raw_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# Lets plot the different features to see the most significant ones
# With the SEM we can see which features are the most significant
# Age does not seem to be that significant for instance
# Of course "Survived" is the most significant ;)
i = 0
f, ax = plt.subplots(8, 1, figsize=(4,12))
f.subplots_adjust(hspace=1.5)
for column in train.columns:
    ax[i].set_title(column)
    ax[i].bar(train.groupby(column).Survived.mean().index,train.groupby(column).Survived.mean(), yerr=train[column].sem())
    i+=1
plt.show()


# Pairplot
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

# Let's do a cross validation for aaaaalll the models
from sklearn.model_selection import train_test_split
X = train[["Pclass", "NameLen", "Sex", "Age", "Cabin", "SibSp", "Parch"]]
y = np.ravel(train["Survived"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# All the classifiers models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"
        ]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=1000),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

from sklearn.model_selection import cross_val_score

# Iterate over classifiers
results = {}
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    results[name] = scores
    
for name, scores in results.items():
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))