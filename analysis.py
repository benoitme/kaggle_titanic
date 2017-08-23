# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Data import
raw_data = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")

# Transform data in something more "readable" for ML algorithms
train = pd.DataFrame()
train["PassengerId"] = raw_data["PassengerId"]
train["Pclass"] = raw_data["Pclass"]
train["NameLen"] = raw_data["Name"].apply(len)
train["Sex"] = raw_data["Sex"].apply(lambda x: 0 if x =="female" else 1)
train["Age"] = raw_data["Age"].apply(lambda x: 0 if math.isnan(x) else int(x))
train["Cabin"] = raw_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
train["SibSp"] = raw_data["SibSp"]
train["Parch"] = raw_data["Parch"]
train["Survived"] = raw_data["Survived"]

# Same thing on test sample
test = pd.DataFrame()
test["PassengerId"] = raw_test["PassengerId"]
test["Pclass"] = raw_test["Pclass"]
test["NameLen"] = raw_test["Name"].apply(len)
test["Sex"] = raw_test["Sex"].apply(lambda x: 0 if x =="female" else 1)
test["Age"] = raw_test["Age"].apply(lambda x: 0 if math.isnan(x) else int(x))
test["Cabin"] = raw_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test["SibSp"] = raw_data["SibSp"]
test["Parch"] = raw_data["Parch"]

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

X_train = train[["Pclass", "NameLen", "Sex", "Age", "Cabin", "SibSp", "Parch"]]
y_train = np.ravel(train["Survived"])
X_test = test[["Pclass", "NameLen", "Sex", "Age", "Cabin", "SibSp", "Parch"]]

# Iterate over classifiers
result = pd.DataFrame()
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    result[name]= clf.predict(X_test)

sub = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":result.mean(axis=1).round().apply(int)})
sub.to_csv("Submission.csv", index=False)