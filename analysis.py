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

# Première tentative en knn neighbors
from sklearn.ensemble import RandomForestClassifier

X_train = train[["Pclass", "NameLen", "Sex", "Age", "Cabin", "SibSp", "Parch"]]
y_train = np.ravel(train["Survived"])
X_test = test[["Pclass", "NameLen", "Sex", "Age", "Cabin", "SibSp", "Parch"]]

rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train, y_train)

final_test = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":rfc.predict(X_test)})
final_test.to_csv("Submission.csv", index=False)