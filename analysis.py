# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Data import
raw_data = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")

# Transform data in something more "readable" for ML algorithms
train = pd.DataFrame()
ytrain = pd.DataFrame()
train["PassengerId"] = raw_data["PassengerId"]
train["Pclass"] = raw_data["Pclass"]
train["NameLen"] = raw_data["Name"].apply(len)
train["Sex"] = raw_data["Sex"].apply(lambda x: 0 if x =="female" else 1)
train["Age"] = raw_data["Age"].apply(lambda x: 0 if math.isnan(x) else int(x))
train["Cabin"] = raw_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
ytrain["Survived"] = raw_data["Survived"]

# Same thing on test sample
test = pd.DataFrame()
ytest = pd.DataFrame()
test["PassengerId"] = raw_test["PassengerId"]
test["Pclass"] = raw_test["Pclass"]
test["NameLen"] = raw_test["Name"].apply(len)
test["Sex"] = raw_test["Sex"].apply(lambda x: 0 if x =="female" else 1)
test["Age"] = raw_test["Age"].apply(lambda x: 0 if math.isnan(x) else int(x))
test["Cabin"] = raw_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Première tentative en knn neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train, ytrain)

final_test = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":knn.predict(test)})
final_test.to_csv("Submission.csv")
print(knn.predict(test))
