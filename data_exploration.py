# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Data import
raw_data = pd.read_csv('data/train.csv')

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
# Transform data in something more "readable" for ML algorithms
train = pd.DataFrame()
train["Pclass"] = raw_data["Pclass"]
train["NameLen"] = raw_data["Name"].apply(len)
train["Sex"] = raw_data["Sex"].apply(lambda x: 0 if x =="female" else 1)
train["Age"] = raw_data["Age"].apply(lambda x: 0 if math.isnan(x) else int(x))
train["Cabin"] = raw_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
train["SibSp"] = raw_data["SibSp"]
train["Parch"] = raw_data["Parch"]
train["Survived"] = raw_data["Survived"]
train.index = raw_data["PassengerId"]

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

