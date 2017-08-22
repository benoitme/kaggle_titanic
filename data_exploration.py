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
data = pd.DataFrame()
data['PassengerId'] = raw_data['PassengerId']
data['Survived'] = raw_data['Survived']
data['Pclass'] = raw_data['Pclass']
data['NameLen'] = raw_data['Name'].apply(len)
data['Sex'] = raw_data['Sex'].apply(lambda x: 0 if x =='female' else 1)
data['Age'] = raw_data['Age'].apply(lambda x: 0 if math.isnan(x) else int(x))
data['Cabin'] = raw_data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
data['Embarked'] = raw_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Pairplot
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

