# Import packages
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

# Data import
raw_train = pd.read_csv('data/train.csv')
raw_test = pd.read_csv('data/test.csv')

# Transform data in something more 'readable' for ML algorithms
# Add new features based on data provided
train = pd.DataFrame()
train['PassengerId'] = raw_train['PassengerId']
train['Survived'] = raw_train['Survived']
train['Pclass'] = raw_train['Pclass']
train['Sex'] = raw_train['Sex'].apply(lambda x: 0 if x =='female' else 1)
train['NameLen'] = raw_train['Name'].apply(len)
train['Child'] = raw_train['Age'].apply(lambda x:1 if x<18 else 0)
train['HasCabin'] = raw_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
train['Title'] = 0
for index, row in raw_train.iterrows():
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
train['Family'] = 0
train['Family'] = raw_train['SibSp'] + raw_train['Parch'] + 1
train['SibSp'] = raw_train['SibSp']
train['Parch'] = raw_train['Parch']
train['HasFam'] = 0
train['HasFam'][train['Family']>1] = 1
raw_train['Embarked'] = raw_train['Embarked'].fillna(value='S')
train['Embarked'] = raw_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train['Fare'] = raw_train['Fare']
train['CatFare'] = 0
train['CatFare'][train['Fare'] <= 7.91] = 0
train['CatFare'][(train['Fare'] > 7.91) & (train['Fare'] <= 14.454)] = 1
train['CatFare'][(train['Fare'] > 14.454) & (train['Fare'] <= 31)]= 2
train['CatFare'][ train['Fare'] > 31] = 3

# Same for test
test = pd.DataFrame()
test['PassengerId'] = raw_test['PassengerId']
test['Pclass'] = raw_test['Pclass']
test['Sex'] = raw_test['Sex'].apply(lambda x: 0 if x =='female' else 1)
test['NameLen'] = raw_test['Name'].apply(len)
test['Child'] = raw_test['Age'].apply(lambda x:1 if x<18 else 0)
test['HasCabin'] = raw_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Title'] = 0
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
        train['Title'][index] = 5
test['Family'] = 0
test['Family'] = raw_test['SibSp'] + raw_test['Parch'] + 1
test['SibSp'] = raw_test['SibSp']
test['Parch'] = raw_test['Parch']
test['HasFam'] = 0
test['HasFam'][train['Family']>1] = 1
raw_train['Embarked'] = raw_test['Embarked'].fillna(value='S')
test['Embarked'] = raw_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Fare'] = raw_test['Fare']
test['CatFare'] = 0
test['CatFare'][test['Fare'] <= 7.91] = 0
test['CatFare'][(test['Fare'] > 7.91) & (test['Fare'] <= 14.454)] = 1
test['CatFare'][(test['Fare'] > 14.454) & (test['Fare'] <= 31)]= 2
test['CatFare'][test['Fare'] > 31] = 3

# Lets use a Random classifier model
from sklearn.ensemble import RandomForestClassifier

# Select the most importants features
X_train = train[['Pclass', 'Sex', 'Child', 'HasCabin', 'Embarked',  'Title', 'HasFam', 'CatFare', 'Family']]
y_train = np.ravel(train['Survived']) 
X_test = test[['Pclass', 'Sex', 'Child', 'HasCabin', 'Embarked',  'Title', 'HasFam', 'CatFare', 'Family']]

# Fit the model
rfc = RandomForestClassifier(max_features='sqrt', n_estimators=1000, min_samples_leaf=3)
rfc.fit(X_train, y_train)
result = rfc.predict(X_test)

# Save results
sub = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':result})
sub.to_csv('Submission.csv', index=False)