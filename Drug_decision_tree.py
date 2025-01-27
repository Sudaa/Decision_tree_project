## import librabries 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

# import dataset
df = pd.read_csv('drug200.csv')
df.head(5)

#Pre-proocessing 
#declare the vvaribales (X as the feture matrix and y as the target or response vector)
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values


## use LabelEncoder  to convert categotical varibale to numerical variable.
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M']) # defines how to map vlaues  F to 0 , M to 1 
X[:,1] = le_sex.transform(X[:,1]) #selectss al rows from second column

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL', 'HIGH']) #defines how to map LOW to 0, Normal to  1 and high to 2
X[:,2] = le_BP.transform(X[:,2]) #selects all row from third  column

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) #selects all rows from 4th column

##declare target variable
y = df['Drug']

##setting up decision tree (train_test_split)
from sklearn.model_selection import train_test_split

##train_test_split will have 4 different parameters. we will name them
## X_train, X_test , y_train, y_test

## train_test_split will need the parameters
## X, y, test_size=0.3, and random_state = 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

## Modeling 
drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
#next, fit the data wth training feature matrix X_train and training respone vector y_train
drugTree.fit(X_train,y_train)

##prediction on the testing dataset and store in predTree
predTree = drugTree.predict(X_test)

##Print predTree and y_test to visually compare the predictions to actual values
print (predTree[0:5])
print (y_test [0:5])

##Evaluation using metrics froom sklearn to check accuracy of our model
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test,predTree))

print('\nClassification Report:\n', classification_report(y_test, predTree))

# Plot the decision tree
##tree.plot_tree(drugTree)
tree.plot_tree(drugTree, filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], class_names=np.unique(y).astype(str), fontsize=5)

plt.show()
