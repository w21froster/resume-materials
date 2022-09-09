# https://www.datacamp.com/community/tutorials/random-forests-classifier-python
import time
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# Load dataset into Pandas dataframe
data = pd.read_csv('datasets/dataset_full.csv')

# Generating Model on our top 10 features
top_10_features = pd.read_csv('top_10_features.csv', index_col=0)
top_10_features = top_10_features['Features'].tolist()
top_10_features.append('phishing')
x = data[top_10_features]

# Split data into features and tags (x = features, y = tags)
y = x.phishing
x = x.drop('phishing',axis=1)


# Split data into testing and training datasets (70% training/30% testing)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

# Create Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

####
# Model Accuracy, how often is the classifier correct?
# print("Accuracy on Test Data (70% training 30% testing):",metrics.accuracy_score(y_test, y_pred))
target_names = ["Not Phishing","Phishing"]
print("---------------MODEL ACCURACY ON TESTING DATA---------------")
print(classification_report(y_test, y_pred, target_names=target_names))

### RESULTS
# Using ALL 110 features, accuracy is 0.9714608009024253
# Using only top 10 features, accuracy is 0.9599172776837751

# Load seperate dataset into Pandas dataframe
new_data = pd.read_csv('datasets/stripped_features/michellkrogza-processed-full.csv', index_col=0)
new_data['phishing'] = 1 # Assign label of 1 to all instances
top_10_features.remove('phishing')
new_pred = clf.predict(new_data[top_10_features])

# Print sklearn classification report on wild data
print("---------------MODEL ACCURACY ON WILD DATA---------------")
print(classification_report(new_data['phishing'], new_pred, target_names=target_names, zero_division=0))
