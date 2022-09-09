# https://www.datacamp.com/community/tutorials/random-forests-classifier-python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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

############ Confusion Matrix for Dataset 1 ############
cf_matrix = confusion_matrix(y_test, y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Random Forest Confusion Matrix on G. Vrbančič et al. Dataset\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Load seperate dataset into Pandas dataframe
new_data = pd.read_csv('datasets/stripped_features/michellkrogza-processed-full.csv', index_col=0)
new_data['phishing'] = 1 # Assign label of 1 to all instances
top_10_features.remove('phishing')
new_pred = clf.predict(new_data[top_10_features])

############ Confusion Matrix for Dataset 2 ############
cf_matrix = confusion_matrix(new_data['phishing'], new_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='BuPu')

ax.set_title('Random Forest Confusion Matrix on M. Krog\'s Dataset\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()