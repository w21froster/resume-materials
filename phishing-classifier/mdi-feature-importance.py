# Process described in https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# Split data into features and tags (x = features, y = tags)
data = pd.read_csv('datasets/dataset_small.csv')
y = data.phishing
x = data.drop('phishing',axis=1)

# Split data into testing and training datasets (80% training/20% testing)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# model fitting and feature selection
feature_names = x.columns.tolist()
forest = RandomForestClassifier(random_state=0)
forest.fit(x_train, y_train)

# feature importance based on mean decrease in inpurity
start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

# plot impurity-based importance
forest_importances = pd.Series(importances, index=feature_names).nlargest(20)

fig = forest_importances.plot(kind='bar')
fig.set_title("Feature importances using MDI")
fig.set_ylabel("Mean decrease in impurity")
plt.tight_layout()
plt.show()

# Export top 20 features to csv (we needed more than the 10 we wanted because 3 of the main features relied on an API)
top_20_features = pd.DataFrame(forest_importances.axes[0].tolist())
top_20_features.to_csv("top_20_features.csv")
