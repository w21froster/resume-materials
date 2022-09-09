from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load dataset into Pandas dataframe
data = pd.read_csv('datasets/dataset_full.csv')

# Strip dataset down to 10 features (that we found with MDI feature importance)
top_10_features = pd.read_csv('top_10_features.csv', index_col=0)
feat_cols = pd.read_csv('top_10_features.csv', index_col=0)
top_10_features = top_10_features['Features'].tolist()
feat_cols = feat_cols['Features'].tolist()
top_10_features.append('phishing')
df = data[top_10_features]
df = df.rename(columns= {'phishing':'y'})
df['label'] = df['y'].apply(lambda i: str(i))

X,y = None,None

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

N = 50000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
y_labels = df_subset['y'].astype(bool)
y_labels.name = "Phishing Link"

plt.figure(figsize=(16,10))
scatter_plot = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue=y_labels,
    data=df_subset,
    palette={False:"#9b59b6", True:"#3498db"},
    legend="full",
    alpha=0.3
)
scatter_plot.set(title="t-SNE Visualization - 2 Components", xlabel=None, ylabel=None)
plt.show()