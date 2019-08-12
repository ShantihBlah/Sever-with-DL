import pandas as pd
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
import seaborn as sns 
# %matplotlib inline

df = pd.read_csv('movie_metadata.csv')
print(df.head())

str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in df.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = df.columns.difference(str_list)

df_num = df[num_list]
#del movie # Get rid of movie df as we won't need it now
print(df_num.head())

df_num = df_num.fillna(value=0, axis=1)

X = df_num.values
# Data Normalization
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

df.plot(y= 'imdb_score', x ='duration',kind='hexbin',
        gridsize=45, sharex=False, colormap='cubehelix', 
        title='Hexbin of Imdb_Score and Duration')
df.plot(y= 'imdb_score', x ='gross',kind='hexbin',
        gridsize=45, sharex=False, colormap='cubehelix', 
        title='Hexbin of Imdb_Score and Gross')
df.plot(y= 'imdb_score', x ='budget',kind='hexbin',
        gridsize=35, sharex=False, colormap='cubehelix', 
        title='Hexbin of Imdb_Score and Budget')
df.plot(y= 'gross', x ='duration',kind='hexbin',
        gridsize=35, sharex=False, colormap='cubehelix', 
        title='Hexbin of Gross and Duration')

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
plt.title('Pearson Correlation of Movie Features')
# Draw the heatmap using seaborn
sns.heatmap(df_num.astype(float).corr(),linewidths=0.4,vmax=1.0, 
            square=True, cmap="YlGnBu", linecolor='black')

plt.show()

# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)
# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
plt.figure(figsize=(8, 5))
plt.bar(range(16), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
plt.step(range(16), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=9)
x_9d = pca.fit_transform(X_std)

from sklearn.cluster import KMeans

# Set a 3 KMeans clustering
kmeans = KMeans(n_clusters=3)
# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(x_9d)

# Define our own color map
LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# Plot the scatter digram
plt.figure(figsize = (7,7))
plt.scatter(x_9d[:,0],x_9d[:,2], c= label_color, alpha=0.5) 
plt.show()
