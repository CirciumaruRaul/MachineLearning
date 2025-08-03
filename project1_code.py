import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from minisom import MiniSom
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, adjusted_rand_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import random

train = "data/exoTrain.csv"

'''
Getting general info about the dataset
'''
data = pd.read_csv(train)
print(data["FLUX.1"].describe())
print(data.info())
print(data.dtypes)
print(data["LABEL"].unique())


'''
  CLEAN THE DATA IF NECESSARY
'''
result = "Yes" if data.isnull().any().any() else "No"
print("Is there any null/NaN values? ", result)
new_data = data.drop_duplicates()
if len(new_data) - len(data) == 0:
    result = "No duplicates found. \n"
else:
    result = "Number of duplicates droped: " + len(new_data) - len(data)
    data = data.drop_duplicates()
print(result)

# plot the distribution of classes
plt.hist(data["LABEL"], color="red")


'''
  SEPARATING LABES FROM FEATURES
'''
X_train, y_train = data.drop(["LABEL"], axis=1), data["LABEL"]

print(X_train.info())
print(y_train.info())

'''
  RESAMPLING TO BALANCE THE CLASSES
'''
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

plt.hist(y_resampled, color='magenta')


'''
  PLOTTING INITIAL DISTRIBUTION OF CLASSES USING PCA
'''
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)

# Scatter plot for true labels
plt.figure(figsize=(8, 6))
for label in np.unique(y_resampled):
    plt.scatter(
        X_reduced[y_resampled == label, 0], 
        X_reduced[y_resampled == label, 1], 
        label=f"Class {label}", alpha=0.7
    )

plt.title("Dataset Visualization with True Labels using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()


'''
  PLOTTING INITIAL DISTRIBUTION OF CLASSES USING t-SNE
'''
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_reduced_tsne = tsne.fit_transform(X_train)

# Scatter plot with color-coded true labels
plt.figure(figsize=(8, 6))
for label in np.unique(y_train):
    plt.scatter(
        X_reduced_tsne[y_train == label, 0], 
        X_reduced_tsne[y_train == label, 1], 
        label=f"Class {label}", alpha=0.7
    )

plt.title("Dataset Visualization with t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.show()

'''
  SCALING EVERYTHING TO IMPROVE TRAINING TIME
'''
minMaxScaler = MinMaxScaler()      # works better with SOM 
standardScaler = StandardScaler()  # works better with mean-shift

df_SOM = minMaxScaler.fit_transform(X_resampled)
df_MeanShift = standardScaler.fit_transform(X_resampled)

print(df_SOM.shape)
print(df_MeanShift.shape)

'''
  EXTRACTING FEATURES USING IncrementalPCA
'''
n_components_SOM = 100  
batch_size_SOM = 404    
ipca_SOM = IncrementalPCA(n_components=n_components_SOM)

for i in range(0, df_SOM.shape[0], batch_size_SOM):
    X_batch_SOM = df_SOM[i:i + batch_size_SOM]
    ipca_SOM.partial_fit(X_batch_SOM)

n_components_ms = 35  
batch_size_ms = 100  
ipca_MeanShift = IncrementalPCA(n_components=n_components_ms)
for i in range(0, df_MeanShift.shape[0], batch_size_ms):
    X_batch_MeanShift = df_MeanShift[i:i + batch_size_ms]
    ipca_MeanShift.partial_fit(X_batch_MeanShift)
    
X_reduced_SOM = ipca_SOM.transform(df_SOM)
X_reduced_MeanShift = ipca_MeanShift.transform(df_MeanShift)

print(X_reduced_SOM.shape)
print(X_reduced_MeanShift.shape)

'''
  FEATURE ENGINEER DIFFERENT STATS FOR LATER COMPARISON
'''
df_SOM_Feature = pd.DataFrame()
df_SOM_Feature['mean'] = np.mean(df_SOM, axis=1)
df_SOM_Feature['std'] = np.std(df_SOM, axis=1)
df_SOM_Feature['skew'] = skew(df_SOM, axis=1)
df_SOM_Feature['kurtosis'] = kurtosis(df_SOM, axis=1)

df_MS_Feature = pd.DataFrame()
df_MS_Feature['mean'] = np.mean(df_MeanShift, axis=1)
df_MS_Feature['std'] = np.std(df_MeanShift, axis=1)
df_MS_Feature['skew'] = skew(df_MeanShift, axis=1)
df_MS_Feature['kurtosis'] = kurtosis(df_MeanShift, axis=1)

print(df_SOM_Feature.head(3))
print(df_MS_Feature.head(3))

'''
  FUNCTION TO DEFINE THE RANDOM STATE
'''
def generate_random_labels(n_samples, n_clusters):
    return [random.randint(0, n_clusters - 1) for _ in range(n_samples)]

'''
  HYPERPARAMETER TUNNING FOR SOM
'''
grid_sizes = [5, 10, 15]
learning_rates = [0.01, 0.05, 0.25]
iterations = 1000
distance_metrics = ['euclidean', 'cosine']

results = []
i = 1

for grid_size in grid_sizes:
    for lr in learning_rates:
        for metric in distance_metrics:
            print(f"Starting training with: grid_size:{grid_size}, learning_rate: {lr}, metric: {metric}")
            som = MiniSom(grid_size, grid_size, X_reduced_SOM.shape[1], learning_rate=lr, sigma=1, neighborhood_function='bubble')
            som.train_random(X_reduced_SOM, iterations)  
            # Assign cluster labels
            labels = [som.winner(x) for x in X_reduced_SOM]  
            # Convert into unique cluster IDs
            cluster_labels = [winner[0] * grid_size + winner[1] for winner in labels] 
            silhouette = silhouette_score(X_reduced_SOM, cluster_labels, metric=metric)
            ars = adjusted_rand_score(y_resampled, cluster_labels)

            # Generate random clustering labels
            n_clusters_random = grid_size * grid_size  # Randomly assigning clusters based on grid size
            cluster_labels_random = generate_random_labels(len(X_reduced_SOM), n_clusters_random)

            # Silhouette score for random clustering
            if len(set(cluster_labels_random)) > 1:
                silhouette_random = silhouette_score(X_reduced_SOM, cluster_labels_random, metric=metric)
            else:
                silhouette_random = -1  # If only one cluster, set silhouette score to -1

            # Adjusted Rand Score for random clustering
            ars_random = adjusted_rand_score(y_resampled, cluster_labels_random)
            # Compare results
            results.append((i, grid_size, lr, metric, silhouette, ars, silhouette_random, ars_random))
            i += 1

# Print results
print("Results (ID, Grid, Learning Rate, Metric, Silhouette Score, Ajusted Rand Score):")
for result in results:
    print(result)
silh = []
silh_rand = []
ars = []
ars_rand = []
ids = []
i = 0
# indexing and extracting highst points for each metric
for result in results:
    ids.append(i)
    i += 1
    silh.append(result[4])
    silh_rand.append(result[6])
    ars.append(result[5])
    ars_rand.append(result[7])
x_max_silh = silh.index(max(silh))
y_max_silh = max(silh)
x_max_ars = ars.index(max(ars))
y_max_ars = max(ars)
x_max_silh_rand = silh_rand.index(max(silh_rand))
y_max_silh_rand = max(silh_rand)
x_max_ars_rand = ars_rand.index(max(ars_rand))
y_max_ars_rand = max(ars_rand)

print(f"Best params used are: \n \
(ID, Grid, Learning Rate, Metric, Silhouette Score, Ajusted Rand Score): \n \
{results[x_max_silh]}")

print(f"Best params used are: \n \
(ID, Grid, Learning Rate, Metric, Silhouette Score, Ajusted Rand Score): \n \
{results[x_max_ars]}")


'''
  PLOTTING THE RESULTING METRICS FOR SOM IN COMPARISOM WITH RANDOM CHANCE 
'''
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(ids, silh, marker='o', color='blue', linestyle='-', markersize=8)
axes[0, 0].scatter(x_max_silh, y_max_silh, color='red', s=300, marker='*', label="Highest Silhouette")
axes[0, 0].set_title('Silhouette Score for Different Hyperparameter Combinations')
axes[0, 0].set_xlabel('Hyperparameter Combinations (ID)')
axes[0, 0].set_ylabel('Silhouette Score')

axes[1, 0].plot(ids, ars, marker='o', color='green', linestyle='-', markersize=8)
axes[1, 0].scatter(x_max_ars, y_max_ars, color='red', s=300, marker='*', label="Highest Silhouette")
axes[1, 0].set_title('Ajusted Rand Score for Different Hyperparameter Combinations')
axes[1, 0].set_xlabel('Hyperparameter Combinations (ID)')
axes[1, 0].set_ylabel('Ajusted Rand Score')

axes[0, 1].plot(ids, silh_rand, marker='o', color='blue', linestyle='-', markersize=8)
axes[0, 1].scatter(x_max_silh_rand, y_max_silh_rand, color='red', s=300, marker='*', label="Highest Silhouette")
axes[0, 1].set_title('Silhouette Score for Different Hyperparameter Combinations RAND')
axes[0, 1].set_xlabel('Hyperparameter Combinations (ID)')
axes[0, 1].set_ylabel('Silhouette Score Random')

axes[1, 1].plot(ids, ars_rand, marker='o', color='green', linestyle='-', markersize=8)
axes[1, 1].scatter(x_max_ars_rand, y_max_ars_rand, color='red', s=300, marker='*', label="Highest Silhouette")
axes[1, 1].set_title('Ajusted Rand Score for Different Hyperparameter Combinations RAND')
axes[1, 1].set_xlabel('Hyperparameter Combinations (ID)')
axes[1, 1].set_ylabel('Ajusted Rand Score Random')
plt.tight_layout()
plt.show()

'''
  HYPERPARAMETER TUNNING FOR MEAN-SHIFT
'''
bandwidths = [2.86, 6.5, 8, 320] # the 320 value was determined while experimenting with the bandwidths and ajusting it as i needed to have fewer clusters
distance_metrics = ['euclidean', 'manhattan']
results = []
i = 1
for bandwidth in bandwidths:
    for metric in distance_metrics:
        print(f"Starting training with: bandwidth: {bandwidth}, metric: {metric}")
        mean_shift = MeanShift(bandwidth=bandwidth)
        mean_shift.fit(X_reduced_MeanShift)  
        labels = mean_shift.labels_

        silhouette = silhouette_score(X_reduced_MeanShift, labels, metric=metric)  
        ars = adjusted_rand_score(y_resampled, labels)

        # Generate random clustering labels
        n_clusters_random = grid_size * grid_size  # Randomly assigning clusters based on grid size
        cluster_labels_random = generate_random_labels(len(X_reduced_SOM), n_clusters_random)
        # Silhouette score for random clustering
        if len(set(cluster_labels_random)) > 1:
            silhouette_random = silhouette_score(X_reduced_SOM, cluster_labels_random, metric=metric)
        else:
            silhouette_random = -1  # If only one cluster, set silhouette score to -1
        ars_random = adjusted_rand_score(y_resampled, cluster_labels_random)

        results.append((i, bandwidth, metric, silhouette, ars, silhouette_random, ars_random))
        i += 1

# Print results
for result in results:
    print(result)
silh = []
silh_rand = []
ars = []
ars_rand = []
ids = []
i = 0
# indexing and extracting highst points for each metric
for result in results:
    ids.append(i)
    i += 1
    silh.append(result[3])
    silh_rand.append(result[5])
    ars.append(result[4])
    ars_rand.append(result[6])
x_max_silh = silh.index(max(silh))
y_max_silh = max(silh)
x_max_ars = ars.index(max(ars))
y_max_ars = max(ars)
x_max_silh_rand = silh_rand.index(max(silh_rand))
y_max_silh_rand = max(silh_rand)
x_max_ars_rand = ars_rand.index(max(ars_rand))
y_max_ars_rand = max(ars_rand)
print("")
print(f"Best params used are: \n \
(ID, BandWidth, Metric, Silhouette Score, Ajusted Rand Score): \n \
{results[x_max_silh]}")


'''
  PLOTTING THE RESULTING METRICS FOR MEAN-SHIFT IN COMPARISOM WITH RANDOM CHANCE 
'''
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(ids, silh, marker='o', color='blue', linestyle='-', markersize=8)
axes[0, 0].scatter(x_max_silh, y_max_silh, color='red', s=300, marker='*', label="Highest Silhouette")
axes[0, 0].set_title('Silhouette Score for Different Hyperparameter Combinations')
axes[0, 0].set_xlabel('Hyperparameter Combinations (ID)')
axes[0, 0].set_ylabel('Silhouette Score')

axes[1, 0].plot(ids, ars, marker='o', color='green', linestyle='-', markersize=8)
axes[1, 0].scatter(x_max_ars, y_max_ars, color='red', s=300, marker='*', label="Highest Silhouette")
axes[1, 0].set_title('Ajusted Rand Score for Different Hyperparameter Combinations')
axes[1, 0].set_xlabel('Hyperparameter Combinations (ID)')
axes[1, 0].set_ylabel('Ajusted Rand Score')

axes[0, 1].plot(ids, silh_rand, marker='o', color='blue', linestyle='-', markersize=8)
axes[0, 1].scatter(x_max_silh_rand, y_max_silh_rand, color='red', s=300, marker='*', label="Highest Silhouette")
axes[0, 1].set_title('Silhouette Score for Different Hyperparameter Combinations RAND')
axes[0, 1].set_xlabel('Hyperparameter Combinations (ID)')
axes[0, 1].set_ylabel('Silhouette Score Random')

axes[1, 1].plot(ids, ars_rand, marker='o', color='green', linestyle='-', markersize=8)
axes[1, 1].scatter(x_max_ars_rand, y_max_ars_rand, color='red', s=300, marker='*', label="Highest Silhouette")
axes[1, 1].set_title('Ajusted Rand Score for Different Hyperparameter Combinations RAND')
axes[1, 1].set_xlabel('Hyperparameter Combinations (ID)')
axes[1, 1].set_ylabel('Ajusted Rand Score Random')
plt.tight_layout()
plt.show()


'''
  TUNNED MODEL USED FOR PLOTTING SOM:
'''
som = MiniSom(5, 5, X_reduced_SOM.shape[1], learning_rate=0.25, sigma=1, neighborhood_function='bubble')
som.train_random(X_reduced_SOM, iterations)  
unique_neurons = np.array(list(set([som.winner(x) for x in X_reduced_SOM])))

print("Finised, training")
distances = cdist(unique_neurons, unique_neurons, metric='euclidean')
threshold = 1.5  
merged_clusters = {}
for i, neuron in enumerate(unique_neurons):
    for j, neighbor in enumerate(unique_neurons):
        if distances[i, j] <= threshold:
            merged_clusters.setdefault(i, set()).add(j)

# Assign merged cluster IDs
cluster_labels = []
for x in X_reduced_SOM:
    winner = som.winner(x)
    cluster_id = [i for i, cluster in merged_clusters.items() if winner in unique_neurons[list(cluster)]][0]
    cluster_labels.append(cluster_id)
print("Finised labeling!")


'''
  PLOTTING RESULTS WITH PCA FOR TUNNED SOM
'''
pca_2d = PCA(n_components=2)
X_2D_SOM = pca_2d.fit_transform(X_resampled)

plt.figure(figsize=(8, 6))
for i, label in enumerate(set(cluster_labels)):
    cluster_points = X_2D_SOM[[index for index, cluster in enumerate(cluster_labels) if cluster == label]]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")

plt.title("SOM Clusters in 2D Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid()
plt.show()


'''
  TUNNED MODEL USED FOR PLOTTING MEAN-SHIFT:
'''
mean_shift = MeanShift(bandwidth=320)
mean_shift.fit(X_reduced_MeanShift)  
labels = mean_shift.labels_
print("Finised, training and labeling")
'''
  PLOTTING RESULTS WITH PCA FOR TUNNED MEAN-SHIFT
'''
X_2D_MeanShift = pca_2d.fit_transform(X_resampled)

plt.figure(figsize=(8, 6))
for i, label in enumerate(set(labels)):
    cluster_points = X_2D_MeanShift[[index for index, cluster in enumerate(labels) if cluster == label]]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")

plt.title("Mean-Shift Clusters in 2D Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid()
plt.show()