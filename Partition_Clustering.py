import pandas as pd
from sklearn.cluster import KMeans

# Read the data from the CSV file
data = pd.read_csv("customer_data2.csv")

# Extracting relevant features for clustering
X = data.iloc[:, [2, 3, 4]].values  # Age, Annual Income, Spending Score

# Applying K-means to the dataset with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Adding the cluster labels to the dataset
data['Cluster'] = y_kmeans

# Displaying the clustered data
print(data)