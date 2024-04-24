import pandas as pd
from sklearn.cluster import KMeans

# Read the data from the CSV file
data = pd.read_csv("status_data.csv")

# Extracting relevant features for clustering
X = data.iloc[:, 2:].values  # Exclude the status_id and status_type columns

# Applying K-means to the dataset with 2 clusters (for demonstration)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Adding the cluster labels to the dataset
data['Cluster'] = y_kmeans

# Displaying the clustered data
print(data)
