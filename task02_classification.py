# ==============================
# Task 02 - K-Means Clustering
# ==============================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset
# Download Mall_Customers.csv from Kaggle and place it in the same folder
data = pd.read_csv("Mall_Customers.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# Step 3: Select features for clustering
# We'll use Annual Income and Spending Score for simplicity (2D visualization possible)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Normalize features (important for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Use Elbow Method to find optimal number of clusters (k)
wcss = []  # Within Cluster Sum of Squares
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia = total distance to cluster centers

# Plot elbow curve
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.show()

# Step 6: Train final KMeans model (say k=5 from elbow method)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Step 7: Add cluster labels to dataset
data['Cluster'] = clusters

print("\nCluster counts:")
print(data['Cluster'].value_counts())

# Step 8: Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=X['Annual Income (k$)'], y=X['Spending Score (1-100)'],
                hue=data['Cluster'], palette='Set1', s=80, alpha=0.7)
plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:,0],
            scaler.inverse_transform(kmeans.cluster_centers_)[:,1],
            s=300, c='black', marker='X', label='Centroids')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()
