import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Annual_Income': [15, 16, 17, 18, 20, 25, 30, 35, 40, 45],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}

df = pd.DataFrame(data)

X = df[['Annual_Income', 'Spending_Score']]

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X)

df['Cluster'] = kmeans.labels_

print("Centroids:")
print(kmeans.cluster_centers_)

plt.scatter(X['Annual_Income'], X['Spending_Score'], c=df['Cluster'], cmap='viridis')

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X',
    color='red'
)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering')
plt.show()