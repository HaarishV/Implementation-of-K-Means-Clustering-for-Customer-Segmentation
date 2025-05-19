# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Select K Clusters: Decide the number of clusters (K) to divide the dataset into.
2. Initialize Centroids: Randomly choose K data points as the initial cluster centers.
3. Assign Data Points: Calculate the distance between each data point and the centroids; assign each point to the closest centroid.
4. Update Centroids: Compute the new centroid for each cluster by averaging the points assigned to it.
5. Repeat Until Convergence: Continue reassignment and centroid updating until centroids stop changing significantly.
6. Use the Elbow Method: Identify the optimal number of clusters by plotting Within-Cluster Sum of Squares (WCSS) for different values of K.
7. Train the Model: Apply K-Means clustering with the chosen K and fit it to the relevant features of the dataset.
8. Predict Clusters: Assign each customer to a cluster based on spending habits and annual income.
9. Visualize Clusters: Create a scatter plot to represent the segmentation, using different colors for different clusters.
10. Interpret Results: Analyze the clusters to extract meaningful insights about customer behavior and spending patterns.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Haarish V
RegisterNumber:  212223230067


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Mall_Customers (1) (1).csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = []  #Within-Cluster sum of square. 
for i in range(1,11):
  kmeans=KMeans(n_clusters = i,init = "k-means++")
  kmeans.fit(data.iloc[:,3:])
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])
y_pred
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
*/
```

## Output:
![Screenshot 2025-05-19 161446](https://github.com/user-attachments/assets/7cb85ce2-1328-4fb3-bb2f-19449ecb7d82)
![Screenshot 2025-05-19 161452](https://github.com/user-attachments/assets/4ae57d31-2d2e-41c6-aa77-d847669de289)
![Screenshot 2025-05-19 161500](https://github.com/user-attachments/assets/94cd7e64-c8e8-4447-a7ef-5bc211a859a5)

![Screenshot 2025-05-19 161509](https://github.com/user-attachments/assets/f618138d-4e3e-4f20-9794-be7f8c451640)
![Screenshot 2025-05-19 161518](https://github.com/user-attachments/assets/863a1129-83e4-4355-ace3-e155674ba9ae)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
