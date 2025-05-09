from sklearn.cluster import KMeans
import numpy as np

class KMeansModel:
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
    
    def fit(self, data):
        """
        Fit the KMeans model to the data.
        
        Args:
            data (numpy.ndarray): The data to cluster.
            
        Returns:
            numpy.ndarray: Cluster labels for each data point.
        """
        self.kmeans.fit(data)
        return self.kmeans.labels_
    
    def predict(self, data):
        """
        Predict the closest cluster for each data point.
        
        Args:
            data (numpy.ndarray): The data to predict clusters for.
            
        Returns:
            numpy.ndarray: Predicted cluster labels.
        """
        return self.kmeans.predict(data)
    
    def fit_predict(self, data):
        """
        Fit the model and predict the clusters in one step.
        
        Args:
            data (numpy.ndarray): The data to cluster.
            
        Returns:
            numpy.ndarray: Cluster labels for each data point.
        """
        return self.kmeans.fit_predict(data)

    def get_cluster_centers(self):
        """
        Get the coordinates of cluster centers.
        
        Returns:
            numpy.ndarray: Coordinates of cluster centers.
        """
        return self.kmeans.cluster_centers_
