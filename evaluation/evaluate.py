import torch
import numpy as np
from sklearn.metrics import silhouette_score


def evaluate(model, data_loader, criterion, device, return_preds=False, use_soft_gating=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Determine whether to use hard or soft gating
            if use_soft_gating:
                outputs = model(images, gating='soft')
            else:
                outputs = model(images, gating='hard')
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)

    if return_preds:
        return avg_loss, accuracy, all_preds, all_labels
    else:
        return avg_loss, accuracy

def evaluate_clustering_accuracy(model, data_loader, device, kmeans_model):
    """
    Evaluate the clustering accuracy of the model.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the evaluation on.
        kmeans_model (KMeansModel): Pre-trained KMeans model for clustering evaluation.
    
    Returns:
        float: Clustering accuracy.
    """
    model.eval()
    all_preds = []
    all_features = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            features = model.extract_features(images)  # Assuming there's an extract_features method
            all_features.append(features.cpu().numpy())

    all_features = np.vstack(all_features)
    cluster_labels = kmeans_model.predict(all_features)
    
    # Here, you would compare cluster_labels with some ground truth or previous clusters
    # For demonstration, assume we have true labels:
    # true_labels = ...
    # clustering_accuracy = accuracy_score(true_labels, cluster_labels)
    # For now, we return the silhouette score as a proxy for clustering accuracy:
    silhouette_avg = silhouette_score(all_features, cluster_labels)
    return silhouette_avg
