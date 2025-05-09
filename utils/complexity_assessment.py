# complexity_assessment.py

import numpy as np
from sklearn.metrics import silhouette_score

def assess_task_complexity(data, labels):
    """
    Assess the complexity of a task using silhouette score.
    
    Args:
        data (numpy.ndarray): The data for the task.
        labels (numpy.ndarray): The labels for the data.
    
    Returns:
        float: The silhouette score indicating the complexity of the task.
    """
    if len(np.unique(labels)) > 1:
        score = silhouette_score(data, labels)
    else:
        score = 0.0  # When there's only one class, complexity is minimal
    
    return score
