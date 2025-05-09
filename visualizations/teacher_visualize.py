import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_class_wise_accuracy(preds, labels, num_classes=10):
    """Plot and print class-wise accuracy for the teacher model."""
    correct_counts = [0] * num_classes
    total_counts = [0] * num_classes
    
    for p, l in zip(preds, labels):
        if p == l:
            correct_counts[l] += 1
        total_counts[l] += 1
    
    accuracies = [correct_counts[i] / total_counts[i] * 100 if total_counts[i] != 0 else 0 for i in range(num_classes)]

    # Print class-wise accuracies
    for i in range(num_classes):
        print(f"Class {i} Accuracy: {accuracies[i]:.2f}%")
    
    # Plotting class-wise accuracies
    plt.bar(range(num_classes), accuracies)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Teacher Model Class-wise Accuracy')
    plt.show()

def plot_confusion_matrix(preds, labels, num_classes=10):
    """Plot confusion matrix for the teacher model."""
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Teacher Model Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Load saved predictions and labels
    preds = np.load('./visualizations/Pretrained_predictions.npy')
    labels = np.load('./visualizations/Pretrained_labels.npy')

    # Plot class-wise accuracy and print accuracies
    plot_class_wise_accuracy(preds, labels)

    # Plot confusion matrix
    plot_confusion_matrix(preds, labels)
