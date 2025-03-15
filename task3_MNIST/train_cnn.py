import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys

# Add parent directory to path to import CNN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task2_CNN.cnn import CNN, train_model, evaluate_model, plot_training_curves, plot_confusion_matrix
from task3_MNIST.mnist_loader import load_mnist_data, visualize_samples

def main():
    """
    Main function to train and evaluate CNN on MNIST dataset
    """
    # Create images directory
    os.makedirs("task3_MNIST/images", exist_ok=True)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST data
    print("Loading MNIST data...")
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=64, use_cuda=use_cuda)
    
    # Print dataset sizes
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Visualize samples
    print("Visualizing samples...")
    visualize_samples(train_loader, save_path="task3_MNIST/images/mnist_samples.png")
    
    # Create CNN model
    print("Creating CNN model...")
    model = CNN(input_channels=1, num_classes=10)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Train model
    print("\nTraining CNN model...")
    model, train_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device
    )
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        save_path="task3_MNIST/images/cnn_training_curves.png"
    )
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_accuracy, predictions, true_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(
        true_labels=true_labels,
        predictions=predictions,
        classes=class_names,
        title="CNN Confusion Matrix",
        save_path="task3_MNIST/images/cnn_confusion_matrix.png"
    )
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), "task3_MNIST/cnn_model.pth")
    print("Model saved to task3_MNIST/cnn_model.pth")

if __name__ == "__main__":
    main() 