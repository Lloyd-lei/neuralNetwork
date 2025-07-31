import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys

# Add parent directory to path to import transformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task4_Transformer.transformer import TransformerModel, train_model, evaluate_model, plot_training_curves, plot_confusion_matrix
from task3_MNIST.mnist_loader import load_mnist_data, visualize_samples

def main():
    """
    Main function to train and evaluate Transformer on MNIST dataset
    """
    # Create images directory
    os.makedirs("task4_Transformer/images", exist_ok=True)
    
    # Check if MPS is available for Apple Silicon
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load MNIST data
    print("Loading MNIST data...")
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=64, use_cuda=use_mps)
    
    # Print dataset sizes
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Visualize samples
    print("Visualizing samples...")
    visualize_samples(train_loader, save_path="task4_Transformer/images/mnist_samples.png")
    
    # Create Transformer model
    print("Creating Transformer model...")
    model = TransformerModel(
        input_dim=784,  # 28*28 for MNIST
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=10
    )
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("\nTraining Transformer model...")
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
        save_path="task4_Transformer/images/transformer_training_curves.png"
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
        title="Transformer Confusion Matrix",
        save_path="task4_Transformer/images/transformer_confusion_matrix.png"
    )
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), "task4_Transformer/transformer_model.pth")
    print("Model saved to task4_Transformer/transformer_model.pth")
    
    # Save training metrics for comparison
    np.save("task4_Transformer/transformer_train_losses.npy", np.array(train_losses))
    np.save("task4_Transformer/transformer_val_accuracies.npy", np.array(val_accuracies))
    
    # Save test results
    results = {
        "test_accuracy": test_accuracy,
        "training_time": time.time() - train_losses[0],  # Approximate
        "model_size_mb": os.path.getsize("task4_Transformer/transformer_model.pth") / (1024 * 1024)
    }
    
    # Print results
    print("\nResults:")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Model Size: {results['model_size_mb']:.2f} MB")

if __name__ == "__main__":
    main() 