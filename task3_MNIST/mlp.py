import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import itertools

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for MNIST classification
    """
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10):
        """
        Initialize MLP model
        
        Args:
            input_size: Size of input features (28*28=784 for MNIST)
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes (10 for MNIST)
        """
        super(MLP, self).__init__()
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Sequential container
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Forward pass through layers
        x = self.layers(x)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_accuracies: List of validation accuracies
    """
    # Move model to device
    model = model.to(device)
    
    # Lists to store metrics
    train_losses = []
    val_accuracies = []
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Running loss for this epoch
        running_loss = 0.0
        
        # Iterate over batches
        for inputs, labels in train_loader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Evaluate on validation set
        val_accuracy = evaluate_model(model, val_loader, device)[0]
        val_accuracies.append(val_accuracy)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # End timer
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return model, train_losses, val_accuracies

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the model
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cpu' or 'cuda')
        
    Returns:
        accuracy: Test accuracy
        predictions: List of predictions
        true_labels: List of true labels
    """
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store predictions and true labels
    predictions = []
    true_labels = []
    
    # Start timer
    start_time = time.time()
    
    # Disable gradient calculation
    with torch.no_grad():
        # Correct predictions counter
        correct = 0
        total = 0
        
        # Iterate over batches
        for inputs, labels in test_loader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    
    # End timer
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    
    return accuracy, predictions, true_labels

def plot_training_curves(train_losses, val_accuracies, save_path=None):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_accuracies: List of validation accuracies
        save_path: Path to save the figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies, 'r-')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(true_labels, predictions, classes, title=None, normalize=False, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        true_labels: List of true labels
        predictions: List of predictions
        classes: List of class names
        title: Title for the plot
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add title
    if title:
        plt.title(title, fontsize=14)
    
    # Add colorbar
    plt.colorbar()
    
    # Add tick marks and labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    # Add labels
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf() 