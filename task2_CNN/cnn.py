import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

class CNN(nn.Module):
    """
    Convolutional Neural Network implementation using PyTorch
    """
    def __init__(self, input_channels=1, num_classes=10):
        """
        Initialize CNN with default architecture
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
        """
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 16 * 4 * 4)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Train the CNN model
    
    Args:
        model: CNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        
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
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # Update statistics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation accuracy
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    return model, train_losses, val_accuracies

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the CNN model on test data
    
    Args:
        model: Trained CNN model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        accuracy: Test accuracy
        predictions: Predicted labels
        true_labels: True labels
    """
    # Move model to device
    model = model.to(device)
    
    # Evaluation mode
    model.eval()
    
    # Lists to store predictions and true labels
    all_predictions = []
    all_labels = []
    
    # Start timer
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    true_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    print(f'Evaluation completed in {inference_time:.2f} seconds')
    
    return accuracy, predictions, true_labels

def plot_training_curves(train_losses, val_accuracies, save_path=None):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_accuracies: List of validation accuracies
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(true_labels, predictions, classes, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        true_labels: True labels
        predictions: Predicted labels
        classes: List of class names
        title: Title for the plot
        save_path: Path to save the figure
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create a simple dataset for testing
    X = torch.randn(10, 1, 28, 28)
    y = torch.randint(0, 10, (10,))
    
    # Create CNN
    cnn = CNN()
    
    # Print model architecture
    print(cnn) 