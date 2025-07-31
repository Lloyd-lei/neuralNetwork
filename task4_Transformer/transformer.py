import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    Transformer model for image classification
    """
    def __init__(self, input_dim=784, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, 
                 dropout=0.1, num_classes=10):
        """
        Initialize transformer model
        
        Args:
            input_dim: Input dimension (784 for MNIST flattened images)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            num_classes: Number of output classes (10 for MNIST)
        """
        super(TransformerModel, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes)
        
        # Initialize parameters
        self._init_parameters()
        
        # Model dimension
        self.d_model = d_model
    
    def _init_parameters(self):
        """
        Initialize model parameters
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) for MNIST images
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten the image if it's not already flattened
        if len(x.shape) == 4:  # (batch_size, channels, height, width)
            x = x.view(x.size(0), -1)  # (batch_size, input_dim)
        
        # Embed input to d_model dimensions
        x = self.input_embedding(x)  # (batch_size, d_model)
        
        # Add sequence dimension for transformer (treating each feature as a "token")
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        
        # Take the output of the sequence
        x = x.squeeze(1)  # (batch_size, d_model)
        
        # Output layer
        x = self.output_layer(x)  # (batch_size, num_classes)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='mps'):
    """
    Train the transformer model
    
    Args:
        model: Transformer model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('mps' for Apple Silicon GPU)
        
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

def evaluate_model(model, test_loader, device='mps'):
    """
    Evaluate the transformer model on test data
    
    Args:
        model: Trained transformer model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('mps' for Apple Silicon GPU)
        
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
    # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create a simple dataset for testing
    X = torch.randn(10, 1, 28, 28)
    y = torch.randint(0, 10, (10,))
    
    # Create transformer model
    transformer = TransformerModel()
    
    # Print model architecture
    print(transformer) 