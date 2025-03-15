import numpy as np
import matplotlib.pyplot as plt

class MLP:
    """
    Multi-Layer Perceptron implementation using NumPy
    """
    def __init__(self, layer_dims, activation='relu'):
        """
        Initialize MLP with specified layer dimensions
        
        Args:
            layer_dims: List of integers representing the dimensions of each layer
                        (including input and output layers)
            activation: Activation function to use ('relu' or 'sigmoid')
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.activation = activation
        
        # Initialize weights and biases
        self.parameters = {}
        for l in range(1, self.num_layers + 1):
            # He initialization for ReLU, Xavier for sigmoid
            if activation == 'relu':
                scale = np.sqrt(2.0 / layer_dims[l-1])
            else:
                scale = np.sqrt(1.0 / layer_dims[l-1])
                
            self.parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    def _relu(self, Z):
        """
        ReLU activation function
        """
        return np.maximum(0, Z)
    
    def _relu_derivative(self, Z):
        """
        Derivative of ReLU activation function
        """
        return np.where(Z > 0, 1, 0)
    
    def _sigmoid(self, Z):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to avoid overflow
    
    def _sigmoid_derivative(self, Z):
        """
        Derivative of sigmoid activation function
        """
        s = self._sigmoid(Z)
        return s * (1 - s)
    
    def _activate(self, Z):
        """
        Apply activation function
        """
        if self.activation == 'relu':
            return self._relu(Z)
        else:
            return self._sigmoid(Z)
    
    def _activate_derivative(self, Z):
        """
        Apply derivative of activation function
        """
        if self.activation == 'relu':
            return self._relu_derivative(Z)
        else:
            return self._sigmoid_derivative(Z)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data of shape (input_size, batch_size)
            
        Returns:
            A dictionary containing the activations and pre-activations for each layer
        """
        caches = {}
        A = X
        caches['A0'] = X
        
        # Forward propagation through layers
        for l in range(1, self.num_layers + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(W, A_prev) + b
            A = self._activate(Z)
            
            # Save values for backpropagation
            caches[f'Z{l}'] = Z
            caches[f'A{l}'] = A
        
        return A, caches
    
    def backward(self, Y, caches, learning_rate=0.01):
        """
        Backward propagation
        
        Args:
            Y: Ground truth labels of shape (output_size, batch_size)
            caches: Dictionary containing the activations and pre-activations from forward pass
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Updated parameters
        """
        m = Y.shape[1]  # batch size
        gradients = {}
        
        # Output layer
        dA = -(np.divide(Y, caches[f'A{self.num_layers}'] + 1e-8) - 
               np.divide(1 - Y, 1 - caches[f'A{self.num_layers}'] + 1e-8))
        
        # Backward propagation through layers
        for l in reversed(range(1, self.num_layers + 1)):
            Z = caches[f'Z{l}']
            A_prev = caches[f'A{l-1}']
            
            dZ = dA * self._activate_derivative(Z)
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                dA = np.dot(self.parameters[f'W{l}'].T, dZ)
            
            # Update parameters
            self.parameters[f'W{l}'] -= learning_rate * dW
            self.parameters[f'b{l}'] -= learning_rate * db
            
            # Store gradients for analysis if needed
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
        
        return gradients
    
    def compute_loss(self, Y_pred, Y):
        """
        Compute binary cross-entropy loss
        
        Args:
            Y_pred: Predicted values from forward pass
            Y: Ground truth labels
            
        Returns:
            Loss value
        """
        m = Y.shape[1]
        epsilon = 1e-8  # To avoid log(0)
        
        # Binary cross-entropy loss
        loss = -np.sum(Y * np.log(Y_pred + epsilon) + (1 - Y) * np.log(1 - Y_pred + epsilon)) / m
        return loss
    
    def train(self, X, Y, num_epochs=1000, batch_size=32, learning_rate=0.01, verbose=True):
        """
        Train the MLP
        
        Args:
            X: Input data of shape (input_size, num_samples)
            Y: Ground truth labels of shape (output_size, num_samples)
            num_epochs: Number of training epochs
            batch_size: Size of mini-batches
            learning_rate: Learning rate for gradient descent
            verbose: Whether to print progress
            
        Returns:
            List of losses during training
        """
        m = X.shape[1]
        losses = []
        
        for epoch in range(num_epochs):
            # Shuffle data
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                end = min(i + batch_size, m)
                X_batch = X_shuffled[:, i:end]
                Y_batch = Y_shuffled[:, i:end]
                
                # Forward pass
                A_final, caches = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(A_final, Y_batch)
                
                # Backward pass
                self.backward(Y_batch, caches, learning_rate)
            
            # Compute loss on full dataset for monitoring
            A_final, _ = self.forward(X)
            epoch_loss = self.compute_loss(A_final, Y)
            losses.append(epoch_loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        
        return losses
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions
        
        Args:
            X: Input data
            threshold: Classification threshold for sigmoid activation
            
        Returns:
            Predictions
        """
        A_final, _ = self.forward(X)
        
        if self.layer_dims[-1] == 1:  # Binary classification
            return (A_final > threshold).astype(int)
        else:  # Multi-class classification
            return np.argmax(A_final, axis=0)

# Example usage
if __name__ == "__main__":
    # Create a simple dataset for testing
    np.random.seed(42)
    X = np.random.randn(2, 100)
    Y = np.zeros((1, 100))
    Y[0, :] = (X[0, :] > 0) & (X[1, :] > 0)
    
    # Create and train MLP
    mlp = MLP([2, 4, 3, 1], activation='sigmoid')
    losses = mlp.train(X, Y, num_epochs=1000, learning_rate=0.1)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('mlp_loss.png')
    
    # Visualize decision boundary
    h = 0.01
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=Y.flatten(), edgecolors='k', marker='o')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('mlp_decision_boundary.png')
    plt.show() 