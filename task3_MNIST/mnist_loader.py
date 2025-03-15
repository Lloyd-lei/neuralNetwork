import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import struct
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    """
    Load MNIST data from path
    
    Args:
        path: Path to the MNIST data files
        kind: 'train' or 't10k' (test)
        
    Returns:
        images: numpy array of shape (n_samples, 1, 28, 28)
        labels: numpy array of shape (n_samples,)
    """
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = images.reshape(len(labels), 1, 28, 28) / 255.0  # Normalize to [0, 1]
    
    return images, labels

def one_hot_encode(labels, num_classes=10):
    """
    One-hot encode labels
    
    Args:
        labels: numpy array of shape (n_samples,)
        num_classes: Number of classes
        
    Returns:
        one_hot: numpy array of shape (n_samples, num_classes)
    """
    one_hot = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        one_hot[i, labels[i]] = 1
    return one_hot

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        X: Input data
        y: Labels
        test_size: Proportion of data to use for testing
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    test_samples = int(X.shape[0] * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def visualize_samples(images, labels, num_samples=5):
    """
    Visualize random samples from the dataset
    
    Args:
        images: numpy array of shape (n_samples, 1, 28, 28)
        labels: numpy array of shape (n_samples,)
        num_samples: Number of samples to visualize
    """
    indices = np.random.choice(images.shape[0], num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx, 0], cmap='gray')
        axes[i].set_title(f"Label: {labels[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('task3_MNIST/images/mnist_samples.png', dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
        cmap: Colormap
        save_path: Path to save the figure
    """
    from sklearn.metrics import confusion_matrix
    import itertools
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def load_mnist_data(batch_size=64, use_cuda=True):
    """
    Load MNIST dataset using PyTorch's torchvision
    
    Args:
        batch_size: Batch size for data loaders
        use_cuda: Whether to use CUDA for data loading
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Split training data into train and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        **kwargs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        **kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        **kwargs
    )
    
    return train_loader, val_loader, test_loader

def visualize_samples(data_loader, num_samples=10, save_path=None):
    """
    Visualize random samples from the dataset
    
    Args:
        data_loader: DataLoader containing the dataset
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
    """
    # Get a batch of data
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Convert images to numpy for visualization
    images = images.numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Plot images
    for i in range(num_samples):
        axes[i].imshow(images[i, 0], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Create images directory
    os.makedirs("task3_MNIST/images", exist_ok=True)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    
    # Load MNIST data
    train_loader, val_loader, test_loader = load_mnist_data(use_cuda=use_cuda)
    
    # Print dataset sizes
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Visualize samples
    visualize_samples(train_loader, save_path="task3_MNIST/images/mnist_samples.png") 