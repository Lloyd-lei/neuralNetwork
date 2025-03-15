import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import CNN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task2_CNN.cnn import CNN

def visualize_cnn_architecture(save_path=None):
    """
    Visualize CNN architecture
    
    Args:
        save_path: Path to save the figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define layer dimensions and spacing
    layer_width = 0.8
    layer_spacing = 1.2
    
    # Define colors for different layer types
    colors = {
        'input': 'lightblue',
        'conv': 'lightgreen',
        'pool': 'lightyellow',
        'fc': 'lightpink'
    }
    
    # Define layer positions and sizes
    layers = [
        {'name': 'Input', 'type': 'input', 'size': (28, 28, 1), 'pos': 0},
        {'name': 'Conv1', 'type': 'conv', 'size': (24, 24, 6), 'pos': 1, 'kernel': '5x5', 'filters': 6},
        {'name': 'Pool1', 'type': 'pool', 'size': (12, 12, 6), 'pos': 2, 'pool': '2x2'},
        {'name': 'Conv2', 'type': 'conv', 'size': (8, 8, 16), 'pos': 3, 'kernel': '5x5', 'filters': 16},
        {'name': 'Pool2', 'type': 'pool', 'size': (4, 4, 16), 'pos': 4, 'pool': '2x2'},
        {'name': 'FC1', 'type': 'fc', 'size': (120,), 'pos': 5, 'units': 120},
        {'name': 'FC2', 'type': 'fc', 'size': (84,), 'pos': 6, 'units': 84},
        {'name': 'Output', 'type': 'fc', 'size': (10,), 'pos': 7, 'units': 10}
    ]
    
    # Draw layers
    for i, layer in enumerate(layers):
        x = layer['pos'] * layer_spacing
        
        # Adjust height based on layer type
        if layer['type'] == 'input' or layer['type'] == 'conv' or layer['type'] == 'pool':
            height = min(3, max(1, layer['size'][0] / 10))
        else:
            height = min(3, max(0.5, layer['size'][0] / 40))
        
        # Draw rectangle
        rect = plt.Rectangle((x, -height/2), layer_width, height, 
                            facecolor=colors[layer['type']], edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        # Add layer name and size
        if layer['type'] == 'input' or layer['type'] == 'conv' or layer['type'] == 'pool':
            size_text = f"{layer['size'][0]}x{layer['size'][1]}x{layer['size'][2]}"
        else:
            size_text = f"{layer['size'][0]}"
            
        ax.text(x + layer_width/2, 0, layer['name'], ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x + layer_width/2, -height/2 - 0.3, size_text, ha='center', va='top', fontsize=8)
        
        # Add additional info
        if layer['type'] == 'conv':
            info = f"Kernel: {layer['kernel']}\nFilters: {layer['filters']}"
            ax.text(x + layer_width/2, height/2 + 0.2, info, ha='center', va='bottom', fontsize=8)
        elif layer['type'] == 'pool':
            info = f"Pool size: {layer['pool']}"
            ax.text(x + layer_width/2, height/2 + 0.2, info, ha='center', va='bottom', fontsize=8)
        elif layer['type'] == 'fc' and i < len(layers) - 1:
            info = f"Units: {layer['units']}"
            ax.text(x + layer_width/2, height/2 + 0.2, info, ha='center', va='bottom', fontsize=8)
        
        # Draw arrows between layers
        if i < len(layers) - 1:
            ax.arrow(x + layer_width, 0, layer_spacing - layer_width, 0, 
                    head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Set axis limits
    ax.set_xlim(-0.5, (len(layers) - 1) * layer_spacing + layer_width + 0.5)
    ax.set_ylim(-2, 2)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title('CNN Architecture', fontsize=14)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['input'], edgecolor='black', alpha=0.7, label='Input'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['conv'], edgecolor='black', alpha=0.7, label='Convolutional'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['pool'], edgecolor='black', alpha=0.7, label='Pooling'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['fc'], edgecolor='black', alpha=0.7, label='Fully Connected')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()
    
    return fig

def visualize_feature_maps(model, input_image, save_path=None):
    """
    Visualize feature maps of CNN layers
    
    Args:
        model: Trained CNN model
        input_image: Input image tensor of shape (1, channels, height, width)
        save_path: Path to save the figure
    """
    # Set model to evaluation mode
    model.eval()
    
    # Move input to the same device as model
    device = next(model.parameters()).device
    input_image = input_image.to(device)
    
    # Get feature maps
    feature_maps = []
    
    # Input
    feature_maps.append(('Input', input_image))
    
    # Conv1
    conv1_output = model.relu(model.conv1(input_image))
    feature_maps.append(('Conv1', conv1_output))
    
    # Pool1
    pool1_output = model.pool1(conv1_output)
    feature_maps.append(('Pool1', pool1_output))
    
    # Conv2
    conv2_output = model.relu(model.conv2(pool1_output))
    feature_maps.append(('Conv2', conv2_output))
    
    # Pool2
    pool2_output = model.pool2(conv2_output)
    feature_maps.append(('Pool2', pool2_output))
    
    # Create figure
    num_layers = len(feature_maps)
    fig, axes = plt.subplots(num_layers, 4, figsize=(15, 3 * num_layers))
    
    # Plot feature maps
    for i, (name, feature_map) in enumerate(feature_maps):
        # Convert feature map to numpy
        feature_map = feature_map.detach().cpu().numpy()
        
        # Get number of channels
        num_channels = feature_map.shape[1]
        
        # Plot up to 4 channels
        for j in range(min(4, num_channels)):
            ax = axes[i, j] if num_layers > 1 else axes[j]
            
            # Plot feature map
            im = ax.imshow(feature_map[0, j], cmap='viridis')
            ax.set_title(f'{name} - Channel {j+1}')
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()
    
    return fig

def visualize_filters(model, save_path=None):
    """
    Visualize filters of CNN layers
    
    Args:
        model: CNN model
        save_path: Path to save the figure
    """
    # Get filters
    conv1_weights = model.conv1.weight.detach().cpu().numpy()
    conv2_weights = model.conv2.weight.detach().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    
    # Plot Conv1 filters (first 6)
    for i in range(6):
        ax = axes[0, i]
        im = ax.imshow(conv1_weights[i, 0], cmap='viridis')
        ax.set_title(f'Conv1 - Filter {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot Conv2 filters (first 6)
    for i in range(6):
        ax = axes[1, i]
        # Average across input channels for visualization
        im = ax.imshow(np.mean(conv2_weights[i], axis=0), cmap='viridis')
        ax.set_title(f'Conv2 - Filter {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Create images directory
    os.makedirs("task2_CNN/images", exist_ok=True)
    
    # Visualize CNN architecture
    print("Visualizing CNN architecture...")
    visualize_cnn_architecture(save_path="task2_CNN/images/cnn_architecture.png")
    
    # Check if a trained model exists
    model_path = "task3_MNIST/cnn_model.pth"
    if os.path.exists(model_path):
        # Load model
        print("Loading trained model...")
        model = CNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Create a random input image
        input_image = torch.randn(1, 1, 28, 28)
        
        # Visualize feature maps
        print("Visualizing feature maps...")
        visualize_feature_maps(model, input_image, save_path="task2_CNN/images/cnn_feature_maps.png")
        
        # Visualize filters
        print("Visualizing filters...")
        visualize_filters(model, save_path="task2_CNN/images/cnn_filters.png")
    else:
        print(f"Trained model not found at {model_path}. Run task3_MNIST/train_cnn.py first to train a model.") 