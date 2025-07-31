import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

# Add parent directory to path to import transformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task4_Transformer.transformer import TransformerModel

def visualize_attention_weights(model, input_data, layer_idx=0, head_idx=0, save_path=None):
    """
    Visualize attention weights for a specific layer and head
    
    Args:
        model: Trained transformer model
        input_data: Input data tensor (batch_size, 1, 28, 28)
        layer_idx: Index of the transformer layer to visualize
        head_idx: Index of the attention head to visualize
        save_path: Path to save the figure
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Register hook to get attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Extract attention weights from the output
        # The structure depends on PyTorch's TransformerEncoderLayer implementation
        attention_weights.append(output[1])
    
    # Register the hook on the specific transformer layer
    for i, layer in enumerate(model.transformer_encoder.layers):
        if i == layer_idx:
            layer.self_attn.register_forward_hook(hook_fn)
    
    # Forward pass to get attention weights
    with torch.no_grad():
        _ = model(input_data)
    
    # Extract attention weights for the specified head
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    attn = attention_weights[0][0, head_idx].cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap
    colors = [(0.1, 0.1, 0.1), (0.0, 0.5, 1.0)]  # Dark blue to light blue
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Plot attention weights as heatmap
    sns.heatmap(attn, cmap=cmap, annot=False)
    plt.title(f'Attention Weights (Layer {layer_idx+1}, Head {head_idx+1})', fontsize=16)
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('Token Position', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def visualize_positional_encoding(d_model=128, max_len=100, save_path=None):
    """
    Visualize positional encoding
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        save_path: Path to save the figure
    """
    # Create positional encoding matrix
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Convert to numpy for plotting
    pe_np = pe.numpy()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot positional encoding
    plt.imshow(pe_np, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Positional Encoding', fontsize=16)
    plt.xlabel('Embedding Dimension', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def visualize_transformer_architecture(save_path=None):
    """
    Visualize transformer architecture using a diagram
    
    Args:
        save_path: Path to save the figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Set background color
    ax.set_facecolor('#f5f5f5')
    
    # Define components and their positions
    components = [
        {'name': 'Input\n(MNIST Image)', 'pos': (5, 1), 'width': 3, 'height': 1, 'color': '#3498db'},
        {'name': 'Flatten', 'pos': (5, 2.5), 'width': 3, 'height': 0.8, 'color': '#2ecc71'},
        {'name': 'Input Embedding\n(Linear Layer)', 'pos': (5, 4), 'width': 3, 'height': 1, 'color': '#e74c3c'},
        {'name': 'Positional\nEncoding', 'pos': (5, 5.5), 'width': 3, 'height': 1, 'color': '#f39c12'},
        {'name': 'Transformer\nEncoder Layer 1', 'pos': (5, 7), 'width': 3, 'height': 1, 'color': '#9b59b6'},
        {'name': 'Transformer\nEncoder Layer 2', 'pos': (5, 8.5), 'width': 3, 'height': 1, 'color': '#9b59b6'},
        {'name': 'Output Layer\n(Linear)', 'pos': (5, 10), 'width': 3, 'height': 1, 'color': '#e74c3c'},
        {'name': 'Softmax', 'pos': (5, 11.5), 'width': 3, 'height': 0.8, 'color': '#2ecc71'},
        {'name': 'Output\n(10 classes)', 'pos': (5, 13), 'width': 3, 'height': 1, 'color': '#3498db'}
    ]
    
    # Draw components
    for comp in components:
        x, y = comp['pos']
        width, height = comp['width'], comp['height']
        rect = plt.Rectangle((x - width/2, y - height/2), width, height, 
                            facecolor=comp['color'], alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, comp['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw connections
    for i in range(len(components) - 1):
        x1, y1 = components[i]['pos']
        x2, y2 = components[i+1]['pos']
        ax.arrow(x1, y1 + components[i]['height']/2, 0, y2 - y1 - (components[i]['height'] + components[i+1]['height'])/2,
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add attention mechanism detail for transformer encoder layer
    attention_detail = [
        {'name': 'Multi-Head\nAttention', 'pos': (2, 7), 'width': 2, 'height': 0.8, 'color': '#9b59b6'},
        {'name': 'Feed Forward\nNetwork', 'pos': (8, 7), 'width': 2, 'height': 0.8, 'color': '#9b59b6'},
        {'name': 'Add & Norm', 'pos': (5, 7.7), 'width': 2, 'height': 0.4, 'color': '#f39c12'}
    ]
    
    # Draw attention components
    for comp in attention_detail:
        x, y = comp['pos']
        width, height = comp['width'], comp['height']
        rect = plt.Rectangle((x - width/2, y - height/2), width, height, 
                            facecolor=comp['color'], alpha=0.5, edgecolor='black', linestyle='--')
        ax.add_patch(rect)
        ax.text(x, y, comp['name'], ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw connection lines for attention detail
    ax.plot([2, 5], [7, 7], 'k--', alpha=0.5)
    ax.plot([5, 8], [7, 7], 'k--', alpha=0.5)
    
    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Add title
    plt.title('Transformer Architecture for MNIST Classification', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def visualize_attention_on_images(model, images, labels, predictions, save_path=None):
    """
    Visualize attention weights overlaid on input images
    
    Args:
        model: Trained transformer model
        images: Input images (batch_size, 1, 28, 28)
        labels: True labels
        predictions: Model predictions
        save_path: Path to save the figure
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Register hook to get attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Extract attention weights from the output
        attention_weights.append(output[1])
    
    # Register the hook on the last transformer layer
    model.transformer_encoder.layers[-1].self_attn.register_forward_hook(hook_fn)
    
    # Forward pass to get attention weights
    with torch.no_grad():
        _ = model(images)
    
    # Get attention weights from the last layer, average over heads
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    attn = attention_weights[0].mean(dim=1).cpu().numpy()
    
    # Number of images to visualize
    num_images = min(5, len(images))
    
    # Create figure
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        # Original image
        img = images[i, 0].cpu().numpy()
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'True: {labels[i]}, Pred: {predictions[i]}', 
                           color='green' if labels[i] == predictions[i] else 'red')
        axes[0, i].axis('off')
        
        # Attention heatmap
        axes[1, i].imshow(img, cmap='gray', alpha=0.7)
        
        # Reshape attention weights to match image dimensions
        # This is a simplification - in practice, you'd need to handle the specific way
        # your transformer processes the image
        attention = attn[i, 0].reshape(1, -1)  # Take first token's attention to all others
        attention_map = attention.reshape(28, 28)  # Reshape to image dimensions
        
        # Overlay attention heatmap
        axes[1, i].imshow(attention_map, cmap='hot', alpha=0.5)
        axes[1, i].set_title('Attention Map')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Create directory for saving images
    os.makedirs("task4_Transformer/images", exist_ok=True)
    
    # Visualize positional encoding
    print("Visualizing positional encoding...")
    visualize_positional_encoding(save_path="task4_Transformer/images/positional_encoding.png")
    
    # Visualize transformer architecture
    print("Visualizing transformer architecture...")
    visualize_transformer_architecture(save_path="task4_Transformer/images/transformer_architecture.png")
    
    # Check if MPS is available for Apple Silicon
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a sample model and input for testing attention visualization
    model = TransformerModel()
    model.to(device)
    
    # Generate random input data
    batch_size = 5
    input_data = torch.randn(batch_size, 1, 28, 28).to(device)
    
    # Visualize attention weights (this may not work without training the model first)
    try:
        print("Attempting to visualize attention weights...")
        visualize_attention_weights(model, input_data, 
                                   save_path="task4_Transformer/images/attention_weights.png")
    except Exception as e:
        print(f"Could not visualize attention weights: {e}")
        print("This is expected if the model hasn't been trained yet.") 