import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
import os

def draw_mlp_computational_graph(layer_dims, save_path=None):
    """
    Draw the computational graph for a Multi-Layer Perceptron
    
    Args:
        layer_dims: List of integers representing the dimensions of each layer
        save_path: Path to save the figure
    """
    # Create a directed graph
    G_forward = nx.DiGraph()
    G_backward = nx.DiGraph()
    
    # Define node positions
    pos = {}
    node_labels = {}
    edge_labels_forward = {}
    edge_labels_backward = {}
    
    # Create nodes for each layer
    max_neurons = max(layer_dims)
    layer_spacing = 3
    neuron_spacing = 1
    
    # Create input layer nodes
    for i in range(layer_dims[0]):
        node_name = f"x_{i}"
        G_forward.add_node(node_name)
        G_backward.add_node(node_name)
        pos[node_name] = (0, (max_neurons - layer_dims[0])/2 + i * neuron_spacing)
        node_labels[node_name] = f"x_{i}"
    
    # Create hidden and output layer nodes
    for l in range(1, len(layer_dims)):
        for i in range(layer_dims[l]):
            # Pre-activation node
            z_node = f"z^{l}_{i}"
            G_forward.add_node(z_node)
            G_backward.add_node(z_node)
            pos[z_node] = (l * layer_spacing - 0.5, (max_neurons - layer_dims[l])/2 + i * neuron_spacing)
            node_labels[z_node] = f"z^{l}_{i}"
            
            # Post-activation node
            a_node = f"a^{l}_{i}"
            G_forward.add_node(a_node)
            G_backward.add_node(a_node)
            pos[a_node] = (l * layer_spacing, (max_neurons - layer_dims[l])/2 + i * neuron_spacing)
            node_labels[a_node] = f"a^{l}_{i}"
            
            # Connect pre-activation to post-activation
            G_forward.add_edge(z_node, a_node)
            G_backward.add_edge(a_node, z_node)
            edge_labels_forward[(z_node, a_node)] = "σ"
            edge_labels_backward[(a_node, z_node)] = "σ'"
            
            # Connect previous layer to this layer's pre-activation
            for j in range(layer_dims[l-1]):
                prev_node = f"a^{l-1}_{j}" if l > 1 else f"x_{j}"
                G_forward.add_edge(prev_node, z_node)
                G_backward.add_edge(z_node, prev_node)
                edge_labels_forward[(prev_node, z_node)] = f"w^{l}_{i,j}"
                edge_labels_backward[(z_node, prev_node)] = f"∂L/∂w^{l}_{i,j}"
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Draw forward graph
    ax1.set_title("Forward Propagation Graph", fontsize=16)
    nx.draw(G_forward, pos, with_labels=False, node_color='lightblue', 
            node_size=500, arrowsize=15, ax=ax1)
    nx.draw_networkx_labels(G_forward, pos, labels=node_labels, font_size=10, ax=ax1)
    nx.draw_networkx_edge_labels(G_forward, pos, edge_labels=edge_labels_forward, 
                                font_size=8, ax=ax1)
    
    # Draw backward graph
    ax2.set_title("Backward Propagation Graph", fontsize=16)
    nx.draw(G_backward, pos, with_labels=False, node_color='lightgreen', 
            node_size=500, arrowsize=15, ax=ax2)
    nx.draw_networkx_labels(G_backward, pos, labels=node_labels, font_size=10, ax=ax2)
    nx.draw_networkx_edge_labels(G_backward, pos, edge_labels=edge_labels_backward, 
                                font_size=8, ax=ax2)
    
    # Add weight and bias dimensions
    for l in range(1, len(layer_dims)):
        ax1.text(l * layer_spacing - 1.5, max_neurons + 0.5, 
                f"W^{l}: {layer_dims[l]}×{layer_dims[l-1]}", 
                fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
        ax1.text(l * layer_spacing - 1.5, max_neurons + 1, 
                f"b^{l}: {layer_dims[l]}×1", 
                fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
        
        ax2.text(l * layer_spacing - 1.5, max_neurons + 0.5, 
                f"∂L/∂W^{l}: {layer_dims[l]}×{layer_dims[l-1]}", 
                fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
        ax2.text(l * layer_spacing - 1.5, max_neurons + 1, 
                f"∂L/∂b^{l}: {layer_dims[l]}×1", 
                fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Add layer labels
    ax1.text(0, -0.5, f"Input Layer\n{layer_dims[0]} neurons", 
            fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    for l in range(1, len(layer_dims)-1):
        ax1.text(l * layer_spacing, -0.5, f"Hidden Layer {l}\n{layer_dims[l]} neurons", 
                fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    ax1.text((len(layer_dims)-1) * layer_spacing, -0.5, 
            f"Output Layer\n{layer_dims[-1]} neurons", 
            fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    # Add layer labels to backward graph
    ax2.text(0, -0.5, f"Input Layer\n{layer_dims[0]} neurons", 
            fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    for l in range(1, len(layer_dims)-1):
        ax2.text(l * layer_spacing, -0.5, f"Hidden Layer {l}\n{layer_dims[l]} neurons", 
                fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.text((len(layer_dims)-1) * layer_spacing, -0.5, 
            f"Output Layer\n{layer_dims[-1]} neurons", 
            fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    # Create images directory if it doesn't exist
    os.makedirs("task1_MLP/images", exist_ok=True)
    
    # Draw the computational graph for the 4-layer MLP as specified in the task
    layer_dims = [4, 6, 4, 3, 2]  # Input, Hidden1, Hidden2, Hidden3, Output
    fig = draw_mlp_computational_graph(layer_dims, save_path="task1_MLP/images/mlp_computational_graph.png")
    plt.show() 