import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task3_MNIST.mlp import MLP
from task2_CNN.cnn import CNN
from task4_Transformer.transformer import TransformerModel
from task3_MNIST.mnist_loader import load_mnist_data
from task4_Transformer.transformer_visualization import visualize_attention_on_images

# Create directory for saving images
os.makedirs("task4_Transformer/inference_images", exist_ok=True)

def load_models():
    """
    Load the trained MLP, CNN and Transformer models
    
    Returns:
        mlp_model: Trained MLP model
        cnn_model: Trained CNN model
        transformer_model: Trained Transformer model
    """
    # Load MLP model
    mlp_model = MLP(input_size=784, hidden_sizes=[512, 256], num_classes=10)
    mlp_model.load_state_dict(torch.load("task3_MNIST/mlp_model.pth", map_location=torch.device('cpu')))
    mlp_model.eval()
    
    # Load CNN model
    cnn_model = CNN(input_channels=1, num_classes=10)
    cnn_model.load_state_dict(torch.load("task3_MNIST/cnn_model.pth", map_location=torch.device('cpu')))
    cnn_model.eval()
    
    # Load Transformer model if it exists
    transformer_model = TransformerModel(
        input_dim=784,  # 28*28 for MNIST
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=10
    )
    
    try:
        transformer_model.load_state_dict(torch.load("task4_Transformer/transformer_model.pth", map_location=torch.device('cpu')))
        print("Loaded trained Transformer model")
    except:
        print("Could not load trained Transformer model, using untrained model")
    
    transformer_model.eval()
    
    return mlp_model, cnn_model, transformer_model

def get_random_samples(data_loader, num_samples=5, set_type="training"):
    """
    Get random samples from the data loader
    
    Args:
        data_loader: PyTorch DataLoader
        num_samples: Number of samples to get
        set_type: "training" or "test"
        
    Returns:
        images: List of images
        labels: List of labels
        indices: List of indices
    """
    # Get all data from the loader
    all_images = []
    all_labels = []
    
    for images, labels in data_loader:
        all_images.append(images)
        all_labels.append(labels)
    
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    
    # Select random indices
    indices = random.sample(range(len(all_images)), num_samples)
    
    # Get the selected images and labels
    selected_images = [all_images[i] for i in indices]
    selected_labels = [all_labels[i].item() for i in indices]
    
    return selected_images, selected_labels, indices

def predict_with_models(images, mlp_model, cnn_model, transformer_model):
    """
    Make predictions with all models
    
    Args:
        images: List of images
        mlp_model: MLP model
        cnn_model: CNN model
        transformer_model: Transformer model
        
    Returns:
        mlp_predictions: List of MLP predictions
        cnn_predictions: List of CNN predictions
        transformer_predictions: List of Transformer predictions
    """
    mlp_predictions = []
    cnn_predictions = []
    transformer_predictions = []
    
    with torch.no_grad():
        for image in images:
            # Add batch dimension
            image_batch = image.unsqueeze(0)
            
            # MLP prediction
            mlp_output = mlp_model(image_batch)
            mlp_pred = torch.argmax(mlp_output, dim=1).item()
            mlp_predictions.append(mlp_pred)
            
            # CNN prediction
            cnn_output = cnn_model(image_batch)
            cnn_pred = torch.argmax(cnn_output, dim=1).item()
            cnn_predictions.append(cnn_pred)
            
            # Transformer prediction
            transformer_output = transformer_model(image_batch)
            transformer_pred = torch.argmax(transformer_output, dim=1).item()
            transformer_predictions.append(transformer_pred)
    
    return mlp_predictions, cnn_predictions, transformer_predictions

def visualize_predictions(images, true_labels, mlp_predictions, cnn_predictions, transformer_predictions, indices, set_type="training"):
    """
    Visualize the predictions from all models
    
    Args:
        images: List of images
        true_labels: List of true labels
        mlp_predictions: List of MLP predictions
        cnn_predictions: List of CNN predictions
        transformer_predictions: List of Transformer predictions
        indices: List of indices
        set_type: "training" or "test"
    """
    num_samples = len(images)
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    
    # Set background color
    fig.patch.set_facecolor('#121212')
    
    for i in range(num_samples):
        # Convert image to numpy and reshape
        img = images[i].numpy()[0]
        
        # MLP row
        ax = axes[0, i]
        ax.imshow(img, cmap='gray')
        
        # Set title color based on correctness
        title_color = 'green' if mlp_predictions[i] == true_labels[i] else 'red'
        ax.set_title(f"{set_type} image [{indices[i]}] = {true_labels[i]}\nMLP prediction = {mlp_predictions[i]}", 
                    color=title_color, fontsize=10)
        ax.axis('off')
        ax.set_facecolor('#1E1E1E')
        
        # CNN row
        ax = axes[1, i]
        ax.imshow(img, cmap='gray')
        
        # Set title color based on correctness
        title_color = 'green' if cnn_predictions[i] == true_labels[i] else 'red'
        ax.set_title(f"{set_type} image [{indices[i]}] = {true_labels[i]}\nCNN prediction = {cnn_predictions[i]}", 
                    color=title_color, fontsize=10)
        ax.axis('off')
        ax.set_facecolor('#1E1E1E')
        
        # Transformer row
        ax = axes[2, i]
        ax.imshow(img, cmap='gray')
        
        # Set title color based on correctness
        title_color = 'green' if transformer_predictions[i] == true_labels[i] else 'red'
        ax.set_title(f"{set_type} image [{indices[i]}] = {true_labels[i]}\nTransformer prediction = {transformer_predictions[i]}", 
                    color=title_color, fontsize=10)
        ax.axis('off')
        ax.set_facecolor('#1E1E1E')
    
    # Add row labels
    fig.text(0.02, 0.83, 'MLP Model', fontsize=14, fontweight='bold', color='#FF9671', rotation=90, va='center')
    fig.text(0.02, 0.5, 'CNN Model', fontsize=14, fontweight='bold', color='#00D2FC', rotation=90, va='center')
    fig.text(0.02, 0.17, 'Transformer Model', fontsize=14, fontweight='bold', color='#9B59B6', rotation=90, va='center')
    
    # Add title
    plt.suptitle(f'Model Predictions on {set_type.capitalize()} Images', fontsize=16, color='white', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, top=0.9)
    
    # Save figure
    plt.savefig(f"task4_Transformer/inference_images/{set_type}_predictions_all.png", dpi=300, bbox_inches='tight', facecolor='#121212')
    plt.close()

def create_combined_visualization(train_images, train_labels, train_indices, 
                                 test_images, test_labels, test_indices,
                                 mlp_model, cnn_model, transformer_model):
    """
    Create a combined visualization with both training and test images for all models
    
    Args:
        train_images: List of training images
        train_labels: List of training labels
        train_indices: List of training indices
        test_images: List of test images
        test_labels: List of test labels
        test_indices: List of test indices
        mlp_model: MLP model
        cnn_model: CNN model
        transformer_model: Transformer model
    """
    # Get predictions
    mlp_train_preds, cnn_train_preds, transformer_train_preds = predict_with_models(
        train_images, mlp_model, cnn_model, transformer_model
    )
    
    mlp_test_preds, cnn_test_preds, transformer_test_preds = predict_with_models(
        test_images, mlp_model, cnn_model, transformer_model
    )
    
    # Combine all data
    all_images = train_images + test_images
    all_labels = train_labels + test_labels
    all_indices = train_indices + test_indices
    all_mlp_preds = mlp_train_preds + mlp_test_preds
    all_cnn_preds = cnn_train_preds + cnn_test_preds
    all_transformer_preds = transformer_train_preds + transformer_test_preds
    all_types = ["training"] * len(train_images) + ["test"] * len(test_images)
    
    # Create figure
    num_samples = len(all_images)
    rows = 3  # MLP, CNN, Transformer
    cols = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    # Set background color
    fig.patch.set_facecolor('#121212')
    
    # Plot all predictions
    for i in range(num_samples):
        # Convert image to numpy and reshape
        img = all_images[i].numpy()[0]
        
        # MLP row
        ax = axes[0, i]
        ax.imshow(img, cmap='gray')
        
        # Set title color based on correctness
        title_color = 'green' if all_mlp_preds[i] == all_labels[i] else 'red'
        ax.set_title(f"{all_types[i]} image [{all_indices[i]}] = {all_labels[i]}\nMLP prediction = {all_mlp_preds[i]}", 
                    color=title_color, fontsize=10)
        ax.axis('off')
        ax.set_facecolor('#1E1E1E')
        
        # CNN row
        ax = axes[1, i]
        ax.imshow(img, cmap='gray')
        
        # Set title color based on correctness
        title_color = 'green' if all_cnn_preds[i] == all_labels[i] else 'red'
        ax.set_title(f"{all_types[i]} image [{all_indices[i]}] = {all_labels[i]}\nCNN prediction = {all_cnn_preds[i]}", 
                    color=title_color, fontsize=10)
        ax.axis('off')
        ax.set_facecolor('#1E1E1E')
        
        # Transformer row
        ax = axes[2, i]
        ax.imshow(img, cmap='gray')
        
        # Set title color based on correctness
        title_color = 'green' if all_transformer_preds[i] == all_labels[i] else 'red'
        ax.set_title(f"{all_types[i]} image [{all_indices[i]}] = {all_labels[i]}\nTransformer prediction = {all_transformer_preds[i]}", 
                    color=title_color, fontsize=10)
        ax.axis('off')
        ax.set_facecolor('#1E1E1E')
    
    # Add row labels
    fig.text(0.02, 0.83, 'MLP Model', fontsize=14, fontweight='bold', color='#FF9671', rotation=90, va='center')
    fig.text(0.02, 0.5, 'CNN Model', fontsize=14, fontweight='bold', color='#00D2FC', rotation=90, va='center')
    fig.text(0.02, 0.17, 'Transformer Model', fontsize=14, fontweight='bold', color='#9B59B6', rotation=90, va='center')
    
    # Add title
    plt.suptitle('Model Predictions on MNIST Images', fontsize=16, color='white', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, top=0.9)
    
    # Save figure
    plt.savefig("task4_Transformer/inference_images/combined_predictions_all.png", dpi=300, bbox_inches='tight', facecolor='#121212')
    plt.close()

def visualize_transformer_attention(test_images, test_labels, transformer_model):
    """
    Visualize attention weights from the transformer model
    
    Args:
        test_images: List of test images
        test_labels: List of test labels
        transformer_model: Transformer model
    """
    # Convert list of images to tensor
    images_tensor = torch.stack(test_images)
    
    # Get predictions
    with torch.no_grad():
        outputs = transformer_model(images_tensor)
        _, predictions = torch.max(outputs.data, 1)
    
    # Visualize attention on images
    try:
        visualize_attention_on_images(
            model=transformer_model,
            images=images_tensor,
            labels=test_labels,
            predictions=predictions.tolist(),
            save_path="task4_Transformer/inference_images/transformer_attention.png"
        )
    except Exception as e:
        print(f"Could not visualize transformer attention: {e}")
        print("This is expected if the model hasn't been trained yet or doesn't have attention hooks.")

def main():
    """
    Main function
    """
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Check if MPS is available for Apple Silicon
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST data...")
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=64, use_cuda=False)
    
    # Load models
    print("Loading trained models...")
    mlp_model, cnn_model, transformer_model = load_models()
    
    # Get random samples
    print("Getting random samples...")
    train_images, train_labels, train_indices = get_random_samples(train_loader, num_samples=5, set_type="training")
    test_images, test_labels, test_indices = get_random_samples(test_loader, num_samples=5, set_type="test")
    
    # Create visualizations
    print("Creating visualizations...")
    create_combined_visualization(
        train_images, train_labels, train_indices,
        test_images, test_labels, test_indices,
        mlp_model, cnn_model, transformer_model
    )
    
    # Visualize transformer attention
    print("Visualizing transformer attention...")
    visualize_transformer_attention(test_images, test_labels, transformer_model)
    
    print("Visualizations saved to task4_Transformer/inference_images/")

if __name__ == "__main__":
    main() 