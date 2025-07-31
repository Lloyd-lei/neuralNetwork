# MNIST Classification with MLP, CNN and Transformer Models

Visit the web page to see more result: https://lloyd-lei.github.io/neuralNetwork/task4_Transformer/model_comparison.html

## Tasks

- **task1_MLP**: Implementation of the MLP model and visualization
- **task2_CNN**: Implementation of the CNN model and visualization
- **task3_MNIST**: Training and evaluation scripts for MLP and CNN, model comparison, and inference visualization
- **task4_Transformer**: Implementation of the Transformer model with attention mechanism, training scripts, and comprehensive model comparisons

## GPU Config

- Implementation of MLP, CNN, and Transformer architectures using PyTorch
- Apple Silicon GPU acceleration (MPS)

## Performance Comparison

| Metric        | MLP    | CNN    | Transformer |
| ------------- | ------ | ------ | ----------- |
| Test Accuracy | 98.10% | 98.83% | 96.80%      |
| Training Time | 35.37s | 83.70s | 164.68s     |
| Model Size    | 2.0MB  | 177KB  | 4.35MB      |

## Model Introduction

### MLP (Multi-Layer Perceptron)

- **Architecture**: Fully connected neural network
- **Input**: 784 neurons (28×28 pixels flattened)
- **Hidden layers**: 512 and 256 neurons
- **Advantages**: Fast training, simple structure
- **Disadvantages**: No spatial information preservation, large parameter count

### CNN (Convolutional Neural Network)

- **Architecture**: Convolutional layers + pooling + fully connected
- **Features**: Spatial feature extraction, parameter sharing
- **Advantages**: Preserves spatial information, smaller model size, high accuracy
- **Disadvantages**: Longer training time, complex structure

### Transformer

- **Architecture**: Attention-based encoder with positional encoding
- **Features**: Multi-head attention, parallel processing capability
- **Configuration**:
  - Model dimension: 128
  - Attention heads: 8
  - Encoder layers: 2
  - Feed-forward dimension: 512
- **Advantages**: Parallel computation, can capture long-range dependencies
- **Disadvantages**: Large parameter count, longer training time

## Visualization Result

- Model architectures comparison
- Training and validation curves for all models
- Performance radar charts and bar comparisons
- Confusion matrices
- Inference results on sample images
- Transformer-specific visualizations:
  - Positional encoding patterns
  - Attention weight heatmaps
  - Architecture diagram with attention mechanism details

## Findings

1. **CNN remains the best choice** for MNIST classification with the highest accuracy (98.83%) and reasonable training time
2. **MLP offers the fastest training** (35.37s) while maintaining good accuracy (98.10%)
3. **Transformer shows its complexity** with the longest training time (164.68s) and largest model size (4.35MB), but achieves good accuracy (96.80%)
4. For simple image classification tasks like MNIST, traditional architectures (CNN/MLP) are more efficient than Transformers
5. Transformers would likely show their advantages on more complex, larger-scale tasks

## Requirements

- Python 3.x
- PyTorch (with MPS support for Apple Silicon)
- NumPy
- Matplotlib
- scikit-learn
- seaborn
- pandas

## Usage

### Training Individual Models

1. Train the MLP model:

   ```bash
   python task3_MNIST/train_mlp.py
   ```

2. Train the CNN model:

   ```bash
   python task3_MNIST/train_cnn.py
   ```

3. Train the Transformer model:
   ```bash
   python task4_Transformer/train_transformer.py
   ```

### Generate Visualizations

4. Generate MLP vs CNN comparison:

   ```bash
   python task3_MNIST/visualize_comparison.py
   ```

5. Generate comprehensive three-model comparison:

   ```bash
   python task4_Transformer/compare_models.py
   ```

6. Generate model inference visualizations:

   ```bash
   python task4_Transformer/model_inference_visualization.py
   ```

7. Generate Transformer-specific visualizations:
   ```bash
   python task4_Transformer/transformer_visualization.py
   ```

### View Results

8. View results in the HTML pages:
   - `task3_MNIST/model_comparison.html` (MLP vs CNN)
   - `task3_MNIST/model_inference.html` (MLP vs CNN inference)
   - `task4_Transformer/model_comparison.html` (All three models comparison)

## File Structure

```
section9B/
├── task1_MLP/
│   ├── mlp.py                     # NumPy-based MLP implementation
│   └── mlp_visualization.py       # MLP visualization tools
├── task2_CNN/
│   ├── cnn.py                     # PyTorch CNN implementation
│   └── cnn_visualization.py       # CNN visualization tools
├── task3_MNIST/
│   ├── mlp.py                     # PyTorch MLP for MNIST
│   ├── train_mlp.py               # MLP training script
│   ├── train_cnn.py               # CNN training script
│   ├── mnist_loader.py            # MNIST data loading utilities
│   ├── visualize_comparison.py    # MLP vs CNN comparison
│   └── model_inference_visualization.py
├── task4_Transformer/
│   ├── transformer.py             # Transformer implementation
│   ├── train_transformer.py       # Transformer training script
│   ├── transformer_visualization.py # Transformer-specific visualizations
│   ├── compare_models.py          # Three-model comparison
│   ├── model_inference_visualization.py # All models inference
│   └── model_comparison.html      # Comprehensive comparison report
└── data/                          # MNIST dataset storage
```

## Conclusion

This project demonstrates the evolution of deep learning architectures from simple fully-connected networks (MLP) to specialized convolutional networks (CNN) and modern attention-based models (Transformer). Each architecture has its own advantages:

- **Choose MLP** easy training and simple implementation
- **Choose CNN** for the best balance of accuracy, efficiency, and model size for image tasks
- **Choose Transformer** for parallel processing capabilities
