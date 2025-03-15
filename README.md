# MNIST Classification with MLP and CNN Models

This repository contains the implementation and comparison of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models for MNIST handwritten digit classification.

## Project Structure

- **task1_MLP**: Implementation of the MLP model and visualization
- **task2_CNN**: Implementation of the CNN model and visualization
- **task3_MNIST**: Training and evaluation scripts, model comparison, and inference visualization

## Key Features

- Implementation of MLP and CNN architectures from scratch using PyTorch
- Comprehensive training and evaluation pipeline
- Detailed performance comparison between models
- Visualization of model architectures, training curves, and inference results
- Interactive HTML pages for result presentation

## Performance Comparison

| Metric | MLP | CNN |
|--------|-----|-----|
| Test Accuracy | 98.10% | 98.83% |
| Training Time | 35.37s | 83.70s |
| Model Size | 2.0MB | 177KB |

## Visualizations

The project includes various visualizations:
- Model architectures
- Training and validation curves
- Performance radar charts
- Inference results on sample images

## Inference Results

The inference visualization shows how both models predict digits from the MNIST dataset, with color-coded results indicating correct (green) and incorrect (red) predictions.

## Conclusion

Both models perform well on the MNIST dataset, with the CNN achieving slightly higher accuracy at the cost of longer training time. The MLP model, despite its simpler architecture, requires more parameters to achieve comparable performance.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Usage

1. Train the MLP model:
   ```
   python task3_MNIST/train_mlp.py
   ```

2. Train the CNN model:
   ```
   python task3_MNIST/train_cnn.py
   ```

3. Generate comparison visualizations:
   ```
   python task3_MNIST/visualize_comparison.py
   ```

4. Visualize model inference:
   ```
   python task3_MNIST/model_inference_visualization.py
   ```

5. View results in the HTML pages:
   - `task3_MNIST/model_comparison.html`
   - `task3_MNIST/model_inference.html` 