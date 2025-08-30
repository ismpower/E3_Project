#!/usr/bin/env python3
"""
Simplified E3 Training Script - Minimal Dependencies
==================================================

Trains a basic neural network on BYU UNP anomalies without complex dependencies.
Uses only numpy and basic Python libraries.

Author: E3 Development Team  
Date: 2025-07-31
Status: MINIMAL WORKING VERSION
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Check for required files
def check_prerequisites():
    """Check if required files from integration step exist."""
    required_files = [
        'byu_unp_anomaly_dataset.json',
        'byu_unp_graph_dataset.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  • {file}")
        print("\nPlease run the integration script first:")
        print("  python byu_unp_integration.py")
        return False
    
    return True

# Early prerequisite check
if not check_prerequisites():
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('e3_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('E3_Training')

class SimpleNeuralNetwork:
    """
    Simple feedforward neural network for anomaly prediction.
    Pure numpy implementation - no external ML libraries required.
    """
    
    def __init__(self, input_size=5, hidden_size=32, output_sizes={'temp': 1, 'iaw': 1, 'class': 2}):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_sizes = output_sizes
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Output heads for different tasks
        self.W_temp = np.random.randn(hidden_size, output_sizes['temp']) * np.sqrt(2.0 / hidden_size)
        self.b_temp = np.zeros((1, output_sizes['temp']))
        
        self.W_iaw = np.random.randn(hidden_size, output_sizes['iaw']) * np.sqrt(2.0 / hidden_size)
        self.b_iaw = np.zeros((1, output_sizes['iaw']))
        
        self.W_class = np.random.randn(hidden_size, output_sizes['class']) * np.sqrt(2.0 / hidden_size)
        self.b_class = np.zeros((1, output_sizes['class']))
        
        logger.info(f"Initialized neural network: {input_size} -> {hidden_size} -> multi-output")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        # Clamp x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Multi-task outputs
        outputs = {}
        outputs['temp'] = self.sigmoid(np.dot(self.a1, self.W_temp) + self.b_temp)
        outputs['iaw'] = np.maximum(0, np.dot(self.a1, self.W_iaw) + self.b_iaw)  # ReLU for amplitude
        outputs['class'] = self.softmax(np.dot(self.a1, self.W_class) + self.b_class)
        
        return outputs
    
    def compute_loss(self, outputs, targets):
        n_samples = len(targets['temp_mask'])
        total_loss = 0
        
        # Temperature loss (for temperature anomaly cases)
        temp_mask = targets['temp_mask']
        if np.any(temp_mask):
            temp_pred = outputs['temp'][temp_mask]
            temp_true = targets['temp_true'][temp_mask]
            temp_loss = np.mean((temp_pred - temp_true) ** 2)
            total_loss += temp_loss
        
        # IAW loss (for IAW anomaly cases)
        iaw_mask = targets['iaw_mask']
        if np.any(iaw_mask):
            iaw_pred = outputs['iaw'][iaw_mask]
            iaw_true = targets['iaw_true'][iaw_mask]
            iaw_loss = np.mean((iaw_pred - iaw_true) ** 2)
            total_loss += iaw_loss
        
        # Classification loss (cross-entropy)
        class_pred = outputs['class']
        class_true = targets['class_true']
        class_loss = -np.mean(np.sum(class_true * np.log(class_pred + 1e-15), axis=1))
        total_loss += 0.1 * class_loss  # Weighted
        
        return total_loss / 3  # Average across tasks
    
    def backward(self, X, outputs, targets, learning_rate=0.001):
        n_samples = X.shape[0]
        
        # Compute output gradients
        grad_temp = np.zeros_like(outputs['temp'])
        grad_iaw = np.zeros_like(outputs['iaw'])
        grad_class = outputs['class'] - targets['class_true']
        
        # Temperature gradient
        temp_mask = targets['temp_mask']
        if np.any(temp_mask):
            grad_temp[temp_mask] = 2 * (outputs['temp'][temp_mask] - targets['temp_true'][temp_mask]) / np.sum(temp_mask)
        
        # IAW gradient
        iaw_mask = targets['iaw_mask']
        if np.any(iaw_mask):
            grad_iaw[iaw_mask] = 2 * (outputs['iaw'][iaw_mask] - targets['iaw_true'][iaw_mask]) / np.sum(iaw_mask)
        
        # Backpropagate to hidden layer
        grad_a1_temp = np.dot(grad_temp, self.W_temp.T)
        grad_a1_iaw = np.dot(grad_iaw, self.W_iaw.T)
        grad_a1_class = 0.1 * np.dot(grad_class, self.W_class.T)  # Weighted
        
        grad_a1 = grad_a1_temp + grad_a1_iaw + grad_a1_class
        grad_z1 = grad_a1 * self.relu_derivative(self.z1)
        
        # Compute weight gradients
        grad_W1 = np.dot(X.T, grad_z1) / n_samples
        grad_b1 = np.mean(grad_z1, axis=0, keepdims=True)
        
        grad_W_temp = np.dot(self.a1.T, grad_temp) / n_samples
        grad_b_temp = np.mean(grad_temp, axis=0, keepdims=True)
        
        grad_W_iaw = np.dot(self.a1.T, grad_iaw) / n_samples
        grad_b_iaw = np.mean(grad_iaw, axis=0, keepdims=True)
        
        grad_W_class = 0.1 * np.dot(self.a1.T, grad_class) / n_samples
        grad_b_class = 0.1 * np.mean(grad_class, axis=0, keepdims=True)
        
        # Update weights
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        
        self.W_temp -= learning_rate * grad_W_temp
        self.b_temp -= learning_rate * grad_b_temp
        
        self.W_iaw -= learning_rate * grad_W_iaw
        self.b_iaw -= learning_rate * grad_b_iaw
        
        self.W_class -= learning_rate * grad_W_class
        self.b_class -= learning_rate * grad_b_class

def load_byu_dataset():
    """Load the BYU dataset created by integration script."""
    logger.info("Loading BYU dataset...")
    
    try:
        with open('byu_unp_graph_dataset.json', 'r') as f:
            graph_data = json.load(f)
        
        logger.info(f"Loaded {len(graph_data)} data points")
        return graph_data
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

def prepare_training_data(graph_data):
    """Convert graph data to training arrays."""
    logger.info("Preparing training data...")
    
    X = []  # Input features
    temp_targets = []
    iaw_targets = []
    class_targets = []
    temp_masks = []
    iaw_masks = []
    
    for item in graph_data:
        # Input features
        X.append(item['node_features'])
        
        # Determine anomaly type and create targets
        anomaly_type = item['targets']['anomaly_class']
        
        if anomaly_type == 'temperature':
            # Temperature anomaly case
            temp_targets.append(item['targets']['temperature_ratio'])
            iaw_targets.append(0.0)  # Dummy value
            class_targets.append([1, 0])  # One-hot: [temp_anomaly, iaw_anomaly]
            temp_masks.append(True)
            iaw_masks.append(False)
        else:
            # IAW anomaly case
            temp_targets.append(0.0)  # Dummy value
            iaw_targets.append(item['targets']['iaw_amplitude'])
            class_targets.append([0, 1])  # One-hot: [temp_anomaly, iaw_anomaly]
            temp_masks.append(False)
            iaw_masks.append(True)
    
    # Convert to numpy arrays
    X = np.array(X)
    temp_targets = np.array(temp_targets).reshape(-1, 1)
    iaw_targets = np.array(iaw_targets).reshape(-1, 1)
    class_targets = np.array(class_targets)
    temp_masks = np.array(temp_masks)
    iaw_masks = np.array(iaw_masks)
    
    # Normalize input features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
    X_normalized = (X - X_mean) / X_std
    
    targets = {
        'temp_true': temp_targets,
        'iaw_true': iaw_targets,
        'class_true': class_targets,
        'temp_mask': temp_masks,
        'iaw_mask': iaw_masks
    }
    
    logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Temperature cases: {np.sum(temp_masks)}")
    logger.info(f"IAW cases: {np.sum(iaw_masks)}")
    
    return X_normalized, targets, (X_mean, X_std)

def train_model(model, X, targets, epochs=200, learning_rate=0.001):
    """Train the neural network model."""
    logger.info(f"Training model for {epochs} epochs...")
    
    history = {
        'epoch': [],
        'loss': [],
        'temp_mae': [],
        'iaw_mae': []
    }
    
    n_samples = X.shape[0]
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model.forward(X)
        
        # Compute loss
        loss = model.compute_loss(outputs, targets)
        
        # Backward pass
        model.backward(X, outputs, targets, learning_rate)
        
        # Compute metrics
        temp_mask = targets['temp_mask']
        iaw_mask = targets['iaw_mask']
        
        temp_mae = 0
        if np.any(temp_mask):
            temp_mae = np.mean(np.abs(outputs['temp'][temp_mask] - targets['temp_true'][temp_mask]))
        
        iaw_mae = 0
        if np.any(iaw_mask):
            iaw_mae = np.mean(np.abs(outputs['iaw'][iaw_mask] - targets['iaw_true'][iaw_mask]))
        
        # Record history
        history['epoch'].append(epoch)
        history['loss'].append(loss)
        history['temp_mae'].append(temp_mae)
        history['iaw_mae'].append(iaw_mae)
        
        # Logging
        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch}: Loss={loss:.4f}, Temp MAE={temp_mae:.4f}, IAW MAE={iaw_mae:.4f}")
    
    logger.info("Training completed")
    return history

def evaluate_model(model, X, targets):
    """Evaluate the trained model."""
    logger.info("Evaluating model performance...")
    
    outputs = model.forward(X)
    final_loss = model.compute_loss(outputs, targets)
    
    # Temperature metrics
    temp_mask = targets['temp_mask']
    temp_mae = 0
    temp_r2 = 0
    if np.any(temp_mask):
        temp_pred = outputs['temp'][temp_mask].flatten()
        temp_true = targets['temp_true'][temp_mask].flatten()
        temp_mae = np.mean(np.abs(temp_pred - temp_true))
        temp_var = np.var(temp_true)
        temp_r2 = 1 - np.sum((temp_true - temp_pred)**2) / (np.sum((temp_true - np.mean(temp_true))**2) + 1e-8)
    
    # IAW metrics
    iaw_mask = targets['iaw_mask']
    iaw_mae = 0
    iaw_r2 = 0
    if np.any(iaw_mask):
        iaw_pred = outputs['iaw'][iaw_mask].flatten()
        iaw_true = targets['iaw_true'][iaw_mask].flatten()
        iaw_mae = np.mean(np.abs(iaw_pred - iaw_true))
        iaw_r2 = 1 - np.sum((iaw_true - iaw_pred)**2) / (np.sum((iaw_true - np.mean(iaw_true))**2) + 1e-8)
    
    # Classification accuracy
    class_pred = np.argmax(outputs['class'], axis=1)
    class_true = np.argmax(targets['class_true'], axis=1)
    class_accuracy = np.mean(class_pred == class_true)
    
    metrics = {
        'final_loss': final_loss,
        'temp_mae': temp_mae,
        'temp_r2': temp_r2,
        'iaw_mae': iaw_mae,
        'iaw_r2': iaw_r2,
        'classification_accuracy': class_accuracy
    }
    
    logger.info("=== MODEL EVALUATION RESULTS ===")
    logger.info(f"Final Loss: {final_loss:.4f}")
    logger.info(f"Temperature MAE: {temp_mae:.4f}, R²: {temp_r2:.4f}")
    logger.info(f"IAW MAE: {iaw_mae:.4f}, R²: {iaw_r2:.4f}")
    logger.info(f"Classification Accuracy: {class_accuracy:.2%}")
    
    return metrics

def plot_training_results(history, metrics):
    """Plot training curves and results."""
    logger.info("Generating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    axes[0, 0].plot(history['epoch'], history['loss'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    
    # Temperature MAE
    axes[0, 1].plot(history['epoch'], history['temp_mae'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Temperature MAE')
    axes[0, 1].set_title('Temperature Prediction Error')
    axes[0, 1].grid(True)
    
    # IAW MAE
    axes[1, 0].plot(history['epoch'], history['iaw_mae'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IAW MAE')
    axes[1, 0].set_title('IAW Prediction Error')
    axes[1, 0].grid(True)
    
    # Performance summary
    summary_text = f"""FINAL PERFORMANCE

Temperature Anomaly:
  MAE: {metrics['temp_mae']:.4f}
  R²:  {metrics['temp_r2']:.3f}

IAW Anomaly:
  MAE: {metrics['iaw_mae']:.4f}  
  R²:  {metrics['iaw_r2']:.3f}

Classification:
  Accuracy: {metrics['classification_accuracy']:.1%}

Overall Status: {'GOOD' if metrics['temp_r2'] > 0.6 else 'NEEDS IMPROVEMENT'}"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue'))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('e3_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("✓ Training plots saved: e3_training_results.png")

def save_trained_model(model, metrics, normalization_params):
    """Save the trained model and results."""
    logger.info("Saving trained model...")
    
    # Save model weights
    model_data = {
        'model_type': 'SimpleNeuralNetwork',
        'architecture': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'output_sizes': model.output_sizes
        },
        'weights': {
            'W1': model.W1.tolist(),
            'b1': model.b1.tolist(),
            'W_temp': model.W_temp.tolist(),
            'b_temp': model.b_temp.tolist(),
            'W_iaw': model.W_iaw.tolist(),
            'b_iaw': model.b_iaw.tolist(),
            'W_class': model.W_class.tolist(),
            'b_class': model.b_class.tolist()
        },
        'normalization': {
            'X_mean': normalization_params[0].tolist(),
            'X_std': normalization_params[1].tolist()
        },
        'training_completed': datetime.now().isoformat()
    }
    
    with open('e3_simple_model.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    
    # Save training results
    results = {
        'training_info': {
            'dataset': 'BYU_UNP_2023',
            'model_type': 'SimpleNeuralNetwork',
            'training_date': datetime.now().isoformat(),
            'anomaly_types': ['temperature_persistence', 'ion_acoustic_waves']
        },
        'performance_metrics': metrics,
        'model_file': 'e3_simple_model.json',
        'training_status': 'COMPLETED'
    }
    
    with open('e3_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Model saved: e3_simple_model.json")
    logger.info("✓ Results saved: e3_training_results.json")

def test_anomaly_predictions(model, normalization_params):
    """Test the model on specific BYU scenarios."""
    logger.info("Testing anomaly predictions...")
    
    X_mean, X_std = normalization_params
    
    # Test cases based on BYU experiments
    test_cases = [
        {
            'name': 'Unmagnetized_48K',
            'features': [20, 40.078, 48, 0, 1.0],
            'expected': 'Normal temperature decay'
        },
        {
            'name': 'Magnetized_200G_48K',
            'features': [20, 40.078, 48, 200, 1.0],
            'expected': 'Temperature anomaly'
        },
        {
            'name': 'IAW_183G_100K',
            'features': [20, 40.078, 100, 183, 1.0],
            'expected': 'IAW oscillations'
        }
    ]
    
    logger.info("Testing specific scenarios:")
    
    for case in test_cases:
        # Normalize input
        x = np.array([case['features']])
        x_norm = (x - X_mean) / X_std
        
        # Predict
        outputs = model.forward(x_norm)
        
        temp_pred = outputs['temp'][0, 0]
        iaw_pred = outputs['iaw'][0, 0]
        class_probs = outputs['class'][0]
        
        logger.info(f"\n{case['name']}:")
        logger.info(f"  Expected: {case['expected']}")
        logger.info(f"  Temperature ratio: {temp_pred:.3f}")
        logger.info(f"  IAW amplitude: {iaw_pred:.3f}")
        logger.info(f"  Classification: Temp={class_probs[0]:.2f}, IAW={class_probs[1]:.2f}")

def main():
    """Main training workflow."""
    logger.info("=== E3 SIMPLE TRAINING STARTING ===")
    
    try:
        # Load dataset
        graph_data = load_byu_dataset()
        if graph_data is None:
            raise Exception("Failed to load dataset")
        
        # Prepare training data
        X, targets, normalization_params = prepare_training_data(graph_data)
        
        # Initialize model
        model = SimpleNeuralNetwork()
        
        # Train model
        history = train_model(model, X, targets, epochs=200)
        
        # Evaluate model
        metrics = evaluate_model(model, X, targets)
        
        # Plot results
        plot_training_results(history, metrics)
        
        # Save model
        save_trained_model(model, metrics, normalization_params)
        
        # Test predictions
        test_anomaly_predictions(model, normalization_params)
        
        # Success summary
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        print("\n" + "="*60)
        print("🎉 E3 TRAINING SUCCESSFUL!")
        print("="*60)
        print(f"Temperature Prediction R²: {metrics['temp_r2']:.3f}")
        print(f"IAW Prediction R²: {metrics['iaw_r2']:.3f}")
        print(f"Classification Accuracy: {metrics['classification_accuracy']:.1%}")
        print("\nFiles generated:")
        print("  • e3_simple_model.json")
        print("  • e3_training_results.json")
        print("  • e3_training_results.png")
        print("  • e3_training.log")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ TRAINING FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
