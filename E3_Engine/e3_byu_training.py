#!/usr/bin/env python3
"""
E3 Engine Training on BYU UNP Anomalies
=======================================

Trains the Elemental Embedding Engine on the verified BYU ultracold neutral 
plasma anomalies to predict temperature-dependent regime transitions.

Author: E3 Development Team
Date: 2025-07-31
Anomaly Target: Magnetized UNP expansion anomalies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
from sklearn.metrics import mean_absolute_error, r2_score
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('E3_BYU_Training')

class E3NetworkBYU(nn.Module):
    """
    E3 Network specialized for BYU UNP anomaly prediction.
    
    Architecture designed to learn context-aware elemental embeddings
    that can predict magnetized plasma behavior anomalies.
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, embedding_dim: int = 32):
        super(E3NetworkBYU, self).__init__()
        
        # Elemental embedding layer - learns context-aware Ca representations
        self.element_embedding = nn.Linear(input_dim, embedding_dim)
        
        # Graph convolution layers for local environment awareness
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        
        # Context attention mechanism
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)
        
        # Physics-informed prediction heads
        self.temperature_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Normalized temperature ratio
        )
        
        self.iaw_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(), 
            nn.Linear(32, 1),
            nn.ReLU()  # Oscillation amplitude
        )
        
        # Anomaly classification head
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [temperature_anomaly, iaw_anomaly]
            nn.Softmax(dim=1)
        )
        
        logger.info(f"Initialized E3 Network: {input_dim} -> {embedding_dim} -> predictions")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Create context-aware elemental embeddings
        h = self.element_embedding(x.float())
        
        # Graph convolutions for environment awareness
        h = torch.relu(self.conv1(h, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = self.conv3(h, edge_index)
        
        # Global pooling for graph-level representation
        h_global = global_mean_pool(h, batch)
        
        # Self-attention for context refinement
        h_attended, _ = self.attention(h_global.unsqueeze(1), h_global.unsqueeze(1), h_global.unsqueeze(1))
        h_final = h_attended.squeeze(1)
        
        # Multi-task predictions
        temp_pred = self.temperature_predictor(h_final)
        iaw_pred = self.iaw_predictor(h_final)
        anomaly_class = self.anomaly_classifier(h_final)
        
        return {
            'temperature': temp_pred,
            'iaw_amplitude': iaw_pred,
            'anomaly_class': anomaly_class,
            'embedding': h_final
        }

class BYUAnomalyTrainer:
    """
    Trainer for E3 Engine on BYU UNP anomalies.
    """
    
    def __init__(self, model: E3NetworkBYU, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.training_history = {
            'epoch': [],
            'total_loss': [],
            'temp_loss': [],
            'iaw_loss': [],
            'class_loss': [],
            'temp_mae': [],
            'iaw_mae': []
        }
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def prepare_data(self, graph_dataset: List[Data]) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.
        """
        # Split dataset
        n_total = len(graph_dataset)
        n_train = int(0.8 * n_total)
        
        train_data = graph_dataset[:n_train]
        val_data = graph_dataset[n_train:]
        
        # Add anomaly type labels for classification
        for data in train_data + val_data:
            if 'temperature' in data.anomaly_type:
                data.anomaly_label = torch.tensor([1, 0], dtype=torch.float)  # Temperature anomaly
            else:
                data.anomaly_label = torch.tensor([0, 1], dtype=torch.float)  # IAW anomaly
        
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
        
        logger.info(f"Data prepared: {len(train_data)} training, {len(val_data)} validation samples")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        temp_loss_sum = 0
        iaw_loss_sum = 0
        class_loss_sum = 0
        
        temp_preds, temp_targets = [], []
        iaw_preds, iaw_targets = [], []
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate losses based on anomaly type
            temp_loss = torch.tensor(0.0, device=self.device)
            iaw_loss = torch.tensor(0.0, device=self.device)
            
            temp_mask = batch.anomaly_label[:, 0] == 1  # Temperature anomaly samples
            iaw_mask = batch.anomaly_label[:, 1] == 1   # IAW anomaly samples
            
            if temp_mask.any():
                temp_loss = self.mse_loss(outputs['temperature'][temp_mask], batch.y[temp_mask])
                temp_preds.extend(outputs['temperature'][temp_mask].cpu().detach().numpy())
                temp_targets.extend(batch.y[temp_mask].cpu().detach().numpy())
            
            if iaw_mask.any():
                iaw_loss = self.mse_loss(outputs['iaw_amplitude'][iaw_mask], batch.y[iaw_mask])
                iaw_preds.extend(outputs['iaw_amplitude'][iaw_mask].cpu().detach().numpy())
                iaw_targets.extend(batch.y[iaw_mask].cpu().detach().numpy())
            
            # Classification loss
            class_loss = self.ce_loss(outputs['anomaly_class'], batch.anomaly_label.argmax(dim=1))
            
            # Total loss
            loss = temp_loss + iaw_loss + 0.1 * class_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            temp_loss_sum += temp_loss.item()
            iaw_loss_sum += iaw_loss.item()
            class_loss_sum += class_loss.item()
        
        # Calculate metrics
        temp_mae = mean_absolute_error(temp_targets, temp_preds) if temp_targets else 0
        iaw_mae = mean_absolute_error(iaw_targets, iaw_preds) if iaw_targets else 0
        
        return {
            'total_loss': total_loss / len(train_loader),
            'temp_loss': temp_loss_sum / len(train_loader),
            'iaw_loss': iaw_loss_sum / len(train_loader),
            'class_loss': class_loss_sum / len(train_loader),
            'temp_mae': temp_mae,
            'iaw_mae': iaw_mae
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model performance.
        """
        self.model.eval()
        total_loss = 0
        temp_preds, temp_targets = [], []
        iaw_preds, iaw_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                
                # Calculate losses
                temp_mask = batch.anomaly_label[:, 0] == 1
                iaw_mask = batch.anomaly_label[:, 1] == 1
                
                temp_loss = torch.tensor(0.0, device=self.device)
                iaw_loss = torch.tensor(0.0, device=self.device)
                
                if temp_mask.any():
                    temp_loss = self.mse_loss(outputs['temperature'][temp_mask], batch.y[temp_mask])
                    temp_preds.extend(outputs['temperature'][temp_mask].cpu().numpy())
                    temp_targets.extend(batch.y[temp_mask].cpu().numpy())
                
                if iaw_mask.any():
                    iaw_loss = self.mse_loss(outputs['iaw_amplitude'][iaw_mask], batch.y[iaw_mask])
                    iaw_preds.extend(outputs['iaw_amplitude'][iaw_mask].cpu().numpy())
                    iaw_targets.extend(batch.y[iaw_mask].cpu().numpy())
                
                class_loss = self.ce_loss(outputs['anomaly_class'], batch.anomaly_label.argmax(dim=1))
                loss = temp_loss + iaw_loss + 0.1 * class_loss
                total_loss += loss.item()
        
        temp_mae = mean_absolute_error(temp_targets, temp_preds) if temp_targets else 0
        iaw_mae = mean_absolute_error(iaw_targets, iaw_preds) if iaw_targets else 0
        temp_r2 = r2_score(temp_targets, temp_preds) if temp_targets else 0
        iaw_r2 = r2_score(iaw_targets, iaw_preds) if iaw_targets else 0
        
        return {
            'val_loss': total_loss / len(val_loader),
            'temp_mae': temp_mae,
            'iaw_mae': iaw_mae,
            'temp_r2': temp_r2,
            'iaw_r2': iaw_r2
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        """
        Full training loop.
        """
        logger.info(f"Starting training for {epochs} epochs")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Update history
            self.training_history['epoch'].append(epoch)
            self.training_history['total_loss'].append(train_metrics['total_loss'])
            self.training_history['temp_loss'].append(train_metrics['temp_loss'])
            self.training_history['iaw_loss'].append(train_metrics['iaw_loss'])
            self.training_history['class_loss'].append(train_metrics['class_loss'])
            self.training_history['temp_mae'].append(val_metrics['temp_mae'])
            self.training_history['iaw_mae'].append(val_metrics['iaw_mae'])
            
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'e3_byu_best_model.pt')
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_metrics['total_loss']:.4f}, "
                          f"Val Loss={val_metrics['val_loss']:.4f}, "
                          f"Temp MAE={val_metrics['temp_mae']:.4f}, "
                          f"IAW MAE={val_metrics['iaw_mae']:.4f}")
            
            if patience_counter >= 20:
                logger.info("Early stopping triggered")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('e3_byu_best_model.pt'))
        logger.info("Training completed")
    
    def analyze_embeddings(self, data_loader: DataLoader) -> Dict:
        """
        Analyze learned elemental embeddings for different conditions.
        """
        self.model.eval()
        embeddings = []
        conditions = []
        anomaly_types = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                
                embeddings.append(outputs['embedding'].cpu().numpy())
                
                # Extract experimental conditions
                for i in range(len(batch.x)):
                    conditions.append({
                        'atomic_number': batch.x[i][0].item(),
                        'atomic_mass': batch.x[i][1].item(), 
                        'electron_temp': batch.x[i][2].item(),
                        'magnetic_field': batch.x[i][3].item(),
                        'density': batch.x[i][4].item()
                    })
                    anomaly_types.append(batch.anomaly_type[i] if hasattr(batch, 'anomaly_type') else 'unknown')
        
        embeddings = np.vstack(embeddings)
        
        return {
            'embeddings': embeddings,
            'conditions': conditions,
            'anomaly_types': anomaly_types
        }
    
    def plot_training_curves(self):
        """
        Plot training curves and save results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['total_loss'], label='Total Loss')
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['temp_loss'], label='Temperature Loss')
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['iaw_loss'], label='IAW Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE curves
        axes[0, 1].plot(self.training_history['epoch'], self.training_history['temp_mae'], label='Temperature MAE')
        axes[0, 1].plot(self.training_history['epoch'], self.training_history['iaw_mae'], label='IAW MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Classification loss
        axes[1, 0].plot(self.training_history['epoch'], self.training_history['class_loss'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Classification Loss')
        axes[1, 0].set_title('Anomaly Classification Loss')
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].text(0.5, 0.5, f'Final Performance:\nTemp MAE: {self.training_history["temp_mae"][-1]:.4f}\nIAW MAE: {self.training_history["iaw_mae"][-1]:.4f}', 
                       transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 1].set_title('Final Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('e3_byu_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def test_anomaly_predictions(model: E3NetworkBYU, device: str):
    """
    Test the trained model on specific BYU anomaly scenarios.
    """
    model.eval()
    
    # Test cases from BYU experiments
    test_cases = [
        # Unmagnetized baseline (should predict normal behavior)
        {
            'name': 'Unmagnetized_48K',
            'features': [20, 40.078, 48, 0, 1.0],  # Ca, 48K, no B-field
            'expected': 'Normal temperature decay'
        },
        # Magnetized anomaly (should predict elevated temperature)
        {
            'name': 'Magnetized_200G_48K', 
            'features': [20, 40.078, 48, 200, 1.0],  # Ca, 48K, 200G B-field
            'expected': 'Elevated temperature persistence'
        },
        # IAW anomaly (transverse direction)
        {
            'name': 'Transverse_IAW_183G_100K',
            'features': [20, 40.078, 100, 183, 1.0],  # Ca, 100K, 183G transverse
            'expected': 'Ion acoustic wave oscillations'
        }
    ]
    
    logger.info("Testing E3 model on BYU anomaly scenarios:")
    
    with torch.no_grad():
        for case in test_cases:
            # Create test data
            x = torch.tensor([case['features']], dtype=torch.float).to(device)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
            batch = torch.zeros(1, dtype=torch.long).to(device)
            
            # Create mock data object
            class TestData:
                def __init__(self):
                    self.x = x
                    self.edge_index = edge_index  
                    self.batch = batch
            
            test_data = TestData()
            
            # Predict
            outputs = model(test_data)
            
            temp_pred = outputs['temperature'].cpu().item()
            iaw_pred = outputs['iaw_amplitude'].cpu().item()
            anomaly_class = outputs['anomaly_class'].cpu().numpy()
            
            logger.info(f"\n{case['name']}:")
            logger.info(f"  Expected: {case['expected']}")
            logger.info(f"  Temperature ratio: {temp_pred:.3f}")
            logger.info(f"  IAW amplitude: {iaw_pred:.3f}")
            logger.info(f"  Anomaly classification: Temp={anomaly_class[0][0]:.3f}, IAW={anomaly_class[0][1]:.3f}")

def main():
    """
    Main integration and training pipeline.
    """
    logger.info("=== E3 ENGINE TRAINING ON BYU UNP ANOMALIES ===")
    
    # Load integrated dataset
    try:
        graph_dataset = torch.load('byu_unp_e3_dataset.pt')
        logger.info(f"Loaded {len(graph_dataset)} samples from integrated dataset")
    except FileNotFoundError:
        logger.error("Dataset not found. Please run BYU integration script first.")
        return
    
    # Initialize model
    model = E3NetworkBYU(input_dim=5, hidden_dim=64, embedding_dim=32)
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = BYUAnomalyTrainer(model, device)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(graph_dataset)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=100)
    
    # Final validation
    final_metrics = trainer.validate(val_loader)
    logger.info("=== FINAL PERFORMANCE ===")
    logger.info(f"Temperature Prediction MAE: {final_metrics['temp_mae']:.4f}")
    logger.info(f"Temperature Prediction R²: {final_metrics['temp_r2']:.4f}")
    logger.info(f"IAW Prediction MAE: {final_metrics['iaw_mae']:.4f}")
    logger.info(f"IAW Prediction R²: {final_metrics['iaw_r2']:.4f}")
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Analyze embeddings
    embedding_analysis = trainer.analyze_embeddings(val_loader)
    logger.info(f"Analyzed {len(embedding_analysis['embeddings'])} elemental embeddings")
    
    # Test on specific BYU scenarios
    test_anomaly_predictions(model, device)
    
    # Save final results
    results = {
        'model_architecture': 'E3NetworkBYU',
        'training_dataset': 'BYU_UNP_2023',
        'anomalies_learned': [
            'Elevated ion temperature persistence in magnetized plasmas',
            'Ion acoustic wave signatures in transverse expansion'
        ],
        'final_metrics': final_metrics,
        'training_history': trainer.training_history,
        'embedding_dimensions': 32,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'training_completed': datetime.now().isoformat()
    }
    
    with open('e3_byu_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
    logger.info("E3 Engine now capable of predicting BYU UNP anomalies")
    logger.info("Files saved: e3_byu_best_model.pt, e3_byu_training_results.json, e3_byu_training_curves.png")
    
    return model, trainer, results

if __name__ == "__main__":
    model, trainer, results = main()
