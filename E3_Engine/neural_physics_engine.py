# neural_physics_engine.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from davp_utils import setup_logger

class PhysicsEngine(nn.Module):
    """Simple neural network to learn plasma physics"""
    
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super(PhysicsEngine, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class PhysicsTrainer:
    """Train and validate physics predictions"""
    
    def __init__(self):
        self.logger = setup_logger("physics_trainer")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🖥️  Using device: {self.device}")
        
    def load_test_data(self):
        """Load the generated physics test data"""
        with open('debye_test_data.json', 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data_points'])
        self.logger.info(f"📊 Loaded {len(df)} data points")
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for neural network training"""
        
        # Features: temperature and density
        X = df[['temperature', 'density']].values
        
        # Target: Debye length
        y = df['debye_length_true'].values.reshape(-1, 1)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features (important for neural networks!)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(self.device)
        
        return (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, 
                X_test, y_test)  # Keep unscaled for comparison
    
    def train_physics_engine(self, epochs=1000):
        """Train the neural network to learn Debye physics"""
        
        self.logger.info("🚀 Starting physics engine training...")
        
        # Load and prepare data
        df = self.load_test_data()
        X_train, X_test, y_train, y_test, X_test_raw, y_test_raw = self.prepare_data(df)
        
        # Create model
        model = PhysicsEngine().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
        
        # Test the model
        model.eval()
        with torch.no_grad():
            test_predictions_scaled = model(X_test)
            test_predictions = self.scaler_y.inverse_transform(
                test_predictions_scaled.cpu().numpy()
            )
        
        # Calculate accuracy
        relative_errors = np.abs(test_predictions.flatten() - y_test_raw.flatten()) / y_test_raw.flatten()
        mean_error = np.mean(relative_errors) * 100
        
        self.logger.info(f"🎯 Training complete!")
        self.logger.info(f"📊 Mean relative error: {mean_error:.2f}%")
        
        # Show some predictions vs true values
        print("\n🔬 PHYSICS ENGINE PREDICTIONS vs TRUE VALUES:")
        print(f"{'True λD (m)':<15} {'Predicted λD (m)':<15} {'Error %':<10}")
        print("-" * 45)
        
        for i in range(min(10, len(test_predictions))):
            true_val = y_test_raw.flatten()[i]
            pred_val = test_predictions.flatten()[i]
            error_pct = abs(pred_val - true_val) / true_val * 100
            
            print(f"{true_val:<15.2e} {pred_val:<15.2e} {error_pct:<10.2f}")
        
        return model, losses, mean_error

if __name__ == "__main__":
    trainer = PhysicsTrainer()
    
    print("🧠 PHYSICS ENGINE TRAINING")
    print("Goal: Learn λD = sqrt(ε₀kT / (n*e²)) from data")
    print("=" * 50)
    
    # Train the physics engine
    model, losses, error = trainer.train_physics_engine()
    
    print(f"\n🎉 PHYSICS ENGINE TRAINED!")
    print(f"🎯 Can predict Debye length with {error:.1f}% average error")
    print(f"🧠 Your engine just learned fundamental plasma physics!")
