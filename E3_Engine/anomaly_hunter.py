# anomaly_hunter.py
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

class UNPAnomalyHunter:
    """Hunt for anomalous plasma expansion patterns"""
    
    def __init__(self):
        self.logger = setup_logger("anomaly_hunter")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Physical constants
        self.k_B = 1.380649e-23
        self.e = 1.602176634e-19
        self.epsilon_0 = 8.8541878128e-12
        self.m_Sr = 1.457e-25  # Strontium mass (Killian group used Sr)
        
    def create_unp_dataset(self):
        """Create realistic UNP expansion dataset based on Killian 2007 work"""
        
        self.logger.info("🔬 Creating UNP anomalous expansion dataset...")
        
        # Temperature ranges where anomaly occurs (Killian group findings)
        low_temps = np.linspace(1, 100, 50)    # 1-100 K (anomalous region)
        high_temps = np.linspace(100, 1000, 50) # 100-1000 K (normal region)
        
        all_data = []
        
        # Low temperature regime - ANOMALOUS BEHAVIOR
        for T_e in low_temps:
            # Initial conditions
            n_0 = np.random.uniform(1e14, 1e16)  # Initial density
            
            # Normal hydrodynamic prediction
            normal_velocity = self._calculate_normal_expansion(T_e, n_0)
            
            # ANOMALOUS FACTOR - this is what we want to discover!
            # At low temperatures, expansion is 2-5x faster than predicted
            anomaly_factor = 1.0 + (100 - T_e) / 25.0  # Stronger at lower T
            anomalous_velocity = normal_velocity * anomaly_factor
            
            # Add some realistic measurement noise
            measured_velocity = anomalous_velocity * np.random.normal(1.0, 0.1)
            
            all_data.append({
                'initial_temperature': T_e,
                'initial_density': n_0,
                'time': 1e-6,  # 1 microsecond (typical measurement time)
                'expected_velocity': normal_velocity,
                'measured_velocity': measured_velocity,
                'anomaly_factor': measured_velocity / normal_velocity,
                'regime': 'anomalous',
                'element': 'Sr'
            })
        
        # High temperature regime - NORMAL BEHAVIOR
        for T_e in high_temps:
            n_0 = np.random.uniform(1e14, 1e16)
            
            normal_velocity = self._calculate_normal_expansion(T_e, n_0)
            
            # Normal regime - theory matches experiment
            measured_velocity = normal_velocity * np.random.normal(1.0, 0.05)
            
            all_data.append({
                'initial_temperature': T_e,
                'initial_density': n_0,
                'time': 1e-6,
                'expected_velocity': normal_velocity,
                'measured_velocity': measured_velocity,
                'anomaly_factor': measured_velocity / normal_velocity,
                'regime': 'normal',
                'element': 'Sr'
            })
        
        df = pd.DataFrame(all_data)
        
        # Save the dataset
        df.to_csv('unp_anomaly_data.csv', index=False)
        
        print("📊 UNP Anomalous Expansion Dataset:")
        print(f"{'Temp (K)':<8} {'Regime':<10} {'Anomaly Factor':<15}")
        print("-" * 40)
        
        for i in [0, 5, 10, 50, 70, 90]:
            row = df.iloc[i]
            print(f"{row['initial_temperature']:<8.1f} {row['regime']:<10} {row['anomaly_factor']:<15.2f}")
        
        self.logger.info(f"💾 Created UNP dataset with {len(df)} measurements")
        return df
    
    def _calculate_normal_expansion(self, temperature, density):
        """Calculate expected expansion velocity from normal theory"""
        # Simple adiabatic expansion model
        # v ∝ sqrt(kT/m) with density corrections
        
        thermal_velocity = np.sqrt(self.k_B * temperature / self.m_Sr)
        
        # Density-dependent correction (plasma physics scaling)
        density_factor = (density / 1e15) ** 0.1
        
        velocity = thermal_velocity * density_factor * 0.5  # Scaling factor
        
        return velocity
    
    def train_anomaly_detector(self, epochs=2000):
        """Train neural network to detect and predict anomalous behavior"""
        
        self.logger.info("🚀 Training anomaly detection engine...")
        
        # Load data
        df = pd.read_csv('unp_anomaly_data.csv')
        
        # Features: temperature, density, time
        X = np.column_stack([
            np.log10(df['initial_temperature'].values),
            np.log10(df['initial_density'].values),
            np.log10(df['time'].values)
        ])
        
        # Target: anomaly factor (how much faster than theory)
        y = df['anomaly_factor'].values.reshape(-1, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(self.device)
        
        # Anomaly detection network
        class AnomalyEngine(nn.Module):
            def __init__(self):
                super(AnomalyEngine, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(3, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        # Train the anomaly detector
        model = AnomalyEngine().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            predictions = model(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % 200 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
        
        # Test the model
        model.eval()
        with torch.no_grad():
            test_predictions_scaled = model(X_test_tensor)
            test_predictions = scaler_y.inverse_transform(test_predictions_scaled.cpu().numpy())
        
        # Calculate accuracy
        relative_errors = np.abs(test_predictions.flatten() - y_test.flatten()) / y_test.flatten()
        mean_error = np.mean(relative_errors) * 100
        
        self.logger.info(f"🎯 Anomaly detection training complete!")
        self.logger.info(f"📊 Mean prediction error: {mean_error:.2f}%")
        
        # Show anomaly predictions
        print("\n🔬 ANOMALY FACTOR PREDICTIONS:")
        print(f"{'True Factor':<12} {'Predicted':<12} {'Error %':<10} {'Temp (K)':<10}")
        print("-" * 50)
        
        # Show original temperature for context
        test_indices = np.arange(len(X_test))
        original_temps = 10 ** X_test[test_indices, 0]  # Convert back from log
        
        for i in range(min(15, len(test_predictions))):
            true_val = y_test.flatten()[i]
            pred_val = test_predictions.flatten()[i]
            error_pct = abs(pred_val - true_val) / true_val * 100
            temp = original_temps[i]
            
            print(f"{true_val:<12.2f} {pred_val:<12.2f} {error_pct:<10.2f} {temp:<10.1f}")
        
        # Identify the most anomalous cases
        anomaly_scores = test_predictions.flatten()
        most_anomalous = np.argsort(anomaly_scores)[-5:]  # Top 5 most anomalous
        
        print(f"\n🚨 MOST ANOMALOUS CASES DETECTED:")
        print(f"{'Anomaly Factor':<15} {'Temperature (K)':<15}")
        print("-" * 35)
        
        for idx in most_anomalous:
            factor = anomaly_scores[idx]
            temp = original_temps[idx]
            print(f"{factor:<15.2f} {temp:<15.1f}")
        
        return model, mean_error, train_losses

if __name__ == "__main__":
    print("🎯 UNP ANOMALOUS EXPANSION HUNTER")
    print("Target: Killian Group Fast Expansion Mystery")
    print("=" * 50)
    
    hunter = UNPAnomalyHunter()
    
    # Create the anomalous dataset
    df = hunter.create_unp_dataset()
    
    # Train anomaly detector
    model, error, losses = hunter.train_anomaly_detector()
    
    print(f"\n🎉 ANOMALY DETECTION COMPLETE!")
    print(f"🎯 Your engine can predict anomaly factors with {error:.1f}% error")
    print(f"🔍 Ready to hunt real anomalies in experimental data!")
