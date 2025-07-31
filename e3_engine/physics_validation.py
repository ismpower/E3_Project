# physics_validation.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from davp_utils import setup_logger
import json

class PlasmaPhysicsValidator:
    """Validate engine against known plasma physics"""
    
    def __init__(self):
        self.logger = setup_logger("physics_validator")
        
        # Physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.e = 1.602176634e-19  # Elementary charge
        self.epsilon_0 = 8.8541878128e-12  # Vacuum permittivity
    
    def calculate_debye_length(self, temperature, density):
        """Calculate theoretical Debye screening length"""
        # λD = sqrt(ε₀kT / (n*e²))
        lambda_d = np.sqrt(
            (self.epsilon_0 * self.k_B * temperature) / (density * self.e**2)
        )
        return lambda_d
    
    def generate_known_data(self, n_points=50):
        """Generate dataset with known Debye length relationships"""
        
        # Create realistic plasma parameter ranges
        temperatures = np.logspace(2, 4, n_points)  # 100K to 10,000K
        densities = np.logspace(18, 22, n_points)   # 10^18 to 10^22 m^-3
        
        data = []
        for i in range(n_points):
            T = temperatures[i % len(temperatures)]
            n = densities[i % len(densities)]
            
            # Calculate TRUE Debye length (what we want to predict)
            lambda_d_true = self.calculate_debye_length(T, n)
            
            data.append({
                'temperature': T,
                'density': n,
                'debye_length_true': lambda_d_true,
                'point_id': i
            })
        
        return pd.DataFrame(data)
    
    def test_known_physics(self):
        """Test engine against known Debye screening physics"""
        
        self.logger.info("🔬 Generating known physics test data...")
        
        # Generate test dataset
        df = self.generate_known_data(100)
        
        # Show some examples
        print("📊 Known Physics Test Data (first 5 points):")
        print(f"{'Temp (K)':<10} {'Density (m⁻³)':<15} {'λD (m)':<15}")
        print("-" * 50)
        
        for i in range(5):
            row = df.iloc[i]
            print(f"{row['temperature']:<10.1f} {row['density']:<15.2e} {row['debye_length_true']:<15.2e}")
        
        # Save test data
        test_file = "debye_test_data.json"
        test_data = {
            'description': 'Known Debye screening length test data',
            'physics_law': 'λD = sqrt(ε₀kT / (n*e²))',
            'data_points': df.to_dict('records')
        }
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        self.logger.info(f"💾 Saved test data to: {test_file}")
        
        return df

if __name__ == "__main__":
    validator = PlasmaPhysicsValidator()
    
    print("🎯 PHYSICS VALIDATION TEST")
    print("Testing: Debye Screening Length Prediction")
    print("Known Law: λD = sqrt(ε₀kT / (n*e²))")
    print()
    
    # Generate known data
    test_df = validator.test_known_physics()
    
    print(f"\n✅ Generated {len(test_df)} test points")
    print("🎯 Next: Build simple neural network to predict λD from T and n")
    print("🎯 Then: Compare predictions vs. known physics!")
