#!/usr/bin/env python3
"""
MDQT Data Processor for E3 Engine
Processes output from Vrinceanu/Killian MDQT simulations for E3 training
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

class MDQTProcessor:
    """Process MDQT simulation output for E3 Engine training/validation"""
    
    def __init__(self, mdqt_output_dir):
        """
        Initialize processor for MDQT simulation output
        
        Args:
            mdqt_output_dir: Path to MDQT simulation output folder
        """
        self.output_dir = Path(mdqt_output_dir)
        self.simulation_params = {}
        self.time_series_data = {}
        self.physics_features = {}
        
    def load_simulation_data(self):
        """Load all MDQT output files"""
        
        # Load energy evolution data
        energies_file = self.output_dir / "energies.dat"
        if energies_file.exists():
            self.time_series_data['energies'] = pd.read_csv(
                energies_file, 
                sep='\t',
                names=['time', 'KE_x', 'KE_y', 'KE_z', 'PE', 'PE_change', 'v_exp_x']
            )
        
        # Load simulation parameters
        sim_params_files = list(self.output_dir.glob("simParams_timestep*.dat"))
        if sim_params_files:
            params_df = pd.read_csv(sim_params_files[0], sep='\t', header=None)
            self.simulation_params = dict(zip(params_df[0], params_df[1]))
            
        # Load final particle conditions
        conditions_files = list(self.output_dir.glob("conditions_timestep*.dat"))
        if conditions_files:
            self.time_series_data['final_conditions'] = pd.read_csv(
                conditions_files[0],
                sep='\t', 
                names=['x', 'y', 'z', 'vx', 'vy', 'vz']
            )
            
        # Load velocity distributions at different times
        vel_files = list(self.output_dir.glob("vel_distX_time*.dat"))
        self.time_series_data['velocity_distributions'] = {}
        for vel_file in vel_files:
            time_step = vel_file.stem.split('_')[-1]
            axis = vel_file.stem.split('_')[1][-1]  # Extract x, y, or z
            self.time_series_data['velocity_distributions'][f'{axis}_{time_step}'] = pd.read_csv(
                vel_file, sep='\t', names=['velocity', 'probability']
            )
    
    def extract_physics_features(self):
        """Extract physics features for E3 Engine"""
        
        if 'energies' not in self.time_series_data:
            raise ValueError("Energy data not loaded. Run load_simulation_data() first.")
            
        energies = self.time_series_data['energies']
        
        # Basic plasma parameters
        density = float(self.simulation_params.get('density', 0)) * 1e14  # Convert to m^-3
        ge = float(self.simulation_params.get('Ge', 0))
        n_particles = int(self.simulation_params.get('N0', 0))
        
        # Calculate derived parameters
        # Wigner-Seitz radius
        a_ws = (3/(4*np.pi*density))**(1/3)
        
        # Plasma frequency (assuming typical values)
        e = 1.602e-19  # Elementary charge
        eps0 = 8.854e-12  # Vacuum permittivity  
        m_ion = 87.6 * 1.66e-27  # Strontium mass in kg
        omega_pi = np.sqrt(density * e**2 / (eps0 * m_ion))
        
        # Temperature from Coulomb coupling
        k_B = 1.381e-23
        T_e = e**2 / (4*np.pi*eps0*k_B*a_ws*ge) if ge > 0 else 1.0
        
        # Extract time evolution features
        total_energy = energies['KE_x'] + energies['KE_y'] + energies['KE_z'] + energies['PE']
        expansion_velocity = energies['v_exp_x'].iloc[-1] if len(energies) > 0 else 0
        
        # Detect anomalous behavior
        # Look for non-monotonic energy evolution (sign of complex dynamics)
        energy_gradient = np.gradient(total_energy)
        anomaly_score = np.std(energy_gradient) / np.mean(np.abs(energy_gradient)) if np.mean(np.abs(energy_gradient)) > 0 else 0
        
        # Check for disorder-induced heating signature
        initial_ke = energies[['KE_x', 'KE_y', 'KE_z']].iloc[0].sum()
        final_ke = energies[['KE_x', 'KE_y', 'KE_z']].iloc[-1].sum()
        heating_ratio = final_ke / initial_ke if initial_ke > 0 else 1.0
        
        self.physics_features = {
            # Basic parameters
            'density': density,
            'coupling_parameter': ge,
            'temperature': T_e,
            'n_particles': n_particles,
            'plasma_frequency': omega_pi,
            'debye_length': np.sqrt(eps0*k_B*T_e/(density*e**2)),
            
            # Dynamic properties
            'expansion_velocity': expansion_velocity,
            'final_total_energy': total_energy.iloc[-1] if len(total_energy) > 0 else 0,
            'heating_ratio': heating_ratio,
            'anomaly_score': anomaly_score,
            
            # Time evolution characterization
            'simulation_time': energies['time'].iloc[-1] if len(energies) > 0 else 0,
            'energy_change_rate': np.mean(np.gradient(total_energy)),
            
            # Laser parameters
            'detuning': float(self.simulation_params.get('detuning', 0)),
            'rabi_frequency': float(self.simulation_params.get('Om', 0)),
        }
        
        return self.physics_features
    
    def create_e3_graph(self):
        """Create PyTorch Geometric graph for E3 Engine"""
        
        if not self.physics_features:
            self.extract_physics_features()
            
        # Create node features (representing the plasma as a system)
        # Each "node" represents a different aspect of the plasma state
        node_features = torch.tensor([
            [self.physics_features['density']],
            [self.physics_features['temperature']], 
            [self.physics_features['coupling_parameter']],
            [self.physics_features['plasma_frequency']],
            [self.physics_features['debye_length']],
            [self.physics_features['expansion_velocity']],
            [self.physics_features['heating_ratio']],
            [self.physics_features['anomaly_score']]
        ], dtype=torch.float)
        
        # Create fully connected edge structure (plasma components interact)
        num_nodes = node_features.shape[0]
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        # Make edges bidirectional
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Edge features (physical relationships between components)
        edge_attr = torch.ones(edge_index.shape[1], 1)
        
        # Global features (system-level properties)
        global_features = torch.tensor([
            self.physics_features['n_particles'],
            self.physics_features['simulation_time'],
            self.physics_features['detuning'],
            self.physics_features['rabi_frequency']
        ]).unsqueeze(0)
        
        # Create target (what we want to predict)
        # For now, let's predict whether this shows anomalous behavior
        anomaly_threshold = 0.5  # Tunable threshold
        target = torch.tensor([1.0 if self.physics_features['anomaly_score'] > anomaly_threshold else 0.0])
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target,
            global_attr=global_features
        )
        
        return data
    
    def generate_training_dataset(self, output_file='mdqt_training_data.json'):
        """Generate training dataset entry"""
        
        features = self.extract_physics_features()
        graph_data = self.create_e3_graph()
        
        # Create training data entry
        training_entry = {
            'source': 'MDQT_simulation',
            'simulation_params': self.simulation_params,
            'physics_features': features,
            'anomaly_detected': bool(features['anomaly_score'] > 0.5),
            'data_quality': 'high',  # Simulation data is clean
            'metadata': {
                'processor_version': '1.0',
                'extraction_date': pd.Timestamp.now().isoformat(),
                'simulation_type': 'molecular_dynamics_quantum_trajectories'
            }
        }
        
        return training_entry, graph_data
    
    def visualize_results(self, save_path=None):
        """Create visualization of MDQT results"""
        
        if 'energies' not in self.time_series_data:
            print("No energy data to visualize")
            return
            
        energies = self.time_series_data['energies']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Energy evolution
        axes[0,0].plot(energies['time'], energies['KE_x'] + energies['KE_y'] + energies['KE_z'], 
                      label='Kinetic Energy')
        axes[0,0].plot(energies['time'], energies['PE'], label='Potential Energy')
        axes[0,0].set_xlabel('Time (ω_pE^-1)')
        axes[0,0].set_ylabel('Energy (E_c)')
        axes[0,0].set_title('Energy Evolution')
        axes[0,0].legend()
        
        # Expansion velocity
        axes[0,1].plot(energies['time'], energies['v_exp_x'])
        axes[0,1].set_xlabel('Time (ω_pE^-1)')
        axes[0,1].set_ylabel('Expansion Velocity')
        axes[0,1].set_title('Plasma Expansion')
        
        # Physics parameters summary
        params_text = f"""
        Density: {self.physics_features.get('density', 0):.2e} m⁻³
        Temperature: {self.physics_features.get('temperature', 0):.1f} K
        Coupling Γ: {self.physics_features.get('coupling_parameter', 0):.3f}
        Anomaly Score: {self.physics_features.get('anomaly_score', 0):.3f}
        """
        axes[1,0].text(0.1, 0.5, params_text, transform=axes[1,0].transAxes, 
                      fontsize=10, verticalalignment='center')
        axes[1,0].set_title('Physics Parameters')
        axes[1,0].axis('off')
        
        # Anomaly detection result
        anomaly_detected = self.physics_features.get('anomaly_score', 0) > 0.5
        axes[1,1].bar(['Normal', 'Anomalous'], 
                     [0 if anomaly_detected else 1, 1 if anomaly_detected else 0],
                     color=['green' if not anomaly_detected else 'lightgreen',
                           'red' if anomaly_detected else 'lightcoral'])
        axes[1,1].set_title('E3 Anomaly Detection')
        axes[1,1].set_ylabel('Classification')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def batch_process_mdqt_simulations(simulations_dir, output_file='mdqt_dataset.json'):
    """Process multiple MDQT simulation outputs"""
    
    simulations_path = Path(simulations_dir)
    dataset = []
    
    # Find all job directories
    job_dirs = [d for d in simulations_path.iterdir() if d.is_dir() and d.name.startswith('job')]
    
    print(f"Found {len(job_dirs)} MDQT simulation outputs")
    
    for job_dir in job_dirs:
        try:
            processor = MDQTProcessor(job_dir)
            processor.load_simulation_data()
            
            training_entry, graph_data = processor.generate_training_dataset()
            dataset.append(training_entry)
            
            print(f"Processed {job_dir.name}: Anomaly={training_entry['anomaly_detected']}")
            
        except Exception as e:
            print(f"Error processing {job_dir.name}: {e}")
    
    # Save dataset
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated dataset with {len(dataset)} entries saved to {output_file}")
    return dataset

if __name__ == "__main__":
    # Example usage
    print("MDQT Processor for E3 Engine")
    print("Usage: processor = MDQTProcessor('path/to/mdqt/output')")
    print("       processor.load_simulation_data()")
    print("       features = processor.extract_physics_features()")
    print("       graph_data = processor.create_e3_graph()")
