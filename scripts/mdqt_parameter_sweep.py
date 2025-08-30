#!/usr/bin/env python3
"""
MDQT Parameter Sweep Generator
Creates multiple MDQT simulations with different parameters for E3 training
"""

import numpy as np
import os
import subprocess
from pathlib import Path
import json

class MDQTParameterSweep:
    """Generate parameter sweeps for MDQT simulations"""
    
    def __init__(self, mdqt_code_path):
        self.mdqt_path = Path(mdqt_code_path)
        self.cpp_file = self.mdqt_path / "PlasmaMDQTSimulation.cpp"
        self.original_code = None
        
    def backup_original_code(self):
        """Backup the original C++ code"""
        if self.original_code is None:
            with open(self.cpp_file, 'r') as f:
                self.original_code = f.read()
    
    def modify_parameters(self, params):
        """Modify parameters in the C++ code"""
        
        if self.original_code is None:
            self.backup_original_code()
        
        modified_code = self.original_code
        
        # Parameter mapping from Python to C++ variable names
        param_mapping = {
            'density': 'density',
            'coupling_parameter': 'Ge', 
            'particles': 'N0',
            'max_time': 'tmax',
            'detuning': 'detuning',
            'rabi_frequency': 'Om',
            'save_directory': 'saveDirectory'
        }
        
        # Replace parameters in code
        for py_param, cpp_param in param_mapping.items():
            if py_param in params:
                value = params[py_param]
                
                # Handle string parameters
                if isinstance(value, str):
                    replacement = f'string {cpp_param} = "{value}";'
                else:
                    replacement = f'double {cpp_param} = {value};'
                
                # Find and replace parameter lines
                # This is a simplified approach - in practice, you'd want more robust parsing
                import re
                pattern = rf'(double|string|int)\s+{cpp_param}\s*=.*?;'
                if re.search(pattern, modified_code):
                    modified_code = re.sub(pattern, replacement, modified_code)
        
        # Write modified code
        with open(self.cpp_file, 'w') as f:
            f.write(modified_code)
    
    def compile_simulation(self):
        """Compile the MDQT simulation"""
        os.chdir(self.mdqt_path)
        result = subprocess.run([
            'g++', '-std=c++11', '-fopenmp', '-o', 'runFile', '-O3', 
            'PlasmaMDQTSimulation.cpp', '-lm', '-larmadillo'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed: {result.stderr}")
        
        return True
    
    def run_simulation(self, job_number=1):
        """Run a single simulation"""
        os.chdir(self.mdqt_path)
        result = subprocess.run(['./runFile', str(job_number)], 
                               capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode != 0:
            print(f"Simulation warning/error: {result.stderr}")
        
        return result.returncode == 0
    
    def generate_parameter_sets(self):
        """Generate systematic parameter variations for E3 training"""
        
        parameter_sets = []
        
        # Base parameters (typical UNP conditions)
        base_params = {
            'density': 2.0,           # 2 x 10^14 m^-3
            'coupling_parameter': 0.05, # Moderate coupling
            'particles': 1000,        # Manageable number for quick runs
            'max_time': 5.0,          # Short simulation time
            'detuning': 0.0,          # On resonance
            'rabi_frequency': 0.1,    # Moderate laser intensity
        }
        
        # Parameter variations to explore different physics regimes
        variations = {
            # Density sweep (affects plasma frequency, screening)
            'density_sweep': [
                {**base_params, 'density': d, 'save_directory': f'"density_{d}"'} 
                for d in [0.5, 1.0, 2.0, 5.0, 10.0]
            ],
            
            # Coupling parameter sweep (weak to strong coupling)
            'coupling_sweep': [
                {**base_params, 'coupling_parameter': g, 'save_directory': f'"coupling_{g}"'} 
                for g in [0.01, 0.03, 0.05, 0.08, 0.1]
            ],
            
            # Laser detuning sweep (different temperature regimes)
            'detuning_sweep': [
                {**base_params, 'detuning': d, 'save_directory': f'"detuning_{d}"'} 
                for d in [-10, -5, -1, 0, 1, 5, 10]
            ],
            
            # Laser intensity sweep (cooling efficiency)
            'intensity_sweep': [
                {**base_params, 'rabi_frequency': om, 'save_directory': f'"intensity_{om}"'} 
                for om in [0.01, 0.05, 0.1, 0.5, 1.0]
            ],
            
            # Combined sweeps for anomaly generation
            'anomaly_conditions': [
                # High density + strong coupling (strongly correlated regime)
                {**base_params, 'density': 10.0, 'coupling_parameter': 0.1, 
                 'save_directory': '"anomaly_strong_coupling"'},
                
                # Low density + weak coupling + off-resonant (expansion regime)
                {**base_params, 'density': 0.5, 'coupling_parameter': 0.01, 'detuning': -10,
                 'save_directory': '"anomaly_expansion"'},
                
                # High intensity + high coupling (nonlinear regime)
                {**base_params, 'coupling_parameter': 0.08, 'rabi_frequency': 1.0,
                 'save_directory': '"anomaly_nonlinear"'},
            ]
        }
        
        # Flatten all parameter sets
        for sweep_name, param_list in variations.items():
            for i, params in enumerate(param_list):
                params['sweep_type'] = sweep_name
                params['sweep_index'] = i
                parameter_sets.append(params)
        
        return parameter_sets
    
    def run_parameter_sweep(self, max_simulations=20):
        """Run systematic parameter sweep"""
        
        parameter_sets = self.generate_parameter_sets()
        
        # Limit number of simulations for testing
        if len(parameter_sets) > max_simulations:
            parameter_sets = parameter_sets[:max_simulations]
        
        results = []
        successful_runs = 0
        
        print(f"Starting parameter sweep with {len(parameter_sets)} simulations...")
        
        for i, params in enumerate(parameter_sets):
            print(f"\n=== Simulation {i+1}/{len(parameter_sets)} ===")
            print(f"Parameters: {params}")
            
            try:
                # Modify parameters
                self.modify_parameters(params)
                
                # Compile
                print("Compiling...")
                self.compile_simulation()
                
                # Run simulation
                print("Running simulation...")
                success = self.run_simulation(job_number=i+1)
                
                if success:
                    successful_runs += 1
                    print(f"✅ Simulation {i+1} completed successfully")
                else:
                    print(f"❌ Simulation {i+1} failed")
                
                results.append({
                    'simulation_id': i+1,
                    'parameters': params,
                    'success': success,
                    'output_directory': params.get('save_directory', f'job{i+1}')
                })
                
            except Exception as e:
                print(f"❌ Error in simulation {i+1}: {e}")
                results.append({
                    'simulation_id': i+1,
                    'parameters': params,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n=== Parameter Sweep Complete ===")
        print(f"Successful simulations: {successful_runs}/{len(parameter_sets)}")
        
        # Save results
        results_file = self.mdqt_path / "parameter_sweep_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Restore original code
        if self.original_code:
            with open(self.cpp_file, 'w') as f:
                f.write(self.original_code)
            print("Original code restored")
        
        return results

def quick_test_simulation(mdqt_path):
    """Run a single quick test simulation"""
    
    sweep = MDQTParameterSweep(mdqt_path)
    
    # Simple test parameters
    test_params = {
        'density': 2.0,
        'coupling_parameter': 0.05,
        'particles': 500,  # Small for quick test
        'max_time': 1.0,   # Very short
        'save_directory': '"test_run"'
    }
    
    print("Running quick test simulation...")
    
    try:
        sweep.modify_parameters(test_params)
        sweep.compile_simulation()
        success = sweep.run_simulation(job_number=1)
        
        if success:
            print("✅ Test simulation successful!")
            print(f"Output should be in: {mdqt_path}/test_run/job1/")
        else:
            print("❌ Test simulation failed")
        
        # Restore original code
        if sweep.original_code:
            with open(sweep.cpp_file, 'w') as f:
                f.write(sweep.original_code)
                
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mdqt_path = sys.argv[1]
    else:
        mdqt_path = "./external/mdqt-sim"
    
    print(f"MDQT Parameter Sweep Generator")
    print(f"MDQT Path: {mdqt_path}")
    
    # Run quick test first
    if quick_test_simulation(mdqt_path):
        print("\nReady for full parameter sweep!")
        print("Run: sweep = MDQTParameterSweep(mdqt_path)")
        print("     results = sweep.run_parameter_sweep()")
    else:
        print("\nTest failed. Check MDQT installation and dependencies.")
